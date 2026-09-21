"""H0 方法侧（`src/distributed/attention_hook.py`）的锚点 —— 纯 CPU，不需要 GPU。

为什么这些能在本机验证
----------------------
钩子是 `device-agnostic` 的，而 transformers 可以用一个**本地构造的 tiny Llama**
（`LlamaConfig` + 随机权重，不下载任何东西）在 CPU 上跑完整前向。于是本机能问出
一件 GPU 机上要花很久才问得清的事：**接线到底对不对**（形状、转置、GQA 分组、
位置、因果、还原、本地块契约）。

最强的是两条端到端数值断言：

- `test_h1_...`：dense 参照臂（`identity_compact`，β ≡ 0、B = L_s）必须复现原生
  HF 的前向。它一条就同时覆盖了形状/转置/GQA/位置/因果/还原。
- `test_h11_...`：目的端**带本地上下文**（分三次前向：源段 → 本地段 → continuation）
  仍必须复现"把三段拼起来一次前向"的原生结果。它覆盖的是 2026-09-21 新增的
  **本地块契约**（本地块 = 传入 cache 全体），旧契约（取尾部 Lq）会把它算错。

容差为什么不是"逐位"
--------------------
钩子走 `attention_kernel` 的**显式**路径（matmul + logsumexp + matmul），而原生
HF 走 SDPA 的**融合核**。两条路径的浮点求和顺序不同 ⇒ 只能 ULP 级一致。
实测（float32，S=24 / dest=6）：最大绝对差 1.5e-07（约 2 ULP）。
若哪天真变成逐位相同，那说明有一侧的核被换掉了，值得回头看一眼。
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from experiments.gpu import _hf  # noqa: E402
from src.distributed import attention_hook as A  # noqa: E402

TINY = dict(
    vocab_size=64, hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=512,
)
VOCAB = TINY["vocab_size"]
SRC_LEN, DST_LEN = 24, 6
BUDGET = 12

# 实测值（float32）：dense 臂与原生前向的路径差 1.5e-07。留一倍余量放进断言，
# 但**不放到 1e-4** —— 那会把"某个 head 走错了"这种量级的错放过去。
NATIVE_PATH_ATOL = 1e-6


@pytest.fixture(scope="module")
def tiny():
    """本地构造的 tiny Llama（随机权重，无需网络）。"""
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**TINY)).eval()


def _ids(seed: int, n: int = SRC_LEN + DST_LEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, n), generator=g)


def native_logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def run_two_phase(model, ids, *, mode, budget=None, budget_ratio=None,
                  dcc_world=1, budget_mode="per_edge"):
    """按钩子的两段式跑一遍，返回 (dst 段 logits, state)。

    ⚠️ 目的端前向**不传 past**（`past_key_values=None`）：本模块的契约是
    「传入的 cache 只含目的端本地 KV」，源端一律由 state 携带。传 `past=None`
    既是最小化的写法，也正好覆盖这条契约。
    """
    src, dst = ids[:, :SRC_LEN], ids[:, SRC_LEN:]
    pos = torch.arange(SRC_LEN, SRC_LEN + DST_LEN).unsqueeze(0)
    cfg = A.HookConfig(budget=budget, budget_ratio=budget_ratio, mode=mode,
                       dcc_world=dcc_world, budget_mode=budget_mode)
    with torch.no_grad():
        with A.dcc_attention(model, cfg) as st:
            st.source_phase()
            model(src, use_cache=True)
            st.destination_phase(compacts="build")
            second = model(dst, use_cache=True, position_ids=pos)
    return second.logits, st


# =============================================================================
# A. 接线正确性
# =============================================================================

def test_h1_dense_reference_arm_reproduces_native_forward(tiny):
    """dense 参照臂（β ≡ 0、B = L_s）必须复现原生 HF 的目的端输出。

    这是本文件最强的一条。形状错、转置错、GQA 分组错、位置错、因果漏加 ——
    任何一种都会让输出偏离到远大于 1e-7，而它们各自的"局部检查"都很难写。
    用一条端到端数值断言一次盖住，是这里唯一有判别力的做法。

    参照臂必须用 `identity_compact` 而**不是** `build_compact_kv(budget=L_s)`：
    后者带 ridge 正则，解不是恒等，于是"参照"自己就带残差。
    """
    ids = _ids(1)
    base = native_logits(tiny, ids)
    logits, st = run_two_phase(tiny, ids, mode="dense", budget=SRC_LEN)

    assert logits.shape == base[:, SRC_LEN:, :].shape, \
        "形状必须与 HF 期望的 [B, Lq, H_q * D_v] 一致"
    dev = (logits - base[:, SRC_LEN:, :]).abs().max().item()
    assert dev < NATIVE_PATH_ATOL, (
        f"dense 参照臂与原生前向差 {dev:.3e}，超过 {NATIVE_PATH_ATOL:.0e}。"
        "两者只应差一条算子路径的 ULP（实测 1.5e-07）——超出说明接线错了："
        "优先查 GQA 分组（必须按 repeat_interleave 连续切）、"
        "返回值转置（必须是 [B, Lq, H*D]）、以及源端的因果掩码有没有施加。"
    )
    assert st.summary()["n_compacts_built"] == 4      # 2 层 × 2 kv head
    assert st.summary()["n_local_blocks"] == 4


def test_h2_compaction_actually_changes_the_output(tiny):
    """压缩臂的输出必须**明显**偏离参照臂与原生前向。

    这条防的是"钩子空转"：若某天 compact 块被静默跳过（或 budget 被当成
    L_s），`dcc_kv` 行会退化成 dense 行，而所有形状类断言照样全绿。
    量级判据取 1e-4（实测偏离 5.8e-03），比路径差 1.5e-07 高四个数量级 ——
    两个量不在同一档，不会互相冒充。
    """
    ids = _ids(2)
    base = native_logits(tiny, ids)
    dense, _ = run_two_phase(tiny, ids, mode="dense", budget=SRC_LEN)
    comp, st = run_two_phase(tiny, ids, mode="dcc_kv", budget=BUDGET)

    assert (comp - base[:, SRC_LEN:, :]).abs().max().item() > 1e-4, \
        "压缩臂与原生前向几乎没有差别 —— 压缩没生效（钩子空转）"
    assert (comp - dense).abs().max().item() > 1e-4, \
        "压缩臂与 dense 参照臂几乎一致 —— 预算没被真正施加"
    assert st.summary()["n_compacts_built"] == 4


def test_h3_phases_and_compacts_source_must_be_declared_explicitly():
    """阶段是显式声明的；目的端的"块从哪来"也必须显式。

    不能靠"`past_key_values` 是不是 None"去猜：把目的端当源端时，钩子会把
    紧凑块又存一份进 state，产物看着完全正常（模块文档第 2 条）。
    也不能给 `compacts` 一个默认值：评测路径上"用本次 query 条件化"可能是
    泄漏（本次 query 就是被打分的 token），"复用"可能是陈旧块 ——
    两个方向的猜错都不报错（模块文档第 3 条）。
    """
    class _Mod:
        layer_idx = 0

    st = A.DccAttentionState(A.HookConfig(budget=SRC_LEN, mode="dense"))
    hook = A.make_hook(st)
    q = torch.randn(1, 4, 4, 16)
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)

    with pytest.raises(RuntimeError, match="不在任何阶段"):
        hook(_Mod(), q, k, v, scaling=0.25)

    with pytest.raises(ValueError, match="源端"):
        st.destination_phase(compacts="build")

    # compacts 必须显式给（没有默认值）
    with pytest.raises(TypeError):
        st.destination_phase()

    with pytest.raises(ValueError, match="compacts 必须"):
        st.destination_phase(compacts="shared")

    # 跑了源端之后才能进目的端；选了 reuse 但没 build 过也要拦
    st.source_phase()
    hook(_Mod(), q, k, v, scaling=0.25)
    with pytest.raises(ValueError, match="reuse"):
        st.destination_phase(compacts="reuse")
    st.destination_phase(compacts="build")
    assert st.n_destination_calls == 0


def test_h4_registry_and_config_are_restored_and_never_overwrite_eager(tiny):
    """退出后必须严格还原；且**不得**覆盖 `eager` / `sdpa` 这类全局键。

    踩过的形态：`register("eager", fn)` 改的是所有模型的实现，而"还原"写成
    `pop("eager")` 会把 transformers 自带实现**删掉** —— 不是还原，是破坏。
    故本模块注册专用键 `HOOK_KEY`，只改 config 指过去。
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as REG

    impl_before = tiny.config._attn_implementation
    eager_before = REG.get("eager", None)
    sdpa_before = REG.get("sdpa", None)

    with A.dcc_attention(tiny, A.HookConfig(budget=SRC_LEN, mode="dense")):
        assert tiny.config._attn_implementation == A.HOOK_KEY
        assert REG.get("eager", None) is eager_before, "eager 被覆盖了"
        assert REG.get("sdpa", None) is sdpa_before, "sdpa 被覆盖了"

    assert tiny.config._attn_implementation == impl_before
    assert REG.get("eager", None) is eager_before
    assert REG.get("sdpa", None) is sdpa_before

    # 还原之后必须逐位回到原来的数值路径（不只是"看起来还原了"）
    ids = _ids(3)
    assert torch.equal(native_logits(tiny, ids), native_logits(tiny, ids))

    # 嵌套/重复进入必须拒绝，而不是让两个 state 抢同一个键
    with A.dcc_attention(tiny, A.HookConfig(budget=SRC_LEN, mode="dense")):
        with pytest.raises(ValueError, match="嵌套"):
            with A.dcc_attention(tiny, A.HookConfig(budget=SRC_LEN, mode="dense")):
                pass


def test_h5_query_groups_split_head_dim_contiguously():
    """GQA 分组必须是**连续切分且完整覆盖**（= `repeat_interleave` 的语义）。

    若改成"整块复制"（`repeat`）的语义，形状仍然完全正确、数值全错。
    """
    q = torch.arange(2 * 4 * 3 * 2, dtype=torch.float64).reshape(2, 4, 3, 2)
    groups = A.query_groups(q, 2)
    assert len(groups) == 2 and groups[0].shape == (2, 2, 3, 2)

    seen = []
    for h, grp in enumerate(groups):
        for i in range(grp.shape[1]):
            assert torch.equal(grp[0, i], q[0, h * 2 + i])
            seen.append(h * 2 + i)
    assert sorted(seen) == [0, 1, 2, 3], "分组必须恰好覆盖每个 query head 一次"

    with pytest.raises(ValueError, match="整除"):
        A.query_groups(q, 3)


def test_h6_a_fully_blocked_key_column_is_refused():
    """整列被遮蔽（padding / 空块）必须抛错 —— 钩子不消费 mask，看不见就等于没有。"""
    ok = torch.zeros(1, 1, 3, 4)
    A._assert_no_fully_blocked_key(ok)          # 全 0 的掩码：不抛
    A._assert_no_fully_blocked_key(None)

    bad = ok.clone()
    bad[..., 1] = float("-inf")                 # 第 1 个 key 对所有 query 都不可见
    with pytest.raises(ValueError, match="padding"):
        A._assert_no_fully_blocked_key(bad)

    # 纯因果（下三角保留）不该被误报：最后一列总被最后一个 query 看见
    causal = torch.triu(torch.full((1, 1, 4, 4), float("-inf")), diagonal=1)
    A._assert_no_fully_blocked_key(causal)


# =============================================================================
# B. 单卡模拟的切分与预算口径
# =============================================================================

def test_h7_source_partition_is_even_and_covers_everything():
    """分段必须均分（前 rem 段各多 1）、无缝覆盖、且段数 = world。

    与 `_forward.chunk_source_sizes` 同款约定 —— 两处不一致会让"每段长度"
    在两套代码里不同值，而两边都自称在同一 world 下。
    """
    for length, world in ((12, 4), (13, 4), (24, 3), (16, 8), (7, 1)):
        segs = A.source_partition(length, world)
        assert len(segs) == world
        sizes = [e - s for s, e in segs]
        assert sum(sizes) == length
        assert max(sizes) - min(sizes) <= 1
        assert all(sizes[i] >= 1 for i in range(world))
        assert segs[0][0] == 0 and segs[-1][1] == length
        for (s0, e0), (s1, _) in zip(segs, segs[1:]):
            assert e0 == s1, "分段必须无缝"

    with pytest.raises(ValueError, match="空段"):
        A.source_partition(3, 4)


def test_h8_budget_modes_ratio_and_the_identity_short_circuit():
    """预算口径：绝对/相对两种给法、`per_edge` 为主线、`total` 只作敏感性对照。

    `total` 会让总预算随 world 放大 —— 那是"花更多 KV 换质量"，不是机制收益。
    summary 里必须把这条限制写进产物，而不是只写在文档里。
    """
    per = A.HookConfig(budget=64, dcc_world=4, budget_mode="per_edge")
    assert [per.edge_budget(n, 100) for n in (100, 16, 15, 1)] == [16, 16, 15, 1]

    tot = A.HookConfig(budget=64, dcc_world=4, budget_mode="total")
    assert [tot.edge_budget(n, 100) for n in (100, 16)] == [64, 16]

    # 相对口径：每次按当时的源长解析（评测路径的源长逐样本变化）
    rel = A.HookConfig(budget_ratio=0.05, dcc_world=2)
    assert rel.resolve_budget(1000) == 50
    assert rel.resolve_budget(100) == 5
    assert rel.edge_budget(500, 1000) == 25
    # 源太短时解析出来的总预算会 < world ⇒ 必须**报错**而不是把每边夹到 1
    # （夹上去会让实际总预算超出声明值，"每边 B_total/world" 这个口径就不再成立）
    with pytest.raises(ValueError, match="per_edge"):
        rel.resolve_budget(10)                  # round(0.5) = 0 ⇒ 夹到 1 < 2

    # 互斥：两个都给 / 都不给都要拦
    with pytest.raises(ValueError, match="恰好给一个"):
        A.HookConfig(budget=8, budget_ratio=0.5)
    with pytest.raises(ValueError, match="恰好给一个"):
        A.HookConfig()
    with pytest.raises(ValueError, match="budget_ratio"):
        A.HookConfig(budget_ratio=1.5)

    assert "主口径" in A.DccAttentionState(per).summary()["budget_claim_scope"]
    assert "不得" in A.DccAttentionState(tot).summary()["budget_claim_scope"]

    # 每边至少 1 个 key ⇒ 总预算必须 ≥ world（绝对口径在构造时就能查）
    with pytest.raises(ValueError, match="per_edge"):
        A.HookConfig(budget=2, dcc_world=4, budget_mode="per_edge")
    # 相对口径只能在拿到源长时才查得出来 —— 所以它必须在 resolve_budget 里也查
    with pytest.raises(ValueError, match="per_edge"):
        A.HookConfig(budget_ratio=0.01, dcc_world=4, budget_mode="per_edge") \
            .resolve_budget(100)                 # round(1.0) = 1 < 4

    # 预算跨过段长时走 identity：保证"无压缩"这一档**不带**构造链的 ridge 残差
    tiny_cfg = A.HookConfig(budget=8, dcc_world=1, mode="dcc_kv")
    keys = torch.randn(8, 16)
    vals = torch.randn(8, 16)
    qs = torch.randn(5, 16)
    blocks = A.build_edge_compacts(keys, vals, qs, tiny_cfg)
    assert len(blocks) == 1
    assert int(blocks[0].keys.shape[0]) == 8
    assert torch.equal(blocks[0].logit_bias, torch.zeros(8)), \
        "预算 = 段长时必须走 identity_compact（β≡0），否则性质 1 的检验失效"


def test_h9_dcc_world_multiplies_the_edges_but_not_the_claim(tiny):
    """`dcc_world=W` 应产生 W 条边；且单卡下必须如实声明**不能**体现代价化。"""
    ids = _ids(4)
    _, st = run_two_phase(tiny, ids, mode="dcc_kv", budget=BUDGET, dcc_world=3)
    # 2 层 × 2 kv head × 3 段
    assert st.summary()["n_compacts_built"] == 12
    assert st.summary()["conditionalization_marginal_available"] is False
    assert "单卡" in st.summary()["conditionalization_marginal_reason"]
    assert st.summary()["dcc_world"] == 3
    assert st.summary()["resolved_budget_by_source_len"] == {str(SRC_LEN): BUDGET}


# =============================================================================
# C. 本地块契约（2026-09-21 新增）
# =============================================================================

def test_h11_local_context_is_attended_to_not_dropped(tiny):
    """目的端**自己的本地上下文**必须被精确算进去，而不是丢掉。

    旧契约把本地块取成"传入 cache 的尾部 Lq 个"，隐含假设"cache 里除了本次
    token 就只有源端"。prefill 计时恰好满足，**评测路径不满足**：目的端有自己
    的本地段（提示词尾段），按旧规则它既不在 state 里、又被尾部规则排除 ⇒
    整段本地上下文静默消失（形状全对、数值偏、量级也正常）。

    判据：分三次前向（源段 → 本地段 → continuation）的 dense 臂结果，必须等于
    "把三段拼起来一次前向"的原生结果。dense 臂下源段是精确的（identity），所以
    两者本应等价 —— 差别只能来自本地块契约。
    """
    n_src, n_loc, n_cont = 20, 5, 4
    ids = _ids(11, n_src + n_loc + n_cont)
    full = native_logits(tiny, ids)[:, -n_cont:, :]

    ids_src, ids_loc, ids_cont = ids[:, :n_src], ids[:, n_src:n_src + n_loc], ids[:, -n_cont:]
    cfg = A.HookConfig(budget=n_src, mode="dense")
    with torch.no_grad():
        with A.dcc_attention(tiny, cfg) as st:
            st.source_phase()
            tiny(ids_src, use_cache=True)
            # 目的端探针：本地段。块在这里建，用的是本地段的 query
            st.destination_phase(compacts="build")
            loc = tiny(ids_loc, use_cache=True, position_ids=torch.arange(
                n_src, n_src + n_loc).unsqueeze(0))
            # continuation 复用同一批块，并把**本地段**作为 past 传进去
            # （本地 KV 只含本地段：探针那次没传 past）
            st.destination_phase(compacts="reuse")
            cont = tiny(ids_cont, past_key_values=loc.past_key_values,
                        use_cache=True,
                        position_ids=torch.arange(n_src + n_loc, n_src + n_loc + n_cont)
                        .unsqueeze(0))

    dev = (cont.logits - full).abs().max().item()
    assert dev < NATIVE_PATH_ATOL, (
        f"带本地上下文的三段式与原生一次前向差 {dev:.3e}（只应差 ULP 量级）。"
        "若量级在 1e-3 附近，最可能的原因是本地块被取成了'尾部 Lq 个'——"
        "于是本地段（5 个 token）整段没进注意力。"
    )
    # 本地段的 5 个 token + 本次 4 个 query ⇒ 两次目的端调用各算一次本地块
    assert st.summary()["n_local_blocks"] == 2 * 2 * 2
    assert st.summary()["n_builds"] == 2 * 2 * 1, "探针那次建块，continuation 那次复用"


def test_h12_reuse_is_bitwise_identical_and_refuses_a_changed_source(tiny):
    """`reuse` 必须逐位一致、且**不**新增构造；源长变了必须拒绝复用。

    复用是"块只建一次"的机制。两条都容易静默失效：
    ① 若 reuse 其实又建了一次，构造耗时留在计时窗口里，而产物看上去一样；
    ② 换了样本（源长不同）却复用旧块 ⇒ budget / 选键 / β 全对不上，
       形状与量级正常，只有数值偏。
    """
    ids = _ids(12)
    src, dst = ids[:, :SRC_LEN], ids[:, SRC_LEN:]
    pos = torch.arange(SRC_LEN, SRC_LEN + DST_LEN).unsqueeze(0)
    cfg = A.HookConfig(budget=BUDGET, mode="dcc_kv")

    with torch.no_grad():
        with A.dcc_attention(tiny, cfg) as st:
            st.source_phase()
            tiny(src, use_cache=True)
            st.destination_phase(compacts="build")
            o1 = tiny(dst, use_cache=True, position_ids=pos).logits
            builds_after_first = st.summary()["n_builds"]
            st.destination_phase(compacts="reuse")
            o2 = tiny(dst, use_cache=True, position_ids=pos).logits
            assert torch.equal(o1, o2), "reuse 与 build 的输出必须逐位相同"
            assert st.summary()["n_builds"] == builds_after_first, \
                "reuse 不得再触发构造（构造代价会因此留在计时窗口里）"

            # 源长变了：必须拒绝，而不是拿旧块算
            other = _ids(13, SRC_LEN + 3 + DST_LEN)
            st.source_phase()
            tiny(other[:, :SRC_LEN + 3], use_cache=True)
            st.destination_phase(compacts="reuse")
            with pytest.raises(ValueError, match="源长"):
                tiny(dst, use_cache=True, position_ids=pos)


def test_h13_the_source_must_not_be_left_in_the_passed_cache(tiny):
    """指纹守卫：把源端连同本地块一起传进来必须抛错。

    混进去的后果是**静默**的：源端会同时以紧凑块与本地精确块两种形式进入
    softmax（源端质量被算两遍），而形状、dtype、量级全都正常，只有数值略偏。
    守卫是 4 行采样的**启发式**，不是证明 —— 但它要抓的成因只有一个
    （调用方把同一张源端张量又传了进来）。
    """
    ids = _ids(14)
    src, dst = ids[:, :SRC_LEN], ids[:, SRC_LEN:]
    pos = torch.arange(SRC_LEN, SRC_LEN + DST_LEN).unsqueeze(0)
    cfg = A.HookConfig(budget=BUDGET, mode="dcc_kv")

    with torch.no_grad():
        # 正确用法：目的端只带自己的 token（past=None）⇒ 不抛
        with A.dcc_attention(tiny, cfg) as st:
            st.source_phase()
            tiny(src, use_cache=True)
            st.destination_phase(compacts="build")
            tiny(dst, use_cache=True, position_ids=pos)

        # 错误用法：把源端 cache 也传进去 ⇒ 必须抛
        with A.dcc_attention(tiny, cfg) as st:
            st.source_phase()
            c = tiny(src, use_cache=True)
            st.destination_phase(compacts="build")
            with pytest.raises(ValueError, match="源端"):
                tiny(dst, past_key_values=c.past_key_values, use_cache=True,
                     position_ids=pos)

    # 守卫只在该契约可能被破坏时才可能开火：源长 > 传入 cache ⇒ 直接跳过
    st2 = A.DccAttentionState(cfg)
    st2.record_source(0, torch.randn(1, 2, 32, 8), torch.randn(1, 2, 32, 8))
    A._assert_local_cache_excludes_source(st2, 0, torch.randn(1, 2, 4, 8))


def test_h14_summary_carries_the_declarations_that_must_travel_with_numbers(tiny):
    """落盘必须带三条与数字同行的声明 —— 否则下游会把"接线"读成"结论"。"""
    _, st = run_two_phase(tiny, _ids(15), mode="dcc_kv", budget=BUDGET, dcc_world=2)
    s = st.summary()
    for key in ("local_block_contract", "conditionalization_marginal_available",
                "conditionalization_marginal_reason", "budget_claim_scope",
                "resolved_budget_by_source_len", "ignores_attention_mask",
                "build_in_timing_window", "lambda_beta", "n_repr", "projection_dim"):
        assert key in s, f"summary 缺 {key}"
    assert s["ignores_attention_mask"] is True
    assert "只含目的端本地 KV" in s["local_block_contract"]
    assert "含 T_build" in s["build_in_timing_window"], \
        "构造在计时窗口内这件事必须自己申报（它是与 dense 臂的一个额外差异）"


# =============================================================================
# D. measure_prefill 目的端的位置口径
# =============================================================================

class _Out:
    def __init__(self, cache):
        self.past_key_values = cache


class _Recorder:
    """记录每次前向的 (输入长度, past 长度, position_ids 首行)。"""

    def __init__(self):
        self.calls: list = []

    def __call__(self, input_ids=None, past_key_values=None, use_cache=False,
                 position_ids=None, **kw):
        lin = int(input_ids.shape[1])
        past = (int(past_key_values[0][0].shape[-2])
                if past_key_values is not None else None)
        pos = None if position_ids is None else tuple(
            int(x) for x in position_ids[0])
        self.calls.append((lin, past, pos))
        total = lin + (past or 0)
        return _Out([(torch.zeros(1, 4, total, 8), torch.zeros(1, 4, total, 8))
                     for _ in range(2)])


class _StubLM:
    def __init__(self):
        self.model = _Recorder()
        self.device = torch.device("cpu")
        self.precision = "bfloat16"
        self.num_layers = 2
        self.num_kv_heads = 4
        self.head_dim = 8


def _run_prefill(budget_ratio, seq_len=512, dest_len=64):
    lm = _StubLM()
    _hf.measure_prefill(lm, seq_len=seq_len, batch_size=1, warmup=1, iters=2,
                        budget_ratio=budget_ratio, compaction_mode="topk_rms",
                        dest_len=dest_len)
    return lm


def test_h10_destination_positions_are_absolute_and_arm_independent():
    """目的端的位置必须是 S..S+dest_len-1，且**两臂相同**。

    修复前的形态（2026-09-21 取证，tiny Llama + 真实 cache）：
    `DynamicCache.get_seq_length()` 读的是**实际张量长度** —— 裁剪后返回 B
    而不是 S。`LlamaModel.forward` 在 position_ids 缺省时用
    `arange(L) + past_seen_tokens`，于是压缩臂的 dst 从 B 起算（实测数值差
    4.9e-3，是 dense 参照臂路径差 1.5e-07 的 5 个数量级）；而 dense 臂的 cache
    没被裁剪、位置本就是 S..。

    危害不是"少了一点精度"：它让"两臂唯一差别是 KV 长度"这条断言
    在真实模型下**不成立**，而跑在桩模型上的测试看不出来（桩不看 position_ids）。
    """
    seq_len, dest_len, ratio = 512, 64, 0.05
    b = max(1, int(round(ratio * seq_len)))
    want = tuple(range(seq_len, seq_len + dest_len))

    dense = _run_prefill(None, seq_len, dest_len)
    comp = _run_prefill(ratio, seq_len, dest_len)

    dst_dense = [c for c in dense.model.calls if c[0] == dest_len]
    dst_comp = [c for c in comp.model.calls if c[0] == dest_len]
    assert len(dst_dense) == 3 and len(dst_comp) == 3, "warmup(1) + iters(2)"

    assert all(c[2] == want for c in dst_dense), \
        f"精确臂的目的端位置应为 {want[:2]}...，实测 {[c[2] for c in dst_dense]}"
    assert all(c[2] == want for c in dst_comp), (
        f"压缩臂的目的端位置应为 {want[:2]}...，实测 {[c[2] for c in dst_comp]}。"
        f"若是 {b}..{b + dest_len} 那样的起点，说明又退回缺省 position_ids 了"
        "（裁剪后 cache 的 get_seq_length() 返回 B）。"
    )
    assert {c[2] for c in dst_dense} == {c[2] for c in dst_comp}, \
        "两臂的目的端位置必须一致，否则两臂之差里混进了位置错位"

    # 位置一致**不**等于长度一致：收益载体仍是 past 长度（B vs S），别把它抹平
    assert {c[1] for c in dst_comp} == {b}
    assert {c[1] for c in dst_dense} == {seq_len}
