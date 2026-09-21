"""H0 方法侧（`src/distributed/attention_hook.py`）的锚点 —— 纯 CPU，不需要 GPU。

为什么这些能在本机验证
----------------------
钩子是 `device-agnostic` 的，而 transformers 可以用一个**本地构造的 tiny Llama**
（`LlamaConfig` + 随机权重，不下载任何东西）在 CPU 上跑完整前向。于是本机能问出
一件 GPU 机上要花很久才问得清的事：**接线到底对不对**（形状、转置、GQA 分组、
位置、因果、还原）。

最强的一条是 `test_h1_...`：dense 参照臂（`identity_compact`，β ≡ 0、B = L_s）
必须复现原生 HF 的前向。它一条就同时覆盖了上面所有维度 —— 因为它们中任何一个
错了，输出都不可能与原生长时间保持在 1e-6 以内。

容差为什么不是"逐位"
--------------------
钩子走 `attention_kernel` 的**显式**路径（matmul + logsumexp + matmul），而原生
HF 走 SDPA 的**融合核**。两条路径的浮点求和顺序不同 ⇒ 只能 ULP 级一致。
实测（float32，S=24 / dest=6）：最大绝对差 9.7e-08（相对 2.5e-07，约 1.6 ULP）。
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

# 实测值（float32）：dense 臂与原生前向的路径差。留一倍余量放进断言，
# 但**不放到 1e-4** —— 那会把"某个 head 走错了"这种量级的错放过去。
NATIVE_PATH_ATOL = 1e-6


@pytest.fixture(scope="module")
def tiny():
    """本地构造的 tiny Llama（随机权重，无需网络）。"""
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**TINY)).eval()


def _ids(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, SRC_LEN + DST_LEN), generator=g)


def native_logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def run_two_phase(model, ids, *, mode, budget, dcc_world=1, budget_mode="per_edge"):
    """按钩子的两段式跑一遍，返回 (dst 段 logits, state)。"""
    src, dst = ids[:, :SRC_LEN], ids[:, SRC_LEN:]
    pos = torch.arange(SRC_LEN, SRC_LEN + DST_LEN).unsqueeze(0)
    cfg = A.HookConfig(budget=budget, mode=mode, dcc_world=dcc_world,
                       budget_mode=budget_mode)
    with torch.no_grad():
        with A.dcc_attention(model, cfg) as st:
            st.source_phase()
            first = model(src, use_cache=True)
            st.destination_phase()
            second = model(dst, past_key_values=first.past_key_values,
                           use_cache=True, position_ids=pos)
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
    # 目的端的 logits 由**前一个位置**的预测决定，这里比较的是同一批位置，
    # 故可直接逐元素比（两边都是 dst 段的 logits）。
    dev = (logits - base[:, SRC_LEN:, :]).abs().max().item()
    assert dev < NATIVE_PATH_ATOL, (
        f"dense 参照臂与原生前向差 {dev:.3e}，超过 {NATIVE_PATH_ATOL:.0e}。"
        "两者只应差一条算子路径的 ULP（实测 9.7e-08）——超出说明接线错了："
        "优先查 GQA 分组（必须按 repeat_interleave 连续切）、"
        "返回值转置（必须是 [B, Lq, H*D]）、以及源端的因果掩码有没有施加。"
    )
    # 每层每 kv head 一次：2 层 × 2 个 kv head
    assert st.summary()["n_compacts_built"] == 4
    assert st.summary()["n_local_blocks"] == 4


def test_h2_compaction_actually_changes_the_output(tiny):
    """压缩臂的输出必须**明显**偏离参照臂与原生前向。

    这条防的是"钩子空转"：若某天 compact 块被静默跳过（或 budget 被当成
    L_s），`dcc_kv` 行会退化成 dense 行，而所有形状类断言照样全绿。
    量级判据取 1e-4（实测偏离 2.9e-03），比路径差 9.7e-08 高四个数量级 ——
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


def test_h3_phases_must_be_declared_explicitly():
    """阶段是显式声明的，猜错要响亮失败。

    不能靠"`past_key_values` 是不是 None"去猜：把目的端当源端时，钩子会把
    紧凑块又存一份进 state，产物看着完全正常（模块文档第 2 条）。
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
        st.destination_phase()

    # 跑了源端之后才能进目的端
    st.source_phase()
    hook(_Mod(), q, k, v, scaling=0.25)
    st.destination_phase()
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


def test_h8_budget_mode_and_the_identity_short_circuit():
    """预算口径：`per_edge` 为主线，`total` 只作敏感性对照。

    `total` 会让总预算随 world 放大 —— 那是"花更多 KV 换质量"，不是机制收益。
    summary 里必须把这条限制写进产物，而不是只写在文档里。
    """
    per = A.HookConfig(budget=64, dcc_world=4, budget_mode="per_edge")
    assert [per.edge_budget(n) for n in (100, 16, 15, 1)] == [16, 16, 15, 1]

    tot = A.HookConfig(budget=64, dcc_world=4, budget_mode="total")
    assert [tot.edge_budget(n) for n in (100, 16)] == [64, 16]

    assert "主口径" in A.DccAttentionState(per).summary()["budget_claim_scope"]
    assert "不得" in A.DccAttentionState(tot).summary()["budget_claim_scope"]

    # 每边至少 1 个 key ⇒ budget 必须 ≥ world，否则总预算名不副实
    with pytest.raises(ValueError, match="per_edge"):
        A.HookConfig(budget=2, dcc_world=4, budget_mode="per_edge")

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

    # 一次前向里段数与 summary 对得上；但**不得**据此声称"条件化有效"
    assert st.summary()["dcc_world"] == 3


# =============================================================================
# C. measure_prefill 目的端的位置口径
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
    4.9e-3，是 dense 参照臂路径差 9.7e-08 的 5 个数量级）；而 dense 臂的 cache
    没被裁剪、位置本就是 S..。

    危害不是"少了一点精度"：它让 A1b 断言的「两臂唯一差别是 KV 长度」
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
