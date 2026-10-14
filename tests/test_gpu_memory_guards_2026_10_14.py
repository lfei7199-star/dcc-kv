r"""守卫：GPU 前向显存口径（2026-10-14）。

背景
----
``docs/reports/2026-10-14-GPU显存需求预估与机型建议.md`` 在预估时发现两个
**代码级**显存风险（不是"机器不够"）：

1. ``experiments/gpu/_hf.py`` 的各处前向不传 ``logits_to_keep``。
   transformers 5.5.4 的默认值是 0 = "全长过 lm_head"：32K 上下文下
   24K 源段会白物化 ``[1, 24576, 151936]`` 的 bf16 logits ≈ 7.0 GB，
   而打分只用最后一个位置。且 ``score_choices`` 的 ``base_logit``
   是 logits 的**视图**，会把整份 logits 拖到打分结束。
2. 钩子源端 / 本地块走 ``attention_kernel.dense_attention`` 的显式
   logits 路径（刻意不走 SDPA，为的是与融合核交叉验证）。源端
   Lq == Lk == L_s ⇒ 每层物化 ``[g, Lq, Lk]`` 及其 3–4 个同尺寸
   中间张量；Qwen GQA 组宽 g=7、32K ⇒ 单张量 7.9 GB，48 GB 卡
   放不下整条前向。

修复
----
1. ``_hf.py`` 全部 12 处 ``lm.model(...)`` 显式传 ``logits_to_keep``：
   源段/目的段 prefill 传 1；continuation 打分传 ``T``（每个位置都要用）。
   只影响 lm_head 的 M 维 ⇒ 输出在 ULP 量级内不变（E5–E8 从未在 GPU
   上跑过，无旧表可破）。
2. ``dense_attention`` 新增 ``query_chunk``（沿 query 维分块，与
   ``compact_kv_attention`` 同款语义：**显存控制手段，非数值等价手段**）；
   ``attention_hook`` 新增 ``DEFAULT_LOCAL_QUERY_CHUNK = 512`` 与
   ``resolve_query_chunk``，源端与本地块在 ``cfg.query_chunk`` 为 None
   时**默认就分块**——忘了传不会报错、只会在 32K 上 OOM，而那种失败
   看起来像"机器不够"，不像配置遗漏。

本守卫四道：
1. ``_hf.py`` 每个 ``.model(...)`` 调用必须显式含 ``logits_to_keep``
   （AST；含正反例自测的变异闭环）；
2. ``attention_hook.py`` 中 ``K.dense_attention`` / ``K.dcc_kv_attention``
   / ``_source_attention`` 的调用必须传 ``query_chunk``，且钩子两处
   必须走 ``resolve_query_chunk(cfg)``（不能退回裸 ``cfg.query_chunk``
   —— 那等于默认不分块）；
3. ``dense_attention`` 分块后必须与不分块 ULP 级等价（数值）；
4. 钩子源端默认分块真的生效且结果与显式不分块 ULP 级等价（数值）。
"""

import ast
import pathlib

import torch

from src.dcc_kv_ref import dense_attention
from src.distributed import attention_hook as A

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HF_PATH = REPO_ROOT / "experiments" / "gpu" / "_hf.py"
HOOK_PATH = REPO_ROOT / "src" / "distributed" / "attention_hook.py"


# =============================================================================
# 1. _hf.py：每个模型前向都必须显式决定 logits 策略
# =============================================================================

def _calls_missing_logits_to_keep(src: str):
    """返回所有缺少 `logits_to_keep` 关键字参数的 `.model(...)` 调用行号。

    写成"对源码字符串可用"而不是"只对磁盘文件可用"：这样变异测试
    （删掉一处参数）不需要动磁盘，正反例可以在同一进程里自测。
    """
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "model"):
            continue
        kws = {k.arg for k in node.keywords}
        if "logits_to_keep" not in kws:
            bad.append(node.lineno)
    return bad


def test_hf_model_calls_all_pass_logits_to_keep():
    """`_hf.py` 里每个 `lm.model(...)` 都必须显式给 `logits_to_keep`。"""
    src = HF_PATH.read_text(encoding="utf-8")
    bad = _calls_missing_logits_to_keep(src)
    assert not bad, (
        f"_hf.py 有 {len(bad)} 处模型前向没有显式 logits_to_keep：{bad}。"
        "默认值 0 = 全长过 lm_head，32K 源段白占 ~7 GB。"
        "源段/目的段 prefill 传 1，continuation 打分传 T。"
    )


def test_guard_catches_removed_logits_to_keep():
    """正反例自测：把任一处的 `logits_to_keep` 删掉，守卫必须变红。

    只测"删第一个"是不够的——那只能证明第一个参数被盯上了。
    这里对**每一处** `logits_to_keep=1` 各删一次，共 N 次变异，
    全部必须被检出（漏一处 = 守卫有盲区）。
    """
    src = HF_PATH.read_text(encoding="utf-8")
    assert _calls_missing_logits_to_keep(src) == []
    token = ", logits_to_keep=1)"
    assert src.count(token) >= 6, (
        f"预期至少有 6 处 `logits_to_keep=1`，实得 {src.count(token)}；"
        "若修复被回退，本测试应先红在这里"
    )
    for i in range(src.count(token)):
        mutated = src.replace(token, ")", 1) if i == 0 else (
            src[:0] + token.join(  # 只删第 i 处
                [seg.replace(token, ")") for seg in
                 _split_nth(src, token, i)]
            )
        )
        bad = _calls_missing_logits_to_keep(mutated)
        assert bad, f"第 {i} 处变异没有被检出：守卫有盲区"


def _split_nth(src: str, token: str, n: int):
    """把 src 按第 n 个 token 切开（前后两段），便于构造单点变异。"""
    parts = src.split(token)
    assert len(parts) >= n + 2
    head = token.join(parts[: n + 1])
    tail = token.join(parts[n + 1:])
    return [head, tail]


# =============================================================================
# 2. attention_hook.py：分块必须接进源端与本地块
# =============================================================================

def _calls_in(src: str, names):
    """返回 [(unparse(func), {kw: unparse(value)}, lineno)]。"""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in names:
            label = func.attr
        elif isinstance(func, ast.Name) and func.id in names:
            label = func.id
        else:
            continue
        kws = {k.arg: (ast.unparse(k.value) if k.value is not None else None)
               for k in node.keywords}
        out.append((label, kws, node.lineno))
    return out


def test_hook_wiring_passes_query_chunk():
    """钩子里三处调用都必须显式传 query_chunk。

    - `K.dense_attention`（源端参照核）与 `K.dcc_kv_attention`（目的端）
      的调用不带分块 ⇒ 源端/本地块的 `[Lq, Lk]` 显式 logits 不在任何
      显存控制之内，正是 32K OOM 的主因；
    - `_source_attention` 的两个调用点不带 ⇒ 同上。
    """
    src = HOOK_PATH.read_text(encoding="utf-8")
    calls = _calls_in(src, {"dense_attention", "dcc_kv_attention",
                            "_source_attention"})
    assert calls, "一个调用都没找到：扫描目标变了？"
    missing = [(label, ln) for label, kws, ln in calls
               if "query_chunk" not in kws]
    assert not missing, (
        f"attention_hook.py 有 {len(missing)} 处调用没有传 query_chunk："
        f"{missing}。默认 None 在 resolve_query_chunk 处才会变成 512，"
        "调用点漏传就等于这条路径不分块。"
    )


def test_hook_wiring_uses_resolve_query_chunk():
    """钩子内两处必须走 `resolve_query_chunk(cfg)`，不能退回裸 cfg.query_chunk。

    裸 `cfg.query_chunk` 的默认值是 None ⇒ 库层默认不分块 ⇒ "默认就分块"
    的设计被悄悄撤掉，而现象只是长上下文 OOM（看起来像机器不够）。
    """
    src = HOOK_PATH.read_text(encoding="utf-8")
    must_resolve = [(label, ln) for label, kws, ln in
                    _calls_in(src, {"_source_attention", "dcc_kv_attention"})
                    if kws.get("query_chunk") != "resolve_query_chunk(cfg)"]
    assert not must_resolve, (
        f"这些调用点的 query_chunk 不是 resolve_query_chunk(cfg)：{must_resolve}。"
        "请改回 resolve_query_chunk(cfg)（None → DEFAULT_LOCAL_QUERY_CHUNK）。"
    )


def test_resolve_query_chunk_default_and_override():
    """None ⇒ 默认 512；显式给定 ⇒ 该值（不静默改写调用方意图）。"""
    assert A.DEFAULT_LOCAL_QUERY_CHUNK >= 1
    cfg_none = A.HookConfig(budget_ratio=0.1)
    assert cfg_none.query_chunk is None
    assert A.resolve_query_chunk(cfg_none) == A.DEFAULT_LOCAL_QUERY_CHUNK
    cfg_set = A.HookConfig(budget_ratio=0.1, query_chunk=64)
    assert A.resolve_query_chunk(cfg_set) == 64


def test_hook_config_describe_includes_query_chunk():
    """产物必须能看出本次跑的分块口径（含 None = 用了默认）。"""
    d = A.HookConfig(budget_ratio=0.1).describe()
    assert "query_chunk" in d
    assert d["query_chunk"] is None


# =============================================================================
# 3. dense_attention 分块：ULP 量级等价
# =============================================================================

def test_dense_attention_query_chunk_ulp_equivalent():
    """分块改变"怎么算"不改变"算的是什么"，但**不承诺逐位相同**。

    与 `compact_kv_attention` 同款结论（锚点
    `test_attention_kernel.py::test_query_chunk_ulp_equivalent`）：
    query 维是 GEMM 的 M 维，BLAS 按 M 选分块核与累加策略 ⇒ 同一元素的
    浮点累加顺序随 M 改变，实测 1–3 ULP。这里对因果/非因果 ×
    return_lse 两种路径全组合断言 ULP 量级。
    """
    torch.manual_seed(0)
    Lq, Lk, d = 13, 24, 8
    q = torch.randn(Lq, d, dtype=torch.float64)
    k = torch.randn(Lk, d, dtype=torch.float64)
    v = torch.randn(Lk, d, dtype=torch.float64)
    qpos = torch.arange(Lq)
    for causal in (False, True):
        for return_lse in (True, False):
            full = dense_attention(q, k, v, query_positions=qpos, causal=causal,
                                   return_lse=return_lse)
            # 注意：`dense_attention` 是显式路径，**无论 return_lse 都算 lse**
            # （`lse_available` 恒为 True）——这里不断言该字段，只断言数值。
            for chunk in (1, 2, 3, 4, 7):
                got = dense_attention(q, k, v, query_positions=qpos,
                                      causal=causal, return_lse=return_lse,
                                      query_chunk=chunk)
                assert got.out.shape == full.out.shape
                assert torch.allclose(got.out, full.out, atol=1e-13, rtol=0), (
                    f"causal={causal} return_lse={return_lse} "
                    f"query_chunk={chunk} 超出 ULP 量级"
                )
                assert torch.allclose(got.lse, full.lse, atol=1e-11, rtol=0), (
                    f"causal={causal} query_chunk={chunk} 的 lse 超出 ULP 量级"
                )
                assert got.n_keys == full.n_keys


def test_dense_attention_query_chunk_none_is_bit_identical():
    """`query_chunk=None`（默认）必须走原路径 ⇒ 旧调用点逐位不变。"""
    torch.manual_seed(1)
    Lq, Lk, d = 9, 16, 4
    q = torch.randn(Lq, d, dtype=torch.float64)
    k = torch.randn(Lk, d, dtype=torch.float64)
    v = torch.randn(Lk, d, dtype=torch.float64)
    qpos = torch.arange(Lq)
    ref = dense_attention(q, k, v, query_positions=qpos, causal=True)
    for qc in (None, Lq):  # None = 不传；Lq = 不小于 Lq，同样不分块
        got = dense_attention(q, k, v, query_positions=qpos, causal=True,
                              query_chunk=qc)
        assert torch.equal(got.out, ref.out)
        assert torch.equal(got.lse, ref.lse)


# =============================================================================
# 4. 钩子源端：默认分块真的生效
# =============================================================================

def test_source_attention_default_chunk_matches_unchunked():
    """`_source_attention` 的默认分块必须真的生效且不改变答案。

    Lq 取 `DEFAULT_LOCAL_QUERY_CHUNK + 7`：大于默认值 ⇒ 分块条件触发。
    若有人把默认改回"不分块"，本测试仍会过（结果一样）—— 那条路由
    `test_hook_wiring_*` 的源码守卫盯着；这里守的是"分块后数值没错"。
    """
    torch.manual_seed(2)
    L = A.DEFAULT_LOCAL_QUERY_CHUNK + 7
    B, H_kv, d = 1, 2, 8
    g = 2
    H_q = H_kv * g
    q = torch.randn(B, H_q, L, d)
    k = torch.randn(B, H_kv, L, d)
    v = torch.randn(B, H_kv, L, d)
    scale = 1.0 / (d ** 0.5)
    ref = A._source_attention(q, k, v, H_kv, scale, query_chunk=10 ** 9)
    got = A._source_attention(q, k, v, H_kv, scale)  # None ⇒ 默认 512
    assert got.shape == ref.shape
    # `_source_attention` 的累加器是 float32（见其 outs 定义），
    # 故容差按 float32 给；分块只影响 GEMM 的 M 维分组。
    assert torch.allclose(got, ref, atol=1e-5, rtol=0), (
        "默认分块与显式不分块的结果差出 float32 噪声量级"
    )


def test_source_attention_default_chunk_actually_chunks():
    """默认分块必须**真的发生**：Lq > 512 时源端实收的 chunk 是 512。

    行为探针。spy 会同时抓到分块递归（那些调用显式带
    ``query_chunk=None``），故按 (Lq, chunk) 记：外层那一次必须是
    (L, DEFAULT)。若调用点漏传或默认值被改成 None 语义，本测试红。
    """
    torch.manual_seed(3)
    L = A.DEFAULT_LOCAL_QUERY_CHUNK + 3
    B, H_kv, d = 1, 1, 8
    H_q = H_kv * 2
    q = torch.randn(B, H_q, L, d)
    k = torch.randn(B, H_kv, L, d)
    v = torch.randn(B, H_kv, L, d)
    seen = []
    real = A.K.dense_attention

    def spy(*args, **kwargs):
        seen.append((int(args[0].shape[-2]),
                     kwargs.get("query_chunk", "<missing>")))
        return real(*args, **kwargs)

    A.K.dense_attention = spy
    try:
        A._source_attention(q, k, v, H_kv, 0.3)
    finally:
        A.K.dense_attention = real
    assert seen, "spy 没有抓到任何 dense_attention 调用"
    assert (L, A.DEFAULT_LOCAL_QUERY_CHUNK) in seen, (
        f"源端外层调用实收的 query_chunk 不是默认 "
        f"{A.DEFAULT_LOCAL_QUERY_CHUNK}：{seen}"
    )
    assert all(chunk != "<missing>" for _, chunk in seen), (
        f"有调用完全没传 query_chunk：{seen}"
    )
