"""G3 的锚点：三个基线的**向量化算子**必须与 CPU 参考等价。

等价强度是 **ULP 级，不是逐位** —— 向量化把逐 query 的 online softmax 换成
"分块 partial + lse 归并"，求和次序变了（与 G1 的 `query_chunk`、G2 的
`n_chunks` 是同一现象）。所以这里用**相对容差**断言，并把容差写成可解释的数
（float64 下 ≤1e-12，float32 下 ≤1e-5），而不是随手 allclose。

为什么这条断言值钱：三个基线各有一处历史上踩过的坑，它们都不会让代码报错，
只会让数值悄悄不同 ——

1. 因果掩码按**压缩块/锚点数组的行序**切前缀，而不是按原始位置
   （`fast_kv_cpu` / `apb_cpu` 各犯过一次）。$B=L_s$ 时索引恰好升序，
   所以这个错在 $B<L_s$ 时才显形 —— 用 budget < L_s 的用例才能抓住它。
2. APB 的锚点若用**全体 queries** 选（而不是该块之后的 queries），数值会变。
3. FastKV 的 λ_β 若不同源（旧默认 1e-3），数字会变。

因此本文件刻意用 `budget < L_s`、`world_size ≥ 2`、`L_total` 不是 `chunk_size`
整数倍等配置，让上述三处都有机会显形。
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import (  # noqa: E402
    APBConfig, FastKVConfig,
    apb_attention, apb_cpu,
    fast_kv_cpu, fastkv_attention,
    ring_attention, ring_attention_cpu,
)

TOL64 = 1e-12
TOL32 = 1e-5


def _inputs(L_total: int, d_h: int, d_v: int, seed: int = 0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    make = lambda *s: torch.randn(*s, generator=g).to(dtype)  # noqa: E731
    return make(L_total, d_h), make(L_total, d_h), make(L_total, d_v)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(b.abs().max())
    if denom == 0.0:
        return float((a - b).abs().max())
    return float((a - b).abs().max()) / denom


# ---------------------------------------------------------------------------
# Ring：无压缩
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("L_total,chunk,world", [(24, 8, 3), (25, 8, 3), (32, 16, 2)])
def test_ring_attention_matches_the_cpu_reference(
    causal: bool, L_total: int, chunk: int, world: int
) -> None:
    """L_total 刻意取 25（不是 8 的整数倍）—— 最后一块吃掉剩余，容易写错。"""
    q, k, v = _inputs(L_total, 8, 6)
    bounds = [(s * chunk, min((s + 1) * chunk, L_total)) for s in range(world - 1)]
    bounds.append(((world - 1) * chunk, L_total))
    kc = [k[a:b] for a, b in bounds]
    vc = [v[a:b] for a, b in bounds]

    want = ring_attention_cpu(q, kc, vc, causal=causal)
    got = ring_attention(q, kc, vc, causal=causal)
    assert got.shape == want.shape
    assert _rel(got, want) < TOL64


def test_ring_attention_query_chunk_is_the_same_answer() -> None:
    q, k, v = _inputs(24, 8, 6)
    kc, vc = [k[:12], k[12:]], [v[:12], v[12:]]
    base = ring_attention(q, kc, vc)
    for step in (1, 5, 11):
        assert _rel(ring_attention(q, kc, vc, query_chunk=step), base) < TOL64


# ---------------------------------------------------------------------------
# FastKV：共享压缩
# ---------------------------------------------------------------------------

def test_fastkv_attention_matches_the_cpu_reference() -> None:
    """budget < chunk 长度 ⇒ 选键顺序与位置序不同，行序前缀的错误会显形。"""
    L_total, d_h, d_v, chunk, world = 24, 8, 6, 8, 3
    q, k, v = _inputs(L_total, d_h, d_v, seed=1)
    cfg = FastKVConfig(budget=4, num_repr_queries=8, projection_dim=4, seed=7)

    want = fast_kv_cpu(q, k, v, chunk, world, config=cfg)
    got = fastkv_attention(q, k, v, chunk, world, config=cfg)
    assert got.shape == want.shape
    assert _rel(got, want) < TOL64


def test_fastkv_attention_is_not_a_dense_attention() -> None:
    """防止"压缩被绕过"：预算远小于块长时，结果必须与精确注意力分得开。

    若某次重构不小心让 `budget` 失效（例如退回 `budget=len(K_s)`），
    与 CPU 参考仍可能吻合（两者一起退化），但这里会立刻炸。
    """
    L_total, d_h, d_v, chunk, world = 32, 8, 6, 16, 2
    q, k, v = _inputs(L_total, d_h, d_v, seed=2)
    cfg = FastKVConfig(budget=4, num_repr_queries=8, projection_dim=4, seed=7)
    got = fastkv_attention(q, k, v, chunk, world, config=cfg)
    dense = ring_attention(q, [k[:16], k[16:]], [v[:16], v[16:]])
    assert _rel(got, dense) > 1e-3, "压缩预算没生效：结果与精确注意力几乎相同"


def test_fastkv_config_defaults_share_the_lambda_beta_source() -> None:
    """λ_β 必须与主方法同源（曾硬编码 1e-3，对照因此不公平）。"""
    from src.dcc_kv_ref import DEFAULT_LAMBDA_BETA
    assert FastKVConfig().lambda_beta == DEFAULT_LAMBDA_BETA


# ---------------------------------------------------------------------------
# APB：锚点块
# ---------------------------------------------------------------------------

def test_apb_attention_matches_the_cpu_reference() -> None:
    L_total, d_h, d_v, chunk, world = 24, 8, 6, 8, 3
    q, k, v = _inputs(L_total, d_h, d_v, seed=3)
    cfg = APBConfig(anchor_budget=4, num_anchor_queries=8)

    want = apb_cpu(q, k, v, chunk, world, config=cfg)
    got = apb_attention(q, k, v, chunk, world, config=cfg)
    assert got.shape == want.shape
    assert _rel(got, want) < TOL64


def test_apb_anchors_use_only_future_queries() -> None:
    """APB 的锚点用「该块之后的 queries」选 —— 换成全体 queries 会改变数值。

    这是 APB 与 FastKV 的关键分野之一（FastKV 用全体 queries）。
    注意**必须看第 2 块（offset>0）**：第 0 块的 `queries[0:]` 就是全体 queries，
    两种口径按构造恒等，拿它做断言等于什么也没测（第一版就写错过）。
    """
    L_total, d_h, d_v, chunk, world = 24, 8, 6, 8, 3
    q, k, v = _inputs(L_total, d_h, d_v, seed=4)
    cfg = APBConfig(anchor_budget=4, num_anchor_queries=8)

    from src.baselines.apb_cpu import select_anchor_blocks
    from src.baselines.operators import _anchor_compacts, _chunk_bounds
    _kf, _vf, idx_future = select_anchor_blocks(k[8:16], v[8:16], q[8:], budget=4)
    _ka, _va, idx_all = select_anchor_blocks(k[8:16], v[8:16], q, budget=4)
    assert not torch.equal(idx_future, idx_all), (
        "该用例下两种口径恰好同解 ⇒ 本测试没有分辨力，需换 seed"
    )

    # 直接检查算子内部用的是哪一种（比靠数值差更直接）
    blocks = _anchor_compacts(q, k, v, _chunk_bounds(L_total, chunk, world), cfg)
    assert torch.equal(blocks[1][0].selected_indices, idx_future)
    assert not torch.equal(blocks[1][0].selected_indices, idx_all)

    got = apb_attention(q, k, v, chunk, world, config=cfg)
    want = apb_cpu(q, k, v, chunk, world, config=cfg)
    assert _rel(got, want) < TOL64


def test_apb_has_no_beta_bias() -> None:
    """APB 没有质量偏置 β。把压缩链路整套搬过来会给它装上 DCC-KV 的机制。"""
    L_total, d_h, d_v, chunk, world = 24, 8, 6, 8, 3
    q, k, v = _inputs(L_total, d_h, d_v, seed=5)
    from src.baselines.operators import _anchor_compacts, _chunk_bounds
    cfg = APBConfig(anchor_budget=4, num_anchor_queries=8)
    blocks = _anchor_compacts(q, k, v, _chunk_bounds(L_total, chunk, world), cfg)
    assert blocks
    for ck, _off in blocks:
        assert torch.equal(ck.logit_bias, torch.zeros_like(ck.logit_bias))


def test_apb_and_fastkv_are_not_the_same_method() -> None:
    """两个基线必须真的不同 —— 否则主表里的"三个基线"是假的。

    APB 无 β、锚点用后续 queries 选、按 attention mass 挑；FastKV 有 β、
    共享压缩、按 RMS 挑。这个用例是"对照臂没有互相塌缩"的守卫。
    """
    L_total, d_h, d_v, chunk, world = 32, 8, 6, 16, 2
    q, k, v = _inputs(L_total, d_h, d_v, seed=6)
    a = apb_attention(q, k, v, chunk, world,
                      config=APBConfig(anchor_budget=4, num_anchor_queries=8))
    f = fastkv_attention(q, k, v, chunk, world,
                         config=FastKVConfig(budget=4, num_repr_queries=8,
                                             projection_dim=4, seed=7))
    assert _rel(a, f) > 1e-4


# ---------------------------------------------------------------------------
# 通用性质
# ---------------------------------------------------------------------------

def test_all_three_match_the_reference_in_float32() -> None:
    """精度降到 float32 后容差要相应放宽 —— 但不得放宽到"什么都算通过"。"""
    L_total, d_h, d_v, chunk, world = 24, 8, 6, 8, 3
    q, k, v = _inputs(L_total, d_h, d_v, seed=8, dtype=torch.float32)
    kc, vc = [k[:8], k[8:16], k[16:]], [v[:8], v[8:16], v[16:]]

    assert _rel(ring_attention(q, kc, vc),
                ring_attention_cpu(q, kc, vc)) < TOL32
    assert _rel(fastkv_attention(q, k, v, chunk, world,
                                 config=FastKVConfig(budget=4, num_repr_queries=8,
                                                     projection_dim=4, seed=7)),
                fast_kv_cpu(q, k, v, chunk, world,
                            config=FastKVConfig(budget=4, num_repr_queries=8,
                                                projection_dim=4, seed=7))) < TOL32
    assert _rel(apb_attention(q, k, v, chunk, world,
                              config=APBConfig(anchor_budget=4, num_anchor_queries=8)),
                apb_cpu(q, k, v, chunk, world,
                        config=APBConfig(anchor_budget=4, num_anchor_queries=8))) < TOL32


@pytest.mark.parametrize("fn", ["ring_attention", "fastkv_attention", "apb_attention"])
def test_operations_are_dtype_and_device_agnostic(fn: str) -> None:
    """device-agnostic 是可验证性的前提（本机无 CUDA，只能在本机对拍）。

    同时挡住"某处硬写了 float64 / cuda"这类退化。
    """
    import inspect
    from src.baselines import operators as O
    src = inspect.getsource(getattr(O, fn))
    assert "cuda" not in src
    assert "float64" not in src
    assert "double()" not in src


def test_no_per_query_python_loop_in_the_vectorized_layer() -> None:
    """向量化层的正确性判据之一：不得出现逐 query 的循环。

    `for r in range(L_total)` 是 CPU 参考的标志；若它出现在本层，
    E6 的 T_comp 就变成核启动开销的度量（历史上正是这样得出过
    "Ring 比 DCC-KV 快 1.99×" 的假结论）。
    """
    import inspect
    from src.baselines import operators as O
    for fn in (O.ring_attention, O.fastkv_attention, O.apb_attention,
               O._compact_ring):
        src = inspect.getsource(fn)
        assert "range(L_total)" not in src.replace("range(0, L_total", ""), (
            f"{fn.__name__} 里出现了逐 query 循环"
        )
