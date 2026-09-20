"""G3：三个基线的**算子化**实现（device-agnostic，可 GPU）。

缺口
----
`ring_attention_cpu` / `fast_kv_cpu` / `apb_cpu` 都是**逐 query 的 Python 循环**：

    for r in range(L_total):        # L_total 可达 32768
        for s in range(world_size):
            logits = queries[r] @ K.T
            state = online_softmax_from_attention(logits, V)
        outputs.append(merged.o / merged.l)

两个后果，都不是"慢一点"而已：

1. **GPU 上根本用不了**：E6 主表要按方法计时并算任务指标，逐 query 循环在
   L=32k 时是 3 万多次核启动，测出来的 T_comp 是启动开销而非算子代价 ——
   拿它去和 DCC-KV 的向量化路径比，比的是实现质量而不是方法。
2. **归并次序是人写的**：每个 query 独立走一遍 online softmax 再两两合并，
   与"整块一次 softmax"是两套浮点路径。基线之间的差异里会混进归并次序伪影。

本模块把三者改写成对**整个 query 矩阵**一次算的形式，复用 G1 的算子核
（`compact_kv_attention` / `dense_attention` / `merge_partial_attention`）。
语义必须与 CPU 参考**逐条对齐**，包括三处曾经踩过的坑：

- **因果掩码按原始位置**，不是按压缩块/锚点数组的行序切片
  （`fast_kv_cpu` 与 `apb_cpu` 各犯过一次，定位见
  `experiments/cpu/c10_baseline_diagnosis.py`）。这里一律走
  `attention_kernel.causal_visibility`，它只认 `selected_indices`。
- **APB 的锚点用"后续卡的 queries"选**（`queries[offset:]`），不是全体 queries。
- **FastKV 的共享压缩用全体 queries 选**，且 λ_β 与主方法同源。

等价强度：**ULP 级，不是逐位**。向量化把逐 query 的 online softmax 换成
"分块 partial + lse 归并"，求和次序因此改变（与 G1 的 `query_chunk`、
G2 的 `n_chunks` 是同一现象）。测试用相对容差而非 `torch.equal` —— 见
`tests/test_baseline_operators.py`。

`query_chunk` 参数沿 query 维分块，用来把中间张量 `[Lq, L_s]` 的峰值显存压住；
它同样是 ULP 级旋钮，不是逐位旋钮。

**本模块的设备无关性是可验证性的前提**：本机没有 CUDA，只有 device-agnostic
的实现才能与 CPU 参考在本机逐条对拍。GPU 上它是同一段代码。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

try:  # 允许两种导入根：包内 `src.` 导入 与 把 `src/` 直接加入 sys.path
    from ..dcc_kv_ref import CompactKV, build_compact_kv
    from ..dcc_kv_ref import attention_kernel as K
except ImportError:  # pragma: no cover - 兼容把 src/ 直接加入 sys.path 的调用方
    # ⚠️ 首版这里只写了 `from ..dcc_kv_ref import ...`，于是
    # `import baselines`（src/ 在 sys.path 上）会抛
    # "attempted relative import beyond top-level package"，
    # 把 tests/test_dist_equivalence.py 整个模块打成收集错误。
    # 本仓库既有模块（apb_cpu / fast_kv_cpu）都用同一套双根写法，这里必须跟随。
    from dcc_kv_ref import CompactKV, build_compact_kv
    from dcc_kv_ref import attention_kernel as K

from .apb_cpu import APBConfig, select_anchor_blocks
from .fast_kv_cpu import FastKVConfig


def _chunk_bounds(L_total: int, chunk_size: int, world_size: int) -> List[Tuple[int, int]]:
    """块边界 [(start, end)]，与 CPU 版逐字一致。

    CPU 版用的是 `(s+1)*chunk_size if s < world_size-1 else L_total`
    —— 最后一块**吃掉剩下的全部**，不按 chunk_size 截断。这一点必须保留：
    若改成严格等分，`chunk_size * world_size != L_total` 时块数就变了。
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正，得到 {chunk_size}")
    if world_size <= 0:
        raise ValueError(f"world_size 必须为正，得到 {world_size}")
    bounds: List[Tuple[int, int]] = []
    offset = 0
    for s in range(world_size):
        end = (s + 1) * chunk_size if s < world_size - 1 else L_total
        end = min(end, L_total)
        bounds.append((offset, end))
        offset = end
    return bounds


def _iter_query_slices(L_total: int, query_chunk: Optional[int]):
    step = int(query_chunk) if query_chunk else L_total
    step = max(1, step)
    for lo in range(0, L_total, step):
        yield lo, min(L_total, lo + step)


# =============================================================================
# 1) Ring Attention：无压缩，只有块级因果
# =============================================================================

def ring_attention(
    queries: torch.Tensor,
    keys_chunks: Sequence[torch.Tensor],
    values_chunks: Sequence[torch.Tensor],
    *,
    causal: bool = True,
    query_chunk: Optional[int] = None,
) -> torch.Tensor:
    """`ring_attention_cpu` 的向量化版：整块 query 一次算。

    CPU 参考逐个 query 循环 `world_size` 个块；这里对每个来源块算一次
    完整的 `[Lq, L_s]` 注意力（因果掩码交给算子核），再用 lse 归并。

    Args:
        queries: [L_total, d_h]
        keys_chunks / values_chunks: world_size 个 [L_s, d_h] / [L_s, d_v]
        causal: 因果掩码（按**全局位置**）
        query_chunk: 沿 query 维分块（显存控制，非逐位旋钮）

    Returns:
        outputs: [L_total, d_v]
    """
    if len(keys_chunks) != len(values_chunks):
        raise ValueError("keys_chunks 与 values_chunks 长度必须相同")
    if len(keys_chunks) == 0:
        raise ValueError("至少需要一个来源块")
    L_total = int(queries.shape[0])
    d_v = int(values_chunks[0].shape[-1])
    if L_total == 0:
        return queries.new_zeros((0, d_v))

    qp = torch.arange(L_total, device=queries.device)
    outs: List[torch.Tensor] = []

    for lo, hi in _iter_query_slices(L_total, query_chunk):
        q_slice = queries[lo:hi]
        qp_slice = qp[lo:hi]
        partials: List[K.PartialAttention] = []
        offset = 0
        for K_s, V_s in zip(keys_chunks, values_chunks):
            L_s = int(K_s.shape[0])
            if causal:
                # 该块整体位于本切片所有 query 之后 ⇒ 全被掩掉，跳过即可
                if offset > int(qp_slice[-1]):
                    offset += L_s
                    continue
                partials.append(K.dense_attention(
                    q_slice, K_s, V_s, query_positions=qp_slice,
                    block_offset=offset, causal=True, return_lse=True))
            else:
                partials.append(K.dense_attention(
                    q_slice, K_s, V_s, causal=False, return_lse=True))
            offset += L_s
        outs.append(_merge_sources_or_zeros(partials, q_slice, hi - lo, d_v))
    return torch.cat(outs, dim=0)


# =============================================================================
# 2) FastKV + Ring：共享压缩（所有目的端共用一份紧凑 KV）
# =============================================================================

def _shared_compacts(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    bounds: Sequence[Tuple[int, int]],
    config: FastKVConfig,
) -> List[Tuple[CompactKV, int]]:
    """每块独立压缩一次，**所有 query 共享**（这是 FastKV 与 DCC-KV 的分野）。

    与 `fast_kv_cpu` 的构造循环逐字对齐：用**全体 queries** 当"目的端 query"、
    `budget=min(config.budget, len(K_s))`、`seed` 每块相同。

    返回 (紧凑块, 块起始位置) 成对，**不是**两个平行列表 —— 空块被跳过时，
    平行列表的偏移会整体错位一格，而错位后的因果掩码仍然"跑得通"。
    """
    out: List[Tuple[CompactKV, int]] = []
    for lo, hi in bounds:
        K_s = keys[lo:hi]
        V_s = values[lo:hi]
        if int(K_s.shape[0]) == 0:
            continue
        out.append((build_compact_kv(
            source_keys=K_s,
            source_values=V_s,
            destination_queries=queries,
            budget=min(int(config.budget), int(K_s.shape[0])),
            num_representative_queries=int(config.num_repr_queries),
            projection_dim=int(config.projection_dim),
            lambda_beta=float(config.lambda_beta),
            lambda_value=float(config.lambda_value),
            seed=int(config.seed),
        ), lo))
    return out


def fastkv_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    chunk_size: int,
    world_size: int,
    *,
    config: Optional[FastKVConfig] = None,
    causal: bool = True,
    query_chunk: Optional[int] = None,
) -> torch.Tensor:
    """`fast_kv_cpu` 的向量化版。语义与它逐条对齐，含因果掩码按原始位置。"""
    config = config or FastKVConfig()
    bounds = _chunk_bounds(int(queries.shape[0]), chunk_size, world_size)
    return _compact_ring(
        queries, _shared_compacts(queries, keys, values, bounds, config),
        causal=causal, query_chunk=query_chunk)


# =============================================================================
# 3) APB：锚点块（全网共享），无 β
# =============================================================================

def _anchor_compacts(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    bounds: Sequence[Tuple[int, int]],
    config: APBConfig,
) -> List[Tuple[CompactKV, int]]:
    """每块选锚点，用**该块之后的 queries** 选（APB 的设计）。

    返回的 `CompactKV` 的 `logit_bias` 恒为 0 —— APB 没有质量偏置 β。
    这一点必须保住：若沿用压缩链路的 `build_compact_kv`，就会给 APB 装上
    DCC-KV 才有的机制，对照失去意义。
    """
    out: List[Tuple[CompactKV, int]] = []
    for lo, hi in bounds:
        K_s = keys[lo:hi]
        V_s = values[lo:hi]
        if int(K_s.shape[0]) == 0:
            continue
        future = queries[lo:]
        if int(future.shape[0]) == 0:
            future = queries
        a_k, a_v, a_idx = select_anchor_blocks(
            K_s, V_s, future, budget=min(int(config.anchor_budget), int(K_s.shape[0])))
        out.append((CompactKV(
            keys=a_k,
            logit_bias=torch.zeros(int(a_k.shape[0]), dtype=a_k.dtype,
                                   device=a_k.device),
            values=a_v,
            selected_indices=a_idx,
        ), lo))
    return out


def apb_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    chunk_size: int,
    world_size: int,
    *,
    config: Optional[APBConfig] = None,
    causal: bool = True,
    query_chunk: Optional[int] = None,
) -> torch.Tensor:
    """`apb_cpu` 的向量化版。锚点选择与因果口径逐条对齐 CPU 参考。"""
    config = config or APBConfig()
    bounds = _chunk_bounds(int(queries.shape[0]), chunk_size, world_size)
    return _compact_ring(
        queries, _anchor_compacts(queries, keys, values, bounds, config),
        causal=causal, query_chunk=query_chunk)


# =============================================================================
# 共用的「紧凑块 ring」内核
# =============================================================================

def _merge_sources_or_zeros(
    partials: Sequence[K.PartialAttention],
    like: torch.Tensor,
    n_rows: int,
    d_v: int,
) -> torch.Tensor:
    """归并；没有任何可见来源时返回零向量（与 CPU 版的 `torch.zeros` 一致）。

    `merge_partial_attention` 在「全部块都无可见键」时本身就返回全零，
    所以这里不必特判 —— 但**空 partial 列表**必须单独处理，否则会走
    `merge` 的 `ValueError`。CPU 版那时给的正是零向量。
    """
    if not partials:
        return like.new_zeros((int(n_rows), int(d_v)))
    return K.merge_partial_attention(list(partials))


def _compact_ring(
    queries: torch.Tensor,
    blocks: Sequence[Tuple[CompactKV, int]],
    *,
    causal: bool,
    query_chunk: Optional[int],
) -> torch.Tensor:
    """对一串紧凑块做 ring 式注意力：整块 query 一次算 + lse 归并。

    因果可见性一律走 `attention_kernel.causal_visibility`（只认
    `selected_indices`），因此不会退化成"按数组行序切前缀"——那正是
    `fast_kv_cpu` / `apb_cpu` 各犯过一次的错。
    """
    if not blocks:
        raise ValueError("没有可用的紧凑块（源块全为空？）")
    L_total = int(queries.shape[0])
    d_v = int(blocks[0][0].values.shape[-1])
    qp = torch.arange(L_total, device=queries.device)
    outs: List[torch.Tensor] = []

    for lo, hi in _iter_query_slices(L_total, query_chunk):
        q_slice = queries[lo:hi]
        qp_slice = qp[lo:hi]
        partials: List[K.PartialAttention] = []
        for ck, off in blocks:
            if causal:
                vis = K.causal_visibility(ck.selected_indices, qp_slice, int(off))
                if not bool(vis.any()):
                    continue
            else:
                vis = None
            partials.append(K.compact_kv_attention(
                q_slice, ck, visible=vis, return_lse=True))
        outs.append(_merge_sources_or_zeros(partials, q_slice, hi - lo, d_v))
    return torch.cat(outs, dim=0)


__all__ = [
    "ring_attention",
    "fastkv_attention",
    "apb_attention",
    "FastKVConfig",
    "APBConfig",
]
