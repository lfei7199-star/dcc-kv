"""M1.5: CompactKV 数据结构 + 顶层 build_compact_kv。

对应 M1 用户提到的：
@dataclass(frozen=True)
class CompactKV:
    keys: torch.Tensor
    logit_bias: torch.Tensor
    values: torch.Tensor
    selected_indices: torch.Tensor
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .representative_query import select_representative_queries
from .key_selection import select_topk_keys
from .calibration import fit_logit_bias
from .value_regression import fit_compact_value


@dataclass(frozen=True)
class CompactKV:
    """A destination-conditioned compact representation of one source KV block.

    Fields:
        keys:             [budget, head_dim] 紧凑 K
        logit_bias:       [budget]           β 偏置
        values:           [budget, value_dim] 紧凑 V
        selected_indices: [budget]           在原 K/V 中的索引
    """
    keys: torch.Tensor
    logit_bias: torch.Tensor
    values: torch.Tensor
    selected_indices: torch.Tensor

    def __post_init__(self):
        assert self.keys.dim() == 2, f"keys must be 2D [budget, head_dim], got {self.keys.dim()}D"
        assert self.logit_bias.dim() == 1, f"logit_bias must be 1D, got {self.logit_bias.dim()}D"
        assert self.values.dim() == 2, f"values must be 2D, got {self.values.dim()}D"
        assert self.selected_indices.dim() == 1, f"selected_indices must be 1D, got {self.selected_indices.dim()}D"
        B = self.keys.shape[0]
        assert self.logit_bias.shape[0] == B
        assert self.values.shape[0] == B
        assert self.selected_indices.shape[0] == B


def build_compact_kv(
    source_keys: torch.Tensor,         # [L_s, head_dim]
    source_values: torch.Tensor,       # [L_s, value_dim]
    destination_queries: torch.Tensor, # [L_r, head_dim]  目的端的 Q
    budget: int,                       # B_{s,r}
    num_representative_queries: int = 64,
    projection_dim: int = 32,
    lambda_beta: float = 1e-3,
    lambda_value: float = 1e-3,
    seed: int = 42,
) -> CompactKV:
    """DCC-KV 顶层 API：构建目的端条件化的紧凑 KV。

    Pipeline:
    1. select_representative_queries: 从目的端 queries 选 M 个代表
    2. select_topk_keys: RMS 排序选 B 个 key
    3. fit_logit_bias: 拟合 β（用 NNLS）
    4. fit_compact_value: 回归 V（用 ridge）

    Args:
        source_keys: 源块的 K
        source_values: 源块的 V
        destination_queries: 目的端的所有 Q（采样用）
        budget: 压缩预算 B
        num_representative_queries: M
        projection_dim: d_p
        lambda_beta: λ_β
        lambda_value: λ_v
        seed: 随机种子

    Returns:
        CompactKV
    """
    # Step 1: 代表 Query
    repr_queries, _ = select_representative_queries(
        destination_queries,
        num_samples=num_representative_queries,
        projection_dim=projection_dim,
        seed=seed,
    )

    # Step 2: Key 选择
    compact_keys, selected_idx = select_topk_keys(
        source_keys, repr_queries, budget=budget,
    )

    # Step 3: β 拟合 — 需要原始块对代表 Q 的 attention 质量
    M, d_h = repr_queries.shape
    scale = 1.0 / (d_h ** 0.5)
    logits_orig = (repr_queries @ source_keys.T) * scale
    A_orig = torch.softmax(logits_orig, dim=-1)
    block_mass = A_orig.sum(dim=-1)  # [M]

    beta = fit_logit_bias(
        repr_queries, compact_keys, block_mass, lambda_reg=lambda_beta,
    )

    # Step 4: V 回归
    compact_values = fit_compact_value(
        repr_queries, source_keys, source_values, compact_keys, beta,
        lambda_reg=lambda_value,
    )

    return CompactKV(
        keys=compact_keys,
        logit_bias=beta,
        values=compact_values,
        selected_indices=selected_idx,
    )
