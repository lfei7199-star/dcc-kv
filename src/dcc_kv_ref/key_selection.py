"""M1.2: Key 选择（RMS-based TopK）。

论文 DCC-KV eq.(13)(14)：
- A_{s→r} = softmax(Q̂_r K_s^T / √d_h)         [M, L_s]
- u_{s→r}_j = sqrt(1/M * sum_a (A_{s→r}_{a,j})^2)  RMS across queries
- S_{s→r}^K = TopK({u_j}_{j=1}^{L_s}, B_{s,r})
- C_{s→r}^K = K_s[S_{s→r}^K, :]
"""
from __future__ import annotations

import torch


def rms_per_token_score(
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """对 attention 权重按 token 维度算 RMS。

    Args:
        attention_weights: [M, L] M 个代表 query 对 L 个 token 的 attention

    Returns:
        rms_scores: [L] 每个 token 的 RMS 分数
    """
    M = attention_weights.shape[0]
    return torch.sqrt((attention_weights ** 2).sum(dim=0) / M)


def select_topk_keys(
    keys: torch.Tensor,
    representative_queries: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS 排序 TopK Key 选择。

    Args:
        keys: [L_s, head_dim] 源块的 K
        representative_queries: [M, head_dim] 目的端的代表 Q
        budget: B_{s,r} 要保留的 token 数

    Returns:
        selected_keys: [budget, head_dim] 选中的 K 子集
        selected_indices: [budget] 在原 keys 中的索引
    """
    L_s, d_h = keys.shape
    M = representative_queries.shape[0]

    if budget >= L_s:
        return keys, torch.arange(L_s)

    # A = softmax(Q̂ K^T / √d)  [M, L_s]
    scale = 1.0 / (d_h ** 0.5)
    logits = (representative_queries @ keys.T) * scale
    attention = torch.softmax(logits, dim=-1)

    # RMS per token
    rms_scores = rms_per_token_score(attention)

    # TopK
    topk_scores, topk_indices = torch.topk(rms_scores, budget)
    selected_keys = keys[topk_indices]
    return selected_keys, topk_indices
