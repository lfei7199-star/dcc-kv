"""Ring Attention CPU 参考实现（Liu et al. 2023）。

Reference: arXiv:2310.01889

简化版（不优化性能）：
- 序列切到 world_size 块
- 每张卡循环处理其他卡的 KV 块
- Online Softmax 归并

CPU 即可跑。用于 Phase E.1（基线复现）。
"""
from __future__ import annotations

from typing import List, Optional
import torch

from ..dcc_kv_ref import (
    OnlineSoftmaxState,
    online_softmax_from_attention,
    merge_softmax_states,
)


def ring_attention_cpu(
    queries: torch.Tensor,         # [L_total, d_h]
    keys_chunks: List[torch.Tensor],   # List of [L_s, d_h]
    values_chunks: List[torch.Tensor], # List of [L_s, d_v]
    causal: bool = True,
) -> torch.Tensor:
    """单卡视角的 ring attention：循环累加所有 chunk 的 attention。

    Args:
        queries: [L_total, d_h] 完整 queries（单卡视角，模拟全可见）
        keys_chunks: world_size 个块的 K
        values_chunks: world_size 个块的 V
        causal: 因果掩码

    Returns:
        outputs: [L_total, d_v]
    """
    L_total = queries.shape[0]
    d_h = queries.shape[-1]
    scale = 1.0 / (d_h ** 0.5)

    outputs = []
    for r in range(L_total):
        states: List[OnlineSoftmaxState] = []
        offset = 0
        for s, (K_s, V_s) in enumerate(zip(keys_chunks, values_chunks)):
            L_s = K_s.shape[0]
            if causal and offset > r:
                offset += L_s
                continue
            if causal:
                end_in_chunk = min(L_s, r + 1 - offset)
                K_s_visible = K_s[:end_in_chunk]
                V_s_visible = V_s[:end_in_chunk]
            else:
                K_s_visible = K_s
                V_s_visible = V_s

            if len(K_s_visible) == 0:
                offset += L_s
                continue

            logits = (queries[r] @ K_s_visible.T) * scale
            state = online_softmax_from_attention(logits, V_s_visible)
            states.append(state)
            offset += L_s

        if not states:
            outputs.append(torch.zeros(values_chunks[0].shape[-1], dtype=queries.dtype))
        else:
            merged = states[0]
            for st in states[1:]:
                merged = merge_softmax_states(merged, st)
            outputs.append(merged.o / merged.l)

    return torch.stack(outputs, dim=0)


def ring_attention_dense(
    queries: torch.Tensor,    # [L, d_h]
    keys: torch.Tensor,       # [L, d_h]
    values: torch.Tensor,     # [L, d_v]
    chunk_size: int,
    world_size: int,
    causal: bool = True,
) -> torch.Tensor:
    """Dense 版本（用于 cross-check）。"""
    L = queries.shape[0]
    d_h = queries.shape[-1]
    scale = 1.0 / (d_h ** 0.5)

    outputs = []
    for r in range(L):
        end = (r // chunk_size + 1) * chunk_size if causal else L
        end = min(end, L)
        K = keys[:end]
        V = values[:end]
        logits = (queries[r] @ K.T) * scale
        if causal:
            mask = torch.zeros(end, dtype=torch.bool)
            mask[:r + 1] = True
            logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))
        state = online_softmax_from_attention(logits, V)
        outputs.append(state.o / state.l)
    return torch.stack(outputs, dim=0)
