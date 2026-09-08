"""APB CPU mock 实现（Huang et al. ACL 2025）。

Reference: arXiv:2502.12085

APB = Anchor/Passing Blocks：在分布式推理中，每张卡选一些"重要"KV
作为 passing block 发给后续卡。所有卡共享同一组 passing block。

CPU mock 简化版：
- 选 attention mass 最高的 B 个 KV 作为 passing block
- 所有目的卡收到同一份 passing block
- 跟 Ring Attention 配合

跟 DCC-KV 的区别：APB 是"全网共享"，DCC-KV 是"边级独立"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import torch

from ..dcc_kv_ref import (
    OnlineSoftmaxState,
    online_softmax_from_attention,
    merge_softmax_states,
)


@dataclass
class APBConfig:
    """APB 配置。"""
    anchor_budget: int = 128  # 每张卡的 anchor block 大小
    num_anchor_queries: int = 32


def select_anchor_blocks(
    keys: torch.Tensor,         # [L_s, d_h]
    values: torch.Tensor,       # [L_s, d_v]
    queries: torch.Tensor,      # [L_r, d_h]
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """APB 的 anchor block 选择：基于 attention mass。

    Args:
        keys, values: 源块 K/V
        queries: 后续卡的代表 queries
        budget: anchor 数

    Returns:
        anchor_keys: [budget, d_h]
        anchor_values: [budget, d_v]
    """
    d_h = keys.shape[-1]
    scale = 1.0 / (d_h ** 0.5)
    logits = (queries @ keys.T) * scale
    attention = torch.softmax(logits, dim=-1)
    # 每个 token 的总 attention mass
    mass = attention.sum(dim=0)  # [L_s]
    topk = torch.topk(mass, budget).indices
    return keys[topk], values[topk]


def apb_cpu(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    chunk_size: int,
    world_size: int,
    config: Optional[APBConfig] = None,
    causal: bool = True,
) -> torch.Tensor:
    """APB + Ring Attention mock。

    Args:
        queries, keys, values: 完整序列
        chunk_size, world_size: 分块
        config: APB 配置
        causal: 因果

    Returns:
        outputs: [L_total, d_v]
    """
    if config is None:
        config = APBConfig()

    L_total = queries.shape[0]
    d_h = keys.shape[-1]
    scale = 1.0 / (d_h ** 0.5)

    # 每块选 anchor
    anchor_keys_list: List[torch.Tensor] = []
    anchor_values_list: List[torch.Tensor] = []
    chunk_offsets: List[int] = []
    offset = 0
    for s in range(world_size):
        chunk_end = (s + 1) * chunk_size if s < world_size - 1 else L_total
        K_s = keys[offset:chunk_end]
        V_s = values[offset:chunk_end]
        # 用后续卡的 queries 选 anchor（APB 的设计）
        future_queries = queries[offset:]
        if len(future_queries) == 0:
            future_queries = queries
        a_k, a_v = select_anchor_blocks(
            K_s, V_s, future_queries,
            budget=min(config.anchor_budget, len(K_s)),
        )
        anchor_keys_list.append(a_k)
        anchor_values_list.append(a_v)
        chunk_offsets.append(offset)
        offset = chunk_end

    # Ring attention：每张卡用 anchor K/V 替代完整 K/V
    outputs = []
    for r in range(L_total):
        states: List[OnlineSoftmaxState] = []
        for s in range(world_size):
            if causal and chunk_offsets[s] > r:
                continue
            # 因果：只取前 r+1-chunk_offsets[s] 个
            if causal:
                end_in_chunk = min(r + 1 - chunk_offsets[s], anchor_keys_list[s].shape[0])
                if end_in_chunk <= 0:
                    continue
                K_v = anchor_keys_list[s][:end_in_chunk]
                V_v = anchor_values_list[s][:end_in_chunk]
            else:
                K_v = anchor_keys_list[s]
                V_v = anchor_values_list[s]

            logits = (queries[r] @ K_v.T) * scale
            state = online_softmax_from_attention(logits, V_v)
            states.append(state)

        if not states:
            outputs.append(torch.zeros(values.shape[-1], dtype=queries.dtype))
        else:
            merged = states[0]
            for st in states[1:]:
                merged = merge_softmax_states(merged, st)
            outputs.append(merged.o / merged.l)

    return torch.stack(outputs, dim=0)
