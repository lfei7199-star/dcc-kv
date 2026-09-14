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

try:  # 允许两种导入根：包内 `src.` 导入 与 把 `src/` 直接加入 sys.path
    from ..dcc_kv_ref import (
        OnlineSoftmaxState,
        online_softmax_from_attention,
        merge_softmax_states,
    )
except ImportError:  # pragma: no cover - 兼容把 src/ 直接加入 sys.path 的调用方
    from dcc_kv_ref import (
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """APB 的 anchor block 选择：基于 attention mass。

    Args:
        keys, values: 源块 K/V
        queries: 后续卡的代表 queries
        budget: anchor 数

    Returns:
        anchor_keys: [budget, d_h]
        anchor_values: [budget, d_v]
        anchor_indices: [budget]  **按位置升序**的原始索引

    修正（2026-09-14）：返回值新增 ``anchor_indices``。``torch.topk`` 给出的是
    **按 mass 降序**的索引，与位置序无关；因果掩码必须按原始位置做，故调用方
    需要索引本身。这里顺手把索引排成升序，使「位置前缀」就是数组前缀。
    """
    d_h = keys.shape[-1]
    scale = 1.0 / (d_h ** 0.5)
    logits = (queries @ keys.T) * scale
    attention = torch.softmax(logits, dim=-1)
    # 每个 token 的总 attention mass
    mass = attention.sum(dim=0)  # [L_s]
    topk = torch.topk(mass, budget).indices
    topk = torch.sort(topk).values
    return keys[topk], values[topk], topk


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
    anchor_indices_list: List[torch.Tensor] = []
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
        a_k, a_v, a_idx = select_anchor_blocks(
            K_s, V_s, future_queries,
            budget=min(config.anchor_budget, len(K_s)),
        )
        anchor_keys_list.append(a_k)
        anchor_values_list.append(a_v)
        anchor_indices_list.append(a_idx)
        chunk_offsets.append(offset)
        offset = chunk_end

    # Ring attention：每张卡用 anchor K/V 替代完整 K/V
    outputs = []
    for r in range(L_total):
        states: List[OnlineSoftmaxState] = []
        for s in range(world_size):
            if causal and chunk_offsets[s] > r:
                continue
            # 因果掩码按**原始位置**，不是按 anchor 数组的行序切片。
            # 修正（2026-09-14）：anchor 数组的行序由 attention mass 决定，与位置序
            # 无关（实测 topk 索引形如 [0, 33, 27, 22, ...]）。旧实现写的是
            # anchor_keys_list[s][:end_in_chunk]，等于把"mass 最高的前 k 个"当成
            # "位置最靠前的前 k 个"。定位见 experiments/cpu/c10_baseline_diagnosis.py。
            if causal:
                visible = anchor_indices_list[s] < (r + 1 - chunk_offsets[s])
                if not bool(visible.any()):
                    continue
                K_v = anchor_keys_list[s][visible]
                V_v = anchor_values_list[s][visible]
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
