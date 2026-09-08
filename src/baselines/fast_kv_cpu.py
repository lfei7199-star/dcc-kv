"""FastKV + Ring Attention（CPU mock）。

FastKV = Zweiger et al. 2025（MIT-HAN-Lab，Attention Matching）
       + Ring Attention 单机压缩变体

注意：这是"共享压缩"基线——所有目的端共享同一份紧凑 KV。
跟 DCC-KV 的"边级独立压缩"形成对比。

CPU mock：不做性能优化，只保证算法逻辑正确。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import torch

from ..dcc_kv_ref import (
    build_compact_kv, CompactKV, OnlineSoftmaxState,
    online_softmax_from_attention, merge_softmax_states,
)


@dataclass
class FastKVConfig:
    """FastKV 配置。"""
    budget: int = 256
    num_repr_queries: int = 64
    projection_dim: int = 32
    lambda_beta: float = 1e-3
    lambda_value: float = 1e-3
    seed: int = 42


def fast_kv_cpu(
    queries: torch.Tensor,         # [L_total, d_h]
    keys: torch.Tensor,            # [L_total, d_h]
    values: torch.Tensor,          # [L_total, d_v]
    chunk_size: int,
    world_size: int,
    config: Optional[FastKVConfig] = None,
    causal: bool = True,
) -> torch.Tensor:
    """FastKV + Ring：每块独立压缩一次，所有 query 共享该压缩。

    关键差异 vs DCC-KV：每块只有一份 C，**不**根据目的端 query 区分。

    Args:
        queries, keys, values: 完整序列
        chunk_size, world_size: 分块参数
        config: FastKV 配置
        causal: 因果掩码

    Returns:
        outputs: [L_total, d_v]
    """
    if config is None:
        config = FastKVConfig()

    L_total = queries.shape[0]
    d_h = keys.shape[-1]
    scale = 1.0 / (d_h ** 0.5)

    # 切块 + 共享压缩
    chunk_compacts: List[CompactKV] = []
    chunk_offsets: List[int] = []
    offset = 0
    for s in range(world_size):
        chunk_end = (s + 1) * chunk_size if s < world_size - 1 else L_total
        K_s = keys[offset:chunk_end]
        V_s = values[offset:chunk_end]
        # 共享压缩：用全部 queries 作为"目的端 query"
        compact = build_compact_kv(
            source_keys=K_s,
            source_values=V_s,
            destination_queries=queries,
            budget=min(config.budget, len(K_s)),
            num_representative_queries=config.num_repr_queries,
            projection_dim=config.projection_dim,
            lambda_beta=config.lambda_beta,
            lambda_value=config.lambda_value,
            seed=config.seed,
        )
        chunk_compacts.append(compact)
        chunk_offsets.append(offset)
        offset = chunk_end

    # Ring attention：用 compact K/V 替代
    outputs = []
    for r in range(L_total):
        states: List[OnlineSoftmaxState] = []
        for s in range(world_size):
            if causal and chunk_offsets[s] > r:
                continue
            compact = chunk_compacts[s]
            # 因果：只取前 r+1-chunk_offsets[s] 个
            if causal:
                end_in_chunk = min(r + 1 - chunk_offsets[s], compact.keys.shape[0])
                if end_in_chunk <= 0:
                    continue
                K_visible = compact.keys[:end_in_chunk]
                V_visible = compact.values[:end_in_chunk]
                bias_visible = compact.logit_bias[:end_in_chunk]
            else:
                K_visible = compact.keys
                V_visible = compact.values
                bias_visible = compact.logit_bias

            logits = (queries[r] @ K_visible.T) * scale + bias_visible
            state = online_softmax_from_attention(logits, V_visible)
            states.append(state)

        if not states:
            outputs.append(torch.zeros(values.shape[-1], dtype=queries.dtype))
        else:
            merged = states[0]
            for st in states[1:]:
                merged = merge_softmax_states(merged, st)
            outputs.append(merged.o / merged.l)

    return torch.stack(outputs, dim=0)
