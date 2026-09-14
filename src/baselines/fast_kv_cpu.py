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

try:  # 允许两种导入根：包内 `src.` 导入 与 把 `src/` 直接加入 sys.path
    from ..dcc_kv_ref import (
        build_compact_kv, CompactKV, OnlineSoftmaxState,
        online_softmax_from_attention, merge_softmax_states,
        DEFAULT_LAMBDA_BETA,
    )
except ImportError:  # pragma: no cover - 兼容把 src/ 直接加入 sys.path 的调用方
    from dcc_kv_ref import (
        build_compact_kv, CompactKV, OnlineSoftmaxState,
        online_softmax_from_attention, merge_softmax_states,
        DEFAULT_LAMBDA_BETA,
    )


@dataclass
class FastKVConfig:
    """FastKV 配置。"""
    budget: int = 256
    num_repr_queries: int = 64
    projection_dim: int = 32
    # 修正（2026-09-14）：原来硬编码 1e-3，未跟随 dcc_kv_ref 的校准默认值。
    # 该 baseline 与 DCC-KV **共用同一套压缩机制**，区别只在"是否按目的端
    # 条件化"，因此 λ_β 必须与主方法取同一个默认值，否则对照不公平
    # （此时 FastKV 用的是已被替换掉的旧默认）。实测结论不变，但数字会变。
    lambda_beta: float = DEFAULT_LAMBDA_BETA
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
            # 因果掩码按**原始位置**（selected_indices），不是按压缩块的行序切片。
            # 修正（2026-09-14）：压缩块的行序由选键决定（RMS 排序），与位置序无关。
            # 旧实现写的是 compact.keys[:end_in_chunk]，等于把"score 最高的前 k 个"
            # 当成"位置最靠前的前 k 个"，破坏因果语义（$B=L_s$ 时索引恰好升序，
            # 故该 bug 在 $B<L_s$ 时才改变数值）。定位见
            # experiments/cpu/c10_baseline_diagnosis.py。
            if causal:
                visible = compact.selected_indices < (r + 1 - chunk_offsets[s])
                if not bool(visible.any()):
                    continue
                K_visible = compact.keys[visible]
                V_visible = compact.values[visible]
                bias_visible = compact.logit_bias[visible]
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
