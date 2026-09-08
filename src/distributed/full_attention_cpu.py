"""M2 / Phase B.1: 完整分布式 attention 的 CPU 参考实现。

把 K/V 沿序列维度切成 world_size 块，每张卡持一块；循环累加计算。
输出应与单卡 dense attention 在 FP64 容差内一致。

目的：
- CPU 验证分布式 attention 的数值正确性（M2 启动前）
- 给 DCC-KV 同步版做"baseline"对照（M2 主实验）

不需要 GPU。CPU 即可跑 2-4 进程。
"""
from __future__ import annotations

from typing import List, Optional
import torch

from .launch_dist import setup_distributed, cleanup_distributed
from ..dcc_kv_ref import OnlineSoftmaxState, online_softmax_from_attention, merge_softmax_states


def full_distributed_attention_single_rank(
    query: torch.Tensor,         # [d_h]
    all_keys_chunks: List[torch.Tensor],   # List of [L_s, d_h]
    all_values_chunks: List[torch.Tensor], # List of [L_s, d_v]
    causal_mask: Optional[torch.Tensor] = None,  # [L_total] bool, True = allowed
) -> torch.Tensor:
    """单卡视角：循环累加所有 (s, r) 块的 attention。

    Args:
        query: [d_h] 单个 query（单 head）
        all_keys_chunks: world_size 个块的 K
        all_values_chunks: world_size 个块的 V
        causal_mask: [L_total] 因果掩码，True 表示允许 attend

    Returns:
        output: [d_v] attention 输出
    """
    states: List[OnlineSoftmaxState] = []
    offset = 0
    for s, (K_s, V_s) in enumerate(zip(all_keys_chunks, all_values_chunks)):
        L_s = K_s.shape[0]

        # 因果掩码
        if causal_mask is not None:
            allowed = causal_mask[offset:offset + L_s]
            if not allowed.any():
                offset += L_s
                continue
            # 简化：把不允许的 token 设成 -inf logit
            logits = (query @ K_s.T) / (K_s.shape[-1] ** 0.5)
            logits = torch.where(allowed, logits, torch.full_like(logits, float('-inf')))
        else:
            logits = (query @ K_s.T) / (K_s.shape[-1] ** 0.5)

        state = online_softmax_from_attention(logits, V_s)
        states.append(state)
        offset += L_s

    if not states:
        # 全部被因果掩码屏蔽
        return torch.zeros(all_values_chunks[0].shape[-1], dtype=query.dtype)

    merged = states[0]
    for s in states[1:]:
        merged = merge_softmax_states(merged, s)
    return merged.o / merged.l


def full_distributed_attention_dense(
    queries: torch.Tensor,         # [L_total, d_h]
    keys: torch.Tensor,            # [L_total, d_h]
    values: torch.Tensor,          # [L_total, d_v]
    chunk_size: int,
    world_size: int,
    causal: bool = True,
) -> torch.Tensor:
    """单卡 dense attention（参考实现）：整段序列算一次 attention。

    用于 M2 启动后，跟分布式版本逐元素对比。

    Args:
        queries, keys, values: [L_total, d_*]
        chunk_size: 每块大小
        world_size: 卡数
        causal: 是否因果

    Returns:
        outputs: [L_total, d_v]
    """
    L_total = queries.shape[0]
    d_h = queries.shape[-1]
    scale = 1.0 / (d_h ** 0.5)

    outputs = []
    for r in range(L_total):
        # 收集所有 r 能看到的块
        end = (r // chunk_size + 1) * chunk_size if causal else L_total
        end = min(end, L_total)
        K = keys[:end]
        V = values[:end]

        # 算 attention
        logits = (queries[r] @ K.T) * scale
        if causal:
            mask = torch.zeros(end, dtype=torch.bool)
            mask[:r + 1] = True
            logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))

        state = online_softmax_from_attention(logits, V)
        out = state.o / state.l
        outputs.append(out)

    return torch.stack(outputs, dim=0)


def _full_dist_worker(rank, args):
    """M2 worker：跑完整分布式 attention，输出结果（仅 rank 0 保存）。"""
    setup_distributed(rank, args["world_size"], backend="gloo")

    L_total = args["L_total"]
    chunk_size = L_total // args["world_size"]
    d_h = args["d_h"]
    d_v = args["d_v"]
    causal = args["causal"]

    # 每个 rank 持有自己的 chunk
    start = rank * chunk_size
    end = start + chunk_size if rank < args["world_size"] - 1 else L_total

    # 共享的 Q / K / V（用同一份 seed 复现）
    g = torch.Generator().manual_seed(args["seed"])
    full_q = torch.randn(L_total, d_h, generator=g, dtype=torch.float64)
    full_k = torch.randn(L_total, d_h, generator=g, dtype=torch.float64)
    full_v = torch.randn(L_total, d_v, generator=g, dtype=torch.float64)

    my_q = full_q[start:end]
    my_k = full_k[start:end]
    my_v = full_v[start:end]

    # 循环累加：每个 rank 依次处理所有块
    my_outputs = []
    for r in range(start, end):
        # r 视角：收集所有 s <= r 的块
        states = []
        s_start = 0
        for s in range(args["world_size"]):
            s_chunk_end = (s + 1) * chunk_size if s < args["world_size"] - 1 else L_total
            K_s = full_k[s_start:s_chunk_end]
            V_s = full_v[s_start:s_chunk_end]

            if not causal or s_chunk_end <= r + 1:
                # 全部 attend
                logits = (my_q[r - start] @ K_s.T) / (d_h ** 0.5)
                state = online_softmax_from_attention(logits, V_s)
            else:
                # 部分 attend（前 r+1 个 token）
                L_visible = r + 1 - s_start
                K_s = K_s[:L_visible]
                V_s = V_s[:L_visible]
                logits = (my_q[r - start] @ K_s.T) / (d_h ** 0.5)
                state = online_softmax_from_attention(logits, V_s)
            states.append(state)
            s_start = s_chunk_end

        if not states:
            my_outputs.append(torch.zeros(d_v, dtype=torch.float64))
        else:
            merged = states[0]
            for s in states[1:]:
                merged = merge_softmax_states(merged, s)
            my_outputs.append(merged.o / merged.l)

    my_output = torch.stack(my_outputs, dim=0)

    # 收集到 rank 0
    import torch.distributed as dist
    gathered = [torch.zeros_like(my_output) for _ in range(args["world_size"])]
    dist.gather(my_output, gathered if rank == 0 else None, dst=0)

    if rank == 0:
        full_output = torch.cat(gathered, dim=0)
        # 跟 dense 对比
        dense_output = full_distributed_attention_dense(
            full_q, full_k, full_v, chunk_size, args["world_size"], causal=causal,
        )
        max_diff = (full_output - dense_output).abs().max().item()
        mean_diff = (full_output - dense_output).abs().mean().item()
        print(f"max_diff = {max_diff:.2e}, mean_diff = {mean_diff:.2e}")
        assert max_diff < 1e-5, f"Distributed != Dense, max_diff = {max_diff}"

    cleanup_distributed()
    return f"rank_{rank}_ok"
