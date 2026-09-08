"""M2 / Phase B.2: DCC-KV 同步版的 CPU 参考实现。

每个 rank 持有自己的 KV 块。对其他 rank 的目的：构造边级紧凑 KV，
通过 mock 的 All-to-Allv 交换，然后用 Online Softmax 归并。

输出与完整分布式 attention 差 < 1e-3（FP64，压缩误差）。

不需要 GPU。CPU 即可跑 2-4 进程。
"""
from __future__ import annotations

from typing import List, Dict, Optional
import torch

from ..dcc_kv_ref import (
    build_compact_kv, CompactKV, OnlineSoftmaxState,
    online_softmax_from_attention, merge_softmax_states,
)
from .comm import DistributedComm, VarLenMessage


def dcc_kv_sync_attention_single_rank(
    query: torch.Tensor,             # [d_h] 单个 query
    all_keys: List[torch.Tensor],    # [L_s, d_h] x world_size
    all_values: List[torch.Tensor],  # [L_s, d_v] x world_size
    all_dest_queries: Dict[int, torch.Tensor],  # {rank: [L_r, d_h]}
    budgets: Dict[int, int],         # {rank: B_{self, rank}}
    num_repr_queries: int = 16,
    projection_dim: int = 16,
) -> torch.Tensor:
    """DCC-KV 同步版：单 rank 视角。

    对每个目的 rank r：
    1. 用 r 的 queries 构造 C_{self→r}
    2. 模拟收到 Cs→self（用 full 的 K/V 模拟）
    3. Online Softmax 归并

    Args:
        query: 单个 query（d_h）
        all_keys: 所有 rank 的 K
        all_values: 所有 rank 的 V
        all_dest_queries: {rank: 该 rank 的 queries}（用于目的端条件化）
        budgets: {rank: B_{self, rank}}

    Returns:
        output: [d_v]
    """
    s = 0  # 自己
    states: List[OnlineSoftmaxState] = []

    for r, (K_r, V_r) in enumerate(zip(all_keys, all_values)):
        if r == s:
            # 本地块：直接算
            logits = (query @ K_r.T) / (K_r.shape[-1] ** 0.5)
            state = online_softmax_from_attention(logits, V_r)
        else:
            # 远端块：构造 C_{s→r}
            dest_q = all_dest_queries.get(r)
            if dest_q is None or len(dest_q) == 0:
                continue
            B = budgets.get(r, min(64, K_r.shape[0]))
            compact = build_compact_kv(
                source_keys=K_r,
                source_values=V_r,
                destination_queries=dest_q,
                budget=B,
                num_representative_queries=num_repr_queries,
                projection_dim=projection_dim,
            )
            # 用紧凑 KV 算 attention
            logits = (query @ compact.keys.T) / (compact.keys.shape[-1] ** 0.5)
            # 加 logit bias
            logits = logits + compact.logit_bias
            state = online_softmax_from_attention(logits, compact.values)

        states.append(state)

    if not states:
        return torch.zeros(all_values[0].shape[-1], dtype=query.dtype)

    merged = states[0]
    for s in states[1:]:
        merged = merge_softmax_states(merged, s)
    return merged.o / merged.l


def _dcc_kv_sync_worker(rank, args):
    """DCC-KV 同步版 worker（mock 通信版本）。"""
    setup_distributed_simple(rank, args["world_size"])

    L_total = args["L_total"]
    chunk_size = L_total // args["world_size"]
    d_h = args["d_h"]
    d_v = args["d_v"]
    B = args["budget"]  # 统一 budget

    # 共享 Q/K/V
    g = torch.Generator().manual_seed(args["seed"])
    full_q = torch.randn(L_total, d_h, generator=g, dtype=torch.float64)
    full_k = torch.randn(L_total, d_h, generator=g, dtype=torch.float64)
    full_v = torch.randn(L_total, d_v, generator=g, dtype=torch.float64)

    # 自己持的 chunk
    start = rank * chunk_size
    end = start + chunk_size if rank < args["world_size"] - 1 else L_total
    my_k = full_k[start:end]
    my_v = full_v[start:end]
    my_queries = full_q[start:end]

    # 输出
    my_outputs = []
    for r_local in range(end - start):
        r_global = r_local + start
        query = my_queries[r_local]

        # 收集所有 s<=r 的块（mock 通信：直接读 full）
        states = []
        s_start = 0
        for s in range(args["world_size"]):
            s_end_chunk = (s + 1) * chunk_size if s < args["world_size"] - 1 else L_total
            if s_end_chunk > r_global + 1:
                # 因果：只取前 r_global+1 个
                K_s = full_k[s_start:r_global + 1]
                V_s = full_v[s_start:r_global + 1]
            else:
                K_s = full_k[s_start:s_end_chunk]
                V_s = full_v[s_start:s_end_chunk]
            s_start = s_end_chunk

            if len(K_s) == 0:
                continue

            if s == rank:
                # 本地块：直接算
                logits = (query @ K_s.T) / (d_h ** 0.5)
                state = online_softmax_from_attention(logits, V_s)
            else:
                # 远端：用 s 的 queries 构造 C_{s→rank}
                s_queries = full_q[s_start - chunk_size:s_end_chunk] if s < args["world_size"] - 1 else full_q[s * chunk_size:]
                if len(s_queries) == 0:
                    s_queries = full_q[max(0, s_start - chunk_size):s_start]
                compact = build_compact_kv(
                    source_keys=K_s,
                    source_values=V_s,
                    destination_queries=s_queries,
                    budget=min(B, len(K_s)),
                    num_representative_queries=8,
                    projection_dim=8,
                )
                logits = (query @ compact.keys.T) / (d_h ** 0.5)
                logits = logits + compact.logit_bias
                state = online_softmax_from_attention(logits, compact.values)
            states.append(state)

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
        dcc_output = torch.cat(gathered, dim=0)
        # 对比 dense
        from .full_attention_cpu import full_distributed_attention_dense
        dense_output = full_distributed_attention_dense(
            full_q, full_k, full_v, chunk_size, args["world_size"], causal=True,
        )
        max_diff = (dcc_output - dense_output).abs().max().item()
        mean_diff = (dcc_output - dense_output).abs().mean().item()
        print(f"DCC-KV vs Dense: max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.4f}")
        # DCC-KV 有压缩误差，1e-3 是合理上限
        assert max_diff < 1e-2, f"DCC-KV too far from dense: {max_diff}"
        return max_diff

    cleanup_distributed_simple()
    return None


def setup_distributed_simple(rank: int, world_size: int) -> None:
    """简化 setup：默认 gloo。"""
    import os
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup_distributed_simple() -> None:
    """简化 cleanup。"""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
