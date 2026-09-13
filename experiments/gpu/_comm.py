"""通信原语与 bench —— DCC-KV 的变长 All-to-Allv 及其同步/异步流水。

本模块是 A1/A2/A5/A6 共用的计量层。它刻意把三件事分开记账，因为论文
式~(speedup-bound) 的归因依赖这个拆分：

    T_build    逐边构造紧凑 KV 的代价（∝ |E_s| × 构造复杂度）
    T_comm     All-to-Allv 的实际通信耗时
    T_comp     目的端本地注意力计算的耗时
    T_total    wall-clock 总耗时

**只有 T_comm 被与 T_comp 重叠时，异步才有收益**。若构造代价 T_build 被
隐含地塞进 T_comp，异步的收益会被高估；若塞进 T_comm，会被低估。
因此本模块要求调用方显式传入各自的计时区间，不做任何"顺手一起算"。

变长语义
--------
DCC-KV 每条边的预算 B_{s,r} 不同，因此消息是变长的。这里用
`all_to_all_single` 的 `input_split_sizes` / `output_split_sizes` 表达，
而不是先 padding 到等长再发送 —— padding 会把通信量虚高，
使"压缩比"这一核心声明失真。
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.gpu import _env  # noqa: E402


# =============================================================================
# 紧凑 KV 的消息打包
# =============================================================================

def pack_compact_edge(
    keys: torch.Tensor,          # [B, d_h]
    logit_bias: torch.Tensor,    # [B]
    values: torch.Tensor,        # [B, d_v]
) -> torch.Tensor:
    """把一条边的紧凑 KV 打包成单张 [B, d_h + 1 + d_v] 张量。

    为什么把 β 也放进消息：它是权重的一部分，接收端没有它就无法复现
    有偏置的 logit。若把它排除在通信量之外，"压缩比"会被系统性高估
    （β 是 B 个浮点数，相对 B×(d_h+d_v) 很小，但不能因此忽略不计）。
    """
    return torch.cat([keys, logit_bias.unsqueeze(-1), values], dim=-1).contiguous()


def edge_message_bytes(B: int, d_h: int, d_v: int, itemsize: int) -> int:
    """单条边的消息字节数（口径与 pack_compact_edge 一致）。"""
    return int(B * (d_h + 1 + d_v) * itemsize)


@dataclass
class EdgePlan:
    """一次 All-to-Allv 各边的预算与体积计划。

    budgets[src][dst] 是源设备 src 发给目的设备 dst 的压缩预算 B_{s,r}。
    这在 DCC-KV 里本就是逐边量（不同目的端关注不同区域），
    等预算只是它在 A2/A5 里的特例。
    """
    world_size: int
    budgets: List[List[int]]          # budgets[src][dst]
    d_h: int
    d_v: int
    itemsize: int
    full_kv_len: int = 0              # 不压缩时每边的长度，用于算压缩比

    def outbound_bytes(self, src: int) -> int:
        return sum(edge_message_bytes(b, self.d_h, self.d_v, self.itemsize)
                   for b in self.budgets[src])

    def inbound_bytes(self, dst: int) -> int:
        return sum(edge_message_bytes(self.budgets[s][dst], self.d_h, self.d_v, self.itemsize)
                   for s in range(self.world_size))

    def full_outbound_bytes(self, src: int) -> int:
        """等价"不压缩"出站体积（每边发完整 L_s 的 K 与 V，无 β）。"""
        L = self.full_kv_len or max(max(row) for row in self.budgets)
        return int(self.world_size * L * (self.d_h + self.d_v) * self.itemsize)

    def compression_ratio(self, src: int) -> float:
        denom = self.full_outbound_bytes(src)
        return (self.outbound_bytes(src) / denom) if denom else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_size": self.world_size,
            "budgets": self.budgets,
            "d_h": self.d_h, "d_v": self.d_v, "itemsize": self.itemsize,
            "full_kv_len": self.full_kv_len,
        }


def make_uniform_plan(
    world_size: int,
    budget: int,
    d_h: int,
    d_v: int,
    itemsize: int,
    full_kv_len: int = 0,
) -> EdgePlan:
    """等预算计划：所有边同一预算 B（A2 的扫描对象）。"""
    return EdgePlan(
        world_size=world_size,
        budgets=[[budget] * world_size for _ in range(world_size)],
        d_h=d_h, d_v=d_v, itemsize=itemsize, full_kv_len=full_kv_len,
    )


def make_skewed_plan(
    world_size: int,
    base_budget: int,
    d_h: int,
    d_v: int,
    itemsize: int,
    skew: float = 1.0,
    full_kv_len: int = 0,
) -> EdgePlan:
    """偏斜预算计划：B_{s,r} 随 (s, r) 变化。

    设置 skew > 1 用来检验一个容易被忽略的后果：**变长消息的负载不均衡**。
    等预算下 all_to_all 是完美对称的；一旦逐边预算不同，某些 rank 的
    出站/入站体积会显著高于均值，此时报告的"平均通信量"会掩盖长尾。
    skew=1 时退化为等预算。
    """
    budgets = []
    for s in range(world_size):
        row = []
        for r in range(world_size):
            factor = skew ** ((s - r) % world_size / max(1, world_size - 1))
            row.append(max(1, int(round(base_budget * factor))))
        budgets.append(row)
    return EdgePlan(world_size=world_size, budgets=budgets,
                    d_h=d_h, d_v=d_v, itemsize=itemsize, full_kv_len=full_kv_len)


# =============================================================================
# All-to-Allv
# =============================================================================

def all_to_all_v(
    send: torch.Tensor,
    send_sizes: Sequence[int],
    recv_sizes: Sequence[int],
) -> torch.Tensor:
    """变长 All-to-Allv（阻塞版）。

    send: [sum(send_sizes), F]，按 dst 顺序拼接。
    recv: [sum(recv_sizes), F]
    """
    world = dist.get_world_size()
    assert len(send_sizes) == world, f"send_sizes 长度 {len(send_sizes)} != world {world}"
    assert len(recv_sizes) == world
    assert int(send.shape[0]) == int(sum(send_sizes))
    recv = torch.empty(int(sum(recv_sizes)), send.shape[1],
                       dtype=send.dtype, device=send.device)
    dist.all_to_all_single(
        recv, send,
        output_split_sizes=[int(s) for s in recv_sizes],
        input_split_sizes=[int(s) for s in send_sizes],
    )
    return recv


def all_to_all_v_async(
    send: torch.Tensor,
    send_sizes: Sequence[int],
    recv_sizes: Sequence[int],
) -> Tuple[torch.Tensor, Any]:
    """变长 All-to-Allv（非阻塞版），返回 (recv_buffer, handle)。"""
    world = dist.get_world_size()
    recv = torch.empty(int(sum(recv_sizes)), send.shape[1],
                       dtype=send.dtype, device=send.device)
    handle = dist.all_to_all_single(
        recv, send,
        output_split_sizes=[int(s) for s in recv_sizes],
        input_split_sizes=[int(s) for s in send_sizes],
        async_op=True,
    )
    return recv, handle


# =============================================================================
# 同步 / 异步流水
# =============================================================================

@dataclass
class PipelineTiming:
    """一次流水执行的耗时拆解（毫秒）。"""
    total_ms: float
    comm_ms: float
    comp_ms: float
    build_ms: float = 0.0


def run_sync_pipeline(
    payload: torch.Tensor,
    send_sizes: Sequence[int],
    recv_sizes: Sequence[int],
    comp_work,
    n_chunks: int = 1,
) -> PipelineTiming:
    """同步流水：通信与计算串行。

    分块只是为了与异步版分块口径一致（同样的消息切分），
    但每块"发完即等"，因此不存在重叠。
    """
    import time
    chunks = _split_chunks(payload, send_sizes, n_chunks)

    t0 = time.perf_counter()
    comm_ms = 0.0
    comp_ms = 0.0

    for send_c, send_s, recv_s in chunks:
        c0 = time.perf_counter()
        recv_c = all_to_all_v(send_c, [send_s] * dist.get_world_size(),
                              [recv_s] * dist.get_world_size())
        _env.barrier_and_sync()
        comm_ms += (time.perf_counter() - c0) * 1000.0

        c1 = time.perf_counter()
        comp_work(recv_c)
        _env.barrier_and_sync()
        comp_ms += (time.perf_counter() - c1) * 1000.0

    total = (time.perf_counter() - t0) * 1000.0
    return PipelineTiming(total_ms=total, comm_ms=comm_ms, comp_ms=comp_ms)


def run_async_pipeline(
    payload: torch.Tensor,
    send_sizes: Sequence[int],
    recv_sizes: Sequence[int],
    comp_work,
    n_chunks: int = 1,
) -> PipelineTiming:
    """异步流水：第 i 块的通信与第 i-1 块的计算重叠。

    结构（双缓冲）：
        issue(chunk 0)
        for i in range(n):
            if i+1 < n: issue(chunk i+1)      # 非阻塞
            wait(chunk i)                     # 等第 i 块到达
            comp(chunk i)                     # 计算与第 i+1 块的通信重叠
        wait(last)

    `comp_work` 在等待期间执行 —— 这正是 overlap 的来源。若 comp 比 comm 慢，
    total 由 sum(comp) 决定，异步收益趋近 comm 的总量；反之收益趋近
    min(sum(comm), sum(comp))。因此本函数同时返回 comm_ms 与 comp_ms，
    使调用方能按式~(speedup-bound) 判断实测加速比的归因是否合理。
    """
    import time
    chunks = _split_chunks(payload, send_sizes, n_chunks)

    t0 = time.perf_counter()
    comm_ms = 0.0
    comp_ms = 0.0

    pending: Optional[Tuple[torch.Tensor, Any]] = None
    for i, (send_c, send_s, recv_s) in enumerate(chunks):
        c0 = time.perf_counter()
        recv_c, handle = all_to_all_v_async(send_c, [send_s] * dist.get_world_size(),
                                            [recv_s] * dist.get_world_size())
        comm_issue = (time.perf_counter() - c0) * 1000.0

        if pending is not None:
            p_recv, p_handle = pending
            p_handle.wait()
            _env.barrier_and_sync()
            comm_ms += (time.perf_counter() - c0) * 1000.0  # 等待 + 发起 合并计入通信

            c1 = time.perf_counter()
            comp_work(p_recv)
            _env.barrier_and_sync()
            comp_ms += (time.perf_counter() - c1) * 1000.0
        else:
            comm_ms += comm_issue

        pending = (recv_c, handle)

    if pending is not None:
        p_recv, p_handle = pending
        c0 = time.perf_counter()
        p_handle.wait()
        _env.barrier_and_sync()
        comm_ms += (time.perf_counter() - c0) * 1000.0
        c1 = time.perf_counter()
        comp_work(p_recv)
        _env.barrier_and_sync()
        comp_ms += (time.perf_counter() - c1) * 1000.0

    total = (time.perf_counter() - t0) * 1000.0
    return PipelineTiming(total_ms=total, comm_ms=comm_ms, comp_ms=comp_ms)


def _split_chunks(
    payload: torch.Tensor,
    send_sizes: Sequence[int],
    n_chunks: int,
) -> List[Tuple[torch.Tensor, int, int]]:
    """把 outbound payload 按**等预算**切成 n_chunks 块。

    假设各边预算相同（A5 固定其他条件，只变同步/异步），因此每块的
    发/收大小都是 payload.shape[0] // n_chunks。
    """
    total = int(payload.shape[0])
    if n_chunks <= 1:
        per = total
        return [(payload, per, per)]
    per = max(1, total // n_chunks)
    return [(payload[i * per:(i + 1) * per], int(payload[i * per:(i + 1) * per].shape[0]),
             int(payload[i * per:(i + 1) * per].shape[0]))
            for i in range(n_chunks) if payload[i * per:(i + 1) * per].shape[0] > 0]


# =============================================================================
# 计算侧模拟（用于制造可重叠的 compute）
# =============================================================================

def make_comp_work(
    recv_elem: int,
    feature: int,
    d_h: int,
    d_v: int,
    flops_scale: float = 1.0,
):
    """构造一个与接收消息同规模的注意力式计算，用于产生真实 T_comp。

    scale = 1/√d_h 的 QK^T → softmax → @V 是稠密注意力的三段式，
    与接收到的紧凑块规模严格对应。这样 T_comm/T_comp 的比值
    才有物理含义，而不是被人为设定的 sleep 决定。
    """
    ratio = max(1, (d_h + 1 + d_v) // max(1, d_h))
    n_rows = max(1, int(recv_elem // max(1, d_h + 1 + d_v)) * ratio)

    def _work(recv: torch.Tensor) -> torch.Tensor:
        B = int(recv.shape[0])
        if B == 0:
            return recv
        q = torch.randn(n_rows, d_h, device=recv.device, dtype=recv.dtype)
        k = recv[:, :d_h]
        # 紧凑块里 β 与 V 的列位置由 pack_compact_edge 决定
        logits = (q @ k.T) * (1.0 / (d_h ** 0.5)) * flops_scale
        attn = torch.softmax(logits, dim=-1)
        v = recv[:, d_h + 1:]
        return attn @ v

    return _work


__all__ = [
    "pack_compact_edge",
    "edge_message_bytes",
    "EdgePlan",
    "make_uniform_plan",
    "make_skewed_plan",
    "all_to_all_v",
    "all_to_all_v_async",
    "PipelineTiming",
    "run_sync_pipeline",
    "run_async_pipeline",
    "make_comp_work",
]
