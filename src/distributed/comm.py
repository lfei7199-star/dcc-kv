"""分布式通信抽象层。

设计目标：
- 接口统一：gloo (CPU) / nccl (GPU) 切换对上层代码透明
- 变长 All-to-Allv：DCC-KV 的核心通信原语
- mock 模式：单元测试不需要真实 distributed 环境
- 数值正确性：gloo 和 nccl 的 all_to_all_single 在元素级与单卡等价

接口与 torch.distributed 对齐，但增加：
- 变长消息（VarLenMessage）
- mock 模式（DistributedComm(mock=True)）
- 清晰的初始化/清理生命周期
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 数据类
# -----------------------------------------------------------------------------
@dataclass
class VarLenMessage:
    """变长消息描述。

    Attributes:
        payload: 实际数据张量（[N, ...]，N 任意）
        metadata: 元数据（可选），比如发送方 rank、layer idx 等
    """
    payload: torch.Tensor
    metadata: Optional[dict] = field(default_factory=dict)

    def numel(self) -> int:
        """返回第 0 维大小（用于 all-to-allv 大小协商）。"""
        return self.payload.shape[0] if self.payload.dim() > 0 else 1


# -----------------------------------------------------------------------------
# 核心抽象类
# -----------------------------------------------------------------------------
class DistributedComm:
    """分布式通信抽象。

    用法：
        comm = DistributedComm(backend="gloo", mock=False)
        comm.init(rank=0, world_size=2)

        # 等长 all-to-all
        recv = comm.all_to_all_single(send, send_counts, recv_counts)

        # 变长 all-to-allv（DCC-KV 核心）
        msgs = [VarLenMessage(...), ...]
        result = comm.all_to_all_v(msgs, send_sizes, recv_sizes)

        comm.cleanup()

    Args:
        backend: "gloo" (CPU) / "nccl" (GPU) / "mpi" (集群)
        mock: True 时所有操作返回 placeholder，不调用 torch.distributed
    """

    def __init__(self, backend: str = "gloo", mock: bool = False):
        if backend not in ("gloo", "nccl", "mpi"):
            raise ValueError(f"Unknown backend: {backend}; choose gloo/nccl/mpi")
        self.backend = backend
        self.mock = mock
        self.rank: int = 0
        self.world_size: int = 1
        self.initialized: bool = False
        self._device = torch.device("cpu") if backend == "gloo" else torch.device("cuda")

    # -------------------------------------------------------------------------
    # 生命周期
    # -------------------------------------------------------------------------
    def init(self, rank: int, world_size: int) -> None:
        """初始化。假设 torch.distributed 已被 setup_distributed 初始化过。

        Args:
            rank: 当前进程 rank
            world_size: 总进程数
        """
        if self.initialized:
            logger.warning("DistributedComm already initialized; skipping reinit")
            return

        self.rank = rank
        self.world_size = world_size

        if not self.mock:
            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed not initialized; call setup_distributed() first"
                )
            actual_rank = dist.get_rank()
            actual_ws = dist.get_world_size()
            if actual_rank != rank or actual_ws != world_size:
                raise ValueError(
                    f"Comm init rank/ws mismatch: "
                    f"requested ({rank}/{world_size}), actual ({actual_rank}/{actual_ws})"
                )
        self.initialized = True
        logger.info(f"DistributedComm initialized: rank={rank}, ws={world_size}, "
                    f"backend={self.backend}, mock={self.mock}")

    def cleanup(self) -> None:
        """清理资源。"""
        self.initialized = False

    def __enter__(self):
        if not self.initialized:
            raise RuntimeError("DistributedComm not initialized")
        return self

    def __exit__(self, *args):
        self.cleanup()

    # -------------------------------------------------------------------------
    # 集合通信原语
    # -------------------------------------------------------------------------
    def all_to_all_single(
        self,
        send_tensor: torch.Tensor,
        send_counts: Optional[List[int]] = None,
        recv_counts: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """等长 All-to-All。

        Args:
            send_tensor: [sum(send_counts) × ...] 发给所有 rank 的拼接
            send_counts: 每个目标 rank 的发送数量（默认均分）
            recv_counts: 每个源 rank 的接收数量（默认均分）

        Returns:
            recv_tensor: [sum(recv_counts) × ...]
        """
        self._check_init()

        if send_counts is None:
            send_counts = [send_tensor.shape[0] // self.world_size] * self.world_size
        if recv_counts is None:
            recv_counts = send_counts

        if sum(send_counts) != send_tensor.shape[0]:
            raise ValueError(
                f"send_counts sum ({sum(send_counts)}) != "
                f"send_tensor.shape[0] ({send_tensor.shape[0]})"
            )

        if self.mock:
            total = sum(recv_counts)
            shape = list(send_tensor.shape)
            shape[0] = total
            return torch.zeros(shape, dtype=send_tensor.dtype, device=send_tensor.device)

        recv_tensor = torch.empty(
            (sum(recv_counts),) + tuple(send_tensor.shape[1:]),
            dtype=send_tensor.dtype,
            device=send_tensor.device,
        )
        dist.all_to_all_single(
            recv_tensor, send_tensor,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
        return recv_tensor

    def all_to_all_v(
        self,
        messages: List[VarLenMessage],
        send_sizes: Optional[List[int]] = None,
        recv_sizes: Optional[List[int]] = None,
    ) -> List[VarLenMessage]:
        """变长 All-to-Allv（DCC-KV 核心通信原语）。

        gloo 和 nccl 都不直接支持变长——需要：
        Step 1: 交换每条边的 size（用 all_to_all_single）
        Step 2: 拼接 send buffer
        Step 3: 分配 recv buffer
        Step 4: 实际数据交换（用 all_to_all_single）
        Step 5: 切分 recv buffer

        Args:
            messages: 准备发给各 rank 的消息列表（长度必须为 world_size）
            send_sizes: 准备发给各 rank 的第 0 维大小（默认从 messages 推断）
            recv_sizes: 准备从各 rank 接收的第 0 维大小（必须预先知道！）

        Returns:
            接收到的消息列表

        注意：recv_sizes 是必需的，因为 DCC-KV 的 Bs,r 是发送方决定的；
        接收方需要预先知道对方会发多少（通过 budget 配置共享）。
        """
        self._check_init()

        if len(messages) != self.world_size:
            raise ValueError(
                f"messages length ({len(messages)}) != world_size ({self.world_size})"
            )

        if send_sizes is None:
            send_sizes = [m.numel() for m in messages]
        if any(m.numel() != s for m, s in zip(messages, send_sizes)):
            raise ValueError(
                f"send_sizes inconsistent with messages: "
                f"sizes={send_sizes}, actual={[m.numel() for m in messages]}"
            )

        if recv_sizes is None:
            raise ValueError(
                "recv_sizes is required for all_to_all_v "
                "(DCC-KV must share budget config across ranks)"
            )
        if len(recv_sizes) != self.world_size:
            raise ValueError(
                f"recv_sizes length ({len(recv_sizes)}) != world_size ({self.world_size})"
            )

        if self.mock:
            return [
                VarLenMessage(
                    payload=torch.zeros(
                        (s,) + tuple(messages[0].payload.shape[1:]),
                        dtype=messages[0].payload.dtype,
                        device=messages[0].payload.device,
                    ),
                    metadata={"src_rank": i, "mock": True},
                )
                for i, s in enumerate(recv_sizes)
            ]

        # 推断 feature 维度（假设所有 messages 的 feature dim 相同）
        feature_shape = messages[0].payload.shape[1:]
        dtype = messages[0].payload.dtype
        device = messages[0].payload.device

        # Step 1: 交换 size（all_to_all_single 走 int64）
        send_sizes_t = torch.tensor(send_sizes, dtype=torch.long, device=device)
        recv_sizes_t = torch.empty_like(send_sizes_t)
        dist.all_to_all_single(
            recv_sizes_t, send_sizes_t,
            output_split_sizes=[1] * self.world_size,
            input_split_sizes=[1] * self.world_size,
        )
        actual_recv_sizes = recv_sizes_t.tolist()

        # Step 2: 拼接 send buffer
        send_buffer = torch.cat([m.payload for m in messages], dim=0)

        # Step 3: 分配 recv buffer
        recv_buffer = torch.empty(
            (sum(actual_recv_sizes),) + tuple(feature_shape),
            dtype=dtype, device=device,
        )

        # Step 4: 实际数据交换
        dist.all_to_all_single(
            recv_buffer, send_buffer,
            output_split_sizes=actual_recv_sizes,
            input_split_sizes=send_sizes,
        )

        # Step 5: 切分 recv buffer
        result = []
        offset = 0
        for i, size in enumerate(actual_recv_sizes):
            payload = recv_buffer[offset:offset + size]
            offset += size
            result.append(VarLenMessage(
                payload=payload,
                metadata={"src_rank": i, "size": size},
            ))

        return result

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        """All-reduce（in-place）。"""
        self._check_init()
        if self.mock:
            return tensor.clone()
        dist.all_reduce(tensor, op=op)
        return tensor

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """Broadcast（in-place）。"""
        self._check_init()
        if self.mock:
            return tensor.clone()
        dist.broadcast(tensor, src=src)
        return tensor

    def barrier(self) -> None:
        """进程同步。"""
        if not self.mock and dist.is_initialized():
            dist.barrier()

    def gather(
        self,
        tensor: torch.Tensor,
        dst: int = 0,
    ) -> Optional[List[torch.Tensor]]:
        """Gather 到 dst 进程。"""
        self._check_init()
        if self.mock:
            return [tensor.clone() for _ in range(self.world_size)]
        if self.rank == dst:
            gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.gather(tensor, gathered, dst=dst)
            return gathered
        else:
            dist.gather(tensor, None, dst=dst)
            return None

    # -------------------------------------------------------------------------
    # 内部工具
    # -------------------------------------------------------------------------
    def _check_init(self) -> None:
        if not self.initialized:
            raise RuntimeError("DistributedComm not initialized; call init() first")

    def __repr__(self) -> str:
        return (
            f"DistributedComm(backend={self.backend}, mock={self.mock}, "
            f"rank={self.rank}/{self.world_size}, initialized={self.initialized})"
        )
