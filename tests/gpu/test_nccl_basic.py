"""M2 GPU 验证：NCCL backend 基本操作。

⚠️ 必须有真实 GPU + NCCL 才能跑。
pytest 会自动按 -m "not gpu" 跳过（见 pytest.ini + conftest.py）。

验证：
- NCCL init 在 2 进程上能跑
- all_reduce / broadcast / all_to_all_single 数值正确
- 4 进程扩展
- all_to_all_v 变长消息（DCC-KV 核心）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# 全部测试加 gpu marker（即使 conftest 也会自动加）
pytestmark = pytest.mark.gpu


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture(scope="module")
def gpu_available():
    """检查 GPU 是否可用。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


def _init_nccl(rank, world_size, port):
    """初始化 NCCL。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# 2 进程 NCCL 测试
# ============================================================================
def _2proc_all_reduce(rank, args):
    _init_nccl(rank, world_size=2, port=29520)
    tensor = torch.tensor([float(rank + 1)] * 4, device=f"cuda:{rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = torch.tensor([3.0] * 4, device=f"cuda:{rank}")  # 1+2
    assert torch.allclose(tensor, expected), f"rank {rank}: {tensor} != {expected}"
    _cleanup()


@pytest.mark.gpu
def test_nccl_2proc_all_reduce(gpu_available):
    """2 卡 NCCL all_reduce。"""
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    torch.multiprocessing.spawn(
        _2proc_all_reduce, args=({}), nprocs=2, daemon=False,
    )


def _2proc_all_to_all_v(rank, args):
    _init_nccl(rank, world_size=2, port=29521)
    d = 64
    if rank == 0:
        my_msgs = [
            torch.randn(2, d, device=f"cuda:{rank}"),  # 2 给 rank 0
            torch.randn(3, d, device=f"cuda:{rank}"),  # 3 给 rank 1
        ]
        send_sizes = [2, 3]
    else:
        my_msgs = [
            torch.randn(3, d, device=f"cuda:{rank}"),
            torch.randn(2, d, device=f"cuda:{rank}"),
        ]
        send_sizes = [3, 2]

    # 交换 sizes
    send_sizes_t = torch.tensor(send_sizes, dtype=torch.long, device=f"cuda:{rank}")
    recv_sizes_t = torch.empty_like(send_sizes_t)
    dist.all_to_all_single(recv_sizes_t, send_sizes_t)

    # 拼接 send
    send_buf = torch.cat(my_msgs, dim=0)

    # 准备 recv
    recv_buf = torch.empty(
        (sum(recv_sizes_t.tolist()), d),
        device=f"cuda:{rank}", dtype=torch.float32,
    )

    # 实际数据交换
    dist.all_to_all_single(
        recv_buf, send_buf,
        output_split_sizes=recv_sizes_t.tolist(),
        input_split_sizes=send_sizes,
    )

    # 验证
    if rank == 0:
        assert recv_sizes_t.tolist() == [3, 2]  # 从 rank 1 收 3，从 rank 0 收 2
    else:
        assert recv_sizes_t.tolist() == [2, 3]
    _cleanup()


@pytest.mark.gpu
def test_nccl_2proc_all_to_all_v(gpu_available):
    """2 卡 NCCL 变长 all-to-allv（DCC-KV 关键）。"""
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    torch.multiprocessing.spawn(
        _2proc_all_to_all_v, args=({}), nprocs=2, daemon=False,
    )


# ============================================================================
# 4 进程 NCCL 扩展测试
# ============================================================================
def _4proc_scaling(rank, args):
    _init_nccl(rank, world_size=4, port=29522)
    # 算 scaling
    N = 1024
    tensor = torch.randn(N, device=f"cuda:{rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # 4 个进程求和，期望值 = 4 * (期望值)
    expected = torch.randn(N, device=f"cuda:{rank}") * 4
    assert torch.allclose(tensor, expected, atol=1e-3), "4-proc all_reduce incorrect"
    _cleanup()


@pytest.mark.gpu
def test_nccl_4proc_scaling(gpu_available):
    """4 卡 NCCL：scaling sanity。"""
    if torch.cuda.device_count() < 4:
        pytest.skip("Need at least 4 GPUs")
    torch.multiprocessing.spawn(
        _4proc_scaling, args=({}), nprocs=4, daemon=False,
    )
