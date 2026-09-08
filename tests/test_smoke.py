"""Smoke test 入口：M2 启动前的接口验证。

验证范围（不需要 GPU）：
- launch_dist 单进程模式直接调用
- DistributedComm mock 模式接口正确
- 真实 distributed 2 进程 gloo 模式能 init/cleanup
- 变长 All-to-Allv 接口形状正确
- 数值等价性 baseline（mock 模式仅接口形状）

测试分级（用 pytest -m 选择）：
- 默认运行：mock 模式 + 单进程（< 5 秒）
- `-m "distributed"`：2 进程 gloo（< 30 秒，需要 ≥2 CPU 核）
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# 让 src 可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distributed.launch_dist import (
    launch_dist,
    setup_distributed,
    cleanup_distributed,
    get_worker_rank,
    get_worker_world_size,
)
from distributed.comm import DistributedComm, VarLenMessage


# ============================================================================
# Test 1: 单进程模式 launch_dist
# ============================================================================
def test_launch_dist_single_process_returns_value():
    """单进程模式：launch_dist 直接调用 fn，返回 fn 的结果。"""
    results = []

    def fn(rank, args):
        results.append((rank, args))
        return f"result_from_rank_{rank}"

    ret = launch_dist(fn, nproc_per_node=1, args=("hello",))
    assert ret == "result_from_rank_0"
    assert results == [(0, "hello")]


def test_launch_dist_single_process_no_distributed_init():
    """单进程模式：不调用 torch.distributed。"""
    from distributed.launch_dist import cleanup_distributed

    def fn(rank, args):
        # 验证 torch.distributed 没被初始化
        import torch.distributed as dist
        assert not dist.is_initialized(), "should not init distributed in single mode"
        return True

    assert launch_dist(fn, nproc_per_node=1) is True


# ============================================================================
# Test 2: DistributedComm mock 模式
# ============================================================================
def test_comm_mock_init_cleanup():
    """mock 模式 init/cleanup 生命周期。"""
    comm = DistributedComm(backend="gloo", mock=True)
    assert not comm.initialized
    comm.init(rank=0, world_size=2)
    assert comm.initialized
    assert comm.rank == 0
    assert comm.world_size == 2
    comm.cleanup()
    assert not comm.initialized


def test_comm_mock_all_to_all_single_shape():
    """mock 模式 all_to_all_single：返回正确形状的零 tensor。"""
    comm = DistributedComm(backend="gloo", mock=True)
    comm.init(rank=0, world_size=2)

    send = torch.tensor([1.0, 2.0, 3.0, 4.0])  # 4 元素
    recv = comm.all_to_all_single(send, send_counts=[2, 2], recv_counts=[2, 2])

    assert recv.shape == (4,), f"expected (4,), got {recv.shape}"
    assert torch.all(recv == 0), "mock should return zeros"

    comm.cleanup()


def test_comm_mock_all_to_all_v_unequal_sizes():
    """mock 模式 all_to_all_v：变长消息接口。"""
    comm = DistributedComm(backend="gloo", mock=True)
    comm.init(rank=0, world_size=2)

    # 准备发给 2 个 rank 的不同长度消息
    msg0 = VarLenMessage(payload=torch.tensor([1.0, 2.0]))
    msg1 = VarLenMessage(payload=torch.tensor([3.0, 4.0, 5.0]))
    messages = [msg0, msg1]
    send_sizes = [2, 3]
    recv_sizes = [3, 2]  # 假设从 rank 0 收 3 个，从 rank 1 收 2 个

    results = comm.all_to_all_v(messages, send_sizes=send_sizes, recv_sizes=recv_sizes)

    assert len(results) == 2
    assert results[0].payload.shape == (3,), f"expected (3,), got {results[0].payload.shape}"
    assert results[1].payload.shape == (2,), f"expected (2,), got {results[1].payload.shape}"
    assert results[0].metadata["src_rank"] == 0
    assert results[1].metadata["src_rank"] == 1

    comm.cleanup()


def test_comm_mock_all_reduce_no_op():
    """mock 模式 all_reduce：不修改 tensor。"""
    comm = DistributedComm(backend="gloo", mock=True)
    comm.init(rank=0, world_size=2)

    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = comm.all_reduce(tensor)
    assert torch.all(result == tensor)
    assert result.data_ptr() != tensor.data_ptr(), "should be a copy in mock mode"

    comm.cleanup()


def test_comm_mock_broadcast_no_op():
    """mock 模式 broadcast：不修改 tensor。"""
    comm = DistributedComm(backend="gloo", mock=True)
    comm.init(rank=0, world_size=2)

    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = comm.broadcast(tensor, src=0)
    assert torch.all(result == tensor)

    comm.cleanup()


def test_comm_mock_with_multidim_payload():
    """mock 模式 all_to_all_v：多维 payload（K/V 张量是 [N, head_dim]）。"""
    comm = DistributedComm(backend="gloo", mock=True)
    comm.init(rank=0, world_size=2)

    # 模拟 K/V 张量：[N, head_dim=128]
    head_dim = 128
    msg0 = VarLenMessage(payload=torch.randn(2, head_dim))
    msg1 = VarLenMessage(payload=torch.randn(3, head_dim))
    messages = [msg0, msg1]

    results = comm.all_to_all_v(messages, send_sizes=[2, 3], recv_sizes=[3, 2])

    assert results[0].payload.shape == (3, head_dim)
    assert results[1].payload.shape == (2, head_dim)

    comm.cleanup()


# ============================================================================
# Test 3: 真实 distributed 模式（2 进程 gloo）
# ============================================================================
def _worker_dist_comm_real(rank, args):
    """mp.spawn 启动的子进程：测试真实 distributed all_reduce。"""
    setup_distributed(rank, args["world_size"], backend="gloo")

    comm = DistributedComm(backend="gloo", mock=False)
    comm.init(rank=rank, world_size=args["world_size"])

    # 数值正确性测试：all_reduce sum
    tensor = torch.tensor([float(rank + 1), float(rank + 2)])
    result = comm.all_reduce(tensor)

    # rank 0: [1,2], rank 1: [2,3], sum = [3, 5]
    expected = torch.tensor([3.0, 5.0])
    assert torch.allclose(result, expected), (
        f"rank {rank} got {result.tolist()}, expected {expected.tolist()}"
    )

    comm.cleanup()
    cleanup_distributed()
    return f"rank_{rank}_ok"


@pytest.mark.distributed
def test_launch_dist_two_processes_gloo_all_reduce():
    """2 进程 gloo backend：all_reduce 数值正确性。"""
    if os.cpu_count() < 2:
        pytest.skip("Need at least 2 CPU cores")

    launch_dist(
        _worker_dist_comm_real,
        nproc_per_node=2,
        args={"world_size": 2},
        backend="gloo",
        master_port="29501",  # 用不同端口避免冲突
    )


def _worker_dist_comm_broadcast(rank, args):
    """mp.spawn 启动的子进程：测试 broadcast。"""
    setup_distributed(rank, args["world_size"], backend="gloo")

    comm = DistributedComm(backend="gloo", mock=False)
    comm.init(rank=rank, world_size=args["world_size"])

    # 准备 tensor（每个 rank 初始值不同）
    tensor = torch.tensor([rank * 10.0, rank * 100.0])
    # broadcast from rank 0：所有 rank 应该变成 [0, 0]
    comm.broadcast(tensor, src=0)

    expected = torch.tensor([0.0, 0.0])
    assert torch.allclose(tensor, expected), (
        f"rank {rank} got {tensor.tolist()}, expected {expected.tolist()}"
    )

    comm.cleanup()
    cleanup_distributed()
    return f"rank_{rank}_ok"


@pytest.mark.distributed
def test_launch_dist_two_processes_gloo_broadcast():
    """2 进程 gloo backend：broadcast 数值正确性。"""
    if os.cpu_count() < 2:
        pytest.skip("Need at least 2 CPU cores")

    launch_dist(
        _worker_dist_comm_broadcast,
        nproc_per_node=2,
        args={"world_size": 2},
        backend="gloo",
        master_port="29502",
    )


def _worker_dist_comm_all_to_all_v(rank, args):
    """mp.spawn 启动的子进程：测试变长 all-to-allv。"""
    setup_distributed(rank, args["world_size"], backend="gloo")

    comm = DistributedComm(backend="gloo", mock=False)
    comm.init(rank=rank, world_size=args["world_size"])

    # 每个 rank 准备发给 2 个 rank 的不同长度消息
    # rank 0 发：[2, 3]  →  收到 [3, 2]
    # rank 1 发：[3, 2]  →  收到 [2, 3]
    if rank == 0:
        my_messages = [
            VarLenMessage(payload=torch.tensor([10.0, 11.0])),
            VarLenMessage(payload=torch.tensor([20.0, 21.0, 22.0])),
        ]
        my_send_sizes = [2, 3]
        my_recv_sizes = [3, 2]
    else:
        my_messages = [
            VarLenMessage(payload=torch.tensor([100.0, 101.0, 102.0])),
            VarLenMessage(payload=torch.tensor([200.0, 201.0])),
        ]
        my_send_sizes = [3, 2]
        my_recv_sizes = [2, 3]

    results = comm.all_to_all_v(
        my_messages, send_sizes=my_send_sizes, recv_sizes=my_recv_sizes
    )

    # 验证接收内容
    if rank == 0:
        # rank 0 从 rank 1 收到 3 个（payload 来自 rank 1 的 send[0]）
        # rank 0 从 rank 0 收到 2 个（payload 来自 rank 0 的 send[1]）
        # 注意：all-to-allv 的语义是 src_rank → dst_rank
        # recv[i] = src_rank i 发给 dst_rank (=self.rank) 的消息
        # 在我们的约定里，messages[i] 是发给 rank i 的，所以 recv[i] 是从 rank i 收到的
        assert results[0].numel() == 3  # from rank 1: [100, 101, 102]
        assert results[1].numel() == 2  # from rank 0: [20, 21, 22]
        assert torch.allclose(results[0].payload, torch.tensor([100.0, 101.0, 102.0]))
        assert torch.allclose(results[1].payload, torch.tensor([20.0, 21.0, 22.0]))
    else:
        assert results[0].numel() == 2  # from rank 0: [10, 11]
        assert results[1].numel() == 3  # from rank 1: [200, 201, 202]
        assert torch.allclose(results[0].payload, torch.tensor([10.0, 11.0]))
        assert torch.allclose(results[1].payload, torch.tensor([200.0, 201.0, 202.0]))

    comm.cleanup()
    cleanup_distributed()
    return f"rank_{rank}_ok"


@pytest.mark.distributed
def test_launch_dist_two_processes_gloo_all_to_all_v():
    """2 进程 gloo backend：变长 all-to-allv 数值正确性（M2 关键测试）。"""
    if os.cpu_count() < 2:
        pytest.skip("Need at least 2 CPU cores")

    launch_dist(
        _worker_dist_comm_all_to_all_v,
        nproc_per_node=2,
        args={"world_size": 2},
        backend="gloo",
        master_port="29503",
    )


# ============================================================================
# Test 4: 错误处理
# ============================================================================
def test_comm_init_twice_warns():
    """重复 init 给出 warning，不报错。"""
    comm = DistributedComm(backend="gloo", mock=True)
    comm.init(rank=0, world_size=2)
    comm.init(rank=0, world_size=2)  # 第二次应 warn
    comm.cleanup()


def test_comm_uninit_raises():
    """未 init 时调用操作应报错。"""
    comm = DistributedComm(backend="gloo", mock=True)
    tensor = torch.tensor([1.0, 2.0])
    with pytest.raises(RuntimeError, match="not initialized"):
        comm.all_reduce(tensor)


def test_comm_all_to_all_v_size_mismatch_raises():
    """messages 长度 != world_size 时报错。"""
    comm = DistributedComm(backend="gloo", mock=True)
    comm.init(rank=0, world_size=3)
    try:
        with pytest.raises(ValueError, match="length"):
            comm.all_to_all_v(
                [VarLenMessage(payload=torch.tensor([1.0]))],  # 只有 1 条
                send_sizes=[1],
                recv_sizes=[1, 1, 1],
            )
    finally:
        comm.cleanup()


def test_launch_dist_invalid_nproc_raises():
    """nproc < 1 时报错。"""
    with pytest.raises(ValueError, match="nproc_per_node"):
        launch_dist(lambda r, a: None, nproc_per_node=0)


# ============================================================================
# Test 5: 端到端 mini 演示（可用 subprocess 跑完整 pipeline）
# ============================================================================
def test_minimal_distributed_demo_runs():
    """最小可跑的多进程 demo：每个 rank 算 all_reduce，验证求和。"""
    if os.cpu_count() < 2:
        pytest.skip("Need at least 2 CPU cores")

    demo_script = """
import sys
sys.path.insert(0, '{src_dir}')
from distributed.launch_dist import launch_dist, setup_distributed, cleanup_distributed
from distributed.comm import DistributedComm
import torch

def worker_fn(rank, world_size):
    setup_distributed(rank, world_size, backend='gloo')
    comm = DistributedComm(backend='gloo', mock=False)
    comm.init(rank=rank, world_size=world_size)
    tensor = torch.tensor([float(rank)])
    result = comm.all_reduce(tensor)
    # rank 0 + rank 1 = 1.0
    assert result.item() == 1.0, f'rank {{rank}} got {{result.item()}}'
    comm.cleanup()
    cleanup_distributed()
    return f'rank_{{rank}}_ok'

if __name__ == '__main__':
    results = launch_dist(worker_fn, nproc_per_node=2, args=2, backend='gloo', master_port='29504')
    print('OK')
""".format(src_dir=os.path.join(os.path.dirname(__file__), "..", "src"))

    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(demo_script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"Demo failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout, f"Expected 'OK' in stdout, got: {result.stdout}"
    finally:
        os.unlink(script_path)


# ============================================================================
# 配置 pytest
# ============================================================================
def pytest_configure(config):
    """注册自定义 marker。"""
    config.addinivalue_line(
        "markers", "distributed: mark test as requiring multi-process distributed execution"
    )
