"""Phase A 验收测试：3 个文件骨架的核心功能。

这是 M2_pre_launch_checklist Phase A 的官方验收。
所有测试不需要 GPU，在 CPU 上即可跑。
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distributed.launch_dist import launch_dist, setup_distributed, cleanup_distributed
from distributed.comm import DistributedComm, VarLenMessage


# ============================================================================
# A1. launch_dist 接口验收
# ============================================================================
class TestLaunchDist:
    """launch_dist 的 5 项接口验收。"""

    def test_a1_1_single_process_no_distributed_init(self):
        """A1.1 单进程模式：不调用 torch.distributed，直接调用 fn。"""
        import torch.distributed as dist

        called = []

        def fn(rank, args):
            called.append(rank)
            assert not dist.is_initialized()
            return rank

        ret = launch_dist(fn, nproc_per_node=1, args=None)
        assert ret == 0
        assert called == [0]

    def test_a1_2_nproc_must_be_positive(self):
        """A1.2 nproc_per_node 必须 >= 1。"""
        with pytest.raises(ValueError):
            launch_dist(lambda r, a: None, nproc_per_node=0)
        with pytest.raises(ValueError):
            launch_dist(lambda r, a: None, nproc_per_node=-1)

    def test_a1_3_args_passed_through(self):
        """A1.3 args 正确传给 fn。"""
        received = []

        def fn(rank, args):
            received.append((rank, args))
            return None

        launch_dist(fn, nproc_per_node=1, args={"key": "value", "n": 42})
        assert received == [(0, {"key": "value", "n": 42})]

    def test_a1_4_fn_exception_propagates(self):
        """A1.4 fn 抛异常时 launch_dist 应传播。"""
        def fn(rank, args):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            launch_dist(fn, nproc_per_node=1)

    def test_a1_5_multi_process_via_mp_spawn(self):
        """A1.5 多进程模式：用 mp.spawn 启动（2 进程 gloo）。"""
        if os.cpu_count() < 2:
            pytest.skip("Need at least 2 CPU cores")

        completed = []

        def fn(rank, args):
            setup_distributed(rank, args["world_size"], backend="gloo")
            completed.append(rank)
            cleanup_distributed()
            return f"rank_{rank}"

        # launch_dist 内部已经处理 setup/cleanup；这里手动调用一次验证接口
        # 实际 launch_dist 调用会包一层，fn 里不需要再 setup/cleanup
        # 我们用直接 mp.spawn 模拟（绕过 launch_dist 的包装）
        import torch.multiprocessing as mp
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29510"

        def wrapped_fn(rank, args):
            setup_distributed(rank, args["world_size"], backend="gloo")
            completed.append(rank)
            cleanup_distributed()

        mp.spawn(
            wrapped_fn,
            args=({"world_size": 2},),
            nprocs=2,
            join=True,
        )
        assert sorted(completed) == [0, 1]


# ============================================================================
# A2. DistributedComm 接口验收
# ============================================================================
class TestDistributedComm:
    """DistributedComm 的 8 项接口验收。"""

    def setup_method(self):
        self.comm = DistributedComm(backend="gloo", mock=True)
        self.comm.init(rank=0, world_size=2)

    def teardown_method(self):
        self.comm.cleanup()

    def test_a2_1_all_to_all_single_shape(self):
        """A2.1 all_to_all_single 返回正确形状。"""
        send = torch.arange(8).float()
        recv = self.comm.all_to_all_single(send, send_counts=[4, 4], recv_counts=[4, 4])
        assert recv.shape == (8,)

    def test_a2_2_all_to_all_v_unequal_sizes(self):
        """A2.2 all_to_all_v 支持变长消息。"""
        messages = [
            VarLenMessage(payload=torch.tensor([1.0, 2.0])),
            VarLenMessage(payload=torch.tensor([3.0, 4.0, 5.0])),
        ]
        results = self.comm.all_to_all_v(messages, send_sizes=[2, 3], recv_sizes=[3, 2])
        assert results[0].payload.shape == (3,)
        assert results[1].payload.shape == (2,)

    def test_a2_3_all_reduce(self):
        """A2.3 all_reduce 接受 tensor 并返回。"""
        t = torch.tensor([1.0, 2.0])
        out = self.comm.all_reduce(t)
        assert out.shape == t.shape

    def test_a2_4_broadcast(self):
        """A2.4 broadcast 接受 src 参数。"""
        t = torch.tensor([1.0, 2.0])
        out = self.comm.broadcast(t, src=0)
        assert out.shape == t.shape

    def test_a2_5_barrier_no_op(self):
        """A2.5 barrier 不应报错。"""
        self.comm.barrier()  # 不应抛

    def test_a2_6_gather_to_dst(self):
        """A2.6 gather 收集到 dst 进程。"""
        t = torch.tensor([float(self.comm.rank)])
        result = self.comm.gather(t, dst=0)
        if self.comm.rank == 0:
            assert result is not None
            assert len(result) == 2
            assert torch.allclose(result[0], torch.tensor([0.0]))
            assert torch.allclose(result[1], torch.tensor([1.0]))
        else:
            assert result is None

    def test_a2_7_size_validation(self):
        """A2.7 size 不一致时报错。"""
        send = torch.arange(10).float()  # 10 元素
        with pytest.raises(ValueError, match="send_counts sum"):
            self.comm.all_to_all_single(send, send_counts=[4, 4], recv_counts=[4, 4])

    def test_a2_8_multidim_payload(self):
        """A2.8 多维 payload 正确处理。"""
        head_dim = 64
        messages = [
            VarLenMessage(payload=torch.randn(2, head_dim)),
            VarLenMessage(payload=torch.randn(3, head_dim)),
        ]
        results = self.comm.all_to_all_v(messages, send_sizes=[2, 3], recv_sizes=[3, 2])
        assert results[0].payload.shape == (3, head_dim)
        assert results[1].payload.shape == (2, head_dim)


# ============================================================================
# A3. 真实 distributed 集成（2 进程 gloo）
# ============================================================================
def _a3_worker(rank, args):
    """A3 worker：跑全部基础操作并验证。"""
    setup_distributed(rank, args["world_size"], backend="gloo")
    comm = DistributedComm(backend="gloo", mock=False)
    comm.init(rank=rank, world_size=args["world_size"])

    # all_reduce
    t1 = torch.tensor([float(rank + 1)])
    r1 = comm.all_reduce(t1)
    assert torch.allclose(r1, torch.tensor([3.0])), f"all_reduce: rank {rank} got {r1.item()}"

    # broadcast
    t2 = torch.tensor([float(rank * 10)])
    comm.broadcast(t2, src=0)
    assert torch.allclose(t2, torch.tensor([0.0])), f"broadcast: rank {rank} got {t2.item()}"

    # all_to_all_single
    send = torch.tensor([1.0, 2.0, 3.0, 4.0])
    recv = comm.all_to_all_single(send, send_counts=[2, 2], recv_counts=[2, 2])
    assert recv.shape == (4,)

    # barrier
    comm.barrier()

    comm.cleanup()
    cleanup_distributed()
    return f"rank_{rank}_a3_ok"


@pytest.mark.distributed
def test_a3_real_distributed_2proc_gloo():
    """A3 真实 2 进程 gloo：全套基础操作。"""
    if os.cpu_count() < 2:
        pytest.skip("Need at least 2 CPU cores")

    launch_dist(
        _a3_worker,
        nproc_per_node=2,
        args={"world_size": 2},
        backend="gloo",
        master_port="29511",
    )


# ============================================================================
# 配置
# ============================================================================
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "distributed: mark test as requiring multi-process distributed execution"
    )
