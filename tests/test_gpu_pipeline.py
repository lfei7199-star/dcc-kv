"""E5/A5 流水线分块口径的单元测试（纯 CPU，不需要 GPU，也不需要进程组）。

背景（本次审计定位并修复）
--------------------------
`_comm._split_chunks` 原先把 payload 按**连续行**切段，却仍声明
`[块行数] * world` 作为 split sizes。而 payload 的布局是「按 dst 顺序拼接」
（见 `all_to_all_v`），于是：

1. 声明与块内实际行数不相等（`world >= 2` 时恰好差 `world` 倍）——
   必然触发 `all_to_all_v` 的自洽性检查；
2. 即便绕过检查，块的起始行也落在某个 dst 片段的**内部**，
   行与目的端的对应关系同样是错的。

后果：`run_sync_pipeline` 抛 AssertionError，`run_async_pipeline` 抛
`Split sizes doesn't match total dim 0 size` —— 也就是 E5/A5（仓库里唯一的
同步 vs 异步对照）在真实多卡上根本产不出结果，而这一点在无 GPU 的环境里
只靠"跑一下看看"是发现不了的。

本文件把口径本身固定下来：分块必须**尺寸自洽**且**逐 dst 恰好覆盖一次**。

同一次收尾审计还修了两处相关缺陷，一并在此立锚：

3. `run_sync_pipeline` / `run_async_pipeline` 曾在每个计时段收尾调
   `barrier_and_sync()`（含 `dist.barrier()`）；异步版更在 `handle.wait()`
   之后做了一次 **device-wide** 同步，把正在飞的下一块集合通信也等掉
   ⇒ overlap 恒为 0、A5 的实测加速比恒为 1.0。现两条流水线内只用
   `device_sync()`，集合 barrier 一律留给窗口开始之前。
4. `e5` 曾用 `torch.device(f"cuda:{rank}")` 绑定设备 —— 那是"拿全局 rank
   当设备序号"：单节点巧合正确，多节点直接是非法序号。现统一走
   `_env.local_device(rank)`（其索引取自 `LOCAL_RANK`）。
"""
from __future__ import annotations

import itertools
import pathlib
from typing import List, Tuple

import pytest
import torch

from experiments.gpu import _comm, _env

F = 8


class _FakeHandle:
    """stand-in for the async work handle returned by all_to_all_single(async_op=True)."""

    def wait(self) -> None:
        return None


def _stub_all_to_all(send: torch.Tensor, send_sizes, recv_sizes) -> torch.Tensor:
    """替代真实集合通信：只校验入参自洽，再返回一个形状正确的接收缓冲。"""
    assert len(send_sizes) == len(recv_sizes), "两个 sizes 必须等长"
    assert int(send.shape[0]) == int(sum(send_sizes)), (
        f"send 行数 {int(send.shape[0])} != sum(send_sizes) {int(sum(send_sizes))}"
    )
    return torch.zeros(int(sum(recv_sizes)), int(send.shape[1]), dtype=send.dtype)


def _stub_all_to_all_async(send: torch.Tensor, send_sizes, recv_sizes):
    return _stub_all_to_all(send, send_sizes, recv_sizes), _FakeHandle()


@pytest.fixture
def no_barrier(monkeypatch):
    """端口守卫：流水线内部一旦出现集合 barrier 就让用例直接失败。

    旧实现在每个计时段收尾调 `barrier_and_sync()`；这里不做"打桩掩盖"，
    而是把它换成会报错的哨兵 —— 该调用本身就是要禁止的行为。
    """
    def _forbid() -> None:
        raise AssertionError("流水线内部不得调用 barrier_and_sync（它含 dist.barrier）")

    monkeypatch.setattr(_env, "barrier_and_sync", _forbid)


# ============================================================================
# _split_chunks 的口径
# ============================================================================

SPLIT_CASES = [
    (world, B, chunks)
    for world, B, chunks in itertools.product((2, 3, 4), (4, 6, 8, 9), (1, 2, 3, 4))
]


@pytest.mark.parametrize("world,B,n_chunks", SPLIT_CASES)
def test_chunks_are_size_consistent(world: int, B: int, n_chunks: int) -> None:
    """每块的行数必须等于该块声明的 send/recv sizes 之和。"""
    payload = torch.randn(world * B, F)
    sizes = [B] * world
    for i, (chunk, send_sizes, recv_sizes) in enumerate(
            _comm._split_chunks(payload, sizes, sizes, n_chunks)):
        assert int(chunk.shape[0]) == int(sum(send_sizes)), (
            f"chunk{i}: 行数 {int(chunk.shape[0])} != sum(send_sizes) {sum(send_sizes)}"
        )
        assert int(sum(send_sizes)) == int(sum(recv_sizes))
        assert len(send_sizes) == world


@pytest.mark.parametrize("world,B,n_chunks", SPLIT_CASES)
def test_chunks_cover_each_destination_exactly_once(
        world: int, B: int, n_chunks: int) -> None:
    """逐 dst 把各块的片段接回去，必须逐位等于 payload 里该 dst 的原始片段。"""
    payload = torch.randn(world * B, F)
    sizes = [B] * world
    per_dst: List[List[torch.Tensor]] = [[] for _ in range(world)]
    for chunk, send_sizes, _ in _comm._split_chunks(payload, sizes, sizes, n_chunks):
        offset = 0
        for d in range(world):
            k = send_sizes[d]
            per_dst[d].append(chunk[offset:offset + k])
            offset += k
        assert offset == int(chunk.shape[0])
    for d in range(world):
        got = torch.cat(per_dst[d], dim=0) if per_dst[d] else torch.empty(0, F)
        assert torch.equal(got, payload[d * B:(d + 1) * B]), f"dst {d} 的片段被破坏或重复"


def test_chunk_rows_match_declared_sizes_no_sum_inflation() -> None:
    """回归锚点：修复前 `sum(send_sizes) == world * 块行数`，恒不成立。"""
    world, B, n_chunks = 4, 8, 2
    payload = torch.randn(world * B, F)
    sizes = [B] * world
    for chunk, send_sizes, _ in _comm._split_chunks(payload, sizes, sizes, n_chunks):
        rows = int(chunk.shape[0])
        assert sum(send_sizes) == rows
        assert sum(send_sizes) != world * rows, "声明的 sizes 又被放大了 world 倍"


def test_variable_budget_degenerates_to_single_chunk() -> None:
    """变长预算下不做分块（宁可退化为单块，也不产出错误的切分）。"""
    send = [3, 5]
    recv = [5, 3]
    payload = torch.randn(sum(send), F)
    chunks = _comm._split_chunks(payload, send, recv, 4)
    assert len(chunks) == 1
    chunk, s, r = chunks[0]
    assert torch.equal(chunk, payload)
    assert s == send and r == recv


def test_send_recv_length_mismatch_raises() -> None:
    payload = torch.randn(8, F)
    with pytest.raises(ValueError):
        _comm._split_chunks(payload, [4, 4], [4, 4, 4], 1)


def test_payload_row_mismatch_raises() -> None:
    payload = torch.randn(7, F)          # 与 world * B = 8 不符
    with pytest.raises(ValueError):
        _comm._split_chunks(payload, [4, 4], [4, 4], 2)


def test_more_chunks_than_rows_is_safe() -> None:
    """n_chunks 大于每条边的行数时不应崩，也不应漏行。"""
    world, B = 2, 1
    payload = torch.randn(world * B, F)
    sizes = [B] * world
    chunks = _comm._split_chunks(payload, sizes, sizes, 4)
    assert len(chunks) == 1
    assert sum(int(c.shape[0]) for c, _, _ in chunks) == world * B


# ============================================================================
# 流水线把 sizes 传对了没有（用替身集合通信，验证调用接线）
# ============================================================================

@pytest.mark.parametrize("n_chunks", [1, 2, 4])
def test_sync_pipeline_passes_consistent_sizes(monkeypatch, no_barrier, n_chunks: int) -> None:
    world, B = 2, 8
    monkeypatch.setattr(_comm, "all_to_all_v", _stub_all_to_all)
    payload = torch.randn(world * B, F)
    res = _comm.run_sync_pipeline(
        payload, [B] * world, [B] * world,
        comp_work=lambda recv: recv.sum(dim=0), n_chunks=n_chunks,
    )
    assert res.total_ms >= 0.0
    assert res.comm_ms >= 0.0 and res.comp_ms >= 0.0


@pytest.mark.parametrize("n_chunks", [1, 2, 4])
def test_async_pipeline_passes_consistent_sizes(monkeypatch, no_barrier, n_chunks: int) -> None:
    world, B = 2, 8
    monkeypatch.setattr(_comm, "all_to_all_v_async", _stub_all_to_all_async)
    payload = torch.randn(world * B, F)
    res = _comm.run_async_pipeline(
        payload, [B] * world, [B] * world,
        comp_work=lambda recv: recv.sum(dim=0), n_chunks=n_chunks,
    )
    assert res.total_ms >= 0.0
    assert res.comm_ms >= 0.0 and res.comp_ms >= 0.0


def test_async_variant_also_checks_input_sizes(monkeypatch) -> None:
    """非阻塞版原先缺入参自洽检查，尺寸写错只会退化成
    `all_to_all_single` 的 "Split sizes doesn't match total dim 0 size"。"""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    with pytest.raises(AssertionError):
        _comm.all_to_all_v_async(torch.randn(6, F), [4, 4], [4, 4])
    with pytest.raises(AssertionError):
        _comm.all_to_all_v_async(torch.randn(8, F), [4], [4, 4])


def test_blocking_variant_checks_input_sizes(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    with pytest.raises(AssertionError):
        _comm.all_to_all_v(torch.randn(6, F), [4, 4], [4, 4])


def test_benchmark_ms_keeps_collective_barrier_out_of_the_window(monkeypatch) -> None:
    """计时窗口内只做设备同步，不做集合 barrier。

    旧实现在窗口收尾又调了一次 `barrier_and_sync()`，于是每个样本都额外含一次
    `dist.barrier()`，样本值还被最慢的 rank 支配 —— T_comm 与 T_comp 被同等抬高，
    二者的比值与 overlap 归因随之失真。
    """
    calls = {"barrier": 0, "device": 0}
    monkeypatch.setattr(_env, "barrier_and_sync",
                        lambda: calls.__setitem__("barrier", calls["barrier"] + 1))
    monkeypatch.setattr(_env, "device_sync",
                        lambda: calls.__setitem__("device", calls["device"] + 1))

    n_warm, n_iter = 2, 5
    samples = _env.benchmark_ms(lambda: None, warmup=n_warm, iters=n_iter)

    assert len(samples) == n_iter
    assert calls["barrier"] == n_iter + 1, "barrier 只应在窗口之前（含 warmup 后的一次）"
    assert calls["device"] == n_iter, "每个样本的窗口内应恰好一次设备同步"


def test_comp_work_applies_beta_and_is_repeatable() -> None:
    """模拟的计算必须把紧凑块里的 β 列加回 logits（否则测的不是 DCC-KV 的算子）。"""
    B, d_h, d_v = 5, 4, 3
    recv = torch.randn(B, d_h + 1 + d_v)
    work = _comm.make_comp_work(B * (d_h + 1 + d_v), d_h + 1 + d_v, d_h, d_v)
    out_a = work(recv)
    out_b = work(recv)
    assert out_a.shape[0] == out_b.shape[0]
    assert torch.allclose(out_a, out_b), "同一输入两次调用结果应一致（查询矩阵已缓存）"

    # β 生效的证据：只改 β 列，输出必须变化
    recv_beta_zero = recv.clone()
    recv_beta_zero[:, d_h] = 0.0
    assert not torch.allclose(work(recv), work(recv_beta_zero)), "β 列未参与计算"


def test_pipelines_use_device_sync_never_collective_barrier(monkeypatch) -> None:
    """收尾同步必须是 `device_sync()`（只等本设备），不能是 `barrier_and_sync()`。

    这是 A5 能不能测出 overlap 的前提：异步流水在 `handle.wait()` 之后若再来
    一次 device-wide 同步，第 i 块的 comp 就与第 i+1 块的 comm 串行了。
    """
    world, B = 2, 8
    payload = torch.randn(world * B, F)
    monkeypatch.setattr(_comm, "all_to_all_v", _stub_all_to_all)
    monkeypatch.setattr(_comm, "all_to_all_v_async", _stub_all_to_all_async)
    monkeypatch.setattr(_env, "barrier_and_sync",
                        lambda: pytest.fail("流水线内出现集合 barrier"))

    n = {"syncs": 0}
    monkeypatch.setattr(_env, "device_sync",
                        lambda: n.__setitem__("syncs", n["syncs"] + 1))

    _comm.run_sync_pipeline(payload, [B] * world, [B] * world,
                            comp_work=lambda r: r.sum(dim=0), n_chunks=2)
    _comm.run_async_pipeline(payload, [B] * world, [B] * world,
                             comp_work=lambda r: r.sum(dim=0), n_chunks=2)
    assert n["syncs"] >= 4, "两条流水线都应在每个计时段收尾做设备同步"


def test_local_device_uses_local_rank_not_global_rank(monkeypatch) -> None:
    """多节点下全局 rank 不是设备序号：rank 8 在第 2 个 8 卡节点上是 cuda:0。"""
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert _env.local_rank_of(8) == 3
    assert str(_env.local_device(8)) == "cuda:3"

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert _env.local_rank_of(8) == 8
    assert str(_env.local_device(8)) == "cuda:8"


def test_e5_binds_device_through_local_device() -> None:
    """源码级锚点：e5 里不得再出现 `torch.device(f"cuda:{rank}")`。"""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "experiments" / "gpu" / "e5_gpu_ablation.py").read_text(encoding="utf-8")
    assert 'torch.device(f"cuda:{rank}")' not in src
    assert "_env.local_device(rank)" in src
