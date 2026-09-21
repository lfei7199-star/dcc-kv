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

import ast
import itertools
import pathlib
from typing import List, Tuple

import pytest
import torch

from experiments.gpu import _comm, _env, _hf

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

    eff = _comm.effective_chunk_count([B] * world, [B] * world, 2)
    assert eff == 2, "本用例假定分块确实发生（否则就是退化情形，另有用例覆盖）"

    _comm.run_sync_pipeline(payload, [B] * world, [B] * world,
                            comp_work=lambda r: r.sum(dim=0), n_chunks=2)
    n_sync = n["syncs"]

    n["syncs"] = 0
    _comm.run_async_pipeline(payload, [B] * world, [B] * world,
                             comp_work=lambda r: r.sum(dim=0), n_chunks=2)
    n_async = n["syncs"]

    # 计数必须是**精确值**，不能只写下界。原断言是 `>= 4`：实测同步臂 4、异步臂 2、
    # 合计 6，余量恒为 2 ⇒ 「异步臂多一次 device_sync」这种回归完全看不见。
    # 而多一次 device_sync 恰恰就是那条把 overlap 抹平、加速比钉在 1.0 的失效
    # 模式（device-wide 同步会等掉正在飞的下一块）。所以这里按臂分开、取等号：
    #   同步臂：每块「通信段 + 计算段」各收尾一次 ⇒ 2 × 实际块数
    #   异步臂：每轮只在 comp 段收尾一次（+ 末轮）⇒ 实际块数
    assert n_sync == 2 * eff, f"同步臂 device_sync 次数应为 {2 * eff}，实得 {n_sync}"
    assert n_async == eff, f"异步臂 device_sync 次数应为 {eff}，实得 {n_async}"


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


# ============================================================================
# E6 的 sync/async 轴折叠（C20）
# ============================================================================

def _e6_module():
    import importlib
    return importlib.import_module("experiments.gpu.e6_main_table")


def _ns(**kw):
    import argparse
    base = dict(models=["m"], context_lengths=[4096, 8192],
                sync_modes=["sync", "async"],
                methods=["dense", "kv_budget_shared", "dcc_kv", "ring", "apb",
                         "fastkv_official"])
    base.update(kw)
    return argparse.Namespace(**base)


def test_e6_sync_axis_applies_only_to_multi_device_methods() -> None:
    """折叠判据是**两个**条件的合取：既有跨设备通信，本次又真的起了多卡。

    只满足前半个（2026-09-20 的旧判据）会在单卡模拟下凭空造出这条轴：
    `dcc_kv` 的 `gpu_required` 是 2，而 `--dcc-world` 只是把源序列切段，
    `gpu_count_observed` 仍是 1 ⇒ 生成的两行会**逐位相同**。
    """
    e6 = _e6_module()
    single_card = _ns(ranks=1)
    two_cards = _ns(ranks=2)
    # 单卡方法：无论几卡都没有这条轴
    for m in ("dense", "kv_budget_shared"):
        assert e6.sync_axis_applies(m, single_card) is False, m
        assert e6.sync_axis_applies(m, two_cards) is False, m
    # 多卡方法：只有本次真的起了多卡才算自变量
    for m in ("dcc_kv", "ring", "apb", "fastkv_official"):
        assert e6.sync_axis_applies(m, two_cards) is True, m
        assert e6.sync_axis_applies(m, single_card) is False, m


def test_e6_planned_points_counts_sync_axis_per_method() -> None:
    """计划点数按各方法自身的轴累加：折叠轴的方法只贡献 1。

    回归锚点：旧式 `n_models * n_ctx * n_sync * n_methods` 会把"永远不会被生成
    的数据点"算进计划数，计划数与实测数的差额会被误读成"漏跑"。

    2026-09-21 补：折叠判据从「单卡方法」推广成「**本次运行没有真的用多卡**」。
    `--ranks` 默认 1 ⇒ 单卡模拟下 `dcc_kv` 的 sync 轴同样不存在。
    """
    e6 = _e6_module()
    a = _ns(models=["m"], context_lengths=[4096, 8192],
            sync_modes=["sync", "async"], methods=["dense", "dcc_kv"])
    # 单卡：两个方法都折叠 ⇒ 1 × 2 × 1 × 2 = 4
    assert e6.planned_points(a) == 4
    # 旧口径：1 × 2 × 2 × 2 = 8 —— 多算了 4 个永不生成的点
    assert e6.planned_points(a) != 8

    # 真的起了多卡：dcc_kv 的轴展开 ⇒ 1 × 2 × (1 + 2) = 6
    a2 = _ns(models=["m"], context_lengths=[4096, 8192],
             sync_modes=["sync", "async"], methods=["dense", "dcc_kv"], ranks=2)
    assert e6.planned_points(a2) == 6

    # 六个方法全选：单卡全折叠；双卡时 4 个多卡方法展开、2 个单卡方法折叠
    assert e6.planned_points(_ns()) == 1 * 2 * 6
    naive = 1 * 2 * 2 * 6
    assert e6.planned_points(_ns(ranks=2)) == naive - 1 * 2 * (2 - 1) * 2


def test_e6_single_device_rows_use_na_not_a_fake_sync_mode() -> None:
    """折叠后写入行里的 sync_async 必须是 "n/a"，不能是 "sync"。

    用 "sync" 顶上会把"该轴不适用"写成一个真实的测量条件。
    """
    e6 = _e6_module()
    assert e6.SYNC_MODE_NA == "n/a"
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "experiments" / "gpu" / "e6_main_table.py").read_text(encoding="utf-8")
    assert '"sync_mode_applicable": False' not in src, "标注又被写死了"
    # 三处：measure_point、blocked_point、主循环的 except 分支
    assert src.count('"sync_mode_applicable": sync_axis_applies(method, a)') == 3
    assert '[SYNC_MODE_NA]' in src, "主循环没有折叠该轴"
    # 折叠判据必须带本次实参：不带就成了"只看 gpu_required"，单卡模拟下
    # dcc_kv 会被误判为"有这条轴"（见 planned_points 与 sync_axis_applies）
    assert "sync_axis_applies(method)" not in src
    assert "sync_axis_applies(m)" not in src


# ============================================================================
# E6：H0 接线本身（2026-09-21）—— 「表里可测」必须等于「代码真的接了」
# ============================================================================

def test_e6_dcc_row_is_wired_to_the_hook() -> None:
    """结构锚点：`dcc_kv` 行必须**真的**把 `HookConfig` 传下去。

    只把 `measurable` 翻成 True 而不接线，会得到一张"看起来测得出数"、实际
    仍走 `apply_kv_budget` 的表 —— 那时质量差恒为 0，而它会被读成
    "DCC-KV 与共享压缩质量相当"。所以这里断言的是**接线**，不是"能跑起来"。
    """
    e6 = _e6_module()
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "experiments" / "gpu" / "e6_main_table.py").read_text(encoding="utf-8")
    assert "_A.HookConfig(" in src, "没有构造钩子配置"
    assert 'method == "dcc_kv"' in src, "没有 dcc_kv 分支"
    # 两个入口各一次：measure_prefill（计时）与 evaluate（质量）
    assert src.count("attn_hook=hook") == 2
    # 评测必须把同一份源/目的端切分传下去，否则质量差里混进预算不对齐
    assert "dest_fraction=a.dest_fraction" in src
    # 钩子路径的预算归 HookConfig（`_hf` 会对"两处都声明"直接报错）
    assert "budget_ratio=a.budget_ratio," in src
    # 缺 --dcc-world 时拒绝执行，而不是猜一个段数
    assert '"dcc_kv" in a.methods and a.dcc_world is None' in src


def test_e6_measure_point_passes_the_hook_to_both_entry_points() -> None:
    """AST 锚点：`measure_point` 内对两个 `_hf` 入口的调用都带 `attn_hook=`。

    源码级守卫一律建在**结构**上（本仓库纪律）：`attn_hook=hook` 这串字符
    出现在别处（例如注释）也能让上面那条通过，而 AST 锚点只看真实的调用。
    """
    e6 = _e6_module()
    tree = ast.parse(pathlib.Path(e6.__file__).read_text(encoding="utf-8"))
    fns = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "measure_point"]
    assert len(fns) == 1, "measure_point 的定义数不是 1"
    seen = {}
    for call in [n for n in ast.walk(fns[0]) if isinstance(n, ast.Call)]:
        f = call.func
        if not isinstance(f, ast.Attribute):
            continue
        base = f.value.id if isinstance(f.value, ast.Name) else None
        if base == "_hf" and f.attr in ("measure_prefill", "evaluate"):
            seen.setdefault(f.attr, []).append(
                {k.arg: k.value for k in call.keywords if k.arg})
    assert set(seen) == {"measure_prefill", "evaluate"}, sorted(seen)

    # 断言建在**取值**上，不建在"这个词有没有出现"上：后者在
    # `dest_fraction=0.0`、`attn_hook=None` 这类变异下照样通过（实测过）。
    for name, kwsets in seen.items():
        for kws in kwsets:
            v = kws.get("attn_hook")
            assert isinstance(v, ast.Name) and v.id in ("hook", "dense_hook"), (
                name, "attn_hook 必须绑定到钩子配置，实测 "
                + (ast.dump(v) if v is not None else "缺该关键字"))
    assert len(seen["evaluate"]) == 1
    d = seen["evaluate"][0].get("dest_fraction")
    assert isinstance(d, ast.Attribute) and d.attr == "dest_fraction", (
        "evaluate 必须拿到与计时同一份源/目的端切分（a.dest_fraction），实测 "
        + (ast.dump(d) if d is not None else "缺该关键字"))


def test_e6_timing_and_quality_share_the_same_split() -> None:
    """计时路径的源长必须与评测路径同源（都走 `_hf.prompt_split`）。

    实测踩到：计时侧把**整条** ctx_len 当源端再另加 dest_len，于是它压缩的源长
    是 ctx_len、评测压缩的是 (1-f)·ctx_len —— 同一行里两个不同的压缩设置，
    产物里的 `budget_tokens_resolved` 对不上质量那一侧（计时报 32、实际 24），
    而"两臂唯一差别是 KV 长度"这条配对性也建在了一个不是评测所用的源上。
    """
    e6 = _e6_module()
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "experiments" / "gpu" / "e6_main_table.py").read_text(encoding="utf-8")
    assert "_hf.prompt_split(ctx_len, a.dest_fraction)" in src
    assert "a.dest_fraction * ctx_len" not in src, "又在两处各算一遍切分"
    assert "source_len_by_context" in src, "payload 里的切分表也要同源"


def test_e6_kernel_matched_speedup_refuses_none_instead_of_guessing() -> None:
    """kernel-matched 加速比：缺任一端、或分母非正 ⇒ None（**不是** 0/1）。

    None 与 0.0 / 1.0 是三个不同的意思。混用会让"没测出来"看起来像
    "测出来是没有收益"，而这一列正是 H2 的第二个合取项。
    """
    e6 = _e6_module()
    K = e6._kernel_matched_speedup
    assert K({"dest_ms_median": 10.0,
              "dest_ms_median_dense_kernel": 12.0}) == pytest.approx(1.2)
    assert K({"dest_ms_median": 10.0}) is None            # 缺分子
    assert K({"dest_ms_median_dense_kernel": 12.0}) is None   # 缺分母
    assert K({}) is None
    assert K({"dest_ms_median": 0.0,
              "dest_ms_median_dense_kernel": 12.0}) is None   # 分母非正
    assert K({"dest_ms_median": 10.0,
              "dest_ms_median_dense_kernel": float("nan")}) is None
    assert K({"dest_ms_median": 10.0,
              "dest_ms_median_dense_kernel": float("inf")}) is None


def test_e6_dcc_world_has_no_default_and_speedup_columns_are_declared() -> None:
    """`--dcc-world` 无默认值；两列加速比各自有口径声明（不得互相冒充）。"""
    e6 = _e6_module()
    a = e6.build_parser().parse_args(["--plan"])
    assert a.dcc_world is None, "--dcc-world 又有了默认值（= 没人声明过的假设）"
    assert a.dcc_budget_mode == "per_edge", "主口径必须是 per_edge"
    assert a.ranks == 1, "--ranks 默认必须是 1（默认单卡）"
    assert a.lambda_beta is None, "未给时应回落到源码默认，而不是在 CLI 里再抄一份"

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "experiments" / "gpu" / "e6_main_table.py").read_text(encoding="utf-8")
    assert '"prefill_speedup_columns"' in src, "payload 里没有两列口径声明"
    assert "prefill_speedup_numerator" not in src, "旧的单列说法还在，会与两列冲突"
    assert "attach_prefill_speedups(a, rows)" in src, "没有补 native 列的那一步"
    # native 必须在 compute_h2 之前补好（H2 读 kernel-matched，但两列都落盘）
    i_attach = src.index("attach_prefill_speedups(a, rows)")
    i_h2 = src.index('"h2": compute_h2(a, rows),')
    assert i_attach < i_h2, "补列的调用晚于 compute_h2"


# ============================================================================
# apply_kv_budget 的显存峰值（C21）
# ============================================================================

def test_pooled_key_energy_matches_full_float_computation() -> None:
    """分块算能量谱必须与"整张 k.float() 一次算完"逐位等价。

    这是"省内存"这个改动的正确性锚点：省法只有在结果不变时才成立。
    """
    torch.manual_seed(0)
    k = torch.randn(2, 3, 5000, 8).to(torch.bfloat16)

    got = _hf._pooled_key_energy(k, chunk=1024, mode="topk_rms")
    ref = k.float().pow(2).mean(dim=(0, 1, 3))
    assert got.shape == ref.shape == (5000,)
    assert got.dtype == torch.float32
    assert torch.equal(got, ref), "rms 分块与整张不逐位一致"

    got_n = _hf._pooled_key_energy(k, chunk=1024, mode="topk_norm")
    ref_n = k.float().pow(2).sum(dim=(0, 1, 3))
    assert torch.equal(got_n, ref_n), "norm 分块与整张不逐位一致"

    # 分块大小是**显存参数**，不是数值参数：chunk >= 2 时结果逐位相同。
    # （chunk == 1 会让 PyTorch 对 [B,H,1,d] 走另一条归约 kernel，产生 1 ULP
    #   级差异；它不影响保留位置，见下一个测试。默认 chunk=8192 不受影响。）
    for chunk in (2, 7, 64, 999, 8192, 100000):
        assert torch.equal(_hf._pooled_key_energy(k, chunk=chunk), ref), (
            f"chunk={chunk} 改变了结果")


def test_kv_budget_kept_positions_are_bitwise_stable_across_chunk_sizes() -> None:
    """分块算能量**不得改变保留位置** —— 位置变了才会改变准确率。

    这是对原 C22 顾虑（"分块累加会改变求和顺序、进而可能改变保留位置与
    准确率，故不宜在无 GPU 复核时改动"）的直接检验。能量归约沿 (B, H, d) 维、
    分块沿 S 维，两者**正交**，所以每个 s 位置的归约元素集合与顺序都不变。

    实测：任何 chunk（含 1）下保留位置都逐位一致 —— 即"省显存"与"不改变
    任务指标"在这里可以解耦，顾虑不成立。
    """
    torch.manual_seed(1)
    S, d, B = 3000, 6, 256
    k = torch.randn(1, 2, S, d).to(torch.bfloat16)

    idx_ref = torch.topk(k.float().pow(2).mean(dim=(0, 1, 3)), B,
                         largest=True).indices.sort().values
    for chunk in (1, 2, 7, 1024, 8192, 10 ** 9):
        idx = torch.topk(_hf._pooled_key_energy(k, chunk=chunk), B,
                         largest=True).indices.sort().values
        assert torch.equal(idx, idx_ref), f"chunk={chunk} 改变了保留位置"


def test_apply_kv_budget_selects_same_indices_as_full_computation() -> None:
    """预算裁剪选出的位置，必须与整张计算出的 top-B 完全一致。"""
    torch.manual_seed(1)
    S, d = 3000, 6
    k = torch.randn(1, 2, S, d).to(torch.bfloat16)
    v = torch.randn(1, 2, S, d).to(torch.bfloat16)

    B = 256
    cache = [(k.clone(), v.clone())]
    stats = _hf.apply_kv_budget(cache, budget=B, mode="topk_rms")

    assert stats["kept_per_layer"] == [B]
    assert stats["total_per_layer"] == [S]
    kk, vv = cache[0]
    assert int(kk.shape[-2]) == B and int(vv.shape[-2]) == B

    idx = torch.topk(k.float().pow(2).mean(dim=(0, 1, 3)), B,
                     largest=True).indices.sort().values
    assert torch.equal(kk, k[..., idx, :].contiguous())
    assert torch.equal(vv, v[..., idx, :].contiguous())


def test_apply_kv_budget_identity_and_short_cache_untouched() -> None:
    """identity 与"S <= budget"两条短路不得改动缓存。"""
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    cache = [(k.clone(), v.clone())]
    stats = _hf.apply_kv_budget(cache, budget=4, mode="identity")
    assert stats["mean_keep_ratio"] == 1.0
    assert torch.equal(cache[0][0], k)

    cache2 = [(k.clone(), v.clone())]
    stats2 = _hf.apply_kv_budget(cache2, budget=99, mode="topk_rms")
    assert stats2["kept_per_layer"] == [8]
    assert torch.equal(cache2[0][0], k)
