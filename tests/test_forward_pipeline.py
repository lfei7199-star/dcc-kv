"""G2（异步流水接真实前向）的锚点。

核心不变式只有一条：**流水只改变"什么时候算"，不改变"算什么"。**
同步与异步两条路径的输出必须逐位相同。其余测试都是围着它的支撑：
布局的打包/解码互逆、块内源切分、退化为单块时如实上报、以及错误输入必须报错
而不是静默取实际值。

用 stub 顶掉集合通信后可以纯 CPU 运行 —— 这是能在租卡前把这条通路验完的前提。

命名注意：conftest.py 会把名字含 gpu / nccl / end_to_end / async_overlap 的用例
自动标成 gpu 并从默认运行里排除，故本文件刻意避开这些词。
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import hypotheses as H  # noqa: E402
from experiments.gpu import _comm, _env, _forward  # noqa: E402
from experiments.gpu import e5_gpu_ablation as E  # noqa: E402
from src.dcc_kv_ref import CompactKV  # noqa: E402
from src.dcc_kv_ref import attention_kernel as K  # noqa: E402

D_H, D_V = 8, 6


# ---------------------------------------------------------------------------
# 集合通信 stub
# ---------------------------------------------------------------------------

class _FakeHandle:
    def wait(self) -> None:
        return None


def _payload_from_send(send: torch.Tensor, recv_rows: int) -> torch.Tensor:
    """确定性 stub：按输入内容做一次可复现的"搬运"。

    刻意**不是** zeros —— 全零的块会让注意力退化到一条平凡路径，
    掩盖列错位之类的错误。这里用输入行的滚动拷贝，使每个 dst 段的内容
    与发送内容一一对应（也能让"行错配到别的源"这类错误显形）。
    """
    src = send
    reps = (int(recv_rows) + int(src.shape[0]) - 1) // max(1, int(src.shape[0]))
    if reps <= 1:
        return src[:int(recv_rows)].clone()
    return torch.cat([src] * reps, dim=0)[:int(recv_rows)].clone()


def _stub_all_to_all(send: torch.Tensor, send_sizes, recv_sizes) -> torch.Tensor:
    return _payload_from_send(send, int(sum(recv_sizes)))


def _stub_all_to_all_async(send: torch.Tensor, send_sizes, recv_sizes):
    return _stub_all_to_all(send, send_sizes, recv_sizes), _FakeHandle()


@pytest.fixture
def stubbed_collective(monkeypatch):
    monkeypatch.setattr(_comm, "all_to_all_v", _stub_all_to_all)
    monkeypatch.setattr(_comm, "all_to_all_v_async", _stub_all_to_all_async)
    # 让设备绑定落到 CPU（A5 会调 _env.local_device，本机无 CUDA）
    monkeypatch.setattr(_env, "local_device", lambda rank: torch.device("cpu"))
    return None


# ---------------------------------------------------------------------------
# 分块数：口径必须与 _split_chunks 一致
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("n_chunks", [1, 2, 3, 4, 8])
def test_effective_chunk_count_agrees_with_split_chunks(
    world: int, B: int, n_chunks: int
) -> None:
    """`effective_chunk_count` 与 `_split_chunks` 必须在尺寸网格上逐点一致。

    两条规则一旦漂移，产物里记的"实际块数"就是错的，而它正是判断
    「这条轴上到底有没有重叠窗口」的依据。`_split_chunks` 是纯函数（不发集合通信），
    故此处无需 stub。
    """
    payload = torch.randn(B * world, D_H + 1 + D_V)
    chunks = _comm._split_chunks(payload, [B] * world, [B] * world, n_chunks)
    assert _comm.effective_chunk_count([B] * world, [B] * world, n_chunks) == len(chunks)


def test_effective_chunk_count_degenerates_on_variable_budget() -> None:
    """逐边预算不等 ⇒ 不做分块 ⇒ 块数 1（异步没有可重叠的窗口）。"""
    assert _comm.effective_chunk_count([8, 4], [8, 4], 4) == 1
    assert _comm.effective_chunk_count([8, 8], [7, 8], 4) == 1
    # 等预算且 B>=2 时才真的分块
    assert _comm.effective_chunk_count([8, 8], [8, 8], 4) == 4
    assert _comm.effective_chunk_count([1, 1], [1, 1], 4) == 1


def test_effective_chunk_count_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        _comm.effective_chunk_count([8, 8], [8], 2)


# ---------------------------------------------------------------------------
# 打包 / 解码
# ---------------------------------------------------------------------------

def _edges(budgets, d_h: int = D_H, d_v: int = D_V, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for b in budgets:
        out.append(CompactKV(
            keys=torch.randn(int(b), d_h, generator=g),
            logit_bias=torch.randn(int(b), generator=g),
            values=torch.randn(int(b), d_v, generator=g),
            selected_indices=torch.arange(int(b), dtype=torch.long),
        ))
    return out


def test_pack_then_decode_is_lossless():
    """打包与解码必须互逆，且 β 列不能错位。

    β 丢了或串到 K/V 上，算出来的就是另一个算子 —— 而这在数值上只会
    表现为"结果略有不同"，不会报错。故逐位比较。
    """
    edges = _edges([5, 3])
    layout = _forward.EdgeLayout(world=2, d_h=D_H, d_v=D_V,
                                 send_sizes=(5, 3), recv_sizes=(5, 3))
    payload = _forward.pack_edges(edges, layout)
    assert tuple(payload.shape) == (8, D_H + 1 + D_V)

    back = _forward.decode_edges(payload, [5, 3], D_H, D_V)
    assert len(back) == 2
    for a, b in zip(edges, back):
        assert torch.equal(a.keys, b.keys)
        assert torch.equal(a.logit_bias, b.logit_bias)
        assert torch.equal(a.values, b.values)
        assert torch.equal(b.selected_indices,
                           torch.arange(b.keys.shape[0], dtype=torch.long))


def test_pack_edges_rejects_row_mismatch():
    edges = _edges([5, 3])
    layout = _forward.EdgeLayout(world=2, d_h=D_H, d_v=D_V,
                                 send_sizes=(4, 4), recv_sizes=(4, 4))
    with pytest.raises(ValueError):
        _forward.pack_edges(edges, layout)
    with pytest.raises(ValueError):
        _forward.pack_edges(edges[:1], layout)


def test_decode_edges_rejects_shape_mismatch():
    rows = torch.randn(6, D_H + 1 + D_V)
    with pytest.raises(ValueError):
        _forward.decode_edges(rows, [5], D_H, D_V)           # 行数对不上
    with pytest.raises(ValueError):
        _forward.decode_edges(rows, [6], D_H, D_V + 1)       # 特征维对不上


def test_decode_edges_skips_empty_edges():
    rows = torch.randn(3, D_H + 1 + D_V)
    back = _forward.decode_edges(rows, [3, 0], D_H, D_V)
    assert len(back) == 1


# ---------------------------------------------------------------------------
# 块内源切分
# ---------------------------------------------------------------------------

def test_chunk_source_sizes_handles_both_regimes():
    # 等预算 + 分块：块内均分
    assert _forward.chunk_source_sizes(6, [8, 8], 2) == [3, 3]
    # 变长：块 = 完整接收缓冲
    assert _forward.chunk_source_sizes(12, [8, 4], 2) == [8, 4]
    # 等预算 + 单块：两条规则同值
    assert _forward.chunk_source_sizes(16, [8, 8], 2) == [8, 8]
    assert _forward.chunk_source_sizes(0, [8, 8], 2) == [0, 0]


def test_chunk_source_sizes_refuses_to_guess():
    """行数既非总接收量也不能均分时必须报错。

    静默猜测会把某些行错配到别的源上 —— 结果仍然"像模像样"，
    只是把 A 源的内容当成了 B 源的键。
    """
    with pytest.raises(ValueError):
        _forward.chunk_source_sizes(7, [8, 8], 2)
    with pytest.raises(ValueError):
        _forward.chunk_source_sizes(4, [8, 8], 3)


# ---------------------------------------------------------------------------
# 核心不变式：同步 == 异步（逐位）
# ---------------------------------------------------------------------------

def _run_both(payload, layout, query, n_chunks, **kw):
    sync = _forward.pipelined_attention(query, payload, layout,
                                        mode="sync", n_chunks=n_chunks, **kw)
    asy = _forward.pipelined_attention(query, payload, layout,
                                       mode="async", n_chunks=n_chunks, **kw)
    return sync, asy


@pytest.mark.parametrize("n_chunks", [1, 2, 4, 8])
def test_sync_and_async_agree_bitwise(stubbed_collective, n_chunks: int) -> None:
    """流水不得改变答案 —— 本模块存在的理由。

    若两路只在 ULP 量级一致，A5 的加速比与 E6 的准确率就建立在两套数值上，
    测出的差异里混着归并顺序伪影（E0 已证归并顺序在 FP32 下可辨识）。
    """
    edges = _edges([8, 8])
    layout = _forward.make_uniform_layout(2, 8, D_H, D_V)
    payload = _forward.pack_edges(edges, layout)
    g = torch.Generator().manual_seed(3)
    query = torch.randn(4, D_H, generator=g)

    sync, asy = _run_both(payload, layout, query, n_chunks)
    _forward.assert_same_answer(sync, asy)
    assert sync.n_partials == asy.n_partials
    assert torch.equal(sync.out, asy.out)


def test_sync_and_async_agree_with_the_local_block(stubbed_collective) -> None:
    edges = _edges([6, 6])
    layout = _forward.make_uniform_layout(2, 6, D_H, D_V)
    payload = _forward.pack_edges(edges, layout)
    g = torch.Generator().manual_seed(4)
    query = torch.randn(3, D_H, generator=g)
    lk = torch.randn(7, D_H, generator=g)
    lv = torch.randn(7, D_V, generator=g)

    sync, asy = _run_both(payload, layout, query, 3, local_keys=lk, local_values=lv)
    _forward.assert_same_answer(sync, asy)
    assert sync.local_included and asy.local_included
    # 本地块固定是最后一个 partial
    assert sync.n_partials == 2 * 3 + 1


def test_assert_same_answer_detects_a_real_difference(stubbed_collective) -> None:
    """防止"断言恒真"：篡改一个输出必须让它炸。

    扰动幅度刻意取 1e-3（远大于 float32 的 ULP ≈ 6e-8）。用 1e-9 试过一次，
    它加到 O(0.5) 的 float32 上**恰好舍入回原值**，于是断言不炸 ——
    那不是断言失灵，是扰动本身没落到浮点栅格上。
    """
    edges = _edges([6, 6])
    layout = _forward.make_uniform_layout(2, 6, D_H, D_V)
    payload = _forward.pack_edges(edges, layout)
    query = torch.randn(3, D_H)
    sync, asy = _run_both(payload, layout, query, 2)
    asy.out = asy.out * (1.0 + 1e-3)
    assert not torch.equal(sync.out, asy.out)
    with pytest.raises(AssertionError):
        _forward.assert_same_answer(sync, asy)


# ---------------------------------------------------------------------------
# 语义：输出确实是紧凑 KV 的注意力
# ---------------------------------------------------------------------------

def test_output_equals_direct_merge_of_the_decoded_edges(stubbed_collective) -> None:
    """把搬运拆开手算一遍，验证本层没有偷偷换掉算子。

    若解码后拼错列、漏掉 β、或把归并写成了逐块平均，这里会立刻分叉。
    """
    edges = _edges([5, 5])
    layout = _forward.make_uniform_layout(2, 5, D_H, D_V)
    payload = _forward.pack_edges(edges, layout)
    g = torch.Generator().manual_seed(9)
    query = torch.randn(4, D_H, generator=g)

    got = _forward.pipelined_attention(query, payload, layout,
                                       mode="sync", n_chunks=1).out
    want = K.merge_partial_attention([
        K.compact_kv_attention(query, ck, return_lse=True) for ck in edges
    ])
    assert torch.equal(got, want)


def test_chunking_shifts_the_answer_at_ulp_level_only(stubbed_collective) -> None:
    """分块**会**改变答案，但只到 ULP 级 —— 这条与"同步==异步"是两件事。

    最初把这条写成了 `assert torch.equal(out_n, out_1)`（即"分块不改变数值"），
    实测**不成立**：每个源边的 softmax 被切成子 softmax 再用 lse 归并，
    与对整块一次 softmax 是两个不同的浮点求和。float32 下实测差 ≤1.79e-7，
    而 ULP ≈ 5.96e-8 ⇒ 1–3 ULP。

    所以正确的口径是：`n_chunks` 是时延/显存旋钮，不是数值旋钮。
    要逐位复现就必须固定它。
    """
    edges = _edges([8, 8])
    layout = _forward.make_uniform_layout(2, 8, D_H, D_V)
    payload = _forward.pack_edges(edges, layout)
    g = torch.Generator().manual_seed(5)
    query = torch.randn(4, D_H, generator=g)

    base = _forward.pipelined_attention(query, payload, layout, mode="sync",
                                        n_chunks=1).out
    ulp = float(torch.finfo(torch.float32).eps) * float(base.abs().max())
    for n in (2, 3, 4, 8):
        out = _forward.pipelined_attention(query, payload, layout, mode="sync",
                                           n_chunks=n).out
        diff = float((out - base).abs().max())
        assert diff <= 4 * ulp, f"n_chunks={n} 的偏差 {diff:.3e} 超出 4 ULP"

    # float64 下同一比较应落到 1 ULP 量级 —— 证明这是浮点累加次序，不是逻辑错误
    q64 = query.double()
    p64 = payload.double()
    d1 = _forward.pipelined_attention(q64, p64, layout, mode="sync", n_chunks=1).out
    d4 = _forward.pipelined_attention(q64, p64, layout, mode="sync", n_chunks=4).out
    assert float((d1 - d4).abs().max()) <= 8 * float(torch.finfo(torch.float64).eps)


def test_mode_invariance_holds_at_every_chunk_count(stubbed_collective) -> None:
    """调度不变式不得是"只在某一档分块下凑巧成立"。"""
    edges = _edges([8, 8])
    layout = _forward.make_uniform_layout(2, 8, D_H, D_V)
    payload = _forward.pack_edges(edges, layout)
    g = torch.Generator().manual_seed(6)
    query = torch.randn(4, D_H, generator=g)
    for n in (1, 2, 3, 4, 8):
        sync, asy = _run_both(payload, layout, query, n)
        assert torch.equal(sync.out, asy.out), f"n_chunks={n} 时两路输出不等"
        assert sync.n_partials == asy.n_partials


def test_variable_budget_reports_no_overlap_window(stubbed_collective) -> None:
    """变长预算 ⇒ 块数 1 ⇒ 如实上报"这条轴上没有重叠机会"。

    这不是实现故障。把 `speedup≈1.0` 当成实现问题去修，会白烧一轮卡时；
    反之，把"还没测"写成"没有这条轴"也不对 —— 故两者都在产物里显式区分。
    """
    edges = _edges([8, 4])
    layout = _forward.EdgeLayout(world=2, d_h=D_H, d_v=D_V,
                                 send_sizes=(8, 4), recv_sizes=(8, 4))
    payload = _forward.pack_edges(edges, layout)
    query = torch.randn(4, D_H)
    res = _forward.pipelined_attention(query, payload, layout,
                                       mode="async", n_chunks=4)
    assert res.chunks_effective == 1
    assert res.to_dict()["overlap_window_available"] is False

    uni = _forward.make_uniform_layout(2, 8, D_H, D_V)
    res2 = _forward.pipelined_attention(query, _forward.pack_edges(
        _edges([8, 8]), uni), uni, mode="async", n_chunks=4)
    assert res2.chunks_effective == 4
    assert res2.to_dict()["overlap_window_available"] is True


# ---------------------------------------------------------------------------
# 入参自洽性
# ---------------------------------------------------------------------------

def test_pipelined_attention_rejects_bad_mode(stubbed_collective) -> None:
    layout = _forward.make_uniform_layout(2, 4, D_H, D_V)
    payload = _forward.pack_edges(_edges([4, 4]), layout)
    with pytest.raises(ValueError):
        _forward.pipelined_attention(torch.randn(2, D_H), payload, layout,
                                     mode="pipeline")


def test_pipelined_attention_rejects_payload_row_mismatch(stubbed_collective) -> None:
    layout = _forward.make_uniform_layout(2, 4, D_H, D_V)
    payload = _forward.pack_edges(_edges([4, 4]), layout)
    with pytest.raises(ValueError):
        _forward.pipelined_attention(torch.randn(2, D_H), payload[:7], layout)


def test_pipelined_attention_requires_local_keys_and_values_together(
    stubbed_collective,
) -> None:
    layout = _forward.make_uniform_layout(2, 4, D_H, D_V)
    payload = _forward.pack_edges(_edges([4, 4]), layout)
    with pytest.raises(ValueError):
        _forward.pipelined_attention(torch.randn(2, D_H), payload, layout,
                                     local_keys=torch.randn(3, D_H))


def test_pipelined_attention_rejects_empty_everything(stubbed_collective) -> None:
    """所有源边都是 0 行且没有本地块 ⇒ 没有任何键可用，必须报错。"""
    layout = _forward.EdgeLayout(world=2, d_h=D_H, d_v=D_V,
                                 send_sizes=(0, 0), recv_sizes=(0, 0))
    payload = torch.empty(0, D_H + 1 + D_V)
    with pytest.raises(ValueError):
        _forward.pipelined_attention(torch.randn(2, D_H), payload, layout)


# ---------------------------------------------------------------------------
# A5 的两个臂：sim（规模模拟）与 real（G1 算子核）
# ---------------------------------------------------------------------------

A5_ARGS = dict(
    precision="float32", L_s=64, d_h=D_H, d_v=D_V, L_r=4,
    budget=16, M=2, d_p=4, seed=42, a5_chunks=[1, 2], a5_comp_source="both",
    comp_scale=1.0, warmup=1, iters=4, chunks=4,
)
A5_CTX = {"build_location": {"effective": "cpu", "probe": {"ok": False}}}


@pytest.fixture
def nondeterministic_collective(monkeypatch):
    """每调用一次返回一批**新**随机数 —— 模拟传输/构造在两次测量间不稳定。

    这是"调度不改变答案"被破坏时最现实的成因，用来验证脚本会如实上报
    而不是把加速比当成有效数据。
    """
    counter = {"n": 0}

    def _stub(send: torch.Tensor, send_sizes, recv_sizes) -> torch.Tensor:
        counter["n"] += 1
        g = torch.Generator().manual_seed(1000 + counter["n"])
        return torch.randn(int(sum(recv_sizes)), send.shape[1], generator=g,
                           dtype=send.dtype, device=send.device) * 0.1

    def _stub_async(send: torch.Tensor, send_sizes, recv_sizes):
        return _stub(send, send_sizes, recv_sizes), _FakeHandle()

    monkeypatch.setattr(_comm, "all_to_all_v", _stub)
    monkeypatch.setattr(_comm, "all_to_all_v_async", _stub_async)
    monkeypatch.setattr(_env, "local_device", lambda rank: torch.device("cpu"))
    return None


def _a5(a, ctx=A5_CTX):
    return E.a5_async_vs_sync(0, 2, a, ctx)


def test_a5_reports_one_row_per_arm_and_chunk_count(stubbed_collective) -> None:
    res = _a5(A5_ARGS)
    assert res["arms"] == ["sim", "real"]
    assert len(res["rows"]) == 2 * len(A5_ARGS["a5_chunks"])
    assert {r["arm"] for r in res["rows"]} == {"sim", "real"}
    assert {r["chunks"] for r in res["rows"]} == set(A5_ARGS["a5_chunks"])
    for row in res["rows"]:
        assert row["h4_threshold_used"] == H.H4_MIN_P50_SPEEDUP
        assert row["ci_method"] == "paired_percentile_bootstrap"
        assert row["bound_basis"] in ("standalone_baselines",
                                      "sync_pipeline_decomposition")


def test_a5_real_arm_verifies_the_scheduling_invariance(stubbed_collective) -> None:
    """real 臂必须在运行时自检「同步 == 异步（逐位）」。"""
    res = _a5(A5_ARGS)
    real = [r for r in res["rows"] if r["arm"] == "real"]
    assert len(real) == len(A5_ARGS["a5_chunks"])
    for row in real:
        assert row["sync_equals_async"] is True
        assert row["sync_async_max_abs_diff"] == 0.0
        assert row["h4_judged"] is True
        assert row["comp_source"].endswith("attention_kernel.py")
        assert row["comp_scale"] is None
    assert res["status"] == "ok"
    assert res["h4"]["valid"] is True


def test_a5_refuses_to_judge_when_the_invariance_breaks(
    nondeterministic_collective,
) -> None:
    """不变式被破坏时：**不抛错**，但必须拒绝下判定并把诊断落盘。

    抛错会带走整段有效计时与诊断；不标记则会让"拿两套数值比出来的加速比"
    悄悄流进 CSV（CSV 只落 rows，顶层标志走不到那里）。故判据落在**行内**：
    h4_pass 置 None（未判定）、h4_judged=False。
    """
    res = _a5(A5_ARGS)
    assert res["status"] == "numeric_invariance_broken"
    assert res["h4"]["valid"] is False
    assert res["h4"]["invalid_reason"]

    real = [r for r in res["rows"] if r["arm"] == "real"]
    assert real, "real 臂应当仍有行（计时有效，只是判据不可用）"
    for row in real:
        assert row["sync_equals_async"] is False
        assert row["sync_async_max_abs_diff"] > 0.0
        assert row["h4_pass"] is None
        assert row["h4_judged"] is False
        # 计时数据仍然在
        assert row["p50_sync_ms"] >= 0.0 and row["p50_async_ms"] >= 0.0

    # sim 臂与数值不变式无关，不应被牵连
    for row in (r for r in res["rows"] if r["arm"] == "sim"):
        assert row["sync_equals_async"] is None
        assert row["h4_judged"] is True


@pytest.mark.parametrize("source", ["sim", "real"])
def test_a5_can_run_a_single_arm(stubbed_collective, source: str) -> None:
    a = dict(A5_ARGS)
    a["a5_comp_source"] = source
    res = _a5(a)
    assert res["arms"] == [source]
    assert {r["arm"] for r in res["rows"]} == {source}
    assert set(res["h4"]["passes_by_arm"]) == {source}


def test_a5_real_rows_report_the_effective_chunk_count(stubbed_collective) -> None:
    """预算 16、请求 8 块 ⇒ 实际 8 块，重叠窗口存在（B>=2 且等预算）。

    这条轴是"有没有机会重叠"的依据：若它被写成 1，读到 speedup≈1.0 的人
    会去修一个没坏的实现。
    """
    a = dict(A5_ARGS)
    a["a5_chunks"] = [1, 8]
    res = _a5(a)
    real = {r["chunks"]: r for r in res["rows"] if r["arm"] == "real"}
    assert real[1]["chunks_effective"] == 1
    assert real[1]["overlap_window_available"] is False
    assert real[8]["chunks_effective"] == 8
    assert real[8]["overlap_window_available"] is True


def test_a5_declares_which_arm_carries_the_h4_verdict(stubbed_collective) -> None:
    """H4 的判据落在哪个臂上必须显式声明，且两臂不可混报。

    同一份硬件上 sim 臂可能远大于 1、real 臂约 1.0 —— 差别只在"计算量是多少"
    这个前提。把两臂的加速比混在一张表里，同一个实验既能被报成达标也能被报成
    未达标。
    """
    res = _a5(A5_ARGS)
    h4 = res["h4"]
    assert h4["judged_on_arm"] == "sim"
    assert set(h4["passes_by_arm"]) == {"sim", "real"}
    for arm, info in h4["passes_by_arm"].items():
        assert info["n_rows"] == len(A5_ARGS["a5_chunks"])
        assert 0 <= info["n_rows_passing"] <= info["n_rows"]
    assert "不可混报" in h4["rationale"]
