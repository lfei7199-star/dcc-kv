"""A4（压缩 × 异步交互）的锚点 —— 缺口 G4 的回归。

A4 是**唯一**一个把「测量」和「判定」都放在设备无关路径上的消融：
判据在 `experiments/common/hypotheses.py`，统计底座在 `experiments/common/report.py`，
测量在 `experiments/gpu/e5_gpu_ablation.py`。三者都能在无 GPU 的机器上验到，
所以本文件用 stub 顶掉集合通信后**真跑一遍** a4_interaction_grid，
而不是只做源码字符串检查（字符串检查挡不住"接线接反"）。

本文件覆盖三类锚点：

1. 判定语义：区间重叠/不重叠/NaN、档数不足、两条可证伪预测；
2. 接线纪律：判据不得被脚本重新实现、CI 水平不得硬编码；
3. 测量正确性：A4 的计算侧必须是**真实算子核**（不是规模模拟），
   且 T_comm/T_comp 的拆解必须取自同步臂。

命名注意：conftest.py 会把名字含 gpu / nccl / end_to_end / async_overlap 的用例
自动标成 gpu 并从默认运行里排除。本文件是纯 CPU 回归，故刻意避开这些词。
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
from experiments.common import report as R  # noqa: E402
from experiments.gpu import _comm, _env  # noqa: E402
from experiments.gpu import e5_gpu_ablation as E  # noqa: E402

E5_SRC = (REPO_ROOT / "experiments" / "gpu" / "e5_gpu_ablation.py").read_text(
    encoding="utf-8")


# ---------------------------------------------------------------------------
# 判定语义（hypotheses.a4_interaction_verdict）
# ---------------------------------------------------------------------------

def _pt(budget: int, speedup: float, lo: float, hi: float, *,
        comm_comp: float = 1.0, bound: float = 2.0) -> H.A4BudgetPoint:
    return H.A4BudgetPoint(
        budget_ratio=budget / 1024.0,
        budget=budget,
        p50_sync_ms=10.0,
        p50_async_ms=10.0 / speedup,
        speedup=speedup,
        speedup_ci_low=lo,
        speedup_ci_high=hi,
        t_comm_over_t_comp=comm_comp,
        theoretical_bound=bound,
        t_build_share=0.1,
    )


def test_a4_constants_match_the_paper():
    """CI 水平与最少档数是论文 §6.3 明写的，改动必须是有意识的。"""
    assert H.A4_CI_LEVEL == 0.95
    assert H.A4_MIN_BUDGET_LEVELS == 4


def test_a4_all_ci_overlap_allows_separate_citation():
    pts = [
        _pt(32, 1.20, 1.15, 1.25),
        _pt(64, 1.22, 1.17, 1.27),
        _pt(128, 1.24, 1.19, 1.29),
        _pt(256, 1.26, 1.21, 1.31),
    ]
    v = H.a4_interaction_verdict(pts)
    assert v.resolved is True
    assert v.all_ci_overlap is True
    assert v.interaction_observed is False
    assert v.reporting_requirement == "edge_results_may_be_cited_separately"
    assert v.n_overlapping_pairs == v.n_pairs == 6


def test_a4_resolvable_structure_forces_the_2d_grid():
    """第三个区间与第一个完全不相交 ⇒ 必须报二维格。

    顺带锚住**闭区间**语义：第 1、2 档的区间恰好端点相触（1.03 与 1.03），
    算重叠。若日后有人把判据改成开区间，这一对会翻面。
    """
    pts = [
        _pt(32, 1.01, 0.99, 1.03),
        _pt(64, 1.05, 1.03, 1.07),
        _pt(128, 1.20, 1.18, 1.22),
        _pt(256, 1.35, 1.33, 1.37),
    ]
    v = H.a4_interaction_verdict(pts)
    assert v.resolved is True
    assert v.all_ci_overlap is False
    assert v.interaction_observed is True
    assert v.reporting_requirement == "must_report_2d_grid"
    # 相邻相触的那一对必须被算作重叠
    assert v.n_overlapping_pairs >= 1


def test_a4_insufficient_levels_is_not_no_interaction():
    """档数不足 ⇒ resolved=False，且**不得**报成「未观测到交互」。

    这两者混同是 A4 最危险的误读：把「没测够」读成「没有交互」，
    于是两条边缘曲线被当成可以分开引用。
    """
    v = H.a4_interaction_verdict([
        _pt(32, 1.1, 1.0, 1.2), _pt(64, 1.2, 1.1, 1.3), _pt(128, 1.3, 1.2, 1.4),
    ])
    assert v.resolved is False
    assert v.reporting_requirement == "insufficient_data"
    assert v.interaction_observed is False
    assert "不等于" in v.reason
    assert v.n_budgets == 3


def test_a4_nan_interval_is_treated_as_not_overlapping():
    """信息缺失一律当作"不重叠"（保守方向：要求报二维格），绝不当作重叠。

    反向处理的后果是：一次测量失败会静默升级为「无交互、边缘结果可分开引用」，
    也就是让缺数据伪装成更强的结论。
    """
    pts = [
        _pt(32, float("nan"), float("nan"), float("nan")),
        _pt(64, 1.22, 1.17, 1.27),
        _pt(128, 1.24, 1.19, 1.29),
        _pt(256, 1.26, 1.21, 1.31),
    ]
    v = H.a4_interaction_verdict(pts)
    assert v.resolved is True
    assert v.all_ci_overlap is False
    assert v.interaction_observed is True


def test_a4_empty_input_raises():
    with pytest.raises(ValueError):
        H.a4_interaction_verdict([])


def test_a4_both_predictions_are_evaluated_with_fractions():
    """两条可证伪预测同时成立时的取值，且比例量与布尔量同报。

    只报布尔量会把「差一格」读成「被证伪」，故比例必须一起出现。
    """
    pts = [
        _pt(32, 1.5, 1.2, 1.8, comm_comp=4.0, bound=2.0),
        _pt(64, 1.6, 1.3, 1.9, comm_comp=2.0, bound=2.0),
        _pt(128, 1.7, 1.4, 2.0, comm_comp=1.5, bound=2.0),
        _pt(256, 1.8, 1.5, 2.1, comm_comp=1.0, bound=2.0),
    ]
    v = H.a4_interaction_verdict(pts)
    # 预测 (i)：加速比随预算单调不减，且最靠近「T_comm≈T_comp」的那档也是 gap 最小的档
    assert v.monotone_speedup_in_budget is True
    assert v.monotone_fraction == 1.0
    assert v.prediction_i_holds is True
    # 预测 (ii)：与上界之差随预算增大而不增
    assert v.prediction_ii_holds is True
    assert v.prediction_ii_monotone_fraction == 1.0
    assert v.gap_to_bound_at_min_budget == pytest.approx(0.5)
    assert v.gap_to_bound_at_max_budget == pytest.approx(0.2)


def test_a4_prediction_ii_reports_partial_agreement_not_just_boolean():
    """末档反弹一档时：布尔量为 False，但比例量必须显示"大部分成立"。"""
    pts = [
        _pt(32, 1.5, 1.2, 1.8, bound=2.0),
        _pt(64, 1.6, 1.3, 1.9, bound=2.0),
        _pt(128, 1.7, 1.4, 2.0, bound=2.0),
        _pt(256, 1.55, 1.25, 1.85, bound=2.0),
    ]
    v = H.a4_interaction_verdict(pts)
    assert v.prediction_ii_holds is False
    assert v.prediction_ii_monotone_fraction == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 统计底座（report.bootstrap_ratio_ci）
# ---------------------------------------------------------------------------

def test_ratio_ci_is_deterministic_and_ordered():
    s = [10.0, 10.1, 9.9, 10.05, 9.95]
    c = [8.0, 8.1, 7.9, 8.05, 7.95]
    a = R.bootstrap_ratio_ci(s, c, seed=1)
    b = R.bootstrap_ratio_ci(s, c, seed=1)
    assert a.to_dict() == b.to_dict()
    assert a.ci_low <= a.point <= a.ci_high
    assert a.method == "paired_percentile_bootstrap"
    assert a.point == pytest.approx(sum(s) / len(s) / (sum(c) / len(c)))


def test_ratio_ci_refuses_nan_instead_of_returning_a_nan_interval():
    """NaN 必须在源头炸掉，而不是流进 A4 的区间重叠判定。

    若这里返回 NaN 区间，`_ci_overlap` 会把 NaN 判成"不重叠"，
    于是一次统计失败会被读成"有交互、必须报二维格" —— 一个更弱的结论
    伪装成更强的结论。宁可抛错。
    """
    with pytest.raises(ValueError):
        R.bootstrap_ratio_ci([1.0, float("nan")], [1.0, 1.0])
    with pytest.raises(ValueError):
        R.bootstrap_ratio_ci([1.0, 1.0], [0.0, 1.0])


def test_ratio_ci_paired_requires_equal_length():
    with pytest.raises(ValueError):
        R.bootstrap_ratio_ci([1.0, 2.0], [1.0], paired=True)
    # 非配对时允许不等长（两组来自不同 run）
    out = R.bootstrap_ratio_ci([1.0, 2.0, 3.0], [1.0, 1.0], paired=False)
    assert out.method == "independent_percentile_bootstrap"
    assert out.n_numerator == 3 and out.n_denominator == 2


# ---------------------------------------------------------------------------
# 接线纪律
# ---------------------------------------------------------------------------

def test_a4_is_registered_and_judged_through_the_threshold_table():
    assert "a4" in E.PARTS
    assert E.PARTS["a4"] is E.a4_interaction_grid
    assert "H.a4_interaction_verdict(" in E5_SRC, "A4 判定未走阈值表"
    assert "H.A4_CI_LEVEL" in E5_SRC, "CI 水平未取自阈值表"
    assert "H.A4_MIN_BUDGET_LEVELS" in E5_SRC, "档数下限未取自阈值表"


def test_a4_uses_the_real_operator_kernel_not_a_scale_proxy():
    """A4 的计算侧必须是算子核；用 make_comp_work 会把交互"构造"成不存在。

    make_comp_work 的查询行数由接收元素数换算 ⇒ 通信与计算同比例缩 ⇒
    压缩轴与异步轴解耦 ⇒ 无论怎么测都得到「无交互」。那不是测量结果。
    """
    assert "attention_kernel as K" in E5_SRC
    assert "K.compact_kv_attention(" in E5_SRC
    assert "K.merge_partial_attention(" in E5_SRC
    # A4 段里不得回退到规模模拟
    a4_start = E5_SRC.index("def a4_interaction_grid")
    a4_end = E5_SRC.index("def a5_async_vs_sync")
    a4_body = E5_SRC[a4_start:a4_end]
    assert "make_comp_work" not in a4_body


def test_a4_default_parts_include_the_grid():
    a = E.build_parser().parse_args([])
    assert "a4" in a.parts
    assert len(a.a4_budget_ratios) >= H.A4_MIN_BUDGET_LEVELS


def test_a4_parser_accepts_custom_budget_levels():
    a = E.build_parser().parse_args(
        ["--parts", "a4", "--a4-budget-ratios", "0.01", "0.02", "0.05", "0.10"]
    )
    assert a.a4_budget_ratios == [0.01, 0.02, 0.05, 0.10]


def test_upper_median_is_the_upper_one():
    """本仓库口径是 `sorted(v)[n // 2]`（上中位数），不是 statistics.median。

    偶样本时两者不同 —— 混用会让同表数字差 0.0036 量级（曾实测 0.4354 对 0.4318）。
    """
    assert E._upper_median([1.0, 2.0, 3.0, 4.0]) == 3.0
    assert E._upper_median([1.0, 2.0, 3.0]) == 2.0
    assert E._upper_median([]) != E._upper_median([])  # nan


# ---------------------------------------------------------------------------
# 测量正确性
# ---------------------------------------------------------------------------

def test_a4_comp_work_matches_the_bare_kernel():
    """搬运不许走样：_a4_comp_work 的输出必须与裸算子核逐位相同。

    它只是"按 world 切边 → 逐块算 → 归并"的包装。包装里任何一处
    （切分偏移、β 列位置、selected_indices 的类型）搞错，都会被这条断言抓住。
    """
    from src.dcc_kv_ref import CompactKV, attention_kernel as K

    q = torch.randn(4, 8, dtype=torch.float64)
    seg = torch.randn(5, 8 + 1 + 8, dtype=torch.float64)
    work = E._a4_comp_work(q, world=2, d_h=8, d_v=8)
    got = work(torch.cat([seg, seg], dim=0))

    partials = [
        K.compact_kv_attention(q, CompactKV(
            keys=seg[:, :8], logit_bias=seg[:, 8], values=seg[:, 9:],
            selected_indices=torch.arange(5, dtype=torch.long)), return_lse=True)
        for _ in range(2)
    ]
    want = K.merge_partial_attention(partials)
    assert torch.equal(got, want)


def test_a4_comp_work_refuses_to_skip_compute_silently():
    """接收行数不足以均分到各源边时必须报错。

    静默跳过会让 T_comp 趋零 ⇒ 伪造出一个巨大的加速比 ⇒ A4 判成"有交互"。
    """
    work = E._a4_comp_work(torch.randn(2, 8), world=8, d_h=8, d_v=8)
    with pytest.raises(ValueError):
        work(torch.randn(4, 8 + 1 + 8))


# ---------------------------------------------------------------------------
# 端到端：stub 顶掉集合通信后真跑一遍
# ---------------------------------------------------------------------------

class _FakeHandle:
    def wait(self) -> None:
        return None


def _stub_all_to_all(send: torch.Tensor, send_sizes, recv_sizes) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    return torch.randn(int(sum(recv_sizes)), send.shape[1], generator=g,
                       dtype=send.dtype, device=send.device) * 0.1


def _stub_all_to_all_async(send: torch.Tensor, send_sizes, recv_sizes):
    return _stub_all_to_all(send, send_sizes, recv_sizes), _FakeHandle()


@pytest.fixture
def stubbed_collective(monkeypatch):
    monkeypatch.setattr(_comm, "all_to_all_v", _stub_all_to_all)
    monkeypatch.setattr(_comm, "all_to_all_v_async", _stub_all_to_all_async)
    monkeypatch.setattr(_env, "local_device", lambda rank: torch.device("cpu"))
    # 计时窗口里的 device_sync 在 CPU 上是空操作；barrier 因 dist 未初始化而跳过。
    return None


def _a4_args() -> dict:
    """a4_interaction_grid 实际读取的字段（比整个 argparse namespace 小得多）。"""
    return dict(
        precision="float32", L_s=64, d_h=8, d_v=8, L_r=4,
        M=2, d_p=4, chunks=2, seed=42, warmup=1, iters=4,
        a4_budget_ratios=[0.05, 0.10, 0.20, 0.40],
    )


A4_CTX = {"build_location": {"effective": "cpu", "probe": {"ok": False}}}


def test_a4_grid_runs_and_produces_the_full_two_by_four_grid(stubbed_collective):
    res = E.a4_interaction_grid(0, world=2, a=_a4_args(), ctx=A4_CTX)

    assert res["experiment"] == "A4"
    assert res["executed"] is True
    assert res["compute_source"].endswith("attention_kernel.py (G1 operator kernel)")
    assert len(res["grid"]) == 8, "4 档预算 × {同步, 异步} = 8 格"
    assert {c["sync_mode"] for c in res["grid"]} == {"sync", "async"}
    assert len(res["points"]) == 4

    for p in res["points"]:
        assert p["speedup_ci_low"] <= p["speedup"] <= p["speedup_ci_high"]
        assert p["budget"] >= 2
        assert p["theoretical_bound"] >= 1.0

    # 判定必须与阈值表一致（脚本不得自行下结论）
    v = res["verdict"]
    assert v["a4_ci_level"] == H.A4_CI_LEVEL
    assert v["a4_min_budget_levels"] == H.A4_MIN_BUDGET_LEVELS
    assert v["a4_resolved"] is True
    assert v["a4_reporting_requirement"] in (
        "edge_results_may_be_cited_separately", "must_report_2d_grid")


def test_a4_t_comm_over_t_comp_comes_from_the_sync_arm(stubbed_collective):
    """拆解必须取自同步臂。

    异步臂的 comp_ms 会吸收尚未完成的下一块传输，是上界（见
    _comm.run_async_pipeline）。若拿它去算 T_comm/T_comp，overlap 的收益会被
    错记成"计算变慢"，理论上界随之下移 —— 而 A4 的预测 (i) 正是拿上界做参照。
    """
    res = E.a4_interaction_grid(0, world=2, a=_a4_args(), ctx=A4_CTX)
    by_budget = {}
    for cell in res["grid"]:
        by_budget.setdefault(cell["budget"], {})[cell["sync_mode"]] = cell

    for p in res["points"]:
        cell = by_budget[p["budget"]]["sync"]
        assert p["t_comm_over_t_comp"] == pytest.approx(
            cell["t_comm_ms"] / cell["t_comp_ms"])
        assert p["theoretical_bound"] == pytest.approx(
            1.0 + cell["t_comm_ms"] / cell["t_comp_ms"])


def test_a4_grid_refuses_an_underpowered_grid_before_measuring(stubbed_collective):
    """档数不足必须在开始测量前失败，而不是跑完再由 verdict 报 insufficient_data。

    在租来的卡上，"跑完一轮才发现档数不够"就是白烧一轮卡时。
    """
    a = _a4_args()
    a["a4_budget_ratios"] = [0.05, 0.10, 0.20]
    with pytest.raises(ValueError, match="至少"):
        E.a4_interaction_grid(0, world=2, a=a, ctx=A4_CTX)


def test_a4_rows_carry_what_a_report_needs(stubbed_collective):
    res = E.a4_interaction_grid(0, world=2, a=_a4_args(), ctx=A4_CTX)
    assert res["rows"] and len(res["rows"]) == 4
    required = {
        "budget_ratio", "budget", "world_size", "p50_sync_ms", "p50_async_ms",
        "speedup", "speedup_ci_low", "speedup_ci_high", "speedup_ci_level",
        "ci_method", "t_comm_over_t_comp", "theoretical_bound", "t_build_ms",
        "t_build_share", "n_pairs", "chunks", "comp_source",
    }
    for row in res["rows"]:
        assert required <= set(row)
        assert row["ci_method"] == "paired_percentile_bootstrap"
        assert row["n_pairs"] == 4
