"""H1–H5 阈值表的锚点：把「口径」冻结成可执行的断言。

这个文件的重点不是覆盖率，而是**防退化**：

- 边界语义（严格大于 vs 闭区间）一旦被改动，必须有人看见；
- H2 的**双条件**是 2026-09-15 明确裁决的口径，不能悄悄退回单条件；
- 阈值必须有唯一的事实源，实验脚本里不得再出现魔法数字；
- 「判据未接」这个缺口要被记录，而不是随时间消失。
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import hypotheses as H  # noqa: E402

E5 = REPO_ROOT / "experiments" / "gpu" / "e5_gpu_ablation.py"
E3 = REPO_ROOT / "experiments" / "cpu" / "e3_edge_conditioning.py"
RELEASE_CHECKLIST = REPO_ROOT / "docs" / "release_checklist.md"


# ---------------------------------------------------------------------------
# 注册表结构
# ---------------------------------------------------------------------------
def test_registry_covers_h1_to_h5():
    assert set(H.HYPOTHESES) == {"H1", "H2", "H3", "H4", "H5"}


def test_every_spec_is_well_formed():
    for hid, spec in H.HYPOTHESES.items():
        assert spec.hid == hid, f"{hid} 的 hid 字段不一致"
        assert spec.statement, f"{hid} 缺 statement"
        assert spec.thresholds, f"{hid} 缺 thresholds"
        assert spec.boundary in ("closed", "strict"), f"{hid} 的 boundary 取值非法"
        assert spec.code_status in ("judged", "no-judge"), f"{hid} 的 code_status 取值非法"
        assert callable(getattr(H, spec.judge, None)), f"{hid} 的 judge {spec.judge} 不可解析"


def test_boundary_string_matches_judge_behaviour():
    """boundary 字段是给人和文档看的，必须与函数的实际行为一致。

    H1 原文写 ">"，故 strict；其余原文写 "≥"，故 closed。
    这里用**恰好等于阈值**的输入来区分两者。
    """
    strict_at_threshold = H.h1_pass(H.H1_MIN_KL)
    assert strict_at_threshold is False

    closed_cases = [
        (H.h4_pass, H.H4_MIN_P50_SPEEDUP),
        (H.h5_pass, H.H5_MIN_SPEEDUP_AT_2X),
    ]
    for fn, th in closed_cases:
        assert fn(th) is True, f"{fn.__name__} 在阈值处应为闭区间"

    assert H.HYPOTHESES["H1"].boundary == "strict"
    for hid in ("H2", "H3", "H4", "H5"):
        assert H.HYPOTHESES[hid].boundary == "closed"


def test_registry_thresholds_are_the_module_constants():
    """表里的数字必须就是模块常量，防止两处各写一份。"""
    assert H.HYPOTHESES["H1"].thresholds["min_kl_divergence"] == H.H1_MIN_KL
    assert H.HYPOTHESES["H2"].thresholds["min_quality_gain_pp"] == H.H2_MIN_QUALITY_GAIN_PP
    assert H.HYPOTHESES["H2"].thresholds["min_prefill_speedup"] == H.H2_MIN_PREFILL_SPEEDUP
    assert H.HYPOTHESES["H3"].thresholds["min_drop_beta_pp"] == H.H3_MIN_DROP_BETA_PP
    assert H.HYPOTHESES["H3"].thresholds["min_drop_value_pp"] == H.H3_MIN_DROP_VALUE_PP
    assert H.HYPOTHESES["H4"].thresholds["min_p50_speedup"] == H.H4_MIN_P50_SPEEDUP
    assert H.HYPOTHESES["H5"].thresholds["min_speedup_at_2x"] == H.H5_MIN_SPEEDUP_AT_2X


# ---------------------------------------------------------------------------
# 边界语义
# ---------------------------------------------------------------------------
def test_h1_boundary_is_strict_not_closed():
    assert H.h1_pass(0.5) is False, "H1 原文是 '>0.5'，0.5 本身不算达标"
    assert H.h1_pass(0.5000001) is True


def test_h4_boundary_is_closed():
    assert H.h4_pass(1.05) is True
    assert H.h4_pass(1.049999) is False


def test_h5_boundary_is_closed():
    assert H.h5_pass(1.5) is True
    assert H.h5_pass(1.49) is False


def test_h3_requires_both_components():
    assert H.h3_pass(0.5, 1.0) is True
    assert H.h3_pass(0.49, 1.0) is False, "β 侧未达 0.5 点不应通过"
    assert H.h3_pass(0.5, 0.99) is False, "V 回归侧未达 1.0 点不应通过"


# ---------------------------------------------------------------------------
# H2：双条件口径（2026-09-15 裁决）
# ---------------------------------------------------------------------------
def test_h2_requires_both_sides():
    """只满足一侧**不算**达标 —— 这是本轮明确的口径，不得退回单条件。"""
    assert H.h2_pass(1.5, 1.0999, quality_comparable=True) is False  # 仅质量侧达
    assert H.h2_pass(1.4999, 1.10, quality_comparable=True) is False  # 仅性能侧达
    assert H.h2_pass(1.5, 1.10, quality_comparable=True) is True  # 两侧都达
    assert H.h2_pass(9.9, 9.9, quality_comparable=True) is True


def test_h2_respects_the_quality_comparable_premise():
    """即便两个数值都超标，前提不成立也不算达标。"""
    assert H.h2_pass(9.9, 9.9, quality_comparable=False) is False


def test_h2_quality_premise_must_be_explicit():
    """「质量相近」的判定主体未定义，故不设默认值：少传就报 TypeError。

    给默认值会让这个缺口悄悄消失（默认 True 等价于假装前提永远成立）。
    """
    with pytest.raises(TypeError):
        H.h2_pass(2.0, 1.5)  # type: ignore[call-arg]


def test_h2_registry_records_both_sides():
    spec = H.HYPOTHESES["H2"]
    assert set(spec.thresholds) == {"min_quality_gain_pp", "min_prefill_speedup"}
    assert spec.source == "docs/release_checklist.md §4（2026-09-15 定稿为逻辑与）"


# ---------------------------------------------------------------------------
# H2 前提「质量相近」的判定程序（2026-09-16 定稿）
# ---------------------------------------------------------------------------
def test_quality_comparable_reference_is_dense_not_shared_baseline():
    """参照物必须是精确注意力；取共享压缩基线与「>= 1.5 pp」自相矛盾。"""
    assert H.QUALITY_COMPARABLE_REFERENCE == "dense"
    assert H.QUALITY_COMPARABLE_MAX_DELTA_PP == H.H2_MIN_QUALITY_GAIN_PP


def test_quality_comparable_rejects_delta_below_noise_floor():
    """delta < 噪声底线 ⇒ 拒绝执行（这不是「不相近」，是「测不出来」）。"""
    with pytest.raises(ValueError):
        H.quality_comparable_non_inferior(-0.1, delta_pp=0.5, noise_floor_pp=0.8)


def test_quality_comparable_rejects_delta_at_or_above_effect_size():
    """非劣边界不得 >= 1.5 pp，否则前提吞没 H2 前半句所声称的效应。"""
    with pytest.raises(ValueError):
        H.quality_comparable_non_inferior(-0.1, delta_pp=1.5, noise_floor_pp=0.2)
    with pytest.raises(ValueError):
        H.quality_comparable_non_inferior(-0.1, delta_pp=2.0, noise_floor_pp=0.2)


def test_quality_comparable_enforces_the_data_dependent_upper_bound():
    """给了 delta_bad 时，delta 必须小于它 —— 否则共享压缩本身也算「相近」。"""
    with pytest.raises(ValueError):
        H.quality_comparable_non_inferior(
            -0.1, delta_pp=0.9, noise_floor_pp=0.2, delta_bad_pp=0.5
        )
    assert (
        H.quality_comparable_non_inferior(
            -0.1, delta_pp=0.4, noise_floor_pp=0.2, delta_bad_pp=0.5
        )
        is True
    )


def test_quality_comparable_is_one_sided_strict():
    """单侧非劣：只卡下界，且下界恰等于 -delta 时不算通过。"""
    assert (
        H.quality_comparable_non_inferior(-0.4, delta_pp=0.5, noise_floor_pp=0.1) is True
    )
    assert (
        H.quality_comparable_non_inferior(-0.5, delta_pp=0.5, noise_floor_pp=0.1) is False
    )
    # 「显著更优」也通过非劣（但调用方必须披露方向）
    assert (
        H.quality_comparable_non_inferior(0.3, delta_pp=0.5, noise_floor_pp=0.1) is True
    )


def test_quality_comparable_delta_must_be_positive():
    with pytest.raises(ValueError):
        H.quality_comparable_non_inferior(-0.1, delta_pp=0.0, noise_floor_pp=0.0)


def test_quality_comparable_returns_python_bool():
    assert isinstance(
        H.quality_comparable_non_inferior(-0.1, delta_pp=0.5, noise_floor_pp=0.1), bool
    )


# ---------------------------------------------------------------------------
# 返回值类型（要进 JSON，必须是原生 bool）
# ---------------------------------------------------------------------------
def test_judges_return_python_bool():
    assert isinstance(H.h1_pass(1.0), bool)
    assert isinstance(H.h2_pass(2.0, 2.0, quality_comparable=True), bool)
    assert isinstance(H.h3_pass(1.0, 1.0), bool)
    assert isinstance(H.h4_pass(2.0), bool)
    assert isinstance(H.h5_pass(2.0), bool)


# ---------------------------------------------------------------------------
# 落地状态：缺口必须被记录
# ---------------------------------------------------------------------------
def test_h4_judgement_is_wired_into_e5():
    assert H.HYPOTHESES["H4"].code_status == "judged"


def test_h1_judgement_is_wired_into_e3():
    """H1 的判据由 E3 的 h1_criterion_met 产出（2026-09-15 更正）。"""
    assert H.HYPOTHESES["H1"].code_status == "judged"


def test_e3_does_not_hardcode_the_h1_threshold():
    """E3 曾写裸的 `ci_95_lower > 0.5`，绕开阈值表 —— 由独立监督查出。"""
    src = E3.read_text(encoding="utf-8")
    assert "hypotheses as H" in src, "e3 未导入阈值表"
    assert "H.h1_pass(" in src, "E3 的 H1 判据未调用 h1_pass"
    assert "ci_95_lower > 0.5" not in src, "E3 仍硬编码 H1 阈值，应改从 hypotheses 取"


def test_unjudged_gap_is_recorded():
    """H2/H3/H5 的判据尚未接线 —— 这个事实有记录，不会随时间消失。

    H1 已于 2026-09-15 登记为 judged（E3 的 h1_criterion_met），H4 一直如此。
    将来接线扩大覆盖面时，请**同时**更新此处与各 spec 的 code_status
    及 note（本测试失败即提醒）。
    """
    assert set(H.UNJUDGED) == {"H2", "H3", "H5"}
    assert H.UNJUDGED == tuple(
        h for h, s in H.HYPOTHESES.items() if s.code_status != "judged"
    )


# ---------------------------------------------------------------------------
# 源码锚点：阈值不得再以魔法数字出现
# ---------------------------------------------------------------------------
def test_e5_a5_uses_the_threshold_table_not_a_magic_number():
    src = E5.read_text(encoding="utf-8")
    assert "hypotheses as H" in src, "e5 未导入阈值表"
    assert "H.h4_pass(" in src, "A5 未调用 h4_pass"
    assert ">= 1.05" not in src, "A5 仍硬编码 H4 阈值，应改从 hypotheses 取"
    assert "H.H4_MIN_P50_SPEEDUP" in src, "A5 的报告文案未从表中取阈值"


def test_release_checklist_states_h2_as_two_sides():
    """文档侧的 H2 也必须是双条件，不能只有一侧。"""
    doc = RELEASE_CHECKLIST.read_text(encoding="utf-8")
    assert "1.5 pp" in doc
    assert "1.10×" in doc
