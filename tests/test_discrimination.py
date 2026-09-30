"""评测集量具判据（`discrimination.py`）的锚点：把用户裁决冻结成可执行断言。

这个文件的重点不是覆盖率，而是**防退化**。四类断言：

1. **判据字面** —— 用户 2026-09-30 裁定的「不一致对 > 0 且（方向不对称 或
   参照臂两端有余量）」一旦被改动，必须有人看见；
2. **真实黄金值** —— NIAH x20 / LongBench n=20 的逐样本对错**抄自落盘产物**，
   判定结果必须与报告一致。抄进来而不是读文件，是因为落盘产物不在仓库内
   （`_onetime_scripts/_out/`），仓库测试不得依赖仓外文件；
3. **边界语义** —— `b=0, c=5`（p=0.0625）**不算**方向可分辨。这一条最容易
   被后来者「顺手放宽」，它正是 n=8 那轮把方向复现误当结论的那个坑；
4. **声明↔实现绑定** —— 阈值只有一处定义；文档里必须点到该模块。

`passes_discrimination_gate` 是必要条件、`claim_licensed` 才是许可证 —— 两者
**必须分开断言**，否则「过了门槛」会被读成「能下结论」。
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import discrimination as D  # noqa: E402

DOC = REPO_ROOT / "docs" / "writing_scope_and_metrics.md"

# ---------------------------------------------------------------------------
# 黄金值：抄自本机落盘产物（逐样本对错，20 位）
# ---------------------------------------------------------------------------
# _onetime_scripts/_out/niah/niah_eval_x20.json（Qwen2.5-0.5B，NIAH 4k，每档 5 条）
NIAH_DENSE = [1] * 20
NIAH_TOPKDOT25 = [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
NIAH_TOPK0DOT5 = [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
NIAH_RAND0DOT25 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
NIAH_STRIDE0DOT25 = [1] * 20

# _onetime_scripts/_out/niah/niah_eval_lbzh20.json（LongBench passage_retrieval_zh，4k）
LB_DENSE = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
LB_TOPKDOT25 = [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
LB_RAND0DOT25 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]
LB_STRIDE0DOT25 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]

# 一个在「需求分辨率」上足够宽的声明值：探针尺度（预期效应数十 pp），
# 不是论文尺度。✔ 显式写出来，不做成默认值。
PROBE_RESOLUTION_PP = 25.0


# ---------------------------------------------------------------------------
# 1. 阈值本体
# ---------------------------------------------------------------------------
def test_thresholds_are_registered_here_and_only_here():
    assert D.DISCRIMINATION_ALPHA == 0.05
    assert D.DISCRIMINATION_CI_LEVEL == 0.95
    assert D.REF_HEADROOM_MIN_ACC == 0.10


def test_grades_and_licensing_partition():
    assert set(D.GRADES) == {
        "blind",
        "directional",
        "not_detected",
        "underresolved",
        "inconclusive",
    }
    assert set(D.CLAIM_LICENSING_GRADES) == {"directional", "not_detected"}
    assert set(D.CLAIM_LICENSING_GRADES) < set(D.GRADES)


# ---------------------------------------------------------------------------
# 2. 精确 McNemar：边界与小样本
# ---------------------------------------------------------------------------
def test_mcnemar_matches_hand_computed_exact_binomial():
    assert D.mcnemar_exact_p(0, 15) == pytest.approx(2 * 0.5 ** 15)
    assert D.mcnemar_exact_p(0, 14) == pytest.approx(2 * 0.5 ** 14)
    assert D.mcnemar_exact_p(0, 5) == pytest.approx(0.0625)
    assert D.mcnemar_exact_p(0, 6) == pytest.approx(0.03125)
    assert D.mcnemar_exact_p(3, 3) == pytest.approx(1.0)
    assert D.mcnemar_exact_p(0, 1) == pytest.approx(1.0)
    assert D.mcnemar_exact_p(1, 1) == pytest.approx(1.0)


def test_mcnemar_refuses_empty_discordance():
    with pytest.raises(ValueError, match="无定义"):
        D.mcnemar_exact_p(0, 0)


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError, match="不得为负"):
        D.mcnemar_exact_p(-1, 3)


def test_exact_and_chi2_disagree_at_this_sample_size():
    """这正是「必须用精确检验」的实证：``b=0, c=5`` 双尾 0.0625。

    卡方近似（无连续性校正）会给约 0.025，把**不显著**读成显著。
    """
    p = D.mcnemar_exact_p(0, 5)
    assert p > D.DISCRIMINATION_ALPHA, "b=0,c=5 不得判为显著"


# ---------------------------------------------------------------------------
# 3. 判据分量：三条逐一锚定
# ---------------------------------------------------------------------------
def test_not_blind_component():
    same = D.discrimination_verdict(
        NIAH_DENSE, NIAH_STRIDE0DOT25, required_resolution_pp=PROBE_RESOLUTION_PP
    )
    assert same.n_discordant == 0 and same.not_blind is False
    assert same.p_value is None
    assert same.grade == "blind"
    assert same.passes_discrimination_gate is False
    assert same.claim_licensed is False


def test_directional_is_the_exact_test_not_the_sign_of_b_minus_c():
    """``b=0, c=5`` 看着方向一致，但**功效不够** ⇒ 不得当结论。"""
    ref = [1] * 5
    arm = [0] * 5
    v = D.discrimination_verdict(ref, arm, required_resolution_pp=PROBE_RESOLUTION_PP)
    assert (v.b, v.c) == (0, 5)
    assert v.p_value == pytest.approx(0.0625)
    assert v.directional is False
    assert v.ref_headroom is False  # ref_acc = 1.0，钉在端点
    assert v.grade == "inconclusive"
    assert v.passes_discrimination_gate is False
    assert v.claim_licensed is False


def test_directional_flips_at_c_equals_six():
    ref = [1] * 6
    arm = [0] * 6
    v = D.discrimination_verdict(ref, arm, required_resolution_pp=PROBE_RESOLUTION_PP)
    assert v.p_value == pytest.approx(0.03125)
    assert v.directional is True
    assert v.grade == "directional"
    assert v.passes_discrimination_gate is True
    assert v.claim_licensed is True


def test_headroom_component_both_endpoints():
    """参照臂钉在任一端 ⇒ 该端点方向被截断，判据必须拦住。"""
    high = D.discrimination_verdict(
        [1] * 4, [0, 0, 1, 1], required_resolution_pp=PROBE_RESOLUTION_PP
    )
    assert high.ref_acc == 1.0 and high.ref_headroom is False
    assert (high.b, high.c) == (0, 2)
    assert high.grade == "inconclusive"

    low = D.discrimination_verdict(
        [0] * 4, [0, 0, 1, 1], required_resolution_pp=PROBE_RESOLUTION_PP
    )
    assert low.ref_acc == 0.0 and low.ref_headroom is False
    assert (low.b, low.c) == (2, 0)
    assert low.grade == "inconclusive"

    interior = D.discrimination_verdict(
        [1, 0, 1, 0], [1, 1, 1, 0], required_resolution_pp=PROBE_RESOLUTION_PP
    )
    assert interior.ref_acc == 0.5 and interior.ref_headroom is True


def test_resolution_gate_separates_not_detected_from_underresolved():
    """同一个 ``b=3, c=3``：所需分辨率一变，许可证就变。"""
    strict = D.discrimination_verdict(
        LB_DENSE, LB_TOPKDOT25, required_resolution_pp=10.0
    )
    loose = D.discrimination_verdict(
        LB_DENSE, LB_TOPKDOT25, required_resolution_pp=PROBE_RESOLUTION_PP
    )
    for v in (strict, loose):
        assert (v.b, v.c) == (3, 3)
        assert v.p_value == pytest.approx(1.0)
        assert v.directional is False
        assert v.ref_headroom is True
        assert v.passes_discrimination_gate is True  # 判据（必要条件）都通过
    assert strict.resolution_halfwidth_pp > 10.0
    assert strict.resolution_ok is False
    assert strict.grade == "underresolved"
    assert strict.claim_licensed is False  # ← 过门槛 ≠ 能说话

    assert loose.resolution_ok is True
    assert loose.grade == "not_detected"
    assert loose.claim_licensed is True


def test_resolution_readout_is_not_an_inferable_interval():
    """⚠️ 反例锚点：分辨率读数**不得**被当成置信区间来推断。

    ``b=0, c=4, n=20``：读数 19.6 pp < |Δ| = 20.0 pp —— 若把它当区间，
    会得出「显著」。而**精确 McNemar p = 0.125，不显著**。

    这条断言的作用是把这个矛盾**钉住**：将来若有人把该读数改名回 `ci_*`
    或拿它做推断，至少会先看见这里写着「以 p 为准」。
    """
    ref = [1] * 20
    arm = [1] * 16 + [0] * 4
    v = D.discrimination_verdict(ref, arm, required_resolution_pp=PROBE_RESOLUTION_PP)
    assert (v.b, v.c) == (0, 4)
    assert v.p_value == pytest.approx(0.125)
    assert v.directional is False, "精确检验说了不显著，读数不能推翻它"
    assert abs(v.delta_pp) == pytest.approx(20.0)
    assert v.resolution_halfwidth_pp == pytest.approx(19.6, abs=0.05)
    assert v.resolution_halfwidth_pp < abs(v.delta_pp), (
        "读数比 |Δ| 还窄却仍不显著 —— 这正是「读数不是区间」的证据本身"
    )
    assert v.ref_headroom is False  # ref=1.0，端点带内
    assert v.grade == "inconclusive"
    assert v.claim_licensed is False


# ---------------------------------------------------------------------------
# 4. 真实黄金值：与两份复验报告逐项一致
# ---------------------------------------------------------------------------
def test_niah_x20_golden_topk_025_vs_dense():
    v = D.discrimination_verdict(
        NIAH_DENSE, NIAH_TOPKDOT25, required_resolution_pp=PROBE_RESOLUTION_PP
    )
    assert v.n_items == 20
    assert (v.b, v.c) == (0, 15)
    assert v.ref_acc == 1.0 and v.arm_acc == 0.25
    assert v.delta_pp == pytest.approx(-75.0)
    assert v.p_value == pytest.approx(2 * 0.5 ** 15)
    assert v.grade == "directional"
    assert v.claim_licensed is True
    assert v.ref_headroom is False  # ref=1.0，靠「方向不对称」这一支过关


def test_niah_x20_golden_topk_05_vs_dense():
    v = D.discrimination_verdict(
        NIAH_DENSE, NIAH_TOPK0DOT5, required_resolution_pp=PROBE_RESOLUTION_PP
    )
    assert (v.b, v.c) == (0, 11)
    assert v.arm_acc == pytest.approx(0.45)
    assert v.grade == "directional"
    assert v.claim_licensed is True


def test_niah_x20_rand_vs_dense_is_inconclusive_not_a_null():
    """⚠️ 关键锚点：``b=0`` 但只有 1 条 ⇒ **既不是显著、也不是「无差异」**。

    NIAH 上 ``rand_0.25`` 只错 1 条。若有人把「``b=0``」当成「方向已复现」，
    会得出「随机保留无损」—— 那是把 1 条样本当证据。判据必须拦住它。
    """
    v = D.discrimination_verdict(
        NIAH_DENSE, NIAH_RAND0DOT25, required_resolution_pp=PROBE_RESOLUTION_PP
    )
    assert (v.b, v.c) == (0, 1)
    assert v.p_value == pytest.approx(1.0)
    assert v.directional is False
    assert v.ref_headroom is False
    assert v.grade == "inconclusive"
    assert v.passes_discrimination_gate is False
    assert v.claim_licensed is False


def test_longbench_n20_golden_all_three_arms_are_underresolved():
    """LongBench n=20：三条臂全部**过判据但拿不到许可证**。

    这是判据「必要条件 vs 许可证」两层分离的真实现场：``passes`` 全 True，
    但与报告结论一致地 ``claim_licensed`` 全 False。
    """
    expected = {
        "topk_0.25": (LB_TOPKDOT25, (3, 3)),
        "rand_0.25": (LB_RAND0DOT25, (1, 2)),
        "stride_0.25": (LB_STRIDE0DOT25, (1, 1)),
    }
    assert len(expected) == 3
    for name, (arm_ps, bc) in expected.items():
        v = D.discrimination_verdict(
            LB_DENSE, arm_ps, required_resolution_pp=10.0
        )
        assert (v.b, v.c) == bc, name
        assert v.ref_acc == pytest.approx(0.30)
        assert v.ref_headroom is True  # 0.10 <= 0.30 <= 0.90
        assert v.claim_licensed is False, name


def test_longbench_ref_baseline_is_low_which_is_why_headroom_matters():
    """同一个量具，参照臂读数决定了它有没有余量。"""
    assert sum(LB_DENSE) / len(LB_DENSE) == pytest.approx(0.30)
    assert sum(NIAH_DENSE) / len(NIAH_DENSE) == pytest.approx(1.00)


# ---------------------------------------------------------------------------
# 5. 输入校验：不合格输入必须报错，不得静默降级
# ---------------------------------------------------------------------------
def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="长度不等"):
        D.discrimination_verdict([1, 0], [1, 0, 1], required_resolution_pp=10.0)


def test_rejects_non_binary_values():
    with pytest.raises(ValueError, match="不是 0/1"):
        D.discrimination_verdict([1, None], [1, 0], required_resolution_pp=10.0)
    with pytest.raises(ValueError, match="不是 0/1"):
        D.discrimination_verdict([1, 2], [1, 0], required_resolution_pp=10.0)


def test_rejects_empty_samples():
    with pytest.raises(ValueError, match="为空"):
        D.discrimination_verdict([], [], required_resolution_pp=10.0)


def test_required_resolution_has_no_default_and_must_be_positive():
    with pytest.raises(TypeError):
        D.discrimination_verdict([1, 0], [0, 1])  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="必须为正"):
        D.discrimination_verdict([1, 0], [0, 1], required_resolution_pp=0.0)


def test_rejects_impossible_alpha_and_headroom():
    with pytest.raises(ValueError, match="alpha"):
        D.discrimination_verdict([1, 0], [0, 1], required_resolution_pp=10.0, alpha=1.5)
    with pytest.raises(ValueError, match="headroom_min_acc"):
        D.discrimination_verdict(
            [1, 0], [0, 1], required_resolution_pp=10.0, headroom_min_acc=0.5
        )


# ---------------------------------------------------------------------------
# 6. 批量入口
# ---------------------------------------------------------------------------
def test_verdict_from_arms_skips_reference_and_covers_the_rest():
    arms = {
        "dense": {"per_sample": NIAH_DENSE},
        "topk_0.25": {"per_sample": NIAH_TOPKDOT25},
        "stride_0.25": {"per_sample": NIAH_STRIDE0DOT25},
    }
    out = D.discrimination_verdict_from_arms(
        arms, reference="dense", required_resolution_pp=PROBE_RESOLUTION_PP
    )
    assert set(out) == {"topk_0.25", "stride_0.25"}
    assert out["topk_0.25"].grade == "directional"
    assert out["stride_0.25"].grade == "blind"


def test_verdict_from_arms_refuses_missing_per_sample():
    arms = {"dense": {"per_sample": NIAH_DENSE}, "arm": {"accuracy": 0.5}}
    with pytest.raises(ValueError, match="per_sample"):
        D.discrimination_verdict_from_arms(
            arms, reference="dense", required_resolution_pp=10.0
        )


def test_verdict_from_arms_refuses_missing_reference():
    with pytest.raises(KeyError, match="参照臂"):
        D.discrimination_verdict_from_arms(
            {"a": {"per_sample": NIAH_DENSE}},
            reference="dense",
            required_resolution_pp=10.0,
        )


def test_verdict_is_json_serialisable():
    v = D.discrimination_verdict(
        NIAH_DENSE, NIAH_TOPKDOT25, required_resolution_pp=PROBE_RESOLUTION_PP
    )
    blob = json.dumps(v.as_dict(), ensure_ascii=False)
    assert "passes_discrimination_gate" in blob
    assert "claim_licensed" in blob


# ---------------------------------------------------------------------------
# 7. 声明 ↔ 实现绑定
# ---------------------------------------------------------------------------
_PRUNE_DIRS = {"venv", "__pycache__", ".git", ".pytest_cache", "node_modules"}

_SINGLE_SOURCE = "experiments/common/discrimination.py"


def _repo_python_files():
    """仓库内全部 `.py`（剪枝 venv/缓存，避免遍历 1 GB 的 site-packages）。"""
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield pathlib.Path(dirpath) / fn


def _parsed_repo_modules():
    for path in _repo_python_files():
        try:
            yield path, ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue


def test_exactly_one_definition_of_the_criterion_in_the_repo():
    """阈值与检验只有一处实现，脚本里不得再手搓 McNemar。"""
    hits = []
    for path, tree in _parsed_repo_modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "mcnemar_exact_p":
                hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == [_SINGLE_SOURCE], hits


def test_threshold_constants_are_not_assigned_outside_the_single_source():
    """按**结构**判定，不按字符串判定。

    用 AST 找「赋值给这三个名字」的语句，而不是 grep 字符串 ——
    grep 会被 `D.DISCRIMINATION_ALPHA` 这种**正当引用**误伤（本测试文件自己就有）。
    """
    watched = {"DISCRIMINATION_ALPHA", "DISCRIMINATION_CI_LEVEL", "REF_HEADROOM_MIN_ACC"}
    offenders = []
    for path, tree in _parsed_repo_modules():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel == _SINGLE_SOURCE:
            continue
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in watched:
                    offenders.append(f"{rel}:{t.lineno} {t.id}")
    assert offenders == [], f"阈值被第二处赋值：{offenders}"


def test_writing_scope_document_points_at_this_module():
    """判据的文档出处与实现必须互指，否则修订会静默漂移。"""
    text = DOC.read_text(encoding="utf-8")
    assert "experiments/common/discrimination.py" in text
    assert "不一致对" in text
