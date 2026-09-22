"""2026-09-22 工程有效性审查（F1–F7）发现的「静默丢值」类缺陷的回归锚点。

体例与 `test_selfcheck_2026_09_18.py` / `test_adversarial_2026_09_20.py` 一致：
每条对应一个**已修**的缺陷，且都先说明「它本可以不被发现」。

  F5  `e4_dist_equivalence.run_existing_tests()` 之前只记
      ``{"ran": true, "returncode": 1, "summary": ""}`` —— 转发进来的
      pytest 失败时没有任何可复算的证据（跑的是什么、判据是什么、错在哪），
      于是「E4 通过」这句可以在无支撑的情况下被写出来。

  新  `e11_lambda_tuning.print_asymptote()` 只 **打印** 不 **返回**，
      调用方却把它的返回值直接写进 ``summary["asymptote_check"]``
      ⇒ 落盘恒为 ``null``，而论文 §6 引用的 "max|Δ| ≤ 1.61·β_std"
      因此没有任何落盘支撑。**它和 F5 是同一类**：
      检查看起来在跑，实际什么都没留下。

  F7  `e9_knob_localization.conditioning()` 把**整个拟合池**当探针，
      于是 ``rank(X) ≤ min(拟合池, B)`` —— 与 ``M`` 无关；而同格落盘的
      ``rank_upper_bound = min(M, B)`` ⇒ 两者不同源，表格里
      ``rank/min(M,B)`` 的 ``*`` 标记失效（525 格中 255 格 rank 越界）。
      正解是把探针换成 V 回归真正的设计矩阵：**M 行**代表 Query。
      另锁 E9 的**脚本默认值**：它原先停在旧区间（``L_s=256``、``B<=128``），
      默认跑出来的落盘与论文声明的 ``B ≪ L_s`` 量级相反。
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cpu import e11_lambda_tuning as E11  # noqa: E402
from experiments.cpu import e4_dist_equivalence as E4  # noqa: E402


# ---------------------------------------------------------------------------
# 新缺陷：print_asymptote 必须返回落盘用 dict
# ---------------------------------------------------------------------------

def _synthetic_rows(ratio: float = 2.0) -> list:
    """够 print_asymptote 跑完的最小 rows：全规格各 1 格。

    尾档（λ=10/30/100/1000）刻意让 |Δ| = ratio · β_std（四档比值都定死为
    ``ratio``、β_std 严格递减），以便同时断言 c_max 的口径没被改动。
    """
    fields = ("eval_out_err_median", "eval_mixture_median",
              "x_eff_budget_frac", "v_fit_resid_rel")
    rows = []
    for spec in E11.SPEC_ORDER:
        rows.append({"spec": spec, "seed": 0, "dest": 0, "M": 8, "budget": 8,
                     "beta_std": 1.0, "beta_mean": 1.0,
                     **{f: 0.0 for f in fields}})
    base = next(r for r in rows if r["spec"] == "nobeta")
    base["beta_std"] = 0.0
    base["beta_mean"] = 0.0
    # 尾档：β_std 递减，|Δ| = ratio · β_std
    tail = [f"box_lam{E11._lam_tag(l)}" for l in E11.LAMBDA_LADDER[-4:]]
    for i, spec in enumerate(tail):
        r = next(x for x in rows if x["spec"] == spec)
        bstd = 0.8 / (2 ** i)          # 0.8, 0.4, 0.2, 0.1 —— 严格递减
        r["beta_std"] = bstd
        for f in fields:
            r[f] = ratio * bstd        # |Δ| / β_std 恒为 ratio
    return rows


def test_print_asymptote_returns_dict(capsys):
    """回归：本函数原先只打印、不返回 ⇒ asymptote_check 静默落盘为 null。"""
    res = E11.print_asymptote(_synthetic_rows())
    capsys.readouterr()          # print_asymptote 会大量打印，丢弃即可
    assert res is not None, (
        "print_asymptote() 必须返回 dict；返回 None 会让 "
        "summary['asymptote_check'] 静默变成 null（2026-09-22 审查的新缺陷）")
    assert isinstance(res, dict)
    for k in ("fields", "steps", "abs_max_over_beta_std", "c_max", "verdict"):
        assert k in res, f"asymptote_check 缺字段 {k}"


def test_print_asymptote_c_max_is_max_over_beta_std(capsys):
    """口径锚点：c_max = max_spec( max_field max_cell|Δ| / β_std(spec) )。

    合成数据把比值定死为 2.0 —— 若有人把口径改成「逐格 β_std」或
    「只取某一字段」，这条会立刻失败。
    """
    res = E11.print_asymptote(_synthetic_rows())
    capsys.readouterr()
    assert res["c_max"] == pytest.approx(2.0, rel=1e-9)
    assert all(r == pytest.approx(2.0, rel=1e-9)
               for r in res["abs_max_over_beta_std"])
    assert len(res["steps"]) == 4


def test_asymptote_verdict_bound_never_understates_c_max(capsys):
    """上界锚点：verdict 里陈述的 "≤ x.xx·β_std" 必须 ≥ 实测 c_max。

    回归一类「上界向下取整」缺陷（2026-09-22，F2 重跑的衍生）：verdict 原先用
    ``f"{c_max:.2f}"``（四舍五入）拼出上界。c_max = 1.608 时凑巧成立
    （1.61 ≥ 1.608），但 F2 把 FPS 口径换到 newest 之后 c_max = 1.663192，
    四舍五入得 1.66 < 1.663192 —— 「不超过 1.66 倍」于是成了字面为假的声明。
    上界必须**向上**取整。

    合成数据取比值 2.004：四舍五入得 2.00（< 2.004，为假）；向上取整得 2.01。
    """
    ratio = 2.004
    res = E11.print_asymptote(_synthetic_rows(ratio))
    capsys.readouterr()
    assert res["c_max"] == pytest.approx(ratio, rel=1e-9)
    m = re.search(r"≤\s*([0-9]+\.[0-9]+)\s*·\s*β_std", res["verdict"])
    assert m, f"verdict 未按「≤ x.xx·β_std」陈述：{res['verdict']!r}"
    bound = float(m.group(1))
    assert bound >= res["c_max"], (
        f"verdict 上界 {bound} 低于实测 c_max {res['c_max']} —— "
        "上界必须向上取整，否则「不超过」声明为假")


def test_summary_payload_keeps_asymptote_check_non_null():
    """源码级：调用方必须防 regression（返回值 None 时抛错而不是照写）。"""
    src = (REPO_ROOT / "experiments" / "cpu" / "e11_lambda_tuning.py").read_text(
        encoding="utf-8")
    assert "asymptote is None" in src, (
        "调用方缺少 asymptote 返回值守卫 —— 静默 null 会再次发生")


# ---------------------------------------------------------------------------
# F5：转发 pytest 失败必须自带可复算证据
# ---------------------------------------------------------------------------

def test_e4_failed_forwarded_test_is_self_explaining(tmp_path):
    """回归：失败记录必须带 argv / 判据 / stderr 尾巴，否则无法复算。"""
    # 人造一个必然失败的 pytest 目标
    failing = tmp_path / "test_always_fails.py"
    failing.write_text("def test_x():\n    assert False, 'boom'\n",
                       encoding="utf-8")
    payload = E4.run_existing_tests(targets=[str(failing)], cwd=str(tmp_path))
    if payload.get("ran") and payload.get("returncode") == 0:
        pytest.skip("本机 pytest 环境异常，未复现失败路径")
    assert payload.get("returncode") != 0
    for k in ("targets", "command", "pass_criterion", "stderr_tail", "note"):
        assert k in payload, f"失败记录缺 {k} ⇒ 无法复算（F5）"
    assert payload["pass_criterion"]
    assert payload["command"]


# ---------------------------------------------------------------------------
# F7（理论↔实现一致性审计）：E9 秩诊断必须用 V 回归真正的设计矩阵（M 行）
# ---------------------------------------------------------------------------

def test_e9_conditioning_rank_bounded_by_min_m_b():
    """回归：`conditioning()` 的探针必须只有 M 行，rank 才由 M 封顶。

    设计意图：`X` 是 V 回归的设计矩阵 —— 行 = 代表 Query 数 `M`、列 = 键数 `B`，
    故 `rank(X) ≤ min(M, B)`。旧实现把 `probe` 传成 `split.fit_queries[dest]`
    （整个拟合池，默认 1024 行）⇒ `rank(X) ≤ min(拟合池, B)`，**与 M 无关**；
    实测 `M=4, B=8` 报 `rank=8 > 上界 4`，525 格中 **255 格**越界，
    与同格落盘的 `rank_upper_bound=min(M,B)` 不同源 ⇒ 该诊断列不可用。

    这里用最简的假 compact 直接测函数（不构造 split）：`B=16`、`M∈{2,4,8}`，
    rank 必须被 M 封顶；若有人把探针换回拟合池，M=2 会报 16（> 2）而立刻失败。
    """
    import types

    torch = pytest.importorskip("torch")
    from experiments.cpu import e9_knob_localization as E9

    B, d = 16, 8
    torch.manual_seed(0)
    compact = types.SimpleNamespace(
        keys=torch.randn(B, d, dtype=torch.float64),
        logit_bias=torch.zeros(B, dtype=torch.float64),
    )
    for M in (2, 4, 8):
        rep = torch.randn(M, d, dtype=torch.float64)
        _cond, rank = E9.conditioning(compact, rep)
        assert rank <= min(M, B), (
            f"M={M}, B={B} 时 rank={rank} > min(M,B)={min(M, B)} —— "
            "探针必须只有 M 行（V 回归的设计矩阵），否则 rank 上界失效（F7）")


def test_e9_conditioning_probe_is_representative_queries():
    """源码级：调用点传的必须是 `rep_queries`，不能回退到整个拟合池。"""
    src = (REPO_ROOT / "experiments" / "cpu"
           / "e9_knob_localization.py").read_text(encoding="utf-8")
    assert "conditioning(compact, rep_queries)" in src, (
        "conditioning 的探针不是 M 个代表 Query（F7 回归）")
    assert "conditioning(compact, split.fit_queries" not in src, (
        "conditioning 又拿整个拟合池当探针 ⇒ rank 上界与 min(M,B) 不同源（F7）")


def test_e9_defaults_equal_declared_interval():
    """回归：E9 的**脚本默认值**必须等于论文 §5 声明的区间。

    2026-09-22 之前默认停在旧区间（`L_s=256`、`B<=128`、`M<=48`、
    fit 池 128），默认跑出来的落盘 `B/L_s` 最大 `0.50`，与论文声明的
    `B ≪ L_s` 量级**相反** —— 这是一个「默认产物不能作论文依据」的陷阱
    （commit_log 第 36 条遗留 3）。用 AST 读默认值，不启动 torch。
    """
    import ast

    src = (REPO_ROOT / "experiments" / "cpu"
           / "e9_knob_localization.py").read_text(encoding="utf-8")
    defaults = {}
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            flag = getattr(node.args[0], "value", None)
            for kw in node.keywords:
                if kw.arg == "default":
                    defaults[flag] = ast.literal_eval(kw.value)

    assert defaults.get("--L-s") == 2048, "E9 默认 L_s 必须等于声明区间的 2048"
    assert defaults.get("--queries-per-dest") == 2048
    assert defaults.get("--Ms") == [4, 8, 16, 32, 48, 96, 192]
    assert defaults.get("--budgets") == [8, 16, 32, 64, 100]
    # 声明区间的判据：默认网格的最大 B/L_s 必须 <= 0.05
    assert max(defaults["--budgets"]) / defaults["--L-s"] <= 0.05, (
        "E9 默认网格越出论文声明的 B ≪ L_s 量级")


# ---------------------------------------------------------------------------
# §5 P2（可选）：再入清单里的「阈值必须走 hypotheses.py」源码级守卫
# ---------------------------------------------------------------------------

def test_experiment_criteria_route_through_hypotheses():
    """守卫：`experiments/` 下的**判据**不得用裸阈值。

    `experiments/common/hypotheses.py` 是阈值的单一事实源（H1–H5 的判据函数），
    但 `e3_edge_conditioning.py` 的 `h1_criterion_met` 曾写成裸比较
    `ci_95_lower > 0.5` **绕开它** —— 审查报告 §5 P2 指出这"是真实发生过的，
    不是理论担忧"，并建议加源码级守卫。

    判据：凡名字含 `criterion` 的赋值/字典键，其取值只能来自 `H.` /
    `hypotheses.py`，不得出现裸的阈值比较（`> 0.5` 这类数字字面量）。
    """
    import re

    assign = re.compile(r"""^\s*["']?\w*criterion\w*["']?\s*[:=]""", re.I)
    bare_cmp = re.compile(r"[<>]=?\s*-?\d")
    bad = []
    for p in sorted((REPO_ROOT / "experiments").rglob("*.py")):
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not assign.match(ln):
                continue
            if "H." in ln or "hypotheses" in ln:
                continue                     # 走了单一事实源
            if bare_cmp.search(ln):
                bad.append((p.relative_to(REPO_ROOT).as_posix(), i, ln.strip()))
    assert not bad, (
        "以下判据用裸阈值绕过了 hypotheses.py（阈值必须走单一事实源）：\n"
        + "\n".join("  %s:%d  %s" % b for b in bad))
