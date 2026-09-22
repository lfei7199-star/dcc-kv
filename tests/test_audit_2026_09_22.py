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
"""
from __future__ import annotations

import pathlib
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

def _synthetic_rows() -> list:
    """够 print_asymptote 跑完的最小 rows：全规格各 1 格。

    尾档（λ=10/30/100/1000）刻意让 |Δ| = 2 · β_std（四档比值都定死为 2.0、
    β_std 严格递减），以便同时断言 c_max 的口径没被改动。
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
    # 尾档：β_std 递减，|Δ| = 2 · β_std
    tail = [f"box_lam{E11._lam_tag(l)}" for l in E11.LAMBDA_LADDER[-4:]]
    for i, spec in enumerate(tail):
        r = next(x for x in rows if x["spec"] == spec)
        bstd = 0.8 / (2 ** i)          # 0.8, 0.4, 0.2, 0.1 —— 严格递减
        r["beta_std"] = bstd
        for f in fields:
            r[f] = 2.0 * bstd          # |Δ| / β_std 恒为 2.0
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
