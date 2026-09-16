"""E0 的「置换次数」轴：把待补项冻成可执行的断言。

背景：论文 §6 的 E0 小节此前留了一项待补——把「置换数量」与「归并树形状」
作为自变量报告最大相对误差曲线。树形状 E0 本来就有，置换次数一直是固定值、
只报一个 max。本轮给 E0 补上了**收敛阶梯**（`--perm-ladder`）。

这个文件守四件事：

1. 阶梯必须真的覆盖声明的那些 n，且是**累计**最大值（非累计会被抓出来）；
2. 阶梯必须随 n 单调不减——单调性破了说明"累计"写错了；
3. 固定种子必须可复现（结果落盘要能重放）；
4. 派生判据量（`summary`）的键名与取值范围不能悄悄改。

第 5 类断言是**源码级**的：这条轴一旦被整个删掉，测试必须失败。
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cpu import e0_order_invariance as E0  # noqa: E402

E0_SRC_PATH = REPO_ROOT / "experiments" / "cpu" / "e0_order_invariance.py"


@pytest.fixture(scope="module")
def fixture_blocks():
    """小规模固定装置：K=8、每块 16 token、d_v=8、FP32，够快也够有区分度。"""
    logits, values, states = E0.make_blocks(
        num_blocks=8, d_h=8, d_v=8, tokens_per_block=16,
        dtype=torch.float32, seed=7,
    )
    return states, E0.reference_output(logits, values)


def _direct_max_err(states, ref, n, seed):
    """独立实现：同一置换序列下 n 次试验各自误差的最大值（非累计）。"""
    g = torch.Generator().manual_seed(seed)
    errs = []
    for _ in range(n):
        perm = torch.randperm(len(states), generator=g).tolist()
        permuted = [states[i] for i in perm]
        errs.append(
            E0.relative_error(E0.attention_output(E0.merge_left_deep(permuted)), ref)
        )
    return max(errs)


# ---------------------------------------------------------------------------
# 阶梯本身
# ---------------------------------------------------------------------------
def test_ladder_covers_declared_points_in_ascending_order(fixture_blocks):
    states, ref = fixture_blocks
    ladder = [1, 2, 5, 10]
    got = E0.perm_ladder(states, ref, ladder, seed=42)
    assert [r["num_permutations"] for r in got] == sorted(ladder)


def test_ladder_accepts_unsorted_input_and_still_ascends(fixture_blocks):
    """声明的 n 不要求有序，输出必须有序且累计正确。"""
    states, ref = fixture_blocks
    got = E0.perm_ladder(states, ref, [10, 1, 5], seed=42)
    assert [r["num_permutations"] for r in got] == [1, 5, 10]


def test_ladder_is_cumulative_monotone(fixture_blocks):
    states, ref = fixture_blocks
    got = E0.perm_ladder(states, ref, [1, 2, 5, 10, 25, 50], seed=42)
    vals = [r["cumulative_max_rel_err"] for r in got]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals


def test_ladder_endpoint_equals_direct_max(fixture_blocks):
    """端点必须等于"同一置换序列前 n 次的最大值"——这是"累计"的判据。

    若实现把累计误写成"只看第 n 次"，端点会显著小于直接最大值。
    """
    states, ref = fixture_blocks
    n = 25
    got = E0.perm_ladder(states, ref, [1, n], seed=123)
    assert got[-1]["cumulative_max_rel_err"] == _direct_max_err(states, ref, n, 123)


def test_ladder_is_reproducible_for_fixed_seed(fixture_blocks):
    states, ref = fixture_blocks
    a = E0.perm_ladder(states, ref, [1, 5, 25], seed=2026)
    b = E0.perm_ladder(states, ref, [1, 5, 25], seed=2026)
    assert a == b


def test_ladder_rejects_empty_input(fixture_blocks):
    """空阶梯没有意义，应当抛错而不是静默返回空表。"""
    states, ref = fixture_blocks
    with pytest.raises(ValueError):
        E0.perm_ladder(states, ref, [], seed=42)


# ---------------------------------------------------------------------------
# 派生判据量
# ---------------------------------------------------------------------------
def test_summary_shape_and_ranges():
    rows = [
        {"dtype": "float32", "num_blocks": 8, "balanced_better_than_seq": True},
        {"dtype": "float32", "num_blocks": 16, "balanced_better_than_seq": False},
        {"dtype": "float64", "num_blocks": 8, "balanced_better_than_seq": True},
    ]
    ladder = []
    for dt, ratio in (("float32", 1.05), ("float64", 1.09)):
        for n, v in ((200, 1e-7), (1000, 1e-7 * ratio)):
            ladder.append(
                {"dtype": dt, "num_blocks": 8,
                 "num_permutations": n, "cumulative_max_rel_err": v}
            )

    sm = E0.experiment_summary(rows, ladder)
    assert sm["tree_shape_total"] == 3
    assert sm["tree_shape_balanced_wins"] == 2
    assert 0 <= sm["tree_shape_balanced_wins"] <= sm["tree_shape_total"]
    assert sm["ladder_ratio_200_to_1000_max_float32"] == pytest.approx(1.05)
    assert sm["ladder_ratio_200_to_1000_max_float64"] == pytest.approx(1.09)


def test_summary_ratio_is_none_when_points_missing():
    """缺 200 或 1000 档时该精度栏必须是 None，不能拿别的档凑数。"""
    ladder = [
        {"dtype": "float32", "num_blocks": 8,
         "num_permutations": 100, "cumulative_max_rel_err": 1e-7},
    ]
    sm = E0.experiment_summary([], ladder)
    assert sm["ladder_ratio_200_to_1000_max_float32"] is None


# ---------------------------------------------------------------------------
# 源码级防退化：这条轴不能被整体删掉
# ---------------------------------------------------------------------------
def test_script_still_exposes_the_permutation_axis():
    src = E0_SRC_PATH.read_text(encoding="utf-8")
    assert "--perm-ladder" in src, "置换次数轴被删了"
    assert "e0_perm_ladder.csv" in src, "阶梯 CSV 不再落盘"
    assert '"perm_ladder"' in src, "payload 少了 perm_ladder"
    assert '"summary"' in src, "payload 少了 summary"
    assert "def perm_ladder(" in src
    assert "def experiment_summary(" in src


def test_default_ladder_contains_the_two_cited_points():
    """论文引用的是 200→1000 的比值，故默认阶梯必须包含这两档。

    改默认阶梯本身没问题，但一旦把 200 或 1000 拿掉，
    正文引用的那个比值就不再能从落盘产物复现 —— 这条断言就是为了让
    这种改动必然被看见。
    """
    src = E0_SRC_PATH.read_text(encoding="utf-8")
    assert "default=[1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]" in src, (
        "默认阶梯被改动，正文引用的 200/1000 两个档位可能已不可复现"
    )


def test_ladder_rejects_nonpositive_points(fixture_blocks):
    states, ref = fixture_blocks
    with pytest.raises(ValueError):
        E0.perm_ladder(states, ref, [0, 5], seed=42)
    with pytest.raises(ValueError):
        E0.perm_ladder(states, ref, [-3], seed=42)
