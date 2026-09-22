#!/usr/bin/env python
"""E10：β 的稀疏塌缩 —— 判定、根因与修复（CPU 可跑）

问题的来源
----------
E2b 把 β 的口径定下来之后，`docs/commit_log.md` §P.2 留下最后一项未修缺陷：

    非负最小二乘允许 w_j → 0，实测相当比例的 β 落到 clamp 下界
    （log(1e-6) = −13.8155），效果接近"软剔除部分 Key"，
    与 β 的"质量重加权"定位不符。

这条缺陷的特殊性在于：**它不需要我们发明解法**。源论文
Attention Matching (arXiv:2602.16284) 附录 C.2 的 "Stabilizing β" 记录了
完全相同的失效模式，并给出对策与理由：

    A potential failure mode of this two-stage procedure is that mass matching
    can assign extremely small weights to some selected keys (i.e., very
    negative β, or effectively β = −∞). Such keys may have little effect on the
    mass objective yet still be useful for reducing attention-output error;
    however, once β is very negative, the corresponding key cannot contribute
    to the attention output, regardless of Cv.
    To mitigate this, we apply simple stability constraints on β. For Highest
    Attention Keys, we replace NNLS with a bounded least-squares over
    w = exp(β), enforcing e^{−3} ≤ w_j ≤ e^{3} (equivalently β_j ∈ [−3, 3]).

源论文结果表的方法名遵循 ``{选键法}_{准则}_nnls{迭代数}_{下界}_{上界}_{回归法}``，
其中与本仓库同源的那一支写作 ``highest_attn_keys_rms_nnls2_-3_3_lsq``：
选键用 attention 权重排序、准则取跨代表 query 的 RMS（与本仓库
`key_selection.py` 逐字一致）、β 用 2 步投影梯度 + **箱约束 [−3,3]**、
V 用普通最小二乘。即：本仓库的选键方式恰好落在**需要加 β 箱约束的那一支**上。
同节还写明他们试过 ℓ₂ 正则 `(XᵀX + λI)⁻¹`，"在所有正的 λ 上都降低性能"。

设计上的一个修订（冒烟运行后）
------------------------------
初版把主假设定为 H-col1 = "死键数等于设计矩阵的零空间维数 max(0, B − rank G)"。
冒烟运行**否掉了它**：M=8、B=32 时 rank(G)=8、零空间 24 维，实测死键数是 0 而非 24。
同时冒烟运行暴露了两个更实的量：

* β 使**有效 Key 数** `B_eff/B` 由 0.78（无 β）降到 0.65–0.70；
* β 使 **V 回归相对残差**由 0.036（无 β）升到 0.18。

即：真正的现象不是"少数 Key 被打到 −13.8"，而是**β 的离散度系统性压掉了
约 1/3 的输出容量**。因此本版的判据从"是否打到 clamp 下界"扩展为
"β 对输出阶段容量的代价有多大、以及哪个旋钮能把它换回来"。

四组可证伪的命题
----------------
**H-col1（根因不是秩）** — 已由冒烟运行否定，此处仍保留检验以便留证：
    `n_dead` 并不等于零空间维数。

**H-col2（输出阶段的容量代价）**
    施加 β 会降低 `B_eff/B` 并抬高 `v_fit_resid_rel`。这是源论文描述的
    "死键对质量目标无所谓、却对输出误差有用，而 β 一旦很负就再也无法参与输出"
    的直接量化。判据：`v_fit_resid_rel(β on) > v_fit_resid_rel(β off)` 且
    差值随 `beta_std` 增大。

**H-col3（箱约束是否换回容量）**
    源论文的箱约束 [−3,3] 会压低 `beta_std`，从而抬高 `B_eff/B`、降低 V 残差；
    但它同时限制了质量拟合的自由度。净效应必须在**留出 Query** 上定，
    且要分 B ≤ M 与 B > M 两域看（欠定与超定的解集结构不同）。

**H-col4（真正的旋钮可能是 λ 而不是箱）**
    岭正则 λ‖w − 1‖² 的方向本身就是"收缩回 β = 0"，即压制离散度。
    若加强 λ 能同时保住质量拟合并换回容量，则它比箱约束更对症。
    源论文称 ℓ₂ 正则"在一切正 λ 上都降低性能"—— 本实验直接检验该说法的
    可迁移性（我们的目标量级与 M/B 关系与它未必相同）。

判据与诚实边界
--------------
- 全部评估在**留出 Query** 上进行。留出集从未参与选键、β 拟合、V 回归。
- 同时报告**样本内**质量误差（用拟合用的代表 Query）与**留出**质量误差，
  用来区分"更好的拟合"与"更好的泛化"——二者方向可能相反。
- `B_eff` 的定义不依赖任何误差度量：把 X 的列权重的跨 query 均值 s_j 归一化成
  p_j，取 `exp(−Σ p_j log p_j)`。它回答"还有多少 Key 在真正参与输出"。
- 本脚本只产生**机制级**证据（合成数据上的注意力分布与线性输出重构精度），
  不构成任何任务质量主张。
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import synthetic as S          # noqa: E402
from experiments.common import report as R             # noqa: E402
from experiments.common import beta_variants as BV     # noqa: E402


# =============================================================================
# β 稳定性规格
# =============================================================================
#
# 每个规格 = (家族, 预设, w 下界, w 上界, λ_β, 说明)
#
# 阈值来源（均取自源论文附录 C.2，不是我们拍的）：
#   [−3, 3]：源论文给 Highest Attention Keys 支的箱约束（本仓库选键与之同源）
#   [−7, 7]：源论文给 OMP 支的剪枝线 β ≥ −7 与上界 β ≤ 7
#
# `w_lower = 0.0` 时求解器下界是 0，取 log 时由 eps = 1e-6 兜底 →
# β 下界 −13.8155，即"软剔除"。这是仓库现状。

E3_LO, E3_HI = math.exp(-3.0), math.exp(3.0)
E7_LO, E7_HI = math.exp(-7.0), math.exp(7.0)

#: 名称 → 规格
STABILITY_SPECS: Dict[str, Dict[str, Any]] = {
    # ---- 基准：仓库现状 ----
    "unbound": {"family": "基准", "preset": "am",
                "w_lower": 0.0, "w_upper": None, "lam": 1e-3,
                "desc": "仓库现状：w≥0 无上界 + λ_β=1e-3"},

    # ---- 约束族：λ 固定，只动箱 ----
    "box33": {"family": "约束族", "preset": "am",
              "w_lower": E3_LO, "w_upper": E3_HI, "lam": 1e-3,
              "desc": "源论文箱约束 β∈[−3,3]"},
    "box77": {"family": "约束族", "preset": "am",
              "w_lower": E7_LO, "w_upper": E7_HI, "lam": 1e-3,
              "desc": "OMP 支阈值 β∈[−7,7]"},
    "box33_lam1em2": {"family": "约束族", "preset": "am",
                      "w_lower": E3_LO, "w_upper": E3_HI, "lam": 1e-2,
                      "desc": "箱约束 + 加强 λ_β（约束族里的候选解）"},

    # ---- 正则族：无箱，扫 λ ----
    "noridge": {"family": "正则族", "preset": "am",
                "w_lower": 0.0, "w_upper": None, "lam": 0.0,
                "desc": "去掉岭正则（暴露零空间）"},
    "lam1em4": {"family": "正则族", "preset": "am",
                "w_lower": 0.0, "w_upper": None, "lam": 1e-4,
                "desc": "λ_β = 1e-4"},
    "lam1em2": {"family": "正则族", "preset": "am",
                "w_lower": 0.0, "w_upper": None, "lam": 1e-2,
                "desc": "λ_β = 1e-2（加强 10 倍）"},
    "lam1em1": {"family": "正则族", "preset": "am",
                "w_lower": 0.0, "w_upper": None, "lam": 1e-1,
                "desc": "λ_β = 1e-1（加强 100 倍）"},

    # ---- 对照 ----
    "scalar": {"family": "对照", "preset": "am_scalar",
               "w_lower": 0.0, "w_upper": None, "lam": 1e-3,
               "desc": "β 退化为单一标量（per-key 自由度全去掉）"},
    "nobeta": {"family": "对照", "preset": "am_nobeta",
               "w_lower": 0.0, "w_upper": None, "lam": 1e-3,
               "desc": "β 全链路关闭（区分「加约束」与「不用 β」）"},
}

SPEC_ORDER = ("unbound", "box33", "box77", "box33_lam1em2",
              "noridge", "lam1em4", "lam1em2", "lam1em1",
              "scalar", "nobeta")

#: 逐指标的核心规格（用于分域表与配对表）
KEY_SPECS = ("unbound", "box33", "box33_lam1em2", "lam1em2", "scalar", "nobeta")


# =============================================================================
# 单格评估
# =============================================================================

def evaluate_cell(
    split: S.HeldoutSplit,
    dest: int,
    budget: int,
    spec: str,
    num_repr: int,
    projection_dim: int,
    seed: int,
    fixed_ctx: Optional[Tuple[torch.Tensor, torch.Tensor]],
    lambda_value: float = 1e-3,
) -> Dict[str, Any]:
    """在单个 (M, B, dest, seed) 上按指定稳定性规格构造并评估。"""
    c = STABILITY_SPECS[spec]

    compact, diag = BV.build_compact_kv_variant(
        source_keys=split.keys,
        source_values=split.values,
        destination_queries=split.fit_queries[dest],
        budget=budget,
        preset=str(c["preset"]),
        num_representative_queries=num_repr,
        projection_dim=projection_dim,
        lambda_beta=float(c["lam"]),
        lambda_value=lambda_value,
        seed=seed,
        w_lower=float(c["w_lower"]),
        w_upper=(None if c["w_upper"] is None else float(c["w_upper"])),
    )

    eval_q = split.eval_queries[dest]
    fit_q = split.fit_queries[dest]

    # ---- 留出侧（泛化）----
    rel_eval = S.relative_output_error(eval_q, compact, split.keys, split.values)
    absm_eval = S.absolute_mass_error(eval_q, compact, split.keys)
    if fixed_ctx is not None:
        FK, FV = fixed_ctx
        mix_eval = S.mixture_relative_error(
            eval_q, FK, FV, compact, split.keys, split.values)
        mix_med = float(mix_eval.median().item())
    else:
        mix_med = float("nan")

    # ---- 拟合侧（样本内）----
    absm_fit = S.absolute_mass_error(fit_q, compact, split.keys)

    row: Dict[str, Any] = {
        "spec": spec,
        "family": c["family"],
        "seed": seed,
        "dest": dest,
        "M": num_repr,
        "budget": budget,
        "L_s": split.L_s,
        # 留出侧
        "eval_out_err_median": float(rel_eval.median().item()),
        "eval_abs_mass_median": float(absm_eval.median().item()),
        "eval_mixture_median": mix_med,
        # 拟合侧
        "fit_abs_mass_median": float(absm_fit.median().item()),
        # β 的形态
        "beta_mean": diag["beta_mean"],
        "beta_std": diag["beta_std"],
        "beta_min": diag["beta_min"],
        "beta_max": diag["beta_max"],
        "n_beta_le_m3": diag["n_beta_le_m3"],
        "n_beta_le_m7": diag["n_beta_le_m7"],
        "n_beta_at_clamp": diag["n_beta_at_clamp"],
        "frac_beta_le_m3": diag["frac_beta_le_m3"],
        "clamp_rate": diag["n_beta_at_clamp"] / max(budget, 1),
        "support_size": diag["support_size"],
        # 结构量
        "rank_G_numeric": diag["rank_G_numeric"],
        "rank_upper_bound": diag["rank_upper_bound"],
        "cond_G": diag["cond_G"],
        "solver_residual": diag["solver_residual"],
        # 输出阶段容量
        "x_eff_budget": diag["x_eff_budget"],
        "x_eff_budget_frac": diag["x_eff_budget_frac"],
        "x_eff_loss": 1.0 - diag["x_eff_budget_frac"],
        "x_col_mean_ratio": diag["x_col_mean_ratio"],
        "v_fit_resid_rel": diag["v_fit_resid_rel"],
        # 理想量级参照
        "beta_ideal_mass_preserving": math.log(split.L_s / float(budget)),
    }
    return row


# =============================================================================
# 汇总：描述统计
# =============================================================================

def _med(rows: List[Dict[str, Any]], field: str) -> float:
    vals = sorted(r[field] for r in rows if r[field] == r[field])
    if not vals:
        return float("nan")
    return vals[len(vals) // 2]


def _mean(rows: List[Dict[str, Any]], field: str) -> float:
    vals = [r[field] for r in rows if r[field] == r[field]]
    return sum(vals) / len(vals) if vals else float("nan")


def _pearson(rows: List[Dict[str, Any]], fa: str, fb: str) -> float:
    pairs = [(r[fa], r[fb]) for r in rows
             if r[fa] == r[fa] and r[fb] == r[fb]
             and abs(r[fa]) != float("inf")]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def print_main_table(rows: List[Dict[str, Any]]) -> None:
    print("=" * 118)
    print("主表：全部格子（按规格取中位数；clamp率 / β≤−3 为键数占比）")
    print("=" * 118)
    print(f"{'规格':<15}{'留出输出':>9}{'留出归并':>9}{'留出|mass|':>10}"
          f"{'样本内|mass|':>12}{'B_eff/B':>9}{'V残差':>8}{'β_std':>7}"
          f"{'clamp率':>8}{'β≤−3':>7}")
    print("-" * 118)
    fam = None
    for spec in SPEC_ORDER:
        sub = [r for r in rows if r["spec"] == spec]
        if not sub:
            continue
        f = STABILITY_SPECS[spec]["family"]
        if f != fam:
            print(f"  --- {f} ---")
            fam = f
        n_keys = sum(r["budget"] for r in sub)
        clamp = sum(r["n_beta_at_clamp"] for r in sub) / max(n_keys, 1)
        m3 = sum(r["n_beta_le_m3"] for r in sub) / max(n_keys, 1)
        print(f"  {spec:<13}{_med(sub, 'eval_out_err_median'):>9.4f}"
              f"{_med(sub, 'eval_mixture_median'):>9.4f}"
              f"{_med(sub, 'eval_abs_mass_median'):>10.4f}"
              f"{_med(sub, 'fit_abs_mass_median'):>12.4f}"
              f"{_med(sub, 'x_eff_budget_frac'):>9.4f}"
              f"{_med(sub, 'v_fit_resid_rel'):>8.4f}"
              f"{_med(sub, 'beta_std'):>7.3f}"
              f"{clamp:>8.3f}{m3:>7.3f}")


def print_lambda_sweep(rows: List[Dict[str, Any]]) -> None:
    """H-col4：λ_β 是不是比箱约束更对症的旋钮。"""
    print()
    print("=" * 118)
    print("H-col4：λ_β 扫描（无箱约束）。岭正则的方向本就是「收缩回 β=0」，"
          "即压制离散度。")
    print("  源论文称 ℓ₂ 正则「在一切正 λ 上都降低性能」—— 此表直接检验其可迁移性。")
    print("=" * 118)
    print(f"{'λ_β':>8}{'clamp率':>9}{'β≤−3':>8}{'β_std':>8}{'B_eff/B':>9}"
          f"{'V残差':>8}{'留出输出':>10}{'留出归并':>10}{'样本内|mass|':>13}"
          f"{'留出|mass|':>11}")
    print("-" * 118)
    for spec in ("noridge", "lam1em4", "unbound", "lam1em2", "lam1em1"):
        sub = [r for r in rows if r["spec"] == spec]
        if not sub:
            continue
        n_keys = sum(r["budget"] for r in sub)
        clamp = sum(r["n_beta_at_clamp"] for r in sub) / max(n_keys, 1)
        m3 = sum(r["n_beta_le_m3"] for r in sub) / max(n_keys, 1)
        lam = STABILITY_SPECS[spec]["lam"]
        print(f"{lam:>8g}{clamp:>9.3f}{m3:>8.3f}"
              f"{_med(sub, 'beta_std'):>8.3f}"
              f"{_med(sub, 'x_eff_budget_frac'):>9.4f}"
              f"{_med(sub, 'v_fit_resid_rel'):>8.4f}"
              f"{_med(sub, 'eval_out_err_median'):>10.4f}"
              f"{_med(sub, 'eval_mixture_median'):>10.4f}"
              f"{_med(sub, 'fit_abs_mass_median'):>13.4f}"
              f"{_med(sub, 'eval_abs_mass_median'):>11.4f}")


def print_rank_hypothesis(rows: List[Dict[str, Any]]) -> None:
    """H-col1：死键数是否等于零空间维数 max(0, B − rank G)。"""
    print()
    print("=" * 118)
    print("H-col1：塌缩的根因是秩吗？检验 n_dead = B − rank(G)（用 λ=0 规格，"
          "否则正则会给 GᵀG 加满秩扰动）")
    print("=" * 118)
    sub = [r for r in rows if r["spec"] == "noridge"]
    if not sub:
        print("  （缺少 noridge 规格）")
        return
    print(f"{'M':>4}{'B':>5}{'rank(G)':>9}{'零空间':>8}{'clamp数':>9}"
          f"{'β≤−3数':>9}{'B_eff':>8}{'cond(G)':>12}{'留出输出':>10}")
    print("-" * 118)
    diffs = []
    for (m, b) in sorted({(r["M"], r["budget"]) for r in sub}):
        c = [r for r in sub if r["M"] == m and r["budget"] == b]
        if not c:
            continue
        rk = _med(c, "rank_G_numeric")
        nullity = max(0.0, b - rk)
        clamp = _med(c, "n_beta_at_clamp")
        m3 = _med(c, "n_beta_le_m3")
        diffs.append((abs(clamp - nullity), abs(m3 - nullity)))
        print(f"{m:>4}{b:>5}{rk:>9.1f}{nullity:>8.1f}{clamp:>9.1f}"
              f"{m3:>9.1f}{_med(c, 'x_eff_budget'):>8.2f}"
              f"{_med(c, 'cond_G'):>12.3e}"
              f"{_med(c, 'eval_out_err_median'):>10.4f}")
    if diffs:
        mad_clamp = sum(d[0] for d in diffs) / len(diffs)
        mad_m3 = sum(d[1] for d in diffs) / len(diffs)
        print()
        print(f"  平均绝对偏差  |clamp数 − 零空间| = {mad_clamp:.2f}")
        print(f"  平均绝对偏差  |β≤−3数 − 零空间| = {mad_m3:.2f}")
        print("  → " + ("与「塌缩 = 零空间维数」一致" if mad_m3 < 1.0
                        else "与「塌缩 = 零空间维数」不符 —— 根因不是秩"))


def print_domain_split(rows: List[Dict[str, Any]]) -> None:
    """H-col3：按 B ≤ M（超定）与 B > M（欠定）分域。"""
    print()
    print("=" * 118)
    print("H-col3：按 B ≤ M（回归超定）与 B > M（回归欠定）分域")
    print("=" * 118)
    print(f"{'规格':<15}{'域':<7}{'n':>5}{'留出输出':>10}{'留出归并':>10}"
          f"{'样本内|mass|':>13}{'B_eff/B':>9}{'V残差':>8}{'β≤−3':>8}")
    print("-" * 118)
    for spec in KEY_SPECS:
        for tag, cond in (("B≤M", lambda r: r["budget"] <= r["M"]),
                          ("B>M", lambda r: r["budget"] > r["M"])):
            sub = [r for r in rows if r["spec"] == spec and cond(r)]
            if not sub:
                continue
            n_keys = sum(r["budget"] for r in sub)
            m3 = sum(r["n_beta_le_m3"] for r in sub) / max(n_keys, 1)
            print(f"{spec:<15}{tag:<7}{len(sub):>5}"
                  f"{_med(sub, 'eval_out_err_median'):>10.4f}"
                  f"{_med(sub, 'eval_mixture_median'):>10.4f}"
                  f"{_med(sub, 'fit_abs_mass_median'):>13.4f}"
                  f"{_med(sub, 'x_eff_budget_frac'):>9.4f}"
                  f"{_med(sub, 'v_fit_resid_rel'):>8.4f}"
                  f"{m3:>8.3f}")


def print_coupling(rows: List[Dict[str, Any]]) -> None:
    """H-col2：容量代价与 β 离散度的关系。"""
    print()
    print("=" * 118)
    print("H-col2：输出阶段的容量代价")
    print("  源论文的理由：死键「对质量目标影响很小，却对降低输出误差有用」，")
    print("  而 β 一旦很负，该 Key「无论 Cv 取什么都不能参与输出」。")
    print("=" * 118)
    print(f"{'规格':<15}{'样本内|mass|':>13}{'留出|mass|':>12}{'β_std':>8}"
          f"{'B_eff/B':>9}{'损耗':>8}{'V残差':>8}{'留出输出':>10}")
    print("-" * 118)
    for spec in KEY_SPECS:
        sub = [r for r in rows if r["spec"] == spec]
        if not sub:
            continue
        print(f"{spec:<15}{_med(sub, 'fit_abs_mass_median'):>13.4f}"
              f"{_med(sub, 'eval_abs_mass_median'):>12.4f}"
              f"{_med(sub, 'beta_std'):>8.3f}"
              f"{_med(sub, 'x_eff_budget_frac'):>9.4f}"
              f"{_med(sub, 'x_eff_loss'):>8.4f}"
              f"{_med(sub, 'v_fit_resid_rel'):>8.4f}"
              f"{_med(sub, 'eval_out_err_median'):>10.4f}")
    print()
    print("  跨格子相关性（全部规格合并，n=格子数）：")
    for fa, fb, note in (("beta_std", "x_eff_loss", "β 离散度 ↔ 有效 Key 损耗"),
                         ("beta_std", "v_fit_resid_rel", "β 离散度 ↔ V 回归残差"),
                         ("x_eff_loss", "v_fit_resid_rel", "有效 Key 损耗 ↔ V 残差")):
        print(f"    corr({note}) = {_pearson(rows, fa, fb):+.3f}")


# =============================================================================
# main
# =============================================================================

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E10：β 稀疏塌缩的判定与修复")
    p.add_argument("--out", default="results/cpu/e10")
    p.add_argument("--quick", action="store_true", help="缩小规模，用于冒烟")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--num-dest", dest="num_dest", type=int, default=3)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=96)
    p.add_argument("--M", type=int, nargs="+", default=[8, 16, 32],
                   help="代表 Query 数（扫多档是为了让 B≶M 两侧都有格子）")
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=16)
    p.add_argument("--eval-fraction", dest="eval_fraction", type=float, default=0.5)
    p.add_argument("--focus-strength", dest="focus_strength", type=float, default=8.0)
    p.add_argument("--fixed-len", dest="fixed_len", type=int, default=128)
    p.add_argument("--lambda-value", dest="lambda_value", type=float, default=1e-3)
    p.add_argument("--no-paired", dest="paired", action="store_false")
    args = p.parse_args(argv)

    if args.quick:
        args.seeds, args.L_s, args.num_dest = 2, 128, 2
        args.queries_per_dest = 64
        args.M, args.budgets = [8, 16], [8, 32]

    n_cells = len(args.M) * len(args.budgets) * args.num_dest * args.seeds
    print("=" * 118)
    print("E10：β 的稀疏塌缩 —— 判定、根因与修复")
    print("=" * 118)
    print(f"  L_s={args.L_s} d_h={args.d_h} d_v={args.d_v} dest={args.num_dest} "
          f"queries/dest={args.queries_per_dest} eval_frac={args.eval_fraction}")
    print(f"  M={args.M} budgets={args.budgets} seeds={args.seeds} "
          f"λ_v={args.lambda_value:g} focus={args.focus_strength}")
    print(f"  (M,B,dest,seed) 组合 = {n_cells}，× {len(SPEC_ORDER)} 规格 "
          f"= {n_cells * len(SPEC_ORDER)} 次构造")
    print()
    print("  规格（阈值取自源论文附录 C.2，非自拟）：")
    for spec in SPEC_ORDER:
        c = STABILITY_SPECS[spec]
        lo = "−∞" if c["w_lower"] == 0.0 else f"{math.log(c['w_lower']):+.1f}"
        hi = "∞" if c["w_upper"] is None else f"{math.log(c['w_upper']):+.1f}"
        print(f"    {c['family']:<5}{spec:<15} β∈[{lo:>4},{hi:>4}] "
              f"λ_β={c['lam']:<7g} {c['desc']}")
    print()

    rows: List[Dict[str, Any]] = []
    for si in range(args.seeds):
        seed = 42 + si
        scenario = S.make_scenario(
            L_s=args.L_s, d_h=args.d_h, d_v=args.d_v,
            num_dest=args.num_dest, queries_per_dest=args.queries_per_dest,
            focus_strength=args.focus_strength, seed=seed,
        )
        split = S.heldout_split(scenario, eval_fraction=args.eval_fraction)
        fixed_ctx = S.make_fixed_context(
            d_h=args.d_h, d_v=args.d_v, length=args.fixed_len, seed=1000 + si,
        )
        for dest in range(args.num_dest):
            for num_repr in args.M:
                for budget in args.budgets:
                    for spec in SPEC_ORDER:
                        rows.append(evaluate_cell(
                            split, dest, budget, spec, num_repr,
                            args.projection_dim, seed, fixed_ctx,
                            lambda_value=args.lambda_value,
                        ))

    print_main_table(rows)
    print_lambda_sweep(rows)
    print_rank_hypothesis(rows)
    print_domain_split(rows)
    print_coupling(rows)

    # ---- 配对检验 ----
    paired: Dict[str, Any] = {}
    if args.paired:
        print()
        print("=" * 118)
        print("配对检验（逐 (seed, dest, M, B) 配对；A 优于 B ⇔ A 的误差更小）")
        print("=" * 118)

        def keyed(spec: str, field: str,
                  subset: Optional[Callable[[Dict[str, Any]], bool]] = None
                  ) -> Dict[Any, float]:
            return {
                (r["seed"], r["dest"], r["M"], r["budget"]): r[field]
                for r in rows
                if r["spec"] == spec and (subset is None or subset(r))
            }

        def compare(a: str, b: str, field: str, tag: str,
                    subset: Optional[Callable[[Dict[str, Any]], bool]] = None,
                    quiet: bool = False) -> Optional[Dict[str, Any]]:
            ka, kb = keyed(a, field, subset), keyed(b, field, subset)
            common = sorted(set(ka) & set(kb))
            if not common:
                return None
            pr = R.paired_bootstrap([ka[k] for k in common],
                                    [kb[k] for k in common],
                                    metric_name=field, unit="ratio")
            paired[f"{a}_vs_{b}__{tag}__{field}"] = pr.to_dict()
            if not quiet:
                print(f"  [{tag}] {a:<16} vs {b:<16} n={pr.n_pairs:<4} "
                      f"Δ={pr.mean_diff:+.4f} "
                      f"CI95=[{pr.ci_95_lower:+.4f},{pr.ci_95_upper:+.4f}] "
                      f"p={pr.p_value_one_sided:.4f} W/L={pr.a_wins}/{pr.b_wins}")
            return pr.to_dict()

        gt_m: Callable[[Dict[str, Any]], bool] = lambda r: r["budget"] > r["M"]
        le_m: Callable[[Dict[str, Any]], bool] = lambda r: r["budget"] <= r["M"]
        metrics = ("eval_out_err_median", "eval_mixture_median",
                   "eval_abs_mass_median")

        print("  [A. 隔离箱约束（两侧 λ 相同）]")
        for fld in metrics:
            compare("box33", "unbound", fld, "全部")
        print()
        print("  [B. 隔离 λ（无箱）]")
        for fld in metrics:
            compare("lam1em2", "unbound", fld, "λ=1e-2")
        print()
        print("  [C. 候选解：箱 + 加强 λ]")
        for fld in metrics:
            compare("box33_lam1em2", "unbound", fld, "全部")
        for fld in ("eval_out_err_median", "eval_mixture_median"):
            compare("box33_lam1em2", "unbound", fld, "B>M", gt_m)
            compare("box33_lam1em2", "unbound", fld, "B≤M", le_m)
        print()
        print("  [D. 分域：箱约束在 B>M / B≤M 两侧是否不同]")
        for fld in ("eval_out_err_median", "eval_mixture_median"):
            compare("box33", "unbound", fld, "B>M", gt_m)
            compare("box33", "unbound", fld, "B≤M", le_m)
        print()
        print("  [E. 对照：per-key 自由度值不值（scalar）与干脆不用 β（nobeta）]")
        for fld in ("eval_out_err_median", "eval_mixture_median"):
            compare("unbound", "scalar", fld, "全部")
            compare("box33_lam1em2", "nobeta", fld, "全部")

        # 容量侧的配对（越大越好 → higher_is_better=True）
        print()
        print("  [F. 容量侧：B_eff/B 与 V 残差（B_eff 越大越好，残差越小越好）]")
        for a, b, tag in (("unbound", "nobeta", "β on vs off"),
                          ("box33", "unbound", "箱 vs 无箱"),
                          ("lam1em2", "unbound", "λ=1e-2 vs λ=1e-3")):
            ka, kb = keyed(a, "x_eff_budget_frac"), keyed(b, "x_eff_budget_frac")
            common = sorted(set(ka) & set(kb))
            if common:
                pr = R.paired_bootstrap([ka[k] for k in common],
                                        [kb[k] for k in common],
                                        metric_name="x_eff_budget_frac",
                                        unit="ratio", higher_is_better=True)
                paired[f"{a}_vs_{b}__{tag}__x_eff_budget_frac"] = pr.to_dict()
                print(f"  [{tag}] B_eff/B  {a:<16} vs {b:<16} n={pr.n_pairs:<4} "
                      f"Δ={pr.mean_diff:+.4f} p={pr.p_value_one_sided:.4f} "
                      f"W/L={pr.a_wins}/{pr.b_wins}")
            ka, kb = keyed(a, "v_fit_resid_rel"), keyed(b, "v_fit_resid_rel")
            common = sorted(set(ka) & set(kb))
            if common:
                pr = R.paired_bootstrap([ka[k] for k in common],
                                        [kb[k] for k in common],
                                        metric_name="v_fit_resid_rel", unit="ratio")
                paired[f"{a}_vs_{b}__{tag}__v_fit_resid_rel"] = pr.to_dict()
                print(f"  [{tag}] V残差  {a:<16} vs {b:<16} n={pr.n_pairs:<4} "
                      f"Δ={pr.mean_diff:+.4f} p={pr.p_value_one_sided:.4f} "
                      f"W/L={pr.a_wins}/{pr.b_wins}")

    # ---- 判定提示（只陈述数据允许的部分）----
    print()
    print("=" * 118)
    print("判定提示（不做超出数据的断言）")
    print("=" * 118)
    for spec in SPEC_ORDER:
        sub = [r for r in rows if r["spec"] == spec]
        if not sub:
            continue
        n_keys = sum(r["budget"] for r in sub)
        clamp = sum(r["n_beta_at_clamp"] for r in sub) / max(n_keys, 1)
        print(f"  {spec:<15} 留出输出 {_med(sub, 'eval_out_err_median'):.4f}  "
              f"留出归并 {_med(sub, 'eval_mixture_median'):.4f}  "
              f"B_eff/B {_med(sub, 'x_eff_budget_frac'):.4f}  "
              f"V残差 {_med(sub, 'v_fit_resid_rel'):.4f}  "
              f"clamp率 {clamp:.3f}")

    payload = {
        "experiment": "E10",
        "kind": "beta-stability-constraint",
        "specs": STABILITY_SPECS,
        "spec_order": list(SPEC_ORDER),
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "paired": paired,
        "rows": rows,
        "source_grounding": (
            "阈值 [−3,3] 与 [−7,7] 取自 Attention Matching "
            "(arXiv:2602.16284) 附录 C.2 'Stabilizing β'；"
            "本仓库的 RMS 选键对应其 highest_attn_keys_rms 支，该支配套的 β 处理"
            "即箱约束 [−3,3]；同节称 ℓ₂ 正则在一切正 λ 上都降低性能。"
        ),
        "caveat": (
            "机制级证据（合成数据上的注意力分布与线性输出重构精度），"
            "不构成任务质量主张。所有评估均在**留出 Query** 上进行，"
            "留出集从未参与 Key 选择、β 拟合或 V 回归。"
        ),
    }
    R.save_summary(str(pathlib.Path(args.out) / "summary.json"), payload)
    R.save_csv(str(pathlib.Path(args.out) / "rows.csv"), rows)
    print()
    print(f"结果已写入 {args.out}/summary.json 与 rows.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
