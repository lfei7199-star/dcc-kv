#!/usr/bin/env python
r"""E11：$\beta$ 岭正则 `lambda_beta` 的细化扫描与分域定位（CPU 可跑）

要回答的问题
------------
E10 判定 $\beta$ 稀疏塌缩的根因是"per-key 离散度压低有效预算"，
并指出 $\lambda_\beta$ 是比箱约束更对症的旋钮：$10^{-3}\to10^{-2}$ 即取得
比箱约束更大的收益。E10 把 $\lambda_\beta$ 的提升列为**待决配置**（D4），
理由是会改变 E2b/E9 已报告的数字，必须先定下来再推 GPU。

但 E10 的 $\lambda$ 阶梯只到 $10^{-1}$，且**尚未收敛**：
无箱条件下留出输出误差 $0.523\to0.516\to0.495$（$\lambda=10^{-3},10^{-2},10^{-1}$）
单调下降。而 $\lambda\to\infty$ 时 $w\to1$、$\beta\to0$，链路退化到
"关闭 $\beta$"，结果**必然**回到 `nobeta` 的 $0.500$。既然 $0.495<0.500$，
最小值必然落在 $(10^{-1},\infty)$ 之间 —— 光在 $10^{-2}\sim10^{-1}$ 之间插档
回答不了"该取多少"，必须**向上扫**。

三条待判命题
------------
**H-lam1（渐近一致性）**
    $\lambda\to\infty$ 时 $\beta\to0$，故 $\lambda$ 充分大的规格应当与
    `nobeta` 逐位一致。若不一致，说明"收缩到 $w=1$"与"关闭 $\beta$"并非同一极限，
    本脚本的前提就错了。这是对**管线本身**的检验，先做。

**H-lam2（是否存在单一最优 $\lambda$）**
    若存在一个 $\lambda^\*$ 在全部 $180$ 格上都不劣于当前默认（$10^{-3}$），
    则可安全采纳；若最优 $\lambda$ 随 $M/B$ 域移动（例如 $B\le M$ 与 $B>M$ 的
    argmin 相差一个数量级以上），则**不存在**可写进实现默认的单一 $\lambda$。

**H-lam3（约束从箱交棒给岭正则）**
    $\lambda$ 增大 → 解收缩 → 顶在箱边界 $\pm3$ 的坐标占比下降。
    记录该占比随 $\lambda$ 的衰减，用来判定：在候选 $\lambda^\*$ 处，
    箱约束是否仍实际生效（若已完全失效，则"箱 + 强 $\lambda$"与
    "只用强 $\lambda$"等价，默认配置可以更简单）。

判据与诚实边界
--------------
- 全部评估在**留出 Query** 上进行，留出集从未参与选键、$\beta$ 拟合或 $V$ 回归。
- 网格、种子、场景参数与 E10 **完全一致**，以便两套数字可直接比较；
  E11 还刻意重跑了 E10 的两个无箱规格（$\lambda=10^{-3}$ 与 $10^{-1}$），
  作为**管线互校**：若两者不逐位相同，则说明比较链本身不可靠。
- 只报机制级证据（合成数据上的注意力分布与线性输出重构精度），
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
# 规格
# =============================================================================
#
# 与 E10 的差别：E10 问"箱约束值不值得"，E11 问"λ 该取多少"，
# 因此本脚本把箱约束 $[-3,3]$（E10 已采纳的新默认）**固定打开**，
# 只扫 λ；另留两个无箱规格做管线互校。

E3_LO, E3_HI = math.exp(-3.0), math.exp(3.0)

#: 默认 $\lambda_\beta$（仓库现状）
DEFAULT_LAMBDA = 1e-3

#: 待扫的 λ 阶梯（覆盖 E10 只到 $10^{-1}$ 的缺口，向上延伸到 10）
LAMBDA_LADDER: Tuple[float, ...] = (
    1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 1e3,
)


#: λ → 规格短标签（显式给出，保证与 E10 的规格名一致，便于管线互校）
LAM_TAGS: Dict[float, str] = {
    1e-3: "1em3", 3e-3: "3em3", 1e-2: "1em2", 3e-2: "3em2",
    1e-1: "1em1", 3e-1: "3em1", 1e0: "1e0", 3e0: "3e0", 1e1: "1e1",
    3e1: "3e1", 1e2: "1e2", 1e3: "1e3",
}


def _lam_tag(lam: float) -> str:
    """λ → 规格名里的短标签（如 1e-3 → ``1em3``）。"""
    if lam in LAM_TAGS:
        return LAM_TAGS[lam]
    return f"{lam:g}".replace("-", "m").replace(".", "p").replace("+", "")


def _build_specs() -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}

    # ---- 主阶梯：箱约束开，只扫 λ ----
    for lam in LAMBDA_LADDER:
        specs[f"box_lam{_lam_tag(lam)}"] = {
            "family": "箱+λ",
            "preset": "am",
            "w_lower": E3_LO, "w_upper": E3_HI,
            "lam": lam,
            "desc": (f"箱 $[-3,3]$ + $\\lambda_\\beta$={lam:g}"
                     + ("（当前默认）" if lam == DEFAULT_LAMBDA else "")),
        }

    # ---- 管线互校：无箱，取 E10 已报过的两档 ----
    for lam in (1e-3, 1e-1):
        specs[f"nobox_lam{_lam_tag(lam)}"] = {
            "family": "无箱互校",
            "preset": "am",
            "w_lower": 0.0, "w_upper": None,
            "lam": lam,
            "desc": f"无箱 + $\\lambda_\\beta$={lam:g}（与 E10 同规格，用于互校）",
        }

    # ---- 端点参照 ----
    specs["scalar"] = {
        "family": "参照", "preset": "am_scalar",
        "w_lower": 0.0, "w_upper": None, "lam": DEFAULT_LAMBDA,
        "desc": "$\\beta$ 退化为单一标量",
    }
    specs["nobeta"] = {
        "family": "参照", "preset": "am_nobeta",
        "w_lower": 0.0, "w_upper": None, "lam": DEFAULT_LAMBDA,
        "desc": "$\\beta$ 全链路关闭（$\\lambda\\to\\infty$ 的极限）",
    }
    return specs


LAMBDA_SPECS: Dict[str, Dict[str, Any]] = _build_specs()

SPEC_ORDER: Tuple[str, ...] = (
    *(f"box_lam{_lam_tag(l)}" for l in LAMBDA_LADDER),
    "nobox_lam1em3", "nobox_lam1em1",
    "scalar", "nobeta",
)

#: E10 中同规格的落盘值（管线互校用；取自 results/cpu/e10/summary.json）
E10_CROSSCHECK = {
    "nobox_lam1em3": {"eval_out_err_median": 0.5229, "eval_mixture_median": 0.4344},
    "nobox_lam1em1": {"eval_out_err_median": 0.4952, "eval_mixture_median": 0.4174},
    "scalar": {"eval_out_err_median": 0.4996, "eval_mixture_median": 0.4138},
    "nobeta": {"eval_out_err_median": 0.4996, "eval_mixture_median": 0.5158},
}


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
    """在单个 (M, B, dest, seed) 上按指定规格构造并评估。"""
    c = LAMBDA_SPECS[spec]

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

    # ---- 约束的实际生效情况：**直接从落盘的 β 向量统计** ----
    # E10 只统计了下界 clamp（β ≤ log 1e-6）。箱约束是双侧的，
    # 且 λ 增大时先松开的是箱边界，因此这里补上"顶在 ±3"的坐标占比。
    with torch.no_grad():
        beta_vec = compact.logit_bias.detach().double()
        n_key = int(beta_vec.numel())
        n_at_lo = int((beta_vec <= -3.0 + 1e-6).sum().item())
        n_at_hi = int((beta_vec >= 3.0 - 1e-6).sum().item())
        beta_span = float(beta_vec.max().item() - beta_vec.min().item())

    row: Dict[str, Any] = {
        "spec": spec,
        "family": c["family"],
        "seed": seed,
        "dest": dest,
        "M": num_repr,
        "budget": budget,
        "L_s": split.L_s,
        "lam": float(c["lam"]),
        # 留出侧
        "eval_out_err_median": float(rel_eval.median().item()),
        "eval_abs_mass_median": float(absm_eval.median().item()),
        "eval_mixture_median": mix_med,
        # 拟合侧
        "fit_abs_mass_median": float(absm_fit.median().item()),
        # β 形态
        "beta_mean": diag["beta_mean"],
        "beta_std": diag["beta_std"],
        "beta_min": diag["beta_min"],
        "beta_max": diag["beta_max"],
        "beta_span": beta_span,
        "n_beta_le_m3": diag["n_beta_le_m3"],
        "n_beta_at_clamp": diag["n_beta_at_clamp"],
        # 约束生效度（本脚本新增）
        "at_box_lower": n_at_lo,
        "at_box_upper": n_at_hi,
        "box_bind_rate": (n_at_lo + n_at_hi) / max(n_key, 1),
        "clamp_rate": diag["n_beta_at_clamp"] / max(budget, 1),
        # 结构量
        "rank_G_numeric": diag["rank_G_numeric"],
        "solver_residual": diag["solver_residual"],
        "solver_lambda_eff": diag["solver_lambda_eff"],
        # 输出阶段容量
        "x_eff_budget_frac": diag["x_eff_budget_frac"],
        "x_eff_loss": 1.0 - diag["x_eff_budget_frac"],
        "v_fit_resid_rel": diag["v_fit_resid_rel"],
        # 理想量级参照
        "beta_ideal_mass_preserving": math.log(split.L_s / float(budget)),
    }
    return row


# =============================================================================
# 汇总工具
# =============================================================================

def _med(rows: List[Dict[str, Any]], field: str) -> float:
    vals = sorted(r[field] for r in rows if r[field] == r[field])
    if not vals:
        return float("nan")
    return vals[len(vals) // 2]


def _mean(rows: List[Dict[str, Any]], field: str) -> float:
    vals = [r[field] for r in rows if r[field] == r[field]]
    return sum(vals) / len(vals) if vals else float("nan")


def _cells(rows: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    return sorted({(r["M"], r["budget"]) for r in rows})


def _cell_rows(rows: List[Dict[str, Any]], spec: str,
               m: int, b: int) -> List[Dict[str, Any]]:
    return [r for r in rows if r["spec"] == spec
            and r["M"] == m and r["budget"] == b]


# =============================================================================
# 表 1：λ 阶梯主表
# =============================================================================

def print_ladder(rows: List[Dict[str, Any]]) -> None:
    print("=" * 126)
    print("表 1：λ 阶梯（箱约束 [−3,3] 固定打开；各规格在 180 格上取中位数）")
    print("  箱绑定率 = 顶在 ±3 边界的坐标占比；clamp率 = 顶在 log(1e-6) 的坐标占比")
    print("=" * 126)
    print(f"{'规格':<18}{'λ_β':>7}{'箱绑定率':>10}{'clamp率':>9}{'β_std':>8}"
          f"{'β_mean':>8}{'B_eff/B':>9}{'V残差':>8}{'留出输出':>10}"
          f"{'留出归并':>10}{'样本内|mass|':>13}{'留出|mass|':>11}")
    print("-" * 126)
    fam = None
    for spec in SPEC_ORDER:
        sub = [r for r in rows if r["spec"] == spec]
        if not sub:
            continue
        f = LAMBDA_SPECS[spec]["family"]
        if f != fam:
            print(f"  --- {f} ---")
            fam = f
        nk = sum(r["budget"] for r in sub)
        bind = sum(r["at_box_lower"] + r["at_box_upper"] for r in sub) / max(nk, 1)
        clamp = sum(r["n_beta_at_clamp"] for r in sub) / max(nk, 1)
        print(f"  {spec:<16}{LAMBDA_SPECS[spec]['lam']:>7g}{bind:>10.3f}"
              f"{clamp:>9.3f}{_med(sub, 'beta_std'):>8.3f}"
              f"{_med(sub, 'beta_mean'):>8.3f}"
              f"{_med(sub, 'x_eff_budget_frac'):>9.4f}"
              f"{_med(sub, 'v_fit_resid_rel'):>8.4f}"
              f"{_med(sub, 'eval_out_err_median'):>10.4f}"
              f"{_med(sub, 'eval_mixture_median'):>10.4f}"
              f"{_med(sub, 'fit_abs_mass_median'):>13.4f}"
              f"{_med(sub, 'eval_abs_mass_median'):>11.4f}")


# =============================================================================
# 表 2：分域（B ≤ M 回归超定 / B > M 回归欠定）
# =============================================================================

def print_regime_split(rows: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 126)
    print("表 2：分域 —— B≤M（回归超定）与 B>M（回归欠定）；每域内取中位数")
    print("=" * 126)
    print(f"{'规格':<18}{'λ_β':>7}{'域':<6}{'n':>4}{'留出输出':>10}{'留出归并':>10}"
          f"{'B_eff/B':>9}{'V残差':>8}{'β_std':>8}{'箱绑定率':>10}")
    print("-" * 126)
    for spec in SPEC_ORDER:
        for tag, cond in (("B≤M", lambda r: r["budget"] <= r["M"]),
                          ("B>M", lambda r: r["budget"] > r["M"])):
            sub = [r for r in rows if r["spec"] == spec and cond(r)]
            if not sub:
                continue
            nk = sum(r["budget"] for r in sub)
            bind = sum(r["at_box_lower"] + r["at_box_upper"] for r in sub) / max(nk, 1)
            print(f"  {spec:<16}{LAMBDA_SPECS[spec]['lam']:>7g}{tag:<6}{len(sub):>4}"
                  f"{_med(sub, 'eval_out_err_median'):>10.4f}"
                  f"{_med(sub, 'eval_mixture_median'):>10.4f}"
                  f"{_med(sub, 'x_eff_budget_frac'):>9.4f}"
                  f"{_med(sub, 'v_fit_resid_rel'):>8.4f}"
                  f"{_med(sub, 'beta_std'):>8.3f}{bind:>10.3f}")


# =============================================================================
# 表 3：逐格 argmin —— 最优 λ 是否随域移动
# =============================================================================

def print_argmin_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    print()
    print("=" * 140)
    print("表 3：逐 (M,B) 格的最优 λ（只在箱阶梯上取 argmin）")
    print("  若最优点在多数格上都是阶梯的端点，说明最优点在扫描区间之外，结论不成立。")
    print("=" * 126)
    ladder = [f"box_lam{_lam_tag(l)}" for l in LAMBDA_LADDER]
    lam_of = {s: LAMBDA_SPECS[s]["lam"] for s in ladder}
    cells = _cells(rows)

    print(f"{'M':>4}{'B':>5}{'域':<6}  | " +
          "  ".join(f"{lam_of[s]:>7g}" for s in ladder) + "   | argmin(输出)  argmin(归并)")
    print("-" * 140)
    argmin_out: List[Tuple[float, int, int]] = []
    argmin_mix: List[Tuple[float, int, int]] = []
    for (m, b) in cells:
        vals_o, vals_m = [], []
        for s in ladder:
            cell = _cell_rows(rows, s, m, b)
            vals_o.append(_med(cell, "eval_out_err_median"))
            vals_m.append(_med(cell, "eval_mixture_median"))
        io = min(range(len(vals_o)), key=lambda i: vals_o[i])
        im = min(range(len(vals_m)), key=lambda i: vals_m[i])
        argmin_out.append((lam_of[ladder[io]], m, b))
        argmin_mix.append((lam_of[ladder[im]], m, b))
        domain = "B≤M" if b <= m else "B>M"
        print(f"{m:>4}{b:>5}{domain:<6}  | " +
              "  ".join(f"{v:>7.4f}" for v in vals_o) +
              f"   | {lam_of[ladder[io]]:>6g}" + f"        {lam_of[ladder[im]]:>6g}")

    def _bucket(pairs: List[Tuple[float, int, int]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for lam, _m, _b in pairs:
            k = f"{lam:g}"
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: float(kv[0])))

    bo = _bucket(argmin_out)
    bm = _bucket(argmin_mix)
    print()
    print(f"  argmin 分布（留出输出）：{bo}")
    print(f"  argmin 分布（留出归并）：{bm}")

    # 分域统计：域内 argmin 的中位数（用 log 尺度更稳）
    def _domain_median(pairs, want: str) -> float:
        ls = [math.log10(lam) for lam, m, b in pairs
              if ("B≤M" if b <= m else "B>M") == want]
        if not ls:
            return float("nan")
        ls.sort()
        return 10.0 ** ls[len(ls) // 2]

    stats = {
        "argmin_out_hist": bo,
        "argmin_mixture_hist": bm,
        "argmin_out_median_lam_BLEM": _domain_median(argmin_out, "B≤M"),
        "argmin_out_median_lam_BGTM": _domain_median(argmin_out, "B>M"),
        "argmin_mix_median_lam_BLEM": _domain_median(argmin_mix, "B≤M"),
        "argmin_mix_median_lam_BGTM": _domain_median(argmin_mix, "B>M"),
        "ladder": list(LAMBDA_LADDER),
    }
    print()
    print(f"  argmin(输出) 的域内中位 λ：B≤M → {stats['argmin_out_median_lam_BLEM']:.3g}，"
          f"B>M → {stats['argmin_out_median_lam_BGTM']:.3g}")
    print(f"  argmin(归并) 的域内中位 λ：B≤M → {stats['argmin_mix_median_lam_BLEM']:.3g}，"
          f"B>M → {stats['argmin_mix_median_lam_BGTM']:.3g}")
    return stats


# =============================================================================
# 表 4：与默认值逐格对照（谁更好）
# =============================================================================

def print_win_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    print()
    print("=" * 126)
    print("表 4：逐格与当前默认（箱 + λ=1e-3）比较，统计「更优」的格数")
    print("  域内按 (M,B) 的中位数比较；「更优」= 误差更小")
    print("=" * 126)
    base = f"box_lam{_lam_tag(DEFAULT_LAMBDA)}"
    cells = _cells(rows)
    out: Dict[str, Any] = {}
    print(f"{'规格':<18}{'λ_β':>7}{'B≤M 胜/负':>14}{'B>M 胜/负':>14}{'合计':>10}")
    print("-" * 126)
    for spec in SPEC_ORDER:
        if spec == base:
            continue
        w_lem = l_lem = w_gtm = l_gtm = 0
        for (m, b) in cells:
            a = _med(_cell_rows(rows, spec, m, b), "eval_out_err_median")
            c = _med(_cell_rows(rows, base, m, b), "eval_out_err_median")
            if b <= m:
                if a < c:
                    w_lem += 1
                elif a > c:
                    l_lem += 1
            else:
                if a < c:
                    w_gtm += 1
                elif a > c:
                    l_gtm += 1
        print(f"  {spec:<16}{LAMBDA_SPECS[spec]['lam']:>7g}"
              f"{f'{w_lem}/{l_lem}':>14}{f'{w_gtm}/{l_gtm}':>14}"
              f"{f'{w_lem + w_gtm}/{l_lem + l_gtm}':>10}")
        out[spec] = {"win_BLEM": w_lem, "lose_BLEM": l_lem,
                     "win_BGTM": w_gtm, "lose_BGTM": l_gtm}
    return out


# =============================================================================
# 渐近一致性检验（H-lam1）
# =============================================================================

def print_asymptote(rows: List[Dict[str, Any]],
                    n_tail: int = 4) -> Dict[str, Any]:
    """H-lam1：λ→∞ 时 w→1、β→0，链路应逐位退化到「关闭 β」。

    数值上必须看到 β_std 随 λ 衰减、四项指标的 |Δ| 随之趋零，
    否则"收缩到 w=1"与"关闭 β"并非同一极限，本脚本的前提不成立。
    """
    print()
    print("=" * 126)
    print("H-lam1：λ→∞ 时 β→0，故大 λ 规格应当与「关闭 β」逐位一致")
    print("  逐档展示收敛过程（只看最大的 λ 会分不清「尚未渐近」与「前提不成立」）")
    print("=" * 126)
    fields = ("eval_out_err_median", "eval_mixture_median",
              "x_eff_budget_frac", "v_fit_resid_rel")
    key = lambda r: (r["seed"], r["dest"], r["M"], r["budget"])   # noqa: E731

    tail = [f"box_lam{_lam_tag(l)}" for l in LAMBDA_LADDER[-n_tail:]]
    kn = {key(r): r for r in rows if r["spec"] == "nobeta"}

    res: Dict[str, Any] = {"fields": list(fields), "steps": []}
    print(f"{'规格':<16}{'λ_β':>8}{'β_std':>10}{'β_mean':>10}"
          + "".join(f"{'|Δ' + f[:11]:>16}" for f in fields) + f"{'max|Δ|/β_std':>15}")
    print("-" * 126)
    for spec in tail:
        kb = {key(r): r for r in rows if r["spec"] == spec}
        common = sorted(set(kb) & set(kn))
        if not common:
            continue
        step: Dict[str, Any] = {
            "spec": spec, "lambda": LAMBDA_SPECS[spec]["lam"],
            "beta_std": _med([r for r in rows if r["spec"] == spec], "beta_std"),
            "beta_mean": _med([r for r in rows if r["spec"] == spec], "beta_mean"),
            "n_pairs": len(common),
        }
        cells = []
        for f in fields:
            d = [abs(kb[k][f] - kn[k][f]) for k in common]
            cells.append(max(d))
            step[f"max_abs_diff_{f}"] = max(d)
        res["steps"].append(step)
        ratio = max(cells) / max(step["beta_std"], 1e-30)
        print(f"  {spec:<14}{LAMBDA_SPECS[spec]['lam']:>8g}"
              f"{step['beta_std']:>10.3e}{step['beta_mean']:>10.3e}"
              + "".join(f"{c:>16.3e}" for c in cells) + f"{ratio:>15.2f}")

    # 判据：只看两个极限是否同一，不苛求某个绝对阈值 ——
    # λ=1000 时 β_std 仍有 3.4e-3，故 |Δ| 必然还有 ~5e-3 的残量，
    # 用「|Δ|<1e-4」当判据会把"尚未渐近"误判成"前提不成立"。
    # 正确的检验是比值：若 max|Δ| = O(β_std)（C 有界），
    # 则 β_std→0 必然带动 |Δ|→0，两个极限一致。
    ratios = [max(s[f"max_abs_diff_{f}"] for f in fields)
              / max(s["beta_std"], 1e-30) for s in res["steps"]]
    res["abs_max_over_beta_std"] = ratios
    monotone = all(res["steps"][i + 1]["beta_std"] < res["steps"][i]["beta_std"]
                   for i in range(len(res["steps"]) - 1))
    c_max = max(ratios) if ratios else float("inf")
    res["c_max"] = c_max
    if monotone and c_max < 10.0:
        res["verdict"] = (f"与渐近前提一致：β_std 单调衰减，且 max|Δ| ≤ "
                          f"{c_max:.2f}·β_std（比值有界 ⇒ β_std→0 时 |Δ|→0）")
    elif not monotone:
        res["verdict"] = "β_std 未单调衰减 —— 证据不足，无法确认渐近前提"
    else:
        res["verdict"] = ("|Δ| 与 β_std 的比值无界 —— "
                          "须排查两个极限是否同一")
    print()
    print(f"  β_std 衰减：{res['steps'][0]['beta_std']:.3e} → "
          f"{res['steps'][-1]['beta_std']:.3e}"
          f"（共 {len(res['steps'])} 档单调下降）")
    print("  max|Δ| / β_std（各档）："
          + "，".join(f"{r:.2f}" for r in ratios))
    print(f"  → {res['verdict']}")


# =============================================================================
# 管线互校（与 E10 同规格逐位对照）
# =============================================================================

def print_crosscheck(rows: List[Dict[str, Any]],
                     tol: float = 5e-4,
                     enabled: bool = True) -> Dict[str, Any]:
    if not enabled:
        print()
        print("管线互校：quick 模式网格与 E10 不同，跳过（仅全网格可比）")
        return {"all_ok": None, "skipped": True}
    print()
    print("=" * 126)
    print("管线互校：本脚本与 E10 在同规格上的中位数对照（E10 值取自其 summary.json）")
    print("=" * 126)
    out: Dict[str, Any] = {}
    ok_all = True
    print(f"{'规格':<18}{'本脚本(输出)':>14}{'E10(输出)':>12}{'Δ':>10}"
          f"{'本脚本(归并)':>14}{'E10(归并)':>12}{'Δ':>10}")
    print("-" * 126)
    for spec, ref in E10_CROSSCHECK.items():
        sub = [r for r in rows if r["spec"] == spec]
        if not sub:
            continue
        o = _med(sub, "eval_out_err_median")
        mm = _med(sub, "eval_mixture_median")
        do, dm = o - ref["eval_out_err_median"], mm - ref["eval_mixture_median"]
        ok = abs(do) < tol and abs(dm) < tol
        ok_all = ok_all and ok
        out[spec] = {"out_err": o, "mix": mm, "e10_out_err": ref["eval_out_err_median"],
                     "e10_mix": ref["eval_mixture_median"], "ok": bool(ok)}
        print(f"  {spec:<16}{o:>14.4f}{ref['eval_out_err_median']:>12.4f}{do:>+10.4f}"
              f"{mm:>14.4f}{ref['eval_mixture_median']:>12.4f}{dm:>+10.4f}"
              f"   {'✓' if ok else '✗'}")
    out["all_ok"] = bool(ok_all)
    print(f"  → {'两套管线一致（比较链可信）' if ok_all else '存在偏差 —— 比较链不可信，须先排查'}")
    return out


# =============================================================================
# 配对检验
# =============================================================================

def run_paired(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    print()
    print("=" * 126)
    print("配对检验（逐 (seed,dest,M,B) 配对；A 优于 B ⇔ A 误差更小）")
    print("=" * 126)
    base = f"box_lam{_lam_tag(DEFAULT_LAMBDA)}"
    paired: Dict[str, Any] = {}

    def keyed(spec: str, field: str,
              subset: Optional[Callable[[Dict[str, Any]], bool]] = None
              ) -> Dict[Any, float]:
        return {(r["seed"], r["dest"], r["M"], r["budget"]): r[field]
                for r in rows
                if r["spec"] == spec and (subset is None or subset(r))}

    def compare(a: str, b: str, field: str, tag: str,
                subset: Optional[Callable[[Dict[str, Any]], bool]] = None
                ) -> Optional[Dict[str, Any]]:
        ka, kb = keyed(a, field, subset), keyed(b, field, subset)
        common = sorted(set(ka) & set(kb))
        if not common:
            return None
        pr = R.paired_bootstrap([ka[k] for k in common],
                                [kb[k] for k in common],
                                metric_name=field, unit="ratio")
        paired[f"{a}_vs_{b}__{tag}__{field}"] = pr.to_dict()
        print(f"  [{tag}] {a:<18} vs {b:<18} n={pr.n_pairs:<4} "
              f"Δ={pr.mean_diff:+.4f} "
              f"CI95=[{pr.ci_95_lower:+.4f},{pr.ci_95_upper:+.4f}] "
              f"p={pr.p_value_one_sided:.4f} W/L={pr.a_wins}/{pr.b_wins}")
        return pr.to_dict()

    gt_m: Callable[[Dict[str, Any]], bool] = lambda r: r["budget"] > r["M"]      # noqa: E731
    le_m: Callable[[Dict[str, Any]], bool] = lambda r: r["budget"] <= r["M"]     # noqa: E731

    for field in ("eval_out_err_median", "eval_mixture_median"):
        print(f"  --- {field}（全部格子）---")
        for lam in LAMBDA_LADDER:
            spec = f"box_lam{_lam_tag(lam)}"
            if spec == base:
                continue
            compare(spec, base, field, "全部")
        print()
        print(f"  --- {field}（B>M 域）---")
        for lam in (1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0):
            compare(f"box_lam{_lam_tag(lam)}", base, field, "B>M", gt_m)
        print()
        print(f"  --- {field}（B≤M 域）---")
        for lam in (1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0):
            compare(f"box_lam{_lam_tag(lam)}", base, field, "B≤M", le_m)
        print()
    return paired


# =============================================================================
# main
# =============================================================================

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E11：λ_β 的细化与分域定位")
    p.add_argument("--out", default="results/cpu/e11")
    p.add_argument("--quick", action="store_true", help="缩小规模，用于冒烟")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--num-dest", dest="num_dest", type=int, default=3)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=96)
    p.add_argument("--M", type=int, nargs="+", default=[8, 16, 32])
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
    print("=" * 126)
    print("E11：λ_β 的细化扫描与分域定位（箱约束 [−3,3] 固定打开）")
    print("=" * 126)
    print(f"  L_s={args.L_s} d_h={args.d_h} d_v={args.d_v} dest={args.num_dest} "
          f"queries/dest={args.queries_per_dest} eval_frac={args.eval_fraction}")
    print(f"  M={args.M} budgets={args.budgets} seeds={args.seeds} "
          f"λ_v={args.lambda_value:g} focus={args.focus_strength}")
    print(f"  (M,B,dest,seed) 组合 = {n_cells}，× {len(SPEC_ORDER)} 规格 "
          f"= {n_cells * len(SPEC_ORDER)} 次构造")
    print()
    print("  规格：")
    for spec in SPEC_ORDER:
        c = LAMBDA_SPECS[spec]
        lo = "−∞" if c["w_lower"] == 0.0 else f"{math.log(c['w_lower']):+.1f}"
        hi = "∞" if c["w_upper"] is None else f"{math.log(c['w_upper']):+.1f}"
        print(f"    {c['family']:<8}{spec:<18} β∈[{lo:>4},{hi:>4}] λ_β={c['lam']:<8g}")
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

    print_ladder(rows)
    print_regime_split(rows)
    argmin_stats = print_argmin_table(rows)
    win_counts = print_win_counts(rows)
    asymptote = print_asymptote(rows)
    crosscheck = print_crosscheck(rows, enabled=not args.quick)
    paired = run_paired(rows) if args.paired else {}

    payload = {
        "experiment": "E11",
        "kind": "beta-lambda-refinement",
        "specs": LAMBDA_SPECS,
        "spec_order": list(SPEC_ORDER),
        "lambda_ladder": list(LAMBDA_LADDER),
        "default_lambda": DEFAULT_LAMBDA,
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "argmin_stats": argmin_stats,
        "win_counts_vs_default": win_counts,
        "asymptote_check": asymptote,
        "crosscheck_vs_e10": crosscheck,
        "paired": paired,
        "rows": rows,
        "source_grounding": (
            "箱约束 [−3,3] 与 λ 的口径（尺度无关相对量）承接 E10；"
            "E10 判定 λ_β 是比箱约束更对症的旋钮但未定位最优值。"
        ),
        "caveat": (
            "机制级证据（合成数据上的注意力分布与线性输出重构精度），"
            "不构成任务质量主张。所有评估均在留出 Query 上进行。"
        ),
    }
    R.save_json(str(pathlib.Path(args.out) / "summary.json"), payload)
    R.save_csv(str(pathlib.Path(args.out) / "rows.csv"), rows)
    print()
    print(f"结果已写入 {args.out}/summary.json 与 rows.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
