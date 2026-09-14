#!/usr/bin/env python
"""E2b：β 链路的口径判定（CPU 可跑）

要回答的问题
------------
仓库里 β 的拟合与施加存在两处不一致（详见 `experiments/common/beta_variants.py`
的模块文档）：

1. 质量目标退化为常数 1（`compact_kv.py:101` 的 `softmax().sum(-1) ≡ 1`）；
2. 拟合侧 β/M、推理侧 β，相差 M 倍。

本脚本把四种候选口径都跑一遍，用**留出 Query** 上的实测误差判定哪一种最合理：

============  ==================  ==========  ==========
预设          质量目标             拟合系数     推理系数
============  ==================  ==========  ==========
``legacy``    Σ softmax ≡ 1      1/M         1
``shift``     Σ softmax ≡ 1      1/M         1/M
``am``        Σ exp（未归一化）   1           1
``am_over_m`` Σ exp（未归一化）   1/M         1/M
============  ==================  ==========  ==========

判据（三条同时看，缺一不可）
----------------------------
1. **质量误差**：β 的作用就是恢复被压缩块丢掉的质量。若某口径下
   `|signed_mass_error|` 不显著小于 β 关闭时，说明 β 没起作用。
2. **输出误差**：留出 Query 上的相对输出误差中位数。
3. **β 是否落到边界**：`legacy` 下 β 被 clamp 到 log(1e-6) = −13.8155，
   是"目标退化 ⇒ 拟合无解 ⇒ 被正则项推向 0 ⇒ 取 log 后顶到下界"的直接证据。
   一个健康的 β 应当接近 **log(L_s/B)** —— 即"少量 Key 需要被放大多少倍
   才能补上总质量"，B=8、L_s=256 时约 +3.47。

对每个口径还报告 JS 散度：相比 KL，它不会在 B ≪ L_s 时饱和，
因此能区分"略有不同"与"完全不同"。

诚实边界
--------
本脚本只产生**机制级**证据。任何数字都不得用于主张任务质量或通信性能。
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any, Dict, List

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import synthetic as S          # noqa: E402
from experiments.common import report as R             # noqa: E402
from experiments.common import beta_variants as BV     # noqa: E402
from src.dcc_kv_ref import DEFAULT_LAMBDA_BETA         # noqa: E402


# =============================================================================
# 单次评估
# =============================================================================

def evaluate_preset(
    split: S.HeldoutSplit,
    dest: int,
    budget: int,
    preset: str,
    num_repr: int,
    projection_dim: int,
    seed: int,
    lambda_beta: float = DEFAULT_LAMBDA_BETA,
    fixed_ctx: Any = None,
) -> Dict[str, Any]:
    """在单个（目的端, 预算）上按指定口径构造并评估。"""
    cfg = BV.PRESETS[preset]

    compact, diag = BV.build_compact_kv_variant(
        source_keys=split.keys,
        source_values=split.values,
        destination_queries=split.fit_queries[dest],
        budget=budget,
        preset=preset,
        num_representative_queries=num_repr,
        projection_dim=projection_dim,
        lambda_beta=lambda_beta,
        seed=seed,
    )

    eval_q = split.eval_queries[dest]
    # 推理侧的 β 系数由 preset 决定；这里复用 synthetic 的 beta_mode 通道
    beta_mode = "full" if cfg["apply"] == "full" else "over_m"

    rel = S.relative_output_error(
        eval_q, compact, split.keys, split.values,
        beta_mode=beta_mode, num_repr_queries=num_repr,
    )
    sme = S.signed_mass_error(
        eval_q, compact, split.keys,
        beta_mode=beta_mode, num_repr_queries=num_repr,
    )

    # JS：紧凑块诱导分布 vs 完整块分布
    p_compact = S.induced_distribution(
        compact, eval_q, split.L_s,
        beta_mode=beta_mode, num_repr_queries=num_repr,
    )
    p_dense = S.dense_attention_weights(eval_q, split.keys)
    js = S.jensen_shannon_divergence(p_compact, p_dense)

    # β 的主要职责是"补回块质量"，而这只在与其他块归并时才可见 ——
    # 单块 softmax 对 β 的常数分量免疫。故必须补两个归并侧度量。
    abs_mass = S.absolute_mass_error(
        eval_q, compact, split.keys,
        beta_mode=beta_mode, num_repr_queries=num_repr,
    )
    if fixed_ctx is not None:
        FK, FV = fixed_ctx
        mix = S.mixture_relative_error(
            eval_q, FK, FV, compact, split.keys, split.values,
            beta_mode=beta_mode, num_repr_queries=num_repr,
        )
        mix_med = float(mix.median().item())
    else:
        mix_med = float("nan")

    row: Dict[str, Any] = {
        "preset": preset,
        "seed": seed,
        "dest": dest,
        "budget": budget,
        "M": num_repr,
        "L_s": split.L_s,
        "rel_out_err_median": float(rel.median().item()),
        "rel_out_err_mean": float(rel.mean().item()),
        "signed_mass_median": float(sme.median().item()),
        "abs_signed_mass_median": float(sme.abs().median().item()),
        "js_median": float(js.median().item()),
        "mixture_rel_err_median": mix_med,
        "abs_mass_common_median": float(abs_mass.median().item()),
        "beta_mean": diag["beta_mean"],
        "beta_min": diag["beta_min"],
        "beta_max": diag["beta_max"],
        "clamp_hits": diag["clamp_hits"],
        "mass_target_mean": diag["mass_target_mean"],
        "mass_target_std": diag["mass_target_std"],
        "needed_compensation_median": diag["needed_compensation_median"],
        "needed_compensation_spread": diag["needed_compensation_spread"],
        "solver_residual": diag["solver_residual"],
        "rank_upper_bound": diag["rank_upper_bound"],
        "beta_ideal_mass_preserving": math.log(split.L_s / float(budget)),
    }
    return row


# =============================================================================
# main
# =============================================================================

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E2b：β 链路口径判定")
    p.add_argument("--out", default="results/cpu/e2b")
    p.add_argument("--quick", action="store_true", help="缩小规模，用于冒烟")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--num-dest", dest="num_dest", type=int, default=3)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=96)
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--num-repr", dest="num_repr", type=int, default=16)
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=16)
    p.add_argument("--eval-fraction", dest="eval_fraction", type=float, default=0.5)
    p.add_argument("--focus-strength", dest="focus_strength", type=float, default=8.0)
    p.add_argument("--fixed-len", dest="fixed_len", type=int, default=128,
                   help="归并实验里固定上下文块的长度")
    p.add_argument("--lambda-beta", dest="lambda_beta", type=float,
                   default=DEFAULT_LAMBDA_BETA,
                   help="β 拟合的岭正则强度 λ_β（默认取 src 的 DEFAULT_LAMBDA_BETA）")
    p.add_argument("--no-paired", dest="paired", action="store_false")
    args = p.parse_args(argv)

    if args.quick:
        args.seeds, args.L_s, args.num_dest = 2, 128, 2
        args.queries_per_dest, args.budgets = 64, [8, 16]
        args.num_repr, args.projection_dim = 16, 16

    print("=" * 78)
    print("E2b：β 链路口径判定")
    print("=" * 78)
    print(f"  L_s={args.L_s} d_h={args.d_h} d_v={args.d_v} dest={args.num_dest} "
          f"queries/dest={args.queries_per_dest} eval_frac={args.eval_fraction}")
    print(f"  M={args.num_repr} d_p={args.projection_dim} budgets={args.budgets} "
          f"seeds={args.seeds} focus={args.focus_strength} "
          f"λ_β={args.lambda_beta:g}")
    print("  β 箱约束 = 无（E2b 只判定**口径**，故不启用箱约束以隔离该变量；"
          "箱约束与 λ_β 的判定分别在 E10 与 E11）")
    print(f"  β 的理想量级（质量保持）：log(L_s/B) = "
          f"{[round(math.log(args.L_s / b), 3) for b in args.budgets]}")
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
            for budget in args.budgets:
                for preset in BV.PRESET_ORDER:
                    rows.append(evaluate_preset(
                        split, dest, budget, preset,
                        args.num_repr, args.projection_dim, seed,
                        lambda_beta=args.lambda_beta,
                        fixed_ctx=fixed_ctx,
                    ))

    # ---- 汇总 ----
    all_presets = list(BV.PRESET_ORDER)
    print("-" * 78)
    print(f"{'口径':<20} {'单块err':>10} {'归并err':>10} {'JS':>10} "
          f"{'|mass|(自移)':>12} {'|mass|(公共)':>12} {'β_mean':>9} {'clamp':>6}")
    print("-" * 78)
    summary: Dict[str, Dict[str, float]] = {}
    for preset in all_presets:
        sub = [r for r in rows if r["preset"] == preset]
        if not sub:
            continue
        rel = R.summarize([r["rel_out_err_median"] for r in sub], "rel_out_err")
        mass = R.summarize([r["abs_signed_mass_median"] for r in sub], "abs_mass_err")
        js = R.summarize([r["js_median"] for r in sub], "js")
        bmean = sum(r["beta_mean"] for r in sub) / len(sub)
        bmin = min(r["beta_min"] for r in sub)
        bmax = max(r["beta_max"] for r in sub)
        clamp = sum(r["clamp_hits"] for r in sub)
        n_entries = sum(r["budget"] for r in sub)
        summary[preset] = {
            "rel_out_err_median": rel.median,
            "rel_out_err_p5": rel.p5,
            "rel_out_err_p95": rel.p95,
            "abs_mass_err_median": mass.median,
            "js_median": js.median,
            "beta_mean": bmean,
            "beta_min": bmin,
            "beta_max": bmax,
            "clamp_hits_total": clamp,
            "clamp_hit_rate": clamp / n_entries if n_entries else 0.0,
            "n": len(sub),
        }
        mixv = R.summarize([r["mixture_rel_err_median"] for r in sub], "mixture")
        absm = R.summarize([r["abs_mass_common_median"] for r in sub], "abs_mass")
        summary[preset]["mixture_rel_err_median"] = mixv.median
        summary[preset]["abs_mass_common_median"] = absm.median
        print(f"{preset:<20} {rel.median:>10.4e} {mixv.median:>10.4e} "
              f"{js.median:>10.4e} {mass.median:>12.4e} {absm.median:>12.4e} "
              f"{bmean:>9.3f} {clamp:>6d}")

    # ---- 逐预算看：β 是否随 B 变小而变大（质量补偿的定性预测） ----
    print()
    print("-" * 78)
    print("β_mean 随预算的变化（质量保持口径应 ≈ log(L_s/B)）")
    print("-" * 78)
    print(f"{'预算 B':>8} {'log(L_s/B)':>12} " +
          " ".join(f"{p:>12}" for p in BV.PRESET_ORDER))
    for budget in args.budgets:
        ideal = math.log(args.L_s / budget)
        cells = []
        for preset in BV.PRESET_ORDER:
            sub = [r for r in rows if r["preset"] == preset and r["budget"] == budget]
            cells.append(sum(r["beta_mean"] for r in sub) / len(sub) if sub else float("nan"))
        print(f"{budget:>8} {ideal:>12.3f} " +
              " ".join(f"{c:>12.3f}" for c in cells))

    # ---- 配对检验：逐 (seed, dest, budget) 配对 ----
    paired: Dict[str, Any] = {}
    if args.paired:
        print()
        print("-" * 78)
        print("配对比较（rel_out_err，逐 (seed, dest, budget) 配对，越小越好）")
        print("-" * 78)

        def keyed(preset: str, metric: str = "single") -> Dict[Any, float]:
            field = ("mixture_rel_err_median" if metric == "mixture"
                     else "rel_out_err_median")
            return {
                (r["seed"], r["dest"], r["budget"]): r[field]
                for r in rows if r["preset"] == preset
            }

        def compare(name_a: str, name_b: str, metric: str = "single") -> None:
            """A 是否优于 B（配对）。"""
            ka, kb = keyed(name_a, metric), keyed(name_b, metric)
            common = sorted(set(ka) & set(kb))
            if not common:
                return
            a = [ka[k] for k in common]
            b = [kb[k] for k in common]
            pr = R.paired_bootstrap(a, b, metric_name="rel_out_err", unit="ratio")
            # 键里必须带 metric（2026-09-13 修正）：旧写法只有
            # f"{name_a}_vs_{name_b}"，而同一对预设会在"单块"与"归并"两轮里
            # 各写一次，后一轮直接把前一轮覆盖 —— 于是落盘的 paired 里
            # 只剩单块结果，"归并侧 44/1"这类结论再也无法从产物复现。
            # 复现方式：从 rows.csv 按 (seed, dest, budget) 配对重算。
            paired[f"{name_a}_vs_{name_b}__{metric}"] = pr.to_dict()
            print(f"  {name_a:<20} vs {name_b:<12} n={pr.n_pairs:<4} "
                  f"mean_diff={pr.mean_diff:+.4e} "
                  f"CI95=[{pr.ci_95_lower:+.3e},{pr.ci_95_upper:+.3e}] "
                  f"p={pr.p_value_one_sided:.4f}  "
                  f"win/lose/tie={pr.a_wins}/{pr.b_wins}/{pr.ties}")

        print("  [单块误差 rel_out_err]")
        # 关键对照 1：新口径是否优于仓库现状（回答"该不该改"）
        for preset in ("shift", "am", "am_over_m"):
            compare(preset, "legacy")
        print()
        print("  [归并误差 mixture_rel_err —— β 的主战场]")
        for preset in BV.PRESET_ORDER:
            compare(preset, "am_nobeta", metric="mixture")

        print()
        print("  [单块误差视角：β 的跨 Key 离散度效应]")
        for preset in BV.PRESET_ORDER:
            compare(preset, "am_nobeta")

    # ---- 结构性诊断：静态 β 在原理上能不能救回质量 ----
    print()
    print("-" * 78)
    print("结构性诊断：匹配质量所需的逐 query 乘性补偿 c_a = m_a / Σ_j exp(ℓc_a,j)")
    print("静态 β 只能做到'跨 Key 同一组固定抬升'；c_a 跨 query 越分散，β 越不可能有效。")
    print("-" * 78)
    print(f"{'预算 B':>8} {'B 口径':>10} {'c_median':>10} {'c_spread(p95/p5)':>18} "
          f"{'→ 静态 β 可行?':>16}")
    for budget in args.budgets:
        for preset in ("legacy", "am", "am_logfit"):
            sub = [r for r in rows
                   if r["preset"] == preset and r["budget"] == budget]
            if not sub:
                continue
            cm = sum(r["needed_compensation_median"] for r in sub) / len(sub)
            cs = sum(r["needed_compensation_spread"] for r in sub) / len(sub)
            verdict = "可行" if cs < 2.0 else ("勉强" if cs < 5.0 else "不可行")
            print(f"{budget:>8} {preset:>10} {cm:>10.3f} {cs:>18.3f} "
                  f"{verdict:>16}")

    # ---- 判定提示（只陈述数据允许的结论） ----
    print()
    print("-" * 78)
    print("判定依据（不做超出数据的断言）")
    print("-" * 78)
    am = summary.get("am", {})
    lg = summary.get("legacy", {})
    nobe = summary.get("am_nobeta", {})
    sca = summary.get("am_scalar", {})
    if am and lg and nobe:
        print(f"  {'口径':<20} {'归并误差':>12} {'|mass|(公共)':>15}")
        for nm in ("legacy", "legacy_fixed_solver", "shift", "am", "am_logfit",
                   "am_scalar", "am_nobeta"):
            r = summary.get(nm)
            if r:
                print(f"  {nm:<20} {r['mixture_rel_err_median']:>12.4e} "
                      f"{r['abs_mass_common_median']:>15.4e}")
        print()
        # 1. 现状 vs 正确口径（决定要不要改代码）
        p_fix = paired.get("am_vs_legacy__single") or {}
        if p_fix:
            print(f"  · 正确口径 vs 仓库现状（单块误差）："
                  f"mean_diff={p_fix['mean_diff']:+.4e}, p={p_fix['p_value_one_sided']:.4f}, "
                  f"{p_fix['a_wins']}/{p_fix['b_wins']} 配对"
                  f"  → {'应当修' if p_fix['p_value_one_sided'] < 0.05 else '无需改'}")
        # 2. β 在归并侧是否有效（β 的主战场）
        p_mix = paired.get("am_vs_am_nobeta__mixture", {})
        print()
        print("  β 的效应（归并侧 = β 的主战场）：")
        for nm in ("legacy", "shift", "am", "am_logfit", "am_scalar"):
            pm = paired.get(f"{nm}_vs_am_nobeta__mixture")
            if not pm:
                continue
            # 这里 am_nobeta 是 B，nm 是 A；A 更优 => mean_diff < 0
            if pm["mean_diff"] < 0 and pm["p_value_one_sided"] < 0.05:
                tag = "显著更优（β 生效）"
            elif pm["mean_diff"] > 0 and pm["p_value_one_sided"] > 0.95:
                tag = "显著更差（β 有害）"
            else:
                tag = "无显著差异"
            print(f"    {nm:<20} vs β 全关闭: mean_diff={pm['mean_diff']:+.4e} "
                  f"p={pm['p_value_one_sided']:.4f} "
                  f"win/lose={pm['a_wins']}/{pm['b_wins']}  → {tag}")
        print()
        print("  结论性提示（仅陈述数据允许的部分）：")
        print("    · β 的常数分量在单块 softmax 中会被完全抵消，因此'单块误差'")
        print("      天然对 β 主效应免疫 —— 判断 β 必须看归并侧。")
        if sca and nobe:
            ratio = (nobe["abs_mass_common_median"]
                     / max(sca["abs_mass_common_median"], 1e-30))
            print(f"    · 绝对质量误差：β 全关闭 {nobe['abs_mass_common_median']:.4f}"
                  f" → 标量 β {sca['abs_mass_common_median']:.4f}"
                  f"（改善 {ratio:.2f}×）")

    payload = {
        "experiment": "E2b",
        "kind": "beta-convention-decision",
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "beta_bound": None,
        "beta_bound_note": (
            "E2b 只判定 β 的**口径**（质量目标与系数），故不启用箱约束，"
            "以免把稳定性约束混入口径对照；箱约束的效果见 E10，"
            "λ_β 的取值见 E11。"
        ),
        "summary": summary,
        "paired": paired,
        "rows": rows,
        "caveat": (
            "机制级证据（合成数据上的注意力分布与线性输出重构精度），"
            "不构成任务质量主张。所有评估均在**留出 Query** 上进行，"
            "留出集合从未参与 Key 选择、β 拟合或 V 回归。"
        ),
    }
    R.save_json(str(pathlib.Path(args.out) / "summary.json"), payload)
    R.save_csv(str(pathlib.Path(args.out) / "rows.csv"), rows)
    print()
    print(f"结果已写入 {args.out}/summary.json 与 rows.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
