#!/usr/bin/env python
"""E9：旋钮定位 —— 信息瓶颈在 B（预算）还是 M（代表 Query 数）？（CPU 可跑）

问题来源
--------
运行 E2/E3 时发现 Value 回归的设计矩阵 X 只有 M 行（M = 代表 Query 数），
因此 `rank(X) <= M`。当预算 B > M 时 `XᵀX` 奇异，解完全由正则项决定 ——
也就是说，**加大预算 B 并不增加可辨识的信息**。若这一点成立，
论文把「扫描 5 档预算画帕累托前沿」当作主线，就是在优化一个次要旋钮。

本脚本不做这种断言，而是把它变成可测量的对照。

设计
----
网格：M ∈ {4, 8, 16, 32, 48} × B ∈ {8, 16, 32, 64, 128}，每格多个种子。

三条曲线（全部在**留出 Query** 上评估，留出集从未参与选键 / 拟合）：

1. ``fit``     ：DCC-KV 正常链路（用 M 个代表 Query 拟合 V）。
2. ``oracle``  ：**固定同一套选中的 Key 与同一个 β**，但用**评估 Query 自己**
                 去拟合 V。它作弊，因此给出"在预算 B 下最好的可达精度"。
3. ``gap``     ：fit − oracle。这个差就是「M 不够大导致的估计误差」——
                 与 B 无关，只由 M 决定。

于是两个轴的分工变得可分辨：
- **B 轴** 通过 oracle 曲线体现：B 越大，可达上限越高（表示能力）。
- **M 轴** 通过 gap 体现：M 越大，越接近那个上限（估计能力）。

另报三个诊断：
- ``rank_upper_bound = min(M, B)`` 与 `XᵀX + λI` 的条件数；
- 选键质量占比（B/L_s 是名义比，RMS 选键保留的质量远高于它）；
- 沿每个轴的误差变化幅度（归一化），作为"哪个旋钮更有效"的量化。

诚实边界
--------
合成数据 + 机制级指标（注意力分布与线性输出重构精度），
**不构成任何任务质量主张**。真实模型上的结论需 GPU 侧实验（E5/E6）。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import synthetic as S       # noqa: E402
from experiments.common import report as R          # noqa: E402
from src.dcc_kv_ref.value_regression import ridge_regression_value  # noqa: E402


# =============================================================================
# 评估
# =============================================================================

def oracle_value(
    compact,
    fit_queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    lambda_value: float = 1e-3,
):
    """在**固定**的选中 Key 与 β 下，用给定 Query 重新拟合 V。

    传入评估 Query 时它就是"作弊上界"：它回答"若 V 拟合得完美，这个预算
    能达到多好"。传入代表 Query 时则复现正常链路。
    """
    scale = 1.0 / (compact.keys.shape[-1] ** 0.5)
    logits_c = (fit_queries @ compact.keys.T) * scale + compact.logit_bias
    X = torch.softmax(logits_c, dim=-1)
    logits_o = (fit_queries @ keys.T) * scale
    A_o = torch.softmax(logits_o, dim=-1)
    Y = A_o @ values
    C_v = ridge_regression_value(X, Y, lambda_reg=lambda_value)
    from src.dcc_kv_ref import CompactKV
    return CompactKV(keys=compact.keys, logit_bias=compact.logit_bias,
                     values=C_v, selected_indices=compact.selected_indices)


def conditioning(compact, probe: torch.Tensor) -> Tuple[float, int]:
    """返回 (XᵀX + λI 的条件数, X 的数值秩)。"""
    scale = 1.0 / (compact.keys.shape[-1] ** 0.5)
    logits_c = (probe @ compact.keys.T) * scale + compact.logit_bias
    X = torch.softmax(logits_c, dim=-1)
    B = X.shape[1]
    A = X.T @ X + 1e-3 * torch.eye(B, dtype=X.dtype)
    ev = torch.linalg.eigvalsh(A)
    cond = float((ev.max() / ev.clamp(min=1e-300).min()).item())
    rank = int(torch.linalg.matrix_rank(X).item())
    return cond, rank


def retained_mass_fraction(compact, probe, keys) -> float:
    """选中 Key 集合真实承载的注意力质量占完整块的比例（用完整 softmax 权重）。"""
    w = S.dense_attention_weights(probe, keys)          # [N, L_s]
    idx = compact.selected_indices.long()
    return float(w[:, idx].sum(dim=-1).mean().item())


def evaluate_cell(
    split: S.HeldoutSplit,
    dest: int,
    M: int,
    B: int,
    seed: int,
    projection_dim: int,
    fixed_ctx: Any,
    beta_bound: Any,
    lambda_beta: float,
) -> Dict[str, Any]:
    """评估一个 (M, B) 格。"""
    compact, diag = _build(split, dest, M, B, seed, projection_dim,
                           beta_bound, lambda_beta)

    ev = split.eval_queries[dest]
    rel = S.relative_output_error(ev, compact, split.keys, split.values)
    abs_mass = S.absolute_mass_error(ev, compact, split.keys)
    mix = S.mixture_relative_error(ev, fixed_ctx[0], fixed_ctx[1],
                                   compact, split.keys, split.values)

    # oracle：同一套 Key/β，用评估 Query 自己拟合 V
    oracle = oracle_value(compact, ev, split.keys, split.values)
    rel_or = S.relative_output_error(ev, oracle, split.keys, split.values)
    mix_or = S.mixture_relative_error(ev, fixed_ctx[0], fixed_ctx[1],
                                      oracle, split.keys, split.values)

    cond, rank = conditioning(compact, split.fit_queries[dest])
    kept = retained_mass_fraction(compact, ev, split.keys)

    return {
        "seed": seed, "dest": dest, "M": M, "B": B,
        "L_s": split.L_s,
        "nominal_ratio": B / split.L_s,
        "retained_mass_fraction": kept,
        "rel_err": float(rel.median().item()),
        "mix_err": float(mix.median().item()),
        "abs_mass_err": float(abs_mass.median().item()),
        "rel_err_oracle": float(rel_or.median().item()),
        "mix_err_oracle": float(mix_or.median().item()),
        "gap_rel": float((rel - rel_or).median().item()),
        "gap_mix": float((mix - mix_or).median().item()),
        "cond_xtx": cond,
        "numeric_rank": rank,
        "rank_upper_bound": min(M, B),
        "beta_mean": diag["beta_mean"],
        "beta_clamped": diag["clamp_hits"],
    }


def _build(split, dest, M, B, seed, projection_dim, beta_bound, lambda_beta):
    """构造紧凑 KV：代表 Query 数 = M，预算 = B，β 箱约束 = beta_bound。

    `beta_bound` 与 `lambda_beta` 都必须显式传入（由 main 从 CLI 解析），
    以免脚本悄悄跟随 src 的默认值而在产物里留下不可追溯的口径。
    「显式传入」的判据因此是：本脚本产出的每个数字，所用配置都能在
    summary.json 的 config 里原样读回。
    """
    from src.dcc_kv_ref import build_compact_kv
    compact = build_compact_kv(
        source_keys=split.keys,
        source_values=split.values,
        destination_queries=split.fit_queries[dest],
        budget=B,
        num_representative_queries=M,
        projection_dim=projection_dim,
        seed=seed,
        beta_bound=beta_bound,
        lambda_beta=lambda_beta,
    )
    b = compact.logit_bias
    # 下界随箱约束变化（2026-09-14 修正）：旧写法固定比较 -13.0，
    # 在默认箱约束 β∈[−3,3] 下该比较恒为假，会把"约束是否顶住"掩盖成 0。
    lo = -13.8155 if beta_bound is None else -float(beta_bound)
    return compact, {
        "beta_mean": float(b.mean().item()),
        "beta_std": float(b.std().item()) if b.numel() > 1 else 0.0,
        "beta_min": float(b.min().item()),
        "beta_max": float(b.max().item()),
        "beta_bound": beta_bound,
        "clamp_hits": int((b <= lo + 1e-6).sum().item()),
    }


# =============================================================================
# main
# =============================================================================

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E9：M vs B 旋钮定位")
    p.add_argument("--out", default="results/cpu/e9")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--num-dest", dest="num_dest", type=int, default=3)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=128)
    p.add_argument("--eval-fraction", dest="eval_fraction", type=float, default=0.5)
    p.add_argument("--Ms", type=int, nargs="+", default=[4, 8, 16, 32, 48])
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=32)
    p.add_argument("--fixed-len", dest="fixed_len", type=int, default=128)
    p.add_argument("--focus-strength", dest="focus_strength", type=float, default=8.0)
    p.add_argument("--lambda-beta", dest="lambda_beta", type=float,
                   default=None,
                   help="β 拟合的岭正则强度 λ_β；默认取 src 的 DEFAULT_LAMBDA_BETA")
    p.add_argument("--beta-bound", dest="beta_bound", type=float, default=None,
                   help="β 的箱约束半宽；默认取 src 的 DEFAULT_BETA_BOUND")
    p.add_argument("--no-beta-bound", dest="no_beta_bound", action="store_true",
                   help="关闭箱约束（复现 2026-09-14 之前的结果）")
    args = p.parse_args(argv)

    if args.lambda_beta is None:
        from src.dcc_kv_ref import DEFAULT_LAMBDA_BETA
        args.lambda_beta = DEFAULT_LAMBDA_BETA
    if args.beta_bound is None and not args.no_beta_bound:
        from src.dcc_kv_ref import DEFAULT_BETA_BOUND
        args.beta_bound = DEFAULT_BETA_BOUND
    if args.no_beta_bound:
        args.beta_bound = None

    if args.quick:
        args.seeds, args.L_s, args.num_dest = 2, 128, 2
        args.queries_per_dest, args.eval_fraction = 64, 0.5
        args.Ms, args.budgets = [4, 8, 16], [8, 16, 32]

    max_M = max(args.Ms)
    fit_pool = int(round(args.queries_per_dest * (1 - args.eval_fraction)))
    if max_M > fit_pool:
        print(f"!! 最大 M={max_M} 超过留出后的拟合池 {fit_pool}；请提高 "
              f"--queries-per-dest 或降低 M。")
        return 2

    print("=" * 78)
    print("E9：旋钮定位 —— B（预算）还是 M（代表 Query 数）？")
    print("=" * 78)
    print(f"  L_s={args.L_s} d_h={args.d_h} d_v={args.d_v} dest={args.num_dest}")
    print(f"  β 箱约束 = {'无（β ≥ log(1e-6)）' if args.beta_bound is None else f'[-{args.beta_bound:g}, +{args.beta_bound:g}]'}")
    print(f"  λ_β = {args.lambda_beta:g}")
    print(f"  每目的端 Query={args.queries_per_dest}，留出 {args.eval_fraction:.0%} "
          f"→ 拟合池={fit_pool}，评估池={args.queries_per_dest - fit_pool}")
    print(f"  M ∈ {args.Ms}    B ∈ {args.budgets}    seeds={args.seeds}")
    print(f"  网格点数 = {len(args.Ms)} × {len(args.budgets)} × {args.num_dest} "
          f"× {args.seeds} = {len(args.Ms)*len(args.budgets)*args.num_dest*args.seeds}")
    print("  注意：所有指标都在**留出 Query** 上评估；oracle 曲线用评估 Query "
          "自己拟合 V（作弊上界）。")
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
        fixed_ctx = S.make_fixed_context(args.d_h, args.d_v, args.fixed_len,
                                         seed=1000 + si)
        for dest in range(args.num_dest):
            for M in args.Ms:
                for B in args.budgets:
                    rows.append(evaluate_cell(split, dest, M, B, seed,
                                              args.projection_dim, fixed_ctx,
                                              args.beta_bound, args.lambda_beta))
        print(f"  seed={seed} 完成（累计 {len(rows)} 格）")

    def med(field: str, **cond) -> float:
        sub = [r for r in rows
               if all(r[k] == v for k, v in cond.items())]
        if not sub:
            return float("nan")
        return R.summarize([r[field] for r in sub], field).median

    # ---- 表 1：误差 vs (M, B) ----
    print()
    print("-" * 78)
    print("表 1：留出集上的相对输出误差（fit 链路）     行 = M，列 = B")
    print("-" * 78)
    print(f"{'M\\B':>6} " + " ".join(f"{b:>10}" for b in args.budgets))
    for M in args.Ms:
        cells = [med("rel_err", M=M, B=B) for B in args.budgets]
        print(f"{M:>6} " + " ".join(f"{c:>10.4e}" for c in cells))

    print()
    print("表 2：oracle 上界（同 Key/β，用评估 Query 拟合 V）")
    print("-" * 78)
    print(f"{'M\\B':>6} " + " ".join(f"{b:>10}" for b in args.budgets))
    for M in args.Ms:
        cells = [med("rel_err_oracle", M=M, B=B) for B in args.budgets]
        print(f"{M:>6} " + " ".join(f"{c:>10.4e}" for c in cells))

    print()
    print("表 3：估计差距 gap = fit − oracle（越小说明 M 越够用）")
    print("-" * 78)
    print(f"{'M\\B':>6} " + " ".join(f"{b:>10}" for b in args.budgets))
    for M in args.Ms:
        cells = [med("gap_rel", M=M, B=B) for B in args.budgets]
        print(f"{M:>6} " + " ".join(f"{c:>10.4e}" for c in cells))

    print()
    print("表 4：归并侧误差（β 参与其中，更贴近实际使用）")
    print("-" * 78)
    print(f"{'M\\B':>6} " + " ".join(f"{b:>10}" for b in args.budgets))
    for M in args.Ms:
        cells = [med("mix_err", M=M, B=B) for B in args.budgets]
        print(f"{M:>6} " + " ".join(f"{c:>10.4e}" for c in cells))

    # ---- 轴敏感度 ----
    print()
    print("-" * 78)
    print("轴敏感度：固定另一个轴时，沿该轴可获得的相对改善")
    print("-" * 78)
    print(f"{'M\\B':>6} {'沿 B 轴改善':>14} {'沿 M 轴改善':>14} "
          f"{'主/次':>10}")
    axis_rows = []
    for M in args.Ms:
        vals = [med("rel_err", M=M, B=B) for B in args.budgets]
        gB = (vals[0] - min(vals)) / vals[0] if vals[0] > 0 else float("nan")
        axis_rows.append(("M", M, gB))
    for B in args.budgets:
        vals = [med("rel_err", M=M, B=B) for M in args.Ms]
        gM = (vals[0] - min(vals)) / vals[0] if vals[0] > 0 else float("nan")
        axis_rows.append(("B", B, gM))

    def axis_gain(kind: str) -> float:
        sub = [v for k, _, v in axis_rows if k == kind]
        return sum(sub) / len(sub) if sub else float("nan")

    gB_all, gM_all = axis_gain("M"), axis_gain("B")
    print(f"{'—':>6} {gB_all:>14.4f} {gM_all:>14.4f} "
          f"{('B 更有效' if gB_all > gM_all * 1.2 else ('M 更有效' if gM_all > gB_all * 1.2 else '两者相当')):>10}")

    # ---- 秩与条件数诊断 ----
    print()
    print("-" * 78)
    print("秩诊断：rank_upper_bound = min(M,B)；是否触及（= 回归欠定）")
    print("-" * 78)
    print(f"{'M\\B':>6} " + " ".join(f"{b:>10}" for b in args.budgets))
    for M in args.Ms:
        cells = []
        for B in args.budgets:
            sub = [r for r in rows if r["M"] == M and r["B"] == B]
            rank = sub[0]["numeric_rank"] if sub else int("0")
            mun = min(M, B)
            cells.append(f"{rank}/{mun}{'*' if rank < mun else ''}")
        print(f"{M:>6} " + " ".join(f"{c:>10}" for c in cells))
    print("  （* = 数值秩低于上界，说明 XᵀX 在正则下仍退化）")

    # ---- 保留质量 ----
    print()
    print("-" * 78)
    print("选键保留的注意力质量占比（RMS 选键 vs 名义比 B/L_s）")
    print("-" * 78)
    print(f"{'B':>6} {'B/L_s(名义)':>13} {'保留质量(实测)':>16} {'倍数':>8}")
    for B in args.budgets:
        kept = med("retained_mass_fraction", B=B)
        nom = B / args.L_s
        print(f"{B:>6} {nom:>13.4f} {kept:>16.4f} {kept/nom if nom > 0 else float('nan'):>8.2f}×")

    payload = {
        "experiment": "E9",
        "kind": "knob-localization M-vs-B",
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "axis_gain_along_B": gB_all,
        "axis_gain_along_M": gM_all,
        "rows": rows,
        "caveat": (
            "机制级证据（合成数据上的注意力分布与线性输出重构精度）。"
            "不得用于主张任务质量、通信性能或与基线的对比优势 —— "
            "那些需要 GPU 侧的 E5/E6。oracle 曲线使用评估 Query 自身拟合 V，"
            "是**上界**而非可实现结果，只能用于界定 B 轴的表示能力。"
        ),
    }
    R.save_json(str(pathlib.Path(args.out) / "summary.json"), payload)
    R.save_csv(str(pathlib.Path(args.out) / "rows.csv"), rows)
    print()
    print(f"结果已写入 {args.out}/summary.json 与 rows.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
