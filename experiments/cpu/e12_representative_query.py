#!/usr/bin/env python
"""E12：代表 Query 选择的投影维度 d_p 扫描（CPU 可跑）

补的是哪个缺口
--------------
`docs/writing_scope_and_metrics.md` 的 M2：论文 §5 关于「代表 Query 选择」
的开销式(28)(32)与误差项 ε_repr **全是解析的**，`d_p` 取值对投影保真度、
对代表集覆盖、对下游 ε_mass / ε_out 的影响**从未测量**。

为什么这件事不能靠"标准做法"糊过去
---------------------------------
Rademacher 投影 + 最远点采样是**替换**关系而不是近似关系：投影把 d_h 维
距离换成 d_p 维距离，而最远点采样**只看排序**。于是 d_p 一变，被选中的
Query 集合可能整体换掉 —— 这不是"估计误差变小"，而是"选了另一组代表"。
所以必须同时报三件事，缺一不可：

  1. **投影本身**的保真度（JL 距离畸变）；
  2. **代表集**的质量（对留出 Query 的覆盖）与**稳定性**（选中集合的变动）；
  3. **下游**在留出 Query 上的 ε_mass / ε_out。

只报 (1) 会漏掉"投影保真但选错点"，只报 (3) 会看不出机制。

口径与硬约束
------------
- 构造只用 fit Query，**评估只用不相交的 eval Query**（`heldout_split`）。
  这条是本仓库的硬约束：评估 Query 若参与选键/β/V 回归，结论会退化为同义反复。
- ε_mass 用 `absolute_mass_error`（公共偏移），ε_out 用归一化版本；
  二者与 §5 的跨块绝对量不是同一个量，见论文 §6 的度量定义表。
- **多种子**：本脚本是网格型实验，按 `docs/reproducibility.md` §7 的约定
  用 5 个种子（42–46），逐单元格报中位数并给出种子间极差 —— 因为
  「选中集合的 Jaccard」与阈值型覆盖计数对种子本身很敏感，
  单种子会把抽样噪声读成趋势。
- 本脚本**不产生任何任务质量主张**，测的是分布重构精度。

参照物
------
除各 d_p 外，另算一条 **无投影参照**：直接在原始空间（先按余弦归一化）
做最远点采样。它回答"投影这一步到底损失了什么"，只参与选择质量与稳定性两栏。

用法
----
    python experiments/cpu/e12_representative_query.py --out results/cpu/e12
    python experiments/cpu/e12_representative_query.py --quick
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any, Dict, List

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import synthetic as S  # noqa: E402
from experiments.common import report as R  # noqa: E402
from src.dcc_kv_ref.representative_query import (  # noqa: E402
    farthest_point_sampling,
    rademacher_projection,
)


# =============================================================================
# 选择与度量
# =============================================================================

def select_indices_projected(
    queries: torch.Tensor, m: int, projection_dim: int, seed: int
) -> torch.Tensor:
    """走仓库实现的路：Rademacher 投影 → 归一化 → 最远点采样。"""
    proj = rademacher_projection(queries, projection_dim=projection_dim, seed=seed)
    z = queries @ proj.T
    z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    return farthest_point_sampling(z_norm, num_samples=m, seed=seed)


def select_indices_noprojection(
    queries: torch.Tensor, m: int, seed: int
) -> torch.Tensor:
    """无投影参照：直接在原始空间按余弦距离做最远点采样。"""
    q_norm = queries / (queries.norm(dim=-1, keepdim=True) + 1e-8)
    return farthest_point_sampling(q_norm, num_samples=m, seed=seed)


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def jl_distortion(
    queries: torch.Tensor, projection_dim: int, seed: int
) -> Dict[str, float]:
    """投影的距离畸变 |‖z_i−z_j‖ / ‖q_i−q_j‖ − 1|（逐对）。

    投影矩阵按 1/sqrt(d_p) 缩放，故 E‖z‖² = ‖q‖²，比值无系统偏置。
    """
    proj = rademacher_projection(queries, projection_dim=projection_dim, seed=seed)
    z = queries @ proj.T
    d_q = torch.cdist(queries.to(torch.float64), queries.to(torch.float64))
    d_z = torch.cdist(z.to(torch.float64), z.to(torch.float64))
    n = queries.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    ratio = d_z[mask] / (d_q[mask] + 1e-12)
    dist = (ratio - 1.0).abs()
    return {
        "jl_distortion_mean": float(dist.mean()),
        "jl_distortion_p95": float(dist.quantile(0.95)),
    }


def coverage_stats(eval_q: torch.Tensor, repr_q: torch.Tensor) -> Dict[str, float]:
    """代表集对留出 Query 的覆盖。

    ``coverage_ratio`` 用**中位数**（稳健）。另报的
    ``coverage_frac_within_radius`` 以"留出 Query 之间的中位最近邻距离"为半径
    计数，属于阈值计数，在小样本上很不稳（实测同配置跨种子摆动可达 0.15），
    **不得单独作为覆盖结论的依据**，只作辅助。
    """
    d_repr = torch.cdist(eval_q.to(torch.float64), repr_q.to(torch.float64))
    nearest_repr = d_repr.min(dim=-1).values
    d_ee = torch.cdist(eval_q.to(torch.float64), eval_q.to(torch.float64))
    n = eval_q.shape[0]
    d_ee = d_ee + torch.eye(n, dtype=d_ee.dtype) * 1e9  # 排除自己
    nearest_eval = d_ee.min(dim=-1).values

    med_repr = float(nearest_repr.median())
    med_eval = float(nearest_eval.median())
    return {
        "nearest_repr_dist_median": med_repr,
        "nearest_eval_dist_median": med_eval,
        "coverage_ratio": med_repr / (med_eval + 1e-12),
        "coverage_frac_within_radius": float(
            (nearest_repr <= med_eval).to(torch.float64).mean()
        ),
    }


# =============================================================================
# 主流程
# =============================================================================

def run(args) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        for strength in args.focus_strengths:
            scenario = S.make_scenario(
                L_s=args.L_s, d_h=args.d_h, d_v=args.d_v,
                num_dest=1, queries_per_dest=args.queries_per_dest,
                focus_strength=strength, seed=seed, dtype=args._dtype,
            )
            split = S.heldout_split(scenario, eval_fraction=args.eval_fraction)
            dest = 0
            fit_q = split.fit_queries[dest]
            eval_q = split.eval_queries[dest]

            if args.num_repr_queries > fit_q.shape[0]:
                raise ValueError(
                    f"M={args.num_repr_queries} 超过 fit 池 {fit_q.shape[0]}；"
                    f"请调大 --queries-per-dest 或调小 --eval-fraction。"
                )

            base_idx = select_indices_noprojection(
                fit_q, args.num_repr_queries, seed
            )

            for budget in args.budgets:
                budget = min(budget, scenario.L_s)
                # 先算齐各 d_p 的选中集合：这样才谈得上"加大 d_p 是否换了点"。
                # "vs 无投影参照"的 Jaccard 天然偏低 —— 无投影走的是另一条
                # 特征路径，与"投影维度不足"混在一起；故另报 vs 最大 d_p 的稳定性。
                sel_idx = {
                    dp: select_indices_projected(
                        fit_q, args.num_repr_queries, dp, seed
                    )
                    for dp in args.projection_dims
                }
                ref_dp = max(args.projection_dims)

                for dp in args.projection_dims:
                    idx = sel_idx[dp]
                    compact = S.build_compact_kv(
                        source_keys=scenario.keys,
                        source_values=scenario.values,
                        destination_queries=fit_q,
                        budget=budget,
                        num_representative_queries=args.num_repr_queries,
                        projection_dim=dp,
                        seed=seed,
                    )
                    eps_mass = S.absolute_mass_error(
                        eval_q, compact, scenario.keys,
                        beta_mode=args.beta_mode,
                        num_repr_queries=args.num_repr_queries,
                    )
                    eps_out = S.relative_output_error(
                        eval_q, compact, scenario.keys, scenario.values,
                        beta_mode=args.beta_mode,
                        num_repr_queries=args.num_repr_queries,
                    )
                    row: Dict[str, Any] = {
                        "seed": seed,
                        "focus_strength": strength,
                        "budget": budget,
                        "nominal_compression_ratio": budget / scenario.L_s,
                        "projection_dim": dp,
                        "reference_projection_dim": ref_dp,
                        "M": args.num_repr_queries,
                        "n_fit_queries": int(fit_q.shape[0]),
                        "n_eval_queries": int(eval_q.shape[0]),
                        "selection_jaccard_vs_noprojection": jaccard(idx, base_idx),
                        "selection_jaccard_vs_max_dp": jaccard(idx, sel_idx[ref_dp]),
                    }
                    row.update(jl_distortion(fit_q, dp, seed))
                    row.update(coverage_stats(eval_q, fit_q[idx]))
                    row["eps_mass_abscommon_median"] = float(eps_mass.median())
                    row["eps_mass_abscommon_mean"] = float(eps_mass.mean())
                    row["eps_out_rel_median"] = float(eps_out.median())
                    row["eps_out_rel_mean"] = float(eps_out.mean())
                    rows.append(row)

                # 无投影参照行（只报选择质量与稳定性，不建紧凑 KV）
                b_stats = coverage_stats(eval_q, fit_q[base_idx])
                rows.append({
                    "seed": seed,
                    "focus_strength": strength,
                    "budget": budget,
                    "nominal_compression_ratio": budget / scenario.L_s,
                    "projection_dim": None,  # None 表示"无投影参照"
                    "reference_projection_dim": ref_dp,
                    "M": args.num_repr_queries,
                    "n_fit_queries": int(fit_q.shape[0]),
                    "n_eval_queries": int(eval_q.shape[0]),
                    "selection_jaccard_vs_noprojection": 1.0,
                    "selection_jaccard_vs_max_dp": None,
                    "jl_distortion_mean": 0.0,
                    "jl_distortion_p95": 0.0,
                    **b_stats,
                    "eps_mass_abscommon_median": None,
                    "eps_mass_abscommon_mean": None,
                    "eps_out_rel_median": None,
                    "eps_out_rel_mean": None,
                })
            print(f"  seed={seed} strength={strength} done", flush=True)

    return {
        "experiment": "E12",
        "precision": str(args._dtype).replace("torch.", ""),
        "config": {
            "L_s": args.L_s, "d_h": args.d_h, "d_v": args.d_v,
            "budgets": list(args.budgets),
            "projection_dims": list(args.projection_dims),
            "focus_strengths": list(args.focus_strengths),
            "beta_mode": args.beta_mode,
            "num_repr_queries": args.num_repr_queries,
            "queries_per_dest": args.queries_per_dest,
            "eval_fraction": args.eval_fraction,
            "seeds": list(args.seeds),
        },
        "heldout_note": (
            "紧凑 KV 一律用 fit Query 构造、用不相交的 eval Query 评估；"
            "代表 Query 只从 fit 池中选。"
        ),
        "caveat": (
            "机制级证据：测的是代表集覆盖与分布重构精度，不是任务指标。"
            "d_p 与 d_h（=32）同量级时投影近似保距，结论不得外推到 d_p >> d_h；"
            "反过来 d_p > d_h 也不代表更好，只说明该区间内已饱和。"
        ),
        "aggregated": _aggregate(rows, args),
        "rows": rows,
    }


def _aggregate(rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    """跨种子把每个 (strength, budget, d_p) 单元格聚成中位数与极差。

    种子间极差是必须报的量：Jaccard 与阈值型覆盖计数在单种子上会呈现假趋势。
    """
    out: Dict[str, Any] = {}
    for strength in args.focus_strengths:
        for budget in args.budgets:
            cells: Dict[Any, List[Dict[str, Any]]] = {}
            for r in rows:
                if (r["focus_strength"] != strength
                        or r["budget"] != min(budget, args.L_s)):
                    continue
                cells.setdefault(r["projection_dim"], []).append(r)
            entries = []
            for dp in sorted(cells, key=lambda x: (x is None, x)):
                group = cells[dp]

                def med(field, _g=group):
                    vals = [g[field] for g in _g if g[field] is not None]
                    return float(torch.tensor(vals).median()) if vals else None

                def spread(field, _g=group):
                    vals = [g[field] for g in _g if g[field] is not None]
                    return (max(vals) - min(vals)) if vals else None

                entries.append({
                    "projection_dim": dp,
                    "n_seeds": len(group),
                    "jl_distortion_mean_median": med("jl_distortion_mean"),
                    "jl_distortion_mean_range": spread("jl_distortion_mean"),
                    "coverage_ratio_median": med("coverage_ratio"),
                    "coverage_ratio_range": spread("coverage_ratio"),
                    "coverage_frac_median": med("coverage_frac_within_radius"),
                    "coverage_frac_range": spread("coverage_frac_within_radius"),
                    "jaccard_vs_max_dp_median": med("selection_jaccard_vs_max_dp"),
                    "jaccard_vs_max_dp_range": spread("selection_jaccard_vs_max_dp"),
                    "eps_mass_median": med("eps_mass_abscommon_median"),
                    "eps_mass_range": spread("eps_mass_abscommon_median"),
                    "eps_out_median": med("eps_out_rel_median"),
                    "eps_out_range": spread("eps_out_rel_median"),
                })
            out[f"strength={strength},budget={budget}"] = entries
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E12：代表 Query 选择的投影维度 d_p 扫描（CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--projection-dims", dest="projection_dims", type=int, nargs="+",
                   default=[4, 8, 16, 32, 64],
                   help="d_p 扫描点；判定标准要求的 {8,16,32,64} 是其子集")
    p.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--focus-strengths", dest="focus_strengths", type=float, nargs="+",
                   default=[0.0, 8.0])
    p.add_argument("--beta-mode", dest="beta_mode", type=str, default="full")
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=96)
    p.add_argument("--num-repr-queries", dest="num_repr_queries", type=int, default=32)
    p.add_argument("--eval-fraction", dest="eval_fraction", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                   help="网格型实验按 reproducibility.md §7 用 5 个种子")
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--out", type=str, default="results/cpu/e12")
    p.add_argument("--quick", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    support = S.compaction_dtype_support()
    args._dtype = S.dtype_from_name(args.dtype)
    if not support.get(args.dtype, False):
        print(f"错误：{args.dtype} 不可用（见 synthetic.DTYPE_BUG_*）。")
        print(f"  可用精度：{[k for k, v in support.items() if v]}")
        return 2

    if args.quick:
        args.projection_dims = [8, 32]
        args.budgets = [32]
        args.focus_strengths = [8.0]
        args.seeds = [42]
        args.L_s = 128
        args.queries_per_dest = 64
        args.num_repr_queries = 16

    print("=" * 78)
    print("E12：代表 Query 选择的投影维度 d_p 扫描")
    print("=" * 78)
    print(f"  dtype={args.dtype}  L_s={args.L_s}  d_h={args.d_h}  "
          f"M={args.num_repr_queries}  d_p={args.projection_dims}")
    print(f"  β 模式={args.beta_mode}  留出比例={args.eval_fraction}  "
          f"种子={args.seeds}")
    print()

    t0 = time.time()
    payload = run(args)

    print()
    print("=" * 78)
    print("跨种子汇总：中位数 [种子间极差]")
    print("=" * 78)
    for key, entries in payload["aggregated"].items():
        print(f"  {key}")
        print(f"    {'d_p':>5} {'JL畸变':>17} {'覆盖比':>17} "
              f"{'Jaccard vs max':>17} {'ε_mass':>17} {'ε_out':>17}")
        for e in entries:
            if e["projection_dim"] is None:
                print(f"    {'—':>5} {'（无投影参照）':>17} "
                      f"{e['coverage_ratio_median']:>8.3f}[{e['coverage_ratio_range']:.3f}]")
                continue
            print(
                f"    {e['projection_dim']:>5} "
                f"{e['jl_distortion_mean_median']:>8.4f}[{e['jl_distortion_mean_range']:.4f}] "
                f"{e['coverage_ratio_median']:>8.3f}[{e['coverage_ratio_range']:.3f}] "
                f"{e['jaccard_vs_max_dp_median']:>8.3f}[{e['jaccard_vs_max_dp_range']:.3f}] "
                f"{e['eps_mass_median']:>8.4f}[{e['eps_mass_range']:.4f}] "
                f"{e['eps_out_median']:>8.4f}[{e['eps_out_range']:.4f}]"
            )

    print()
    print("读取方式：")
    print("  1. JL 畸变随 d_p 单调下降；但它下降不等于下游误差下降。")
    print("  2. 覆盖比 < 1 表示代表集比留出 Query 互相之间更靠近它们。")
    print("  3. Jaccard vs max 低 ⇒ 改变 d_p 改变了**选中哪些** Query（换点而非变准）。")
    print("  4. ε_mass / ε_out 若在某 d_p 之后落进种子间极差，即视为已饱和。")
    print(f"\n  总耗时 {time.time() - t0:.0f}s")

    out_dir = REPO_ROOT / args.out
    R.save_json(str(out_dir / "e12_results.json"), payload)
    R.save_csv(str(out_dir / "e12_rows.csv"), payload["rows"])
    print(f"结果已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
