#!/usr/bin/env python
"""E3 的 H2 优势在**留出集**下是否仍然成立？（一次性判定探针，保留以便复现）

背景（2026-09-15 由独立监督查出）
--------------------------------
`e3_edge_conditioning.run_h2` 用 `scenario.dest_queries[dest]` 同一批 Query
既构造 DCC 的紧凑 KV，又在这批 Query 上评估输出误差；而 shared 基线从
`scenario.all_queries`（全部目的端）里抽代表 Query。于是 DCC 侧存在
**样本内优势**（它见过评估点），Δ 可能部分是伪影。

`experiments/common/synthetic.py` 的 `heldout_split` 正是为这类问题写的
（它的 docstring 明确警告过「用同一批 Query 同时做拟合与评估，误差会被
系统性低估」）。E2b / E9 / E10 / E11 都已使用，**E3 尚未**。

本探针做什么
------------
在**同一个场景**上并排跑两套口径，只让「评估集是否参与构造」这一件事不同：

  A. in-sample（复刻 E3 现状）
     DCC    从该目的端的 96 条 Query 构造 → 在同样这 96 条上评估
     shared 从全部 384 条 Query 构造      → 在同样这 96 条上评估

  B. heldout（本探针要考察的口径）
     DCC    从该目的端的 fit 48 条构造 → 在**不相交的** eval 48 条上评估
     shared 从全部 fit 192 条构造      → 在同样这 eval 48 条上评估

池子大小是对齐的：A 的 DCC 池 96 / shared 池 384；B 的 DCC fit 池 48 /
shared fit 池 192，恰为 A 的一半，与 E3 原始规格（48 / 192）同构。

⚠️ 因此本探针的 in-sample 列**不等于** E3 已落盘的结果（那边每目的端 48 条）。
它证明的是**样本内偏差的存在与量级**，不是一个可以直接替换论文数字的修正值。

**关键的诊断量是负对照**：`focus_strength=0.0` 时各目的端本无差异，
H2 的机制在此**不可能**有真实效果。此时若某种口径给出显著「DCC 更优」，
那部分优势按定义就是伪影。

两种 Δ 的符号约定（本脚本内部不同，输出里逐处标明）
--------------------------------------------------
- 逐单元 `delta = mean(shared) − mean(DCC)`：**> 0 表示 DCC 误差更小（更优）**
- 归并量 `mean_diff = mean(DCC) − mean(shared)`（沿用 `report.paired_bootstrap`
  的约定，与 E3 一致）：**< 0 表示 DCC 误差更小（更优）**

用法
----
    python experiments/cpu/probe_e3_heldout.py

不写任何结果文件、不改动仓库内其他脚本与论文。只打印。
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

from experiments.common import report as R  # noqa: E402
from experiments.common import synthetic as S  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E3 留出集判定探针（样本内 vs 留出）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--num-dest", dest="num_dest", type=int, default=4)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=96)
    p.add_argument("--num-repr-queries", dest="num_repr_queries", type=int, default=32)
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=32)
    p.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--focus-strengths", type=float, nargs="+",
                   default=[0.0, 2.0, 4.0, 8.0, 16.0])
    p.add_argument("--beta-modes", dest="beta_modes", type=str, nargs="+",
                   default=["full", "over_m", "none"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--eval-fraction", dest="eval_fraction", type=float, default=0.5)
    return p


def _compact(scenario, dest_queries, budget, args, seed):
    from src.dcc_kv_ref import build_compact_kv
    return build_compact_kv(
        source_keys=scenario.keys,
        source_values=scenario.values,
        destination_queries=dest_queries,
        budget=budget,
        num_representative_queries=args.num_repr_queries,
        projection_dim=args.projection_dim,
        seed=seed,
    )


def main() -> int:
    args = build_parser().parse_args()
    t0 = time.time()

    if args.num_repr_queries >= args.queries_per_dest * (1.0 - args.eval_fraction):
        raise ValueError(
            f"M={args.num_repr_queries} 不小于 fit 池 "
            f"{int(args.queries_per_dest * (1 - args.eval_fraction))}；"
            "留出口径下代表 Query 会退化（farthest_point_sampling 走 "
            "`return arange(N)` 分支），请调大 --queries-per-dest。"
        )

    # pooled[beta_mode][protocol] = [list_dcc, list_shared]
    pooled: Dict[str, Dict[str, Any]] = {
        bm: {"in": ([], []), "ho": ([], [])} for bm in args.beta_modes
    }
    per_cell: List[Dict[str, Any]] = []

    for seed in args.seeds:
        for strength in args.focus_strengths:
            scenario = S.make_scenario(
                L_s=args.L_s, d_h=args.d_h, d_v=args.d_v,
                num_dest=args.num_dest,
                queries_per_dest=args.queries_per_dest,
                focus_strength=strength,
                seed=seed,
                dtype=torch.float32,
            )
            split = S.heldout_split(scenario, eval_fraction=args.eval_fraction)

            for budget in args.budgets:
                for bm in args.beta_modes:
                    for dest in sorted(scenario.dest_queries.keys()):
                        q_all = scenario.dest_queries[dest]
                        q_eval = split.eval_queries[dest]

                        # ---- A：in-sample（复刻 E3 现状）----
                        c_dcc_in = _compact(scenario, q_all, budget, args, seed)
                        c_sha_in = _compact(scenario, scenario.all_queries,
                                            budget, args, seed)

                        # ---- B：heldout（评估集不参与构造）----
                        c_dcc_ho = _compact(scenario, split.fit_queries[dest],
                                            budget, args, seed)
                        c_sha_ho = _compact(scenario, split.all_fit_queries,
                                            budget, args, seed)

                        def _err(queries, compact):
                            return S.relative_output_error(
                                queries, compact, scenario.keys, scenario.values,
                                beta_mode=bm,
                                num_repr_queries=args.num_repr_queries,
                            )

                        e_dcc_in, e_sha_in = _err(q_all, c_dcc_in), _err(q_all, c_sha_in)
                        e_dcc_ho, e_sha_ho = _err(q_eval, c_dcc_ho), _err(q_eval, c_sha_ho)

                        # delta = shared − DCC；> 0 表示 DCC 更优
                        d_in = (e_sha_in.mean() - e_dcc_in.mean()).item()
                        d_ho = (e_sha_ho.mean() - e_dcc_ho.mean()).item()

                        pooled[bm]["in"][0].extend(e_dcc_in.tolist())
                        pooled[bm]["in"][1].extend(e_sha_in.tolist())
                        pooled[bm]["ho"][0].extend(e_dcc_ho.tolist())
                        pooled[bm]["ho"][1].extend(e_sha_ho.tolist())

                        per_cell.append({
                            "seed": seed, "strength": strength,
                            "budget": budget, "beta_mode": bm, "dest": dest,
                            "delta_in": d_in, "delta_ho": d_ho,
                        })
        print(f"  seed={seed} done  ({time.time() - t0:.0f}s)", flush=True)

    print()
    print("=" * 92)
    print("归并量 mean_diff = mean(DCC 误差) − mean(shared 误差)")
    print("  ⇒ mean_diff < 0 表示 DCC 误差更小（更优）。这与 E3 的既有约定一致。")
    print("=" * 92)
    print(f"{'beta_mode':<9} {'口径':<6} {'mean_diff':>11} {'CI95 下界':>11} "
          f"{'CI95 上界':>11} {'p(单侧)':>10}  {'DCC 更差的条件数':>16}")
    for bm in args.beta_modes:
        cells = [c for c in per_cell if c["beta_mode"] == bm]
        n_cond = len({(c["strength"], c["budget"]) for c in cells})
        for key, label in (("in", "in"), ("ho", "ho")):
            a, b = pooled[bm][key]
            res = R.paired_bootstrap(
                a, b, metric_name="relative_output_error", unit="ratio",
                higher_is_better=False, seed=42,
            )
            n_worse = sum(
                1 for v in _cond_means(cells, f"delta_{key}").values() if v < 0
            )
            print(f"{bm:<9} {label:<6} {res.mean_diff:>11.6f} "
                  f"{res.ci_95_lower:>11.6f} {res.ci_95_upper:>11.6f} "
                  f"{res.p_value_one_sided:>10.3e}  {n_worse:>8}/{n_cond:<6}")

    print()
    print("-" * 92)
    print("★ 关键诊断：负对照 focus_strength=0.0（各目的端本无差异）")
    print("  此条件下 H2 的机制不可能有真实效果 ⇒ 任何「DCC 更优」都是伪影。")
    print(f"  {'beta_mode':<9} {'Δ_in (shared−DCC)':>20} {'Δ_ho (shared−DCC)':>20}")
    for bm in args.beta_modes:
        cells = [c for c in per_cell
                 if c["beta_mode"] == bm and c["strength"] == 0.0]
        m_in = sum(c["delta_in"] for c in cells) / len(cells)
        m_ho = sum(c["delta_ho"] for c in cells) / len(cells)
        print(f"  {bm:<9} {m_in:>+20.6f} {m_ho:>+20.6f}")

    print()
    print(f"总计 {len(per_cell)} 个 (seed, strength, budget, beta_mode, dest) 单元，"
          f"耗时 {time.time() - t0:.0f}s")
    return 0


def _cond_means(cells: List[Dict[str, Any]], key: str) -> Dict[tuple, float]:
    """把逐单元量按 (strength, budget) 聚合，用于判断逐条件方向。"""
    acc: Dict[tuple, List[float]] = {}
    for c in cells:
        acc.setdefault((c["strength"], c["budget"]), []).append(c[key])
    return {k: sum(v) / len(v) for k, v in acc.items()}


if __name__ == "__main__":
    raise SystemExit(main())
