#!/usr/bin/env python
"""A3 的机制级版本：组件消融（CPU 可跑）

论文 §6.4 把 A3 组件消融列为 **GPU 实验**，因为它测的是任务指标
（LongBench 准确率等）的变化。但 A3 有一个**不依赖模型**的部分：

    移除某个组件后，紧凑块对注意力分布的重构精度变化多少？

这可以在 CPU 上用合成数据测。本脚本做的就是这个，并且它比任务指标
更能**定位**问题 —— 任务指标退化可能来自很多环节，而重构误差直接
指向机制本身。

四个变体（与论文 A3 一致）：
    full       完整 DCC-KV
    no_beta    移除质量偏置 β           —— 对应 beta_mode="none"
    no_value   移除 Value 回归（直接用所选 Key 对应的原始 Value）
    no_both    两者同时移除（退化为纯 Key 选择）

论文 H3 的预测：移除 β 应导致 ≥ 0.5 点退化，移除 Value 回归应 ≥ 1.0 点退化。
那是以**任务指标点数**为单位的，与这里的重构误差不同量纲，因此本脚本
**不**声称验证或否证 H3。它报告的是更上游的量：β 与 Value 回归各自
在重构误差上贡献多少。若 β 的贡献本身接近 0，那么"移除 β 掉 0.5 点"
这个预测就缺少机制基础 —— 这是一个应当先查清的前置问题。

用法
----
    python experiments/cpu/e5a_mechanism_ablation.py --out results/cpu/a3
    python experiments/cpu/e5a_mechanism_ablation.py --quick
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import synthetic as S  # noqa: E402
from experiments.common import report as R  # noqa: E402
from src.dcc_kv_ref import CompactKV  # noqa: E402


def strip_value_regression(compact: CompactKV, source_values: torch.Tensor) -> CompactKV:
    """no_value 变体：紧凑 Value 直接取所选 Key 对应的原始 Value。

    这不改变 Key 集合与 β，只把"拟合出来的 Value"换成"原样搬运的 Value"。
    区分开这两者很重要：如果误差主要来自 Value 回归，
    那么 Value 回归就是必须保留的组件；反之则说明它是无谓开销。
    """
    return CompactKV(
        keys=compact.keys,
        logit_bias=compact.logit_bias,
        values=source_values[compact.selected_indices],
        selected_indices=compact.selected_indices,
    )


VARIANTS = ("full", "no_beta", "no_value", "no_both")


def run(args) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for strength in args.focus_strengths:
        scenario = S.make_scenario(
            L_s=args.L_s, d_h=args.d_h, d_v=args.d_v,
            num_dest=1, queries_per_dest=args.queries_per_dest,
            focus_strength=strength, seed=args.seed, dtype=args._dtype,
        )
        probe = scenario.dest_queries[0]

        for budget in args.budgets:
            budget = min(budget, scenario.L_s)
            base = S.build_compact_kv(
                source_keys=scenario.keys, source_values=scenario.values,
                destination_queries=probe, budget=budget,
                num_representative_queries=args.num_repr_queries,
                projection_dim=args.projection_dim, seed=args.seed,
            )

            variants = {
                "full": (base, "full"),
                "no_beta": (base, "none"),
                "no_value": (strip_value_regression(base, scenario.values), "full"),
                "no_both": (strip_value_regression(base, scenario.values), "none"),
            }

            row: Dict[str, Any] = {
                "focus_strength": strength,
                "budget": budget,
                "nominal_compression_ratio": budget / scenario.L_s,
                # 记录秩诊断：rank(X) <= M，因此 B > M 时 Value 回归欠定
                "M": args.num_repr_queries,
                "value_regression_underdetermined": bool(args.num_repr_queries < budget),
            }

            for name in VARIANTS:
                compact, beta_mode = variants[name]
                rel_out = S.relative_output_error(
                    probe, compact, scenario.keys, scenario.values,
                    beta_mode=beta_mode,
                    num_repr_queries=args.num_repr_queries,
                )
                # M5：改用公共偏移口径。旧实现 mass_error 对 β 的常数分量免疫，
                # 而本消融的自变量含"移除 β"，用旧口径会低估该成分的作用。
                abs_mass = S.absolute_mass_error(
                    probe, compact, scenario.keys,
                    beta_mode=beta_mode,
                    num_repr_queries=args.num_repr_queries,
                )
                out_sum = R.summarize(rel_out.tolist(), f"eps_out[{name}]", "ratio", seed=args.seed)
                mass_sum = R.summarize(
                    abs_mass.tolist(), f"eps_mass_abscommon[{name}]", "ratio", seed=args.seed
                )
                row[f"eps_out_{name}_median"] = out_sum.median
                row[f"eps_out_{name}_ci_lower"] = out_sum.ci_95_lower
                row[f"eps_out_{name}_ci_upper"] = out_sum.ci_95_upper
                row[f"eps_mass_abscommon_{name}_median"] = mass_sum.median

            # 各组件贡献 = 完整版误差 与 移除版误差 之差
            row["beta_contribution"] = (
                row["eps_out_no_beta_median"] - row["eps_out_full_median"]
            )
            row["value_regression_contribution"] = (
                row["eps_out_no_value_median"] - row["eps_out_full_median"]
            )
            rows.append(row)

            print(
                f"  strength={strength:<5} B={budget:<4} "
                f"full={row['eps_out_full_median']:.4f}  "
                f"no_beta={row['eps_out_no_beta_median']:.4f}  "
                f"no_value={row['eps_out_no_value_median']:.4f}  "
                f"no_both={row['eps_out_no_both_median']:.4f}  "
                f"| β 贡献={row['beta_contribution']:+.4f}  "
                f"V回归贡献={row['value_regression_contribution']:+.4f}"
                + ("  [V回归欠定 M<B]" if row["value_regression_underdetermined"] else "")
            )

    # 跨条件汇总各组件贡献
    beta_contrib = R.summarize(
        [r["beta_contribution"] for r in rows], "beta_contribution", "ratio", seed=args.seed
    )
    value_contrib = R.summarize(
        [r["value_regression_contribution"] for r in rows],
        "value_regression_contribution", "ratio", seed=args.seed,
    )

    return {
        "experiment": "A3(mechanism-level)",
        "rows": rows,
        "beta_contribution_summary": beta_contrib.to_dict(),
        "value_regression_contribution_summary": value_contrib.to_dict(),
        "caveat": (
            "本脚本测的是重构误差，不是任务指标。论文 H3 的 0.5/1.0 点阈值"
            "以任务指标为单位，二者不同量纲，因此这里的数字不能直接验证或否证 H3。"
            "它回答的是更上游的前置问题：各组件在机制层面是否还有贡献空间。"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A3 机制级组件消融（CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--focus-strengths", dest="focus_strengths", type=float, nargs="+",
                   default=[0.0, 8.0])
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=48)
    p.add_argument("--num-repr-queries", dest="num_repr_queries", type=int, default=32,
                   help="代表 Query 数 M。**这是紧凑块真正的信息瓶颈**："
                        "Value 回归的设计矩阵 X 只有 M 行，rank(X) <= M，"
                        "故 B > M 时欠定。")
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--out", type=str, default="results/cpu/a3")
    p.add_argument("--quick", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    support = S.compaction_dtype_support()
    args._dtype = S.dtype_from_name(args.dtype)
    if not support.get(args.dtype, False):
        print(f"错误：{args.dtype} 不可用（见 synthetic.DTYPE_BUG_*）")
        return 2

    if args.quick:
        args.budgets = [16, 64]
        args.focus_strengths = [8.0]
        args.L_s = 128
        args.queries_per_dest = 24
        args.num_repr_queries = 16

    print("=" * 78)
    print("A3（机制级）：组件消融 —— β 与 Value 回归各自贡献多少")
    print("=" * 78)
    print(f"  M={args.num_repr_queries}  B={args.budgets}")
    print()

    payload = run(args)
    b = payload["beta_contribution_summary"]
    v = payload["value_regression_contribution_summary"]
    print()
    print(f"  β 贡献中位数        : {b['median']:+.4f}  "
          f"CI95=[{b['ci_95_lower']:+.4f},{b['ci_95_upper']:+.4f}]")
    print(f"  Value 回归贡献中位数: {v['median']:+.4f}  "
          f"CI95=[{v['ci_95_lower']:+.4f},{v['ci_95_upper']:+.4f}]")
    print()
    print(f"  {payload['caveat']}")

    out_dir = REPO_ROOT / args.out
    R.save_json(str(out_dir / "a3_results.json"), payload)
    R.save_csv(str(out_dir / "a3.csv"), payload["rows"])
    print()
    print(f"结果已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
