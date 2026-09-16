#!/usr/bin/env python
"""E2：压缩保真度曲线 ε_mass(B) 与 ε_out(B)（CPU 可跑）

论文 §6.3 的 E2 现状是"部分证据"：仓库测试只断言误差小于阈值，
没有报告误差随预算 B 的变化。待补项原文：

    需要补充的是 ε_mass(B) 与 ε_out(B) 的完整曲线，用于验证 §5.3 中
    性质 2 的预测（无偏置时质量被系统性低估）。

本脚本补这条曲线，并且**顺带检验该预测本身**：

- 用有符号的质量误差 `signed_mass_error`。性质 2 是关于**方向**的命题，
  取绝对值的误差度量会让它不可证伪 —— 差的只是"多少"，
  而预测说的是"偏低"。
- 分别在 β 的三种使用方式下测，因为"无偏置"正对应 beta_mode="none"。
- 同时报告 Key 选择实际保留的质量占比：B/L_s 是"名义压缩比"，
  而 RMS 选择是**非均匀**采样，保留的质量通常远高于名义比例。
  这个差距是解释"为什么压缩到 1% 还能用"的关键量。

精度说明（2026-09-16 更正）
--------------------------
论文 §6 的 E2 规格为 FP64。此前该精度被 `representative_query.py` 的
`.float()` 写死所阻塞（论文 §6「其五」），故早期落盘是 FP32，本文件也
记过该限制。**该实现缺陷已修**：`compaction_dtype_support()` 现返回
`float64: True`。因此本脚本的默认精度改为 **FP64** 以匹配论文规格，
并把本次实际用的 dtype 写进落盘的 `precision` 字段（此前落盘不含该字段，
无法从产物判断用的是哪个精度）。`--dtype float32` 仍可用于对照。

用法
----
    python experiments/cpu/e2_fidelity_curve.py --out results/cpu/e2
    python experiments/cpu/e2_fidelity_curve.py --quick
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


def retained_mass_fraction(
    compact,
    probe_queries: torch.Tensor,
    keys: torch.Tensor,
) -> float:
    """被测 Key 集合真实承载的注意力质量占完整块总质量的比例。

    名义压缩比是 B/L_s，但 RMS 选择挑的是高权重 token，
    所以真实保留的质量显著高于 B/L_s。这个量决定了压缩的可行下界。
    """
    scale = 1.0 / (keys.shape[-1] ** 0.5)
    w = torch.softmax((probe_queries @ keys.T) * scale, dim=-1)  # [N, L_s]
    idx = compact.selected_indices.long()
    kept = w[:, idx].sum(dim=-1).mean().item()
    return kept


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
            compact = S.build_compact_kv(
                source_keys=scenario.keys,
                source_values=scenario.values,
                destination_queries=probe,
                budget=budget,
                num_representative_queries=args.num_repr_queries,
                projection_dim=args.projection_dim,
                seed=args.seed,
            )

            nominal_ratio = budget / scenario.L_s
            kept = retained_mass_fraction(compact, probe, scenario.keys)

            row: Dict[str, Any] = {
                "focus_strength": strength,
                "budget": budget,
                "nominal_compression_ratio": nominal_ratio,
                "retained_mass_fraction": kept,
                "retained_over_nominal": kept / nominal_ratio if nominal_ratio > 0 else float("nan"),
            }

            for beta_mode in args.beta_modes:
                # 口径（2026-09-16，缺口 M5）：
                # 旧实现 mass_error 是"两侧各减自身最大值"，对 β 的常数分量免疫；
                # 此前的列名却写作 eps_mass_abs_*，名字与实现不符。现改用
                # absolute_mass_error（公共偏移 c=max(ℓ_full)），列名同步为
                # eps_mass_abscommon_*，二者不可互比。
                abs_mass = S.absolute_mass_error(
                    probe, compact, scenario.keys,
                    beta_mode=beta_mode, num_repr_queries=args.num_repr_queries,
                )
                # 性质 2 是**方向性**命题（"无偏置时质量被系统性低估"）。
                # 默认偏移（各减自身 max）会削弱该方向，故必须用公共偏移。
                signed_mass = S.signed_mass_error(
                    probe, compact, scenario.keys,
                    beta_mode=beta_mode, num_repr_queries=args.num_repr_queries,
                    common_offset=True,
                )
                rel_out = S.relative_output_error(
                    probe, compact, scenario.keys, scenario.values,
                    beta_mode=beta_mode, num_repr_queries=args.num_repr_queries,
                )

                abs_sum = R.summarize(
                    abs_mass.tolist(), f"eps_mass_abscommon[{beta_mode}]", "ratio", seed=args.seed
                )
                sgn_sum = R.summarize(
                    signed_mass.tolist(), f"eps_mass_signed_common[{beta_mode}]", "ratio", seed=args.seed
                )
                out_sum = R.summarize(rel_out.tolist(), f"eps_out_rel[{beta_mode}]", "ratio", seed=args.seed)

                row[f"eps_mass_abscommon_{beta_mode}_median"] = abs_sum.median
                row[f"eps_mass_signed_{beta_mode}_median"] = sgn_sum.median
                row[f"eps_mass_signed_{beta_mode}_ci_lower"] = sgn_sum.ci_95_lower
                row[f"eps_mass_signed_{beta_mode}_ci_upper"] = sgn_sum.ci_95_upper
                # 性质 2 的判据：有符号质量误差的 CI 整段为负 → 系统性低估
                row[f"mass_underestimated_{beta_mode}"] = bool(sgn_sum.ci_95_upper < 0)
                row[f"eps_out_rel_{beta_mode}_median"] = out_sum.median

            rows.append(row)

            print(
                f"  strength={strength:<5} B={budget:<4} "
                f"B/L={nominal_ratio:6.3f} 保留质量={kept:6.3f} "
                f"({row['retained_over_nominal']:5.1f}x)  "
                f"|ε_mass|_c none={row['eps_mass_abscommon_none_median']:.4f} "
                f"over_m={row['eps_mass_abscommon_over_m_median']:.4f}  "
                f"ε_out none={row['eps_out_rel_none_median']:.4f} "
                f"over_m={row['eps_out_rel_over_m_median']:.4f}"
            )

    return {
        "experiment": "E2",
        # 精度与配置必须随产物落盘：否则无法从结果本身判断用的是哪个口径。
        "precision": str(args._dtype).replace("torch.", ""),
        "config": {
            "L_s": args.L_s, "d_h": args.d_h, "d_v": args.d_v,
            "budgets": list(args.budgets),
            "focus_strengths": list(args.focus_strengths),
            "beta_modes": list(args.beta_modes),
            "num_repr_queries": args.num_repr_queries,
            "projection_dim": args.projection_dim,
            "queries_per_dest": args.queries_per_dest,
            "seed": args.seed,
        },
        "metric_note": (
            "eps_mass_abscommon_* 用公共偏移 c=max(ℓ_full)；"
            "eps_mass_signed_common_* 同偏移并保留符号（用于 §5 性质 2 的方向检验）；"
            "eps_out_rel_* 为单块归一化输出误差。三者与 §5 的跨块绝对量 ε_mass/ε_out "
            "不是同一个量，见论文 §6 的度量定义表。"
        ),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E2：压缩保真度曲线（CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--focus-strengths", dest="focus_strengths", type=float, nargs="+",
                   default=[0.0, 8.0],
                   help="0.0 近似均匀注意力，8.0 近似 one-hot（注意力越发尖峭）")
    p.add_argument("--beta-modes", dest="beta_modes", type=str, nargs="+",
                   default=["none", "over_m", "full"])
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=48)
    p.add_argument("--num-repr-queries", dest="num_repr_queries", type=int, default=32)
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"],
                   help="论文 §6 的 E2 规格为 FP64；dtype 缺陷已修，故默认 FP64")
    p.add_argument("--out", type=str, default="results/cpu/e2")
    p.add_argument("--quick", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    support = S.compaction_dtype_support()
    args._dtype = S.dtype_from_name(args.dtype)
    if not support.get(args.dtype, False):
        print(f"错误：{args.dtype} 不可用（见 synthetic.DTYPE_BUG_*）。")
        print(f"  根因：{S.DTYPE_BUG_FILE} 第 {S.DTYPE_BUG_LINE} 行的 `.float()`")
        print(f"  论文 §6.3 的 E2 规格为 FP64，修复前无法按该规格运行。")
        print(f"  可用精度：{[k for k, v in support.items() if v]}")
        return 2

    if args.quick:
        args.budgets = [16, 64, 256]
        args.focus_strengths = [8.0]
        args.L_s = 128
        args.queries_per_dest = 24
        args.num_repr_queries = 16

    print("=" * 78)
    print("E2：压缩保真度曲线 ε_mass(B) / ε_out(B)")
    print("=" * 78)
    print(f"  dtype={args.dtype}（论文规格为 FP64；'其五'已修，不再阻塞）")
    print(f"  L_s={args.L_s}  d_h={args.d_h}  d_v={args.d_v}  M={args.num_repr_queries}")
    print()

    payload = run(args)

    print()
    print("读取方式：")
    print("  1. 保留质量 >> B/L_s：RMS 选择挑高权重 token，所以名义压缩比")
    print("     严重低估了实际保留的信息。B/L=1% 时保留质量可能仍有几十 %。")
    print("  2. mass_underestimated_* 为真 → §5.3 性质 2 成立（质量被系统性低估）。")
    print("  3. none 与 over_m 两列的差距，就是 β 机制的实际贡献量。")
    print("  4. eps_mass_abscommon_* 为公共偏移口径；与 2026-09-16 之前的落盘")
    print("     （旧口径 mass_error，列名曾误作 eps_mass_abs_*）**不可互比**。")

    out_dir = REPO_ROOT / args.out
    R.save_json(str(out_dir / "e2_results.json"), payload)
    R.save_csv(str(out_dir / "e2.csv"), payload["rows"])
    print()
    print(f"结果已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
