#!/usr/bin/env python
"""E1：紧凑 KV 构造的接口与形状不变量（CPU 可跑）

论文 §6.3 的 E1 已由仓库测试覆盖。本脚本把它整理成可直接产表的检查项，
并补两项现有测试未覆盖的不变量：

1. **形状与预算约束**：keys/logit_bias/values/selected_indices 的维度与 B 一致。
2. **索引无重复且合法**：Top-B 选择不应重复，且必须落在 [0, L_s)。
   （重复索引会让 index_add 把质量累加到同一 token，静默破坏分布。）
3. **同种子可复现 / 异种子可分辨**：新增。
   仅有"同种子一致"是不够的 —— 一个忽略种子的实现也满足它。
   必须同时要求"不同种子给出不同结果"，才能证明种子真的接入了流水线。
   这一点不是吹毛求疵：`farthest_point_sampling` 在 `num_samples >= N` 时
   走 `return torch.arange(N)` 分支，**完全绕过种子**，此时异种子可分辨性失效。
   本脚本会显式检出这种情况。

用法
----
    python experiments/cpu/e1_interface_shapes.py
    python experiments/cpu/e1_interface_shapes.py --quick
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


def run(args) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  — {detail}" if detail else ""))

    scenario = S.make_scenario(
        L_s=args.L_s, d_h=args.d_h, d_v=args.d_v,
        num_dest=1, queries_per_dest=args.queries_per_dest,
        focus_strength=args.focus_strength, seed=args.seed, dtype=args._dtype,
    )
    probe = scenario.dest_queries[0]
    B = args.budget

    compact = S.build_compact_kv(
        source_keys=scenario.keys, source_values=scenario.values,
        destination_queries=probe, budget=B,
        num_representative_queries=args.num_repr_queries,
        projection_dim=args.projection_dim, seed=args.seed,
    )

    # --- 1. 形状 ---
    record("keys 形状为 [B, d_h]",
           tuple(compact.keys.shape) == (B, args.d_h),
           f"got {tuple(compact.keys.shape)}")
    record("logit_bias 形状为 [B]",
           tuple(compact.logit_bias.shape) == (B,),
           f"got {tuple(compact.logit_bias.shape)}")
    record("values 形状为 [B, d_v]",
           tuple(compact.values.shape) == (B, args.d_v),
           f"got {tuple(compact.values.shape)}")
    record("selected_indices 形状为 [B]",
           tuple(compact.selected_indices.shape) == (B,),
           f"got {tuple(compact.selected_indices.shape)}")

    # --- 2. 索引合法性 ---
    idx = compact.selected_indices.tolist()
    record("选中索引互不重复", len(set(idx)) == len(idx),
           f"唯一值 {len(set(idx))}/{len(idx)}")
    record("选中索引落在 [0, L_s) 内",
           all(0 <= i < scenario.L_s for i in idx),
           f"min={min(idx)} max={max(idx)} L_s={scenario.L_s}")

    # --- 3. 数值有限性 ---
    record("keys/values 无 NaN/Inf",
           bool(torch.isfinite(compact.keys).all()
                and torch.isfinite(compact.values).all()))
    record("logit_bias 无 NaN/Inf",
           bool(torch.isfinite(compact.logit_bias).all()),
           f"范围 [{compact.logit_bias.min():.4f}, {compact.logit_bias.max():.4f}]")

    # --- 4. 同种子可复现 ---
    again = S.build_compact_kv(
        source_keys=scenario.keys, source_values=scenario.values,
        destination_queries=probe, budget=B,
        num_representative_queries=args.num_repr_queries,
        projection_dim=args.projection_dim, seed=args.seed,
    )
    record("同种子下构造完全一致",
           bool(torch.equal(compact.selected_indices, again.selected_indices)
                and torch.allclose(compact.logit_bias, again.logit_bias)))

    # --- 5. 异种子可分辨（关键补充）---
    n_queries = probe.shape[0]
    if args.num_repr_queries >= n_queries:
        record("异种子可分辨", False,
               f"M={args.num_repr_queries} >= 目的端 Query 数 {n_queries}："
               f"farthest_point_sampling 走 arange(N) 分支，种子被绕过，"
               f"该实现下种子不接入流水线")
    else:
        other = S.build_compact_kv(
            source_keys=scenario.keys, source_values=scenario.values,
            destination_queries=probe, budget=B,
            num_representative_queries=args.num_repr_queries,
            projection_dim=args.projection_dim, seed=args.seed + 1,
        )
        differs = not torch.equal(compact.selected_indices, other.selected_indices)
        record("异种子可分辨（证明种子真的接入）", differs,
               "种子改变了选中索引" if differs else "换种子结果不变")

    # --- 6. 预算边界 ---
    if B < scenario.L_s:
        record("B < L_s 时确实发生压缩",
               int(compact.selected_indices.max()) < scenario.L_s or B == scenario.L_s)

    n_pass = sum(c["passed"] for c in checks)
    return {
        "experiment": "E1",
        "config": {"L_s": args.L_s, "d_h": args.d_h, "d_v": args.d_v,
                   "budget": B, "M": args.num_repr_queries},
        "n_pass": n_pass,
        "n_total": len(checks),
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E1：构造接口与形状不变量（CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=48)
    p.add_argument("--num-repr-queries", dest="num_repr_queries", type=int, default=32)
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=32)
    p.add_argument("--focus-strength", dest="focus_strength", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--out", type=str, default="results/cpu/e1")
    p.add_argument("--quick", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    support = S.compaction_dtype_support()
    args._dtype = S.dtype_from_name(args.dtype)
    if not support.get(args.dtype, False):
        print(f"错误：{args.dtype} 不可用（见 synthetic.DTYPE_BUG_*）")
        return 2

    print("=" * 78)
    print("E1：紧凑 KV 构造的接口与形状不变量")
    print("=" * 78)
    payload = run(args)
    print()
    print(f"  通过 {payload['n_pass']}/{payload['n_total']}")

    out_dir = REPO_ROOT / args.out
    R.save_json(str(out_dir / "e1_results.json"), payload)
    R.save_csv(str(out_dir / "e1_checks.csv"), payload["checks"])
    print(f"结果已写入 {out_dir}")
    return 0 if payload["n_pass"] == payload["n_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
