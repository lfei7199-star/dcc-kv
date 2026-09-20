#!/usr/bin/env python
"""E8：低精度下的顺序无关性失效边界。

论文 §6.4 把 E8 放在 GPU 侧，因为它要回答的是
"异步实现能否在**真实推理精度**下安全使用"。但它测的算子
（Online Softmax 归并 ⊕）只有逐元素运算与归约，没有矩阵乘，
也没有任何通信 —— 因此它的**结构**可以完整地在 CPU 上跑通，
只是数值结论不迁移。

两种口径，用途不同，不要混用
----------------------------
    --device cuda（默认，权威）   fp16 / bf16 / fp32，与推理实际使用的
                                  tensor core 精度一致。E8 的结论以此为准。
    --device cpu（reduced）       同样的算子、同样的 dtype 名义值，但
                                 CPU 的 bf16/fp16 归约路径与 GPU tensor core
                                 不同，**误差常数的绝对值不可迁移**。
                                 它的价值在于：脚本逻辑可本地验证、
                                 趋势（误差随 K 与树深的增长阶）可预先观察、
                                 GPU 上跑之前就能发现实验设计的问题。

两个度量，含义不同，必须并列报告
--------------------------------
    order_error  —— 各置换顺序的归并结果相对"规范顺序"归并结果的最大相对偏差。
                    这是**顺序无关性本身**的失效度量。异步实现的安全性由它决定。
    total_error  —— 规范顺序归并结果相对 FP64 稠密单次 softmax 的相对偏差。
                    这是**总数值误差**，包含归并顺序之外的量化损失。

把二者混为一谈是常见错误：低精度下 total_error 必然很大（fp16 只有约 3 位
有效十进制数），这不代表顺序无关性失效；异步流水能否用 fp16，取决于
order_error 是否仍淹没在 dtype 自身的舍入量级内。

用法
----
    # 权威运行（GPU）
    python experiments/gpu/e8_low_precision.py --device cuda \
        --dtypes float16 bfloat16 --num-blocks 8 16 32 64 128 --out results/gpu/e8

    # 本地 reduced 复现（CPU，用于验证脚本与观察趋势）
    python experiments/gpu/e8_low_precision.py --device cpu --reduced
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import report as R                        # noqa: E402
from experiments.gpu import _env                                  # noqa: E402
from src.dcc_kv_ref import (                                      # noqa: E402
    OnlineSoftmaxState,
    merge_softmax_states,
    online_softmax_from_attention,
)
from src.dcc_kv_ref.online_softmax import merge_softmax_states_list  # noqa: E402

# 各 dtype 的相对舍入单位，作为"误差是否已淹没在精度噪声里"的参考线
UNIT_ROUNDOFF = {
    "float64": 2.0 ** -52,
    "float32": 2.0 ** -23,
    "bfloat16": 2.0 ** -8,
    "float16": 2.0 ** -10,
}


# =============================================================================
# 归并树形状
# =============================================================================

def merge_left_deep(states: List[OnlineSoftmaxState]) -> OnlineSoftmaxState:
    """左深树（顺序归并）：树深 K-1，舍入误差累积路径最长。"""
    return merge_softmax_states_list(states)


def merge_balanced(states: List[OnlineSoftmaxState]) -> OnlineSoftmaxState:
    """平衡二叉树：树深 ceil(log2 K)，是舍入误差最小的归并形状。"""
    level = list(states)
    while len(level) > 1:
        nxt: List[OnlineSoftmaxState] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(merge_softmax_states(level[i], level[i + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def merge_random_tree(states: List[OnlineSoftmaxState], seed: int) -> OnlineSoftmaxState:
    """随机配对树：模拟不可控的到达序 + 不确定的分组策略（异步的真实情形）。"""
    g = torch.Generator().manual_seed(seed)
    level = list(states)
    while len(level) > 1:
        perm = torch.randperm(len(level), generator=g).tolist()
        level = [level[i] for i in perm]
        nxt: List[OnlineSoftmaxState] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(merge_softmax_states(level[i], level[i + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


TREE_SHAPES = ("left_deep", "balanced", "random")

TREE_FN = {
    "left_deep": lambda states, seed: merge_left_deep(states),
    "balanced": lambda states, seed: merge_balanced(states),
    "random": lambda states, seed: merge_random_tree(states, seed),
}


# =============================================================================
# 数据构造与度量
# =============================================================================

def make_blocks(
    K: int, d_h: int, d_v: int, logit_scale: float,
    dtype: torch.dtype, device: torch.device, seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[OnlineSoftmaxState]]:
    """构造 K 个变长块及其 online softmax 状态。

    块长在 [L/2, 1.5L] 内变化 —— 异步 All-to-Allv 交换的正是变长消息。
    logit 统一乘 logit_scale 以制造较大的动态范围：动态范围小时
    softmax 被单一 token 支配，归并几乎无误差，测不出失效边界。
    """
    g = torch.Generator().manual_seed(seed)
    tokens_per_block = max(8, 2048 // K)
    logits_list: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []
    states: List[OnlineSoftmaxState] = []

    for _ in range(K):
        L = int(tokens_per_block * (0.5 + torch.rand(1, generator=g).item()))
        L = max(L, 2)
        lg = (torch.randn(L, generator=g, dtype=torch.float32) * logit_scale).to(dtype).to(device)
        v = torch.randn(L, d_v, generator=g, dtype=torch.float32).to(dtype).to(device)
        logits_list.append(lg)
        values_list.append(v)
        states.append(online_softmax_from_attention(lg, v))
    return logits_list, values_list, states


def attention_output(s: OnlineSoftmaxState) -> torch.Tensor:
    return s.o / s.l


def dense_reference(
    logits_list: Sequence[torch.Tensor],
    values_list: Sequence[torch.Tensor],
) -> torch.Tensor:
    """真值：把所有块拼起来做一次完整 softmax 注意力，全程 FP64。"""
    lg = torch.cat([x.to(torch.float64) for x in logits_list], dim=0)
    v = torch.cat([x.to(torch.float64) for x in values_list], dim=0)
    # 与分块实现保持同一数值稳定化方式（减 max），避免把实现差异算成误差
    lg = lg - lg.max()
    w = torch.softmax(lg, dim=-1)
    return w @ v


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got.double() - ref.double()).norm() / (ref.double().norm() + 1e-30))


def run_one(
    K: int, dtype: torch.dtype, device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    logits_list, values_list, states = make_blocks(
        K, args.d_h, args.d_v, args.logit_scale, dtype, device, args.seed)
    ref = dense_reference(logits_list, values_list)

    rows: List[Dict[str, Any]] = []
    for shape in args.tree_shapes:
        canonical = TREE_FN[shape](states, args.seed)
        canonical_out = attention_output(canonical)

        # order_error：同一树形下多次置换的散布
        errs: List[float] = []
        for t in range(args.n_perms):
            s = list(states)
            perm = torch.randperm(K, generator=torch.Generator().manual_seed(args.seed + t)).tolist()
            s = [s[i] for i in perm]
            out = attention_output(TREE_FN[shape](s, args.seed + t))
            errs.append(rel_err(out, canonical_out))

        order_stats = R.summarize(errs, "order_error", "ratio", seed=args.seed)
        total = rel_err(canonical_out, ref)
        dt_key = str(dtype).replace("torch.", "")
        u = UNIT_ROUNDOFF[dt_key]
        err_max = max(errs) if errs else 0.0
        rows.append({
            "dtype": dt_key,
            "device": str(device),
            "K": K,
            "tree_shape": shape,
            "n_perms": args.n_perms,
            "order_error_median": order_stats.median,
            "order_error_max": err_max,
            "order_error_ci": [order_stats.ci_95_lower, order_stats.ci_95_upper],
            "total_error": total,
            "unit_roundoff": u,
            # 关键判据量：顺序误差相对该 dtype 自身舍入单位的倍数。
            # ≈1 表示顺序无关性完好（差异就是普通舍入）；
            # >>1 且随 K 增长，说明归并顺序在系统性地放大误差。
            "order_error_over_roundoff": (err_max / u) if u else float("inf"),
            "order_error_at_noise_floor": bool(err_max <= args.noise_tolerance * u),
        })
    return rows


# =============================================================================
# main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E8 低精度数值稳定性（GPU 权威 / CPU reduced）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtypes", nargs="+", default=["float16", "bfloat16"])
    p.add_argument("--num-blocks", dest="num_blocks", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128])
    p.add_argument("--tree-shapes", dest="tree_shapes", nargs="+",
                   default=list(TREE_SHAPES), choices=list(TREE_SHAPES))
    p.add_argument("--n-perms", dest="n_perms", type=int, default=20)
    p.add_argument("--noise-tolerance", dest="noise_tolerance", type=float, default=4.0,
                   help="判据阈值：order_error_max <= tolerance × unit_roundoff 则认为"
                        "误差仍淹没在 dtype 自身舍入内。默认 4（约 2 个 bit 的余量）。")
    p.add_argument("--d-h", dest="d_h", type=int, default=64)
    p.add_argument("--d-v", dest="d_v", type=int, default=64)
    p.add_argument("--logit-scale", dest="logit_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/gpu/e8")
    p.add_argument("--reduced", action="store_true",
                   help="缩小规模（K<=32, n_perms=8）以便本地快速跑通")
    p.add_argument("--allow-cpu", action="store_true",
                   help="显式确认在 CPU 上跑 reduced 版本")
    p.add_argument("--print-env", action="store_true")
    return p


def main() -> int:
    a = build_parser().parse_args()

    if a.print_env:
        return _env.print_env_only("E8 低精度数值稳定性", min_gpus=1, need_nccl=False)

    if a.device == "cpu":
        if not a.allow_cpu:
            print("=" * 78)
            print("阻断：E8 的权威运行在 GPU 上。")
            print("-" * 78)
            print("  --device cpu 只提供 reduced 复现：算子相同，但 CPU 的")
            print("  fp16/bf16 归约路径与 GPU tensor core 不同，误差常数不可迁移。")
            print("  该模式用于验证脚本逻辑与观察趋势，其数值不得作为 E8 结论。")
            print("  若确认要以 reduced 模式运行，请加 --allow-cpu。")
            print("=" * 78)
            return _env.GATE_EXIT_CODE
        print("[reduced] 在 CPU 上运行 —— 数值不可作为 E8 结论，仅用于验证脚本与趋势。")
    else:
        gate = _env.probe(min_gpus=1, need_nccl=False)
        code = _env.enforce(gate, "E8 低精度数值稳定性", "experiments/gpu/e8_low_precision.py")
        if code is not None:
            return code

    if a.reduced:
        a.num_blocks = [8, 16, 32]
        a.n_perms = 8

    device = torch.device(a.device)
    print("=" * 78)
    print(f"E8 低精度顺序无关性  device={a.device}  dtypes={a.dtypes}")
    print(f"  K={a.num_blocks}  tree={a.tree_shapes}  n_perms={a.n_perms}")
    print("=" * 78)

    all_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for dt_name in a.dtypes:
        dtype = _env.dtypes_for(dt_name)
        for K in a.num_blocks:
            try:
                rows = run_one(K, dtype, device, a)
                all_rows.extend(rows)
                for r in rows:
                    flag = "噪声内" if r["order_error_at_noise_floor"] else "**超噪声**"
                    print(f"  {dt_name:<9} K={K:<4} {r['tree_shape']:<10} "
                          f"order_err(max)={r['order_error_max']:.3e}  "
                          f"total_err={r['total_error']:.3e}  "
                          f"u={r['unit_roundoff']:.1e}  "
                          f"倍数={r['order_error_over_roundoff']:6.2f}  {flag}")
            except Exception as e:
                failures.append({"dtype": dt_name, "K": str(K),
                                 "error": f"{type(e).__name__}: {e}",
                                 "where": traceback.format_exc().splitlines()[-3]})
                print(f"  {dt_name:<9} K={K:<4} [不支持] {type(e).__name__}: {e}")

    payload: Dict[str, Any] = {
        "experiment": "E8",
        "device": a.device,
        "authoritative": a.device == "cuda",
        # 元数据此前**完全缺失**：E5/E6/E7 都写了，只有 E8 没有，而 E8 恰恰
        # 是最需要记录设备与精度口径的一个（低精度数值稳定性要按 dtype 解读）。
        "metadata": _env.build_metadata(
            run_id="e8-low-precision",
            model_name="synthetic",
            context_length=0,
            seed=getattr(a, "seed", 42),
            task="low-precision-numerics",
            precision=", ".join(a.dtypes),
            notes=("E8 无重复次数轴：测的是确定性误差界与置换散布，不是计时，"
                   "故 warmup/iters 记 0 —— 含义是「该轴不存在」，不是「没跑」。"),
            warmup=0, iters=0,
        ).to_dict(),
        "rows": all_rows,
        "unsupported_combinations": failures,
        "caveat": (
            "order_error 度量的是顺序无关性本身（各置换结果相对规范顺序结果的散布），"
            "total_error 度量的是相对 FP64 稠密 softmax 的总误差。二者不可混用："
            "低精度下 total_error 必然偏大（fp16 约 3 位有效十进制数），"
            "这不代表异步流水不可用；判据是 order_error 是否仍淹没在 dtype "
            "自身的舍入量级（unit_roundoff）内。"
            + ("" if a.device == "cuda" else
               " 本次为 CPU reduced 运行，偏差常数不可迁移到 GPU。")
        ),
    }

    out = REPO_ROOT / a.out
    R.save_json(str(out / "e8_results.json"), payload)
    if all_rows:
        R.save_csv(str(out / "e8.csv"), all_rows)
    print()
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
