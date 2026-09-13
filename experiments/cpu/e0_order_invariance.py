#!/usr/bin/env python
"""E0：Online Softmax 归并的顺序无关性（CPU 可跑）

论文 §6.3 的 E0 已由仓库测试验证"在 FP64 下成立"，但明确留了一项待补：

    需把"置换数量"与"归并树形状"作为自变量，报告最大相对误差随二者的
    变化曲线，以界定异步实现的数值安全边界。

本脚本就是补这一项。三件事：

1. **置换数量**：随机打乱的次数越多，越可能碰到误差最大的那个顺序。
   逐个记录，看最大相对误差是否随置换数收敛（若发散，说明存在病态顺序）。
2. **归并树形状**：顺序（左深树）vs 平衡二叉树 vs 随机树。
   为什么重要 —— 异步 All-to-Allv 中消息到达顺序不可控，
   但**归并结构可由实现决定**；若平衡树显著优于左深树，
   实现就应当采用平衡归并。这是可以直接落到工程上的结论。
3. **浮点精度**：FP32 vs FP64。E8 要在低精度下重做这件事（需 GPU 规模化），
   这里先给出小规模的精度敏感性参照。

参考实现：Milakov & Gimelshein 2018（arXiv:1805.02867）。
注意归并算子本身**不是本文原创**，本文的原创是把它用于异步到达的变长消息。

用法
----
    python experiments/cpu/e0_order_invariance.py --out results/cpu/e0
    python experiments/cpu/e0_order_invariance.py --quick
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

from experiments.common import report as R  # noqa: E402
from src.dcc_kv_ref import (  # noqa: E402
    OnlineSoftmaxState,
    merge_softmax_states,
    online_softmax_from_attention,
)
from src.dcc_kv_ref.online_softmax import merge_softmax_states_list  # noqa: E402


# =============================================================================
# 归并树形状
# =============================================================================

def merge_left_deep(states: List[OnlineSoftmaxState]) -> OnlineSoftmaxState:
    """左深树（顺序归并）—— 模拟"消息按到达顺序逐个累加"。"""
    return merge_softmax_states_list(states)


def merge_balanced(states: List[OnlineSoftmaxState]) -> OnlineSoftmaxState:
    """平衡二叉树归并 —— 每层两两配对。

    树深为 ceil(log2 K) 而非 K，累积的舍入误差更少。
    对异步实现的意义：缓冲到成对再归并，可以降低误差上界。
    """
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
    """随机配对树 —— 模拟不可控的到达序与不确定的分组策略。"""
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


# =============================================================================
# 误差度量
# =============================================================================

def attention_output(state: OnlineSoftmaxState) -> torch.Tensor:
    return state.o / state.l


def reference_output(
    blocks_logits: List[torch.Tensor],
    blocks_values: List[torch.Tensor],
) -> torch.Tensor:
    """把所有块拼起来，做一次完整（非分块）的 softmax 注意力。

    这才是真值 —— 分块归并只是它的一个数值等价实现。
    """
    logits = torch.cat(blocks_logits, dim=0)
    values = torch.cat(blocks_values, dim=0)
    w = torch.softmax(logits, dim=-1)
    return w @ values


def relative_error(got: torch.Tensor, ref: torch.Tensor) -> float:
    """相对 L2 误差 —— 用真值范数归一化，跨量级可比。"""
    return float((got - ref).norm() / (ref.norm() + 1e-12))


def make_blocks(
    num_blocks: int,
    d_h: int,
    d_v: int,
    tokens_per_block: int,
    dtype: torch.dtype,
    seed: int,
):
    """构造 K 个长度不等的块（模拟变长消息），并给出它们的状态与真值。

    刻意让每块长度不一，因为异步 All-to-Allv 交换的正是**变长**消息。
    """
    g = torch.Generator().manual_seed(seed)
    logits_list: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []
    states: List[OnlineSoftmaxState] = []

    for k in range(num_blocks):
        # 长度在 [tokens_per_block/2, tokens_per_block*3/2] 内变化
        L = int(tokens_per_block * (0.5 + torch.rand(1, generator=g).item()))
        L = max(L, 2)
        lg = torch.randn(L, generator=g, dtype=dtype) * 4.0  # 放大 logit 动态范围
        v = torch.randn(L, d_v, generator=g, dtype=dtype)
        logits_list.append(lg)
        values_list.append(v)
        states.append(online_softmax_from_attention(lg, v))

    return logits_list, values_list, states


# =============================================================================
# 主实验
# =============================================================================

def run(args) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for dtype_name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        for num_blocks in args.num_blocks:
            logits_list, values_list, states = make_blocks(
                num_blocks=num_blocks,
                d_h=args.d_h,
                d_v=args.d_v,
                tokens_per_block=args.tokens_per_block,
                dtype=dtype,
                seed=args.seed,
            )
            ref = reference_output(logits_list, values_list)

            # --- 左深（顺序） ---
            err_seq = relative_error(attention_output(merge_left_deep(states)), ref)

            # --- 平衡树 ---
            err_bal = relative_error(attention_output(merge_balanced(states)), ref)

            # --- 随机置换：逐个累积，记录最大误差 ---
            g = torch.Generator().manual_seed(args.seed)
            errs_perm: List[float] = []
            for _ in range(args.num_permutations):
                perm = torch.randperm(num_blocks, generator=g).tolist()
                permuted = [states[i] for i in perm]
                errs_perm.append(
                    relative_error(attention_output(merge_left_deep(permuted)), ref)
                )

            # --- 随机树 ---
            errs_tree: List[float] = []
            for t in range(args.num_trees):
                errs_tree.append(
                    relative_error(
                        attention_output(merge_random_tree(states, seed=args.seed + t)),
                        ref,
                    )
                )

            perm_sum = R.summarize(errs_perm, "rel_err_permuted", "ratio", seed=args.seed)
            tree_sum = R.summarize(errs_tree, "rel_err_random_tree", "ratio", seed=args.seed)

            rows.append({
                "dtype": dtype_name,
                "num_blocks": num_blocks,
                "tokens_per_block": args.tokens_per_block,
                "num_permutations": args.num_permutations,
                "err_sequential": err_seq,
                "err_balanced_tree": err_bal,
                "err_permuted_max": max(errs_perm),
                "err_permuted_median": perm_sum.median,
                "err_permuted_p95": perm_sum.p95,
                "err_random_tree_median": tree_sum.median,
                "balanced_better_than_seq": bool(err_bal <= err_seq),
                "max_err_all": max(
                    [err_seq, err_bal, max(errs_perm), max(errs_tree)]
                ),
            })

            r = rows[-1]
            print(
                f"  {dtype_name:<8} K={num_blocks:<3} "
                f"顺序={err_seq:9.3e}  平衡树={err_bal:9.3e}  "
                f"置换max={max(errs_perm):9.3e}  随机树={tree_sum.median:9.3e}  "
                f"{'平衡≤顺序' if r['balanced_better_than_seq'] else '平衡>顺序'}"
            )

    return {"experiment": "E0", "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E0：Online Softmax 归并的顺序无关性（CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-blocks", dest="num_blocks", type=int, nargs="+",
                   default=[2, 4, 8, 16, 32, 64], help="块数 K 的扫描")
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--tokens-per-block", dest="tokens_per_block", type=int, default=64,
                   help="每块 token 数的基准值（实际长度在其 0.5~1.5 倍间变化）")
    p.add_argument("--num-permutations", dest="num_permutations", type=int, default=200,
                   help="每个设定下的随机置换次数")
    p.add_argument("--num-trees", dest="num_trees", type=int, default=50,
                   help="每个设定下的随机归并树个数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/cpu/e0")
    p.add_argument("--quick", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.quick:
        args.num_blocks = [2, 8, 32]
        args.num_permutations = 50
        args.num_trees = 20

    print("=" * 78)
    print("E0：Online Softmax 归并的顺序无关性")
    print("    补论文 §6.3 的待补项：置换数 × 归并树形状 → 最大相对误差")
    print("=" * 78)

    payload = run(args)

    print()
    print("读取方式：")
    print("  - 顺序 vs 平衡树：若平衡树误差系统性更小，异步实现应改用平衡归并")
    print("    （消息成对缓冲后归并），因为它把树深从 K 降到 log2(K)。")
    print("  - 置换 max：异步乱序到达的最坏情况误差。这就是数值安全边界。")
    print("  - FP32 vs FP64 的差距给出低精度下的量级参照（E8 需在 GPU 上规模化）。")

    out_dir = REPO_ROOT / args.out
    R.save_json(str(out_dir / "e0_results.json"), payload)
    R.save_csv(str(out_dir / "e0.csv"), payload["rows"])
    print()
    print(f"结果已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
