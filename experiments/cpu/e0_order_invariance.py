#!/usr/bin/env python
"""E0：Online Softmax 归并的顺序无关性（CPU 可跑）

论文 §6.3 的 E0 已由仓库测试验证"在 FP64 下成立"，但明确留了一项待补：

    需把"置换数量"与"归并树形状"作为自变量，报告最大相对误差随二者的
    变化曲线，以界定异步实现的数值安全边界。

本脚本就是补这一项。三件事：

1. **置换数量**：随机打乱的次数越多，越可能碰到误差最大的那个顺序。
   这里不止"逐个记录"，而是给出**累计最大误差随置换次数 n 的收敛阶梯**
   （`--perm-ladder`）：若该量随 n 收敛，说明最大误差存在稳定上界、
   不存在"越试越坏"的病态顺序；若仍单调上升，则已报的最大值只是**下界**。
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
    python experiments/cpu/e0_order_invariance.py --perm-ladder 1 10 100 1000
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
# 置换数量的收敛阶梯
# =============================================================================

def perm_ladder(
    states: List[OnlineSoftmaxState],
    ref: torch.Tensor,
    ladder: List[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """置换数量作为自变量的收敛阶梯。

    对同一条随机置换序列逐次求相对误差，报告**前 n 次观测中的最大误差**。
    读法：若该量随 n 迅速收敛，说明最大误差存在稳定上界，异步乱序到达不会因
    "碰巧遇到某个病态顺序"而失控；若它随 n 持续上升且未见拐点，则当前报出的
    最大值只是**下界**，安全边界尚无证据。

    Args:
        states: K 个块状态（顺序固定）
        ref: 真值输出（把 K 个块拼起来做一次完整 softmax 注意力）
        ladder: 递增的试验次数阶梯
        seed: 随机置换的种子（固定后阶梯可复现）

    Returns:
        [{num_permutations, cumulative_max_rel_err}, ...]，按 ladder 升序

    Raises:
        ValueError: ladder 为空（空阶梯没有可报告的曲线）
    """
    if not ladder:
        raise ValueError("ladder 不能为空：空阶梯没有可报告的收敛曲线")
    if min(ladder) < 1:
        raise ValueError(f"ladder 只能取正整数，得到 {min(ladder)}")
    n_max = max(ladder)
    g = torch.Generator().manual_seed(seed)
    errs: List[float] = []
    for _ in range(n_max):
        perm = torch.randperm(len(states), generator=g).tolist()
        permuted = [states[i] for i in perm]
        errs.append(
            relative_error(attention_output(merge_left_deep(permuted)), ref)
        )

    out: List[Dict[str, Any]] = []
    running_max = 0.0
    cursor = 0
    for n in sorted(ladder):
        while cursor < n:
            running_max = max(running_max, errs[cursor])
            cursor += 1
        out.append({"num_permutations": n, "cumulative_max_rel_err": running_max})
    return out


# =============================================================================
# 派生判据量
# =============================================================================

def experiment_summary(
    rows: List[Dict[str, Any]],
    ladder_all: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """把树形状对比与阶梯汇总成论文可直接引用的判据量。

    两个量：
      - tree_shape_balanced_wins / total：平衡树误差不大于顺序归并的格数。
        读法看**符号分布而非单格**——单次实现的树间比较会被"本次顺序恰好
        是否走运"支配，只有跨 (精度, K) 的整体符号分布才有解释力。
      - ladder_ratio_200_to_1000_max_<dtype>：把试验次数从 200 提到 1000
        （5 倍）时，累计最大相对误差的最大增长比。若该比值接近 1，说明
        报出的最大值是稳定上界；若显著大于 1 且随 n 无拐点，则只是下界。

    注意 FP64 一栏整体处于机器舍入量级（约 1e-16），其"增长比"
    由 1--2 ULP 的抖动主导，不可与 FP32 一栏同口径比较。
    """
    out: Dict[str, Any] = {
        "tree_shape_balanced_wins": sum(
            1 for r in rows if r["balanced_better_than_seq"]
        ),
        "tree_shape_total": len(rows),
    }

    for dt in sorted({r["dtype"] for r in ladder_all}):
        ratios: List[float] = []
        for K in sorted({r["num_blocks"] for r in ladder_all if r["dtype"] == dt}):
            per_n = {
                r["num_permutations"]: r["cumulative_max_rel_err"]
                for r in ladder_all
                if r["dtype"] == dt and r["num_blocks"] == K
            }
            if 200 in per_n and 1000 in per_n and per_n[200] > 0:
                ratios.append(per_n[1000] / per_n[200])
        out[f"ladder_ratio_200_to_1000_max_{dt}"] = (
            max(ratios) if ratios else None
        )

    return out


# =============================================================================
# 主实验
# =============================================================================

def run(args) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    ladder_all: List[Dict[str, Any]] = []

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

            # --- 置换数量的收敛阶梯（补 §6 E0 的待补项） ---
            ladder_rows = perm_ladder(states, ref, args.perm_ladder, seed=args.seed)
            ladder_max = ladder_rows[-1]["cumulative_max_rel_err"]
            ladder_prev = (
                ladder_rows[-2]["cumulative_max_rel_err"]
                if len(ladder_rows) > 1
                else ladder_max
            )
            ladder_tail_rel_growth = (
                (ladder_max - ladder_prev) / ladder_prev if ladder_prev > 0 else 0.0
            )
            for _lr in ladder_rows:
                ladder_all.append({
                    "dtype": dtype_name,
                    "num_blocks": num_blocks,
                    "num_permutations": _lr["num_permutations"],
                    "cumulative_max_rel_err": _lr["cumulative_max_rel_err"],
                })

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
                "ladder_final_max": ladder_max,
                "ladder_tail_rel_growth": ladder_tail_rel_growth,
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
            print(
                f"     置换阶梯 max@{max(args.perm_ladder)}={ladder_max:9.3e}  "
                f"尾部相对增长={ladder_tail_rel_growth:+.2%}"
            )

    return {
        "experiment": "E0",
        "rows": rows,
        "perm_ladder": ladder_all,
        "summary": experiment_summary(rows, ladder_all),
    }


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
    p.add_argument("--perm-ladder", dest="perm_ladder", type=int, nargs="+",
                   default=[1, 2, 5, 10, 25, 50, 100, 200, 500, 1000],
                   help="置换数量的收敛阶梯（累计最大相对误差随试验次数的曲线）")
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
        args.perm_ladder = [1, 5, 25, 100]

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
    print("  - 置换阶梯：看这个 max 随试验次数是否收敛。尾部相对增长接近 0")
    print("    才是稳定上界；若仍在上升，报出的 max 只是下界。")
    print("  - FP32 vs FP64 的差距给出低精度下的量级参照（E8 需在 GPU 上规模化）。")

    sm = payload["summary"]
    print()
    print("派生判据量：")
    print(
        f"  - 平衡树不劣于顺序归并：{sm['tree_shape_balanced_wins']}"
        f"/{sm['tree_shape_total']} 格（看符号分布，不看单格）"
    )
    for dt in sorted(k for k in sm if k.startswith("ladder_ratio_")):
        v = sm[dt]
        print(f"  - {dt}: {v:.4f}" if v is not None else f"  - {dt}: (无)")

    out_dir = REPO_ROOT / args.out
    R.save_json(str(out_dir / "e0_results.json"), payload)
    R.save_csv(str(out_dir / "e0.csv"), payload["rows"])
    R.save_csv(str(out_dir / "e0_perm_ladder.csv"), payload["perm_ladder"])
    print()
    print(f"结果已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
