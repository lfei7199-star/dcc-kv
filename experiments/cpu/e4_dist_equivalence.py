#!/usr/bin/env python
"""E4：分布式等价性 —— 同步 DCC-KV vs 完整注意力（CPU 可跑，gloo）

论文 §6.3 的 E4。仓库已有对应测试
（`tests/test_dist_equivalence.py`、`tests/test_var_len_msg.py`），
本脚本把它们的结论整理成可产表的报告，并补一项测试未覆盖的对照：

    单 rank 视角下，用"逐边条件化紧凑 KV"算出的注意力，
    与用"完整 KV"算出的注意力差多少。

**为什么需要 gcc 的那部分仍然由现有测试承担**：
真多进程（2 进程 gloo）的 all_reduce / broadcast / 变长 all_to_all_v
需要 spawn 子进程，放在 pytest 里更合适。本脚本在最后给出运行命令，
不重复实现。

一个重要说明：E4 的阈值差异（$10^{-2}$ vs Ring Attention 的 $10^{-5}$）
反映的是**压缩近似误差**，不是实现错误。这一点论文已说明，本脚本
通过同时报告"紧凑 vs 完整"和"分块 vs 整体"两条差异来佐证它 ——
若后者很小而前者较大，就证明差异来自压缩而非通信实现。

用法
----
    python experiments/cpu/e4_dist_equivalence.py --out results/cpu/e4
    python experiments/cpu/e4_dist_equivalence.py --quick
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import synthetic as S  # noqa: E402
from experiments.common import report as R  # noqa: E402
from src.dcc_kv_ref import (  # noqa: E402
    merge_softmax_states,
    online_softmax_from_attention,
)
from src.distributed.dcc_kv_sync_cpu import (  # noqa: E402
    dcc_kv_sync_attention_single_rank,
)


def blockwise_no_compression(
    query: torch.Tensor,
    all_keys: List[torch.Tensor],
    all_values: List[torch.Tensor],
) -> torch.Tensor:
    """隔离对照：逐块算注意力后归并，**完全不压缩、不加 β**。

    这是区分"通信/归并实现是否正确"与"压缩近似误差有多大"的关键对照。

    为什么需要一个这样的对照：把 `budget` 设为块长并不能关掉压缩链路 ——
    `build_compact_kv` 仍会跑 Value 回归（在 M < B 时欠定）并施加 β。
    因此"预算=块长"测的是"压缩链路在无预算约束下的行为"，
    而不是通信本身。只有这里这个什么都不做的版本，才能把通信环节单独暴露出来。
    它应当与 dense 只差浮点舍入量级。
    """
    d_h = all_keys[0].shape[-1]
    scale = 1.0 / (d_h ** 0.5)
    states = []
    for K_r, V_r in zip(all_keys, all_values):
        logits = (query @ K_r.T) * scale
        states.append(online_softmax_from_attention(logits, V_r))
    merged = states[0]
    for st in states[1:]:
        merged = merge_softmax_states(merged, st)
    return merged.o / merged.l


def make_chunked(
    L_total: int, d_h: int, d_v: int, world_size: int, dtype: torch.dtype, seed: int,
):
    """把整段序列切成 world_size 块，返回完整张量与分块列表。"""
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(L_total, d_h, generator=g, dtype=dtype)
    K = torch.randn(L_total, d_h, generator=g, dtype=dtype)
    V = torch.randn(L_total, d_v, generator=g, dtype=dtype)

    chunk = L_total // world_size
    all_K: List[torch.Tensor] = []
    all_V: List[torch.Tensor] = []
    for s in range(world_size):
        end = L_total if s == world_size - 1 else (s + 1) * chunk
        all_K.append(K[s * chunk:end])
        all_V.append(V[s * chunk:end])
    return Q, K, V, all_K, all_V


def run(args) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for world_size in args.world_sizes:
        L_total = args.chunk_size * world_size
        Q, K, V, all_K, all_V = make_chunked(
            L_total, args.d_h, args.d_v, world_size, args._dtype, args.seed
        )

        # 真值：整段一次算完的完整注意力
        ref = S.dense_attention(Q, K, V)

        # 目的端 Query 映射：每个 rank 拿到整段的 Q（mock 场景）
        all_dest_queries = {r: Q for r in range(world_size)}
        budgets = {r: args.budget for r in range(world_size)}

        outs = []
        for i in range(L_total):
            outs.append(
                dcc_kv_sync_attention_single_rank(
                    query=Q[i],
                    all_keys=all_K,
                    all_values=all_V,
                    all_dest_queries=all_dest_queries,
                    budgets=budgets,
                    num_repr_queries=args.num_repr_queries,
                    projection_dim=args.projection_dim,
                )
            )
        got = torch.stack(outs, dim=0)

        abs_diff = (got - ref).abs()
        max_abs = float(abs_diff.max())
        rel = float((got - ref).norm() / (ref.norm() + 1e-12))

        # 对照：不做任何压缩的完整注意力（应当几乎精确）
        # 用同一套分块逻辑但预算 = 块长，即"不压缩"
        no_compress_budgets = {r: all_K[r].shape[0] for r in range(world_size)}
        outs_nc = []
        for i in range(L_total):
            outs_nc.append(
                dcc_kv_sync_attention_single_rank(
                    query=Q[i],
                    all_keys=all_K,
                    all_values=all_V,
                    all_dest_queries=all_dest_queries,
                    budgets=no_compress_budgets,
                    num_repr_queries=args.num_repr_queries,
                    projection_dim=args.projection_dim,
                )
            )
        got_nc = torch.stack(outs_nc, dim=0)
        max_abs_nc = float((got_nc - ref).abs().max())

        # 隔离对照：完全不压缩、不加 β 的分块归并
        outs_iso = torch.stack(
            [blockwise_no_compression(Q[i], all_K, all_V) for i in range(L_total)],
            dim=0,
        )
        max_abs_iso = float((outs_iso - ref).abs().max())
        rel_iso = float((outs_iso - ref).norm() / (ref.norm() + 1e-12))

        rows.append({
            "world_size": world_size,
            "chunk_size": args.chunk_size,
            "L_total": L_total,
            "budget": args.budget,
            "max_abs_diff_compressed": max_abs,
            "rel_diff_compressed": rel,
            "within_repo_threshold_1e-2": bool(max_abs < 1e-2),
            "max_abs_diff_budget_eq_chunk": max_abs_nc,
            "max_abs_diff_no_compression_isolated": max_abs_iso,
            "rel_diff_no_compression_isolated": rel_iso,
            # 通信/归并环节是否干净：隔离对照应只差浮点舍入
            "merge_path_correct": bool(max_abs_iso < 1e-5),
            # 误差是否可归因于压缩链路
            "gap_from_compaction_pipeline": bool(max_abs_iso < max_abs / 10.0),
        })

        print(
            f"  world_size={world_size}  B={args.budget:<4}  "
            f"压缩版={max_abs:.3e}  "
            f"预算=块长={max_abs_nc:.3e}  "
            f"隔离对照(无压缩/无β)={max_abs_iso:.3e}  "
            f"{'✓通信干净' if rows[-1]['merge_path_correct'] else '✗通信环节可疑'}"
            f"{'  [偏差来自压缩链路]' if rows[-1]['gap_from_compaction_pipeline'] else ''}"
        )

    return {"experiment": "E4", "rows": rows}


def run_existing_tests(
    targets: Optional[List[str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """调用仓库已有的多进程等价性测试，把结果并入报告。

    真 2 进程 gloo 的 all_reduce / broadcast / 变长 all_to_all_v 由这些
    测试覆盖；本脚本不重复实现，只转发结果。

    为什么失败要自带解释（审查项 F5）：2026-09-16 的落盘里只留下
    ``{"ran": true, "returncode": 1, "summary": ""}`` —— 一个非零返回码
    配一段空 summary，既没有 stderr 也没有命令行，事后**无法追查**是
    收集阶段崩了还是断言失败。因此这里：

    * 把实际 argv 记进 ``command``，让失败可原样复跑；
    * 同时保留 stdout 与 **stderr** 尾部 —— pytest 在收集/启动阶段
      中止时，线索通常只在 stderr；
    * 非零码时补一条 ``note``，说明该返回码该怎么读。

    这样"转发测试失败"这一事件在产物内部就是自解释的，不需要依赖
    外部日志或人工记忆。

    ``targets`` / ``cwd`` 只为**测试注入**而开放（默认即现状），
    使非零返回码这条路径能在本机真跑出来，而不是只做源码字符串检查。
    """
    if targets is None:
        targets = [
            "tests/test_dist_equivalence.py",
            "tests/test_var_len_msg.py",
        ]
    workdir = str(REPO_ROOT) if cwd is None else cwd
    argv = [sys.executable, "-m", "pytest", *targets, "-q", "--no-header"]
    base: Dict[str, Any] = {
        "targets": list(targets),
        "command": " ".join(argv),
        "cwd": workdir,
        "pass_criterion": "returncode == 0；非零即视为未通过，不得写作「E4 通过」。",
    }
    try:
        proc = subprocess.run(
            argv, cwd=workdir, capture_output=True, text=True, timeout=900,
        )
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-6:])
        err_tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        out: Dict[str, Any] = {
            **base,
            "ran": True,
            "returncode": proc.returncode,
            "summary": tail,
            "stderr_tail": err_tail,
        }
        if proc.returncode != 0:
            out["note"] = (
                "转发测试以非零码退出。summary 为空通常意味着 pytest 在收集或启动"
                "阶段就中止（而不是断言失败）——此时线索在 stderr_tail 里。"
                "此前（2026-09-16）的落盘缺这一字段，只留下 returncode=1 与空"
                "summary，导致该失败事后不可追查。"
            )
        return out
    except FileNotFoundError:
        return {**base, "ran": False, "reason": "pytest 未安装"}
    except subprocess.TimeoutExpired:
        return {**base, "ran": False, "reason": "测试超时"}
    except Exception as exc:  # noqa: BLE001
        return {**base, "ran": False, "reason": f"{type(exc).__name__}: {exc}"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E4：分布式等价性（CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--world-sizes", dest="world_sizes", type=int, nargs="+",
                   default=[2, 4], help="模拟的 rank 数（单进程内模拟，非真多进程）")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=64)
    p.add_argument("--budget", type=int, default=16, help="每边的压缩预算 B")
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--num-repr-queries", dest="num_repr_queries", type=int, default=16)
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--skip-pytest", dest="skip_pytest", action="store_true",
                   help="跳过转发仓库 pytest（默认会跑，用于并入多进程结果）")
    p.add_argument("--out", type=str, default="results/cpu/e4")
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
        args.world_sizes = [2]
        args.chunk_size = 32

    print("=" * 78)
    print("E4：分布式等价性 —— 同步 DCC-KV vs 完整注意力")
    print("=" * 78)

    payload = run(args)

    if not args.skip_pytest:
        print()
        print("  转发仓库现有测试（真多进程 gloo 部分）...")
        payload["repo_tests"] = run_existing_tests()
        rt = payload["repo_tests"]
        if rt.get("ran"):
            print(f"    pytest returncode={rt['returncode']}")
            for line in rt["summary"].splitlines():
                print(f"      {line}")
        else:
            print(f"    未运行：{rt.get('reason')}")

    print()
    print("说明：三档对照用于定位误差来源 ——")
    print("  1. 隔离对照（逐块算 attention 后归并，完全不压缩、不加 β）")
    print("     只应差浮点舍入量级。若不是，问题在通信/归并实现。")
    print("  2. 预算=块长：压缩链路在无预算约束下的行为。")
    print("     注意这**不是**无操作 —— build_compact_kv 仍会做 Value 回归并施加 β。")
    print("  3. 压缩版：真实预算下的端到端偏差。")

    out_dir = REPO_ROOT / args.out
    R.save_summary(str(out_dir / "e4_results.json"), payload)
    R.save_csv(str(out_dir / "e4.csv"), payload["rows"])
    print()
    print(f"结果已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
