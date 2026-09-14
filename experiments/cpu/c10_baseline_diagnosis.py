#!/usr/bin/env python
r"""C10：两项压缩基线测试失败的根因定位（CPU 可跑）

要回答的问题
------------
`tests/test_dist_equivalence.py` 里两条断言长期失败：

- `test_fast_kv_vs_dense_within_tolerance`：实测 1.0898，阈值 0.1
- `test_apb_vs_dense_within_tolerance`：实测 1.9944，阈值 0.3

论文早期版本把它们笼统归为「与 $\beta$ 改动无关的既有问题」，未做定位。
本脚本把根因拆成三组可分别验证的对照。

三组对照
--------
**D1（口径）**
    两个 mock 都在**整块**上标定压缩（代表 Query、质量目标、回归目标都取自
    完整块，含未来 token），而因果分支只能按位置前缀使用它。先测
    causal=True / False 的差异，判断误差里有多少来自压缩本身、
    多少来自这种前缀不一致。

**D2（误差分解）**
    FastKV 在 `budget = chunk_size = 64 = L_s` 时 `selected_indices` 恰为
    0..63、`compact.keys` 与原始 K 逐位相同，即**没有发生选键压缩**。
    把 `compact.values` 按 `selected_indices` 换回精确 V，即可把
    $\beta$ + 选键的贡献与 $V$ 回归的贡献分开。

**D3（APB 位置序）**
    `torch.topk` 返回的是**按质量降序**的索引，与位置序无关；旧代码用
    `[:end_in_chunk]` 当位置前缀切片。本脚本对比「行序切片」（原实现）
    与「按 `selected_indices` 的位置掩码」（修正后），并用
    `anchor_budget = 块长` 的精确性作为 mock 自身的正确性锚点。

判据与诚实边界
--------------
- 数据全部合成（随机 Q/K/V），只报机制级数值，不构成任何任务质量主张。
- 本脚本只做定位与记录，不改变任何默认超参。
- 结论口径：这三组对照解释的是*为什么那两条断言不成立*，
  不是*压缩方法本身好不好*。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import (  # noqa: E402
    APBConfig,
    FastKVConfig,
    apb_cpu,
    fast_kv_cpu,
    ring_attention_dense,
)
from src.dcc_kv_ref import (  # noqa: E402
    build_compact_kv,
    merge_softmax_states,
    online_softmax_from_attention,
)


# -----------------------------------------------------------------------------
# 场景
# -----------------------------------------------------------------------------
def make_scene(seed: int = 0, L: int = 256, d_h: int = 32, d_v: int = 32):
    torch.manual_seed(seed)
    Q = torch.randn(L, d_h, dtype=torch.float64)
    K = torch.randn(L, d_h, dtype=torch.float64)
    V = torch.randn(L, d_v, dtype=torch.float64)
    return Q, K, V


def _iter_chunks(queries, keys, values, chunk_size, world_size, cfg, exact_values):
    """按 FastKV 语义构造每块的紧凑表示。

    返回 [(offset, idx, K_b, V_b, bias_b)]，其中 idx 是**位置升序**的原始索引。
    `exact_values=True` 时用原始 V 的对应行替换回归出的紧凑 Value。
    """
    L = queries.shape[0]
    out = []
    for s in range(world_size):
        off = s * chunk_size
        end = (s + 1) * chunk_size if s < world_size - 1 else L
        K_s, V_s = keys[off:end], values[off:end]
        compact = build_compact_kv(
            source_keys=K_s,
            source_values=V_s,
            destination_queries=queries,
            budget=min(cfg.budget, end - off),
            num_representative_queries=cfg.num_repr_queries,
            projection_dim=cfg.projection_dim,
            lambda_beta=cfg.lambda_beta,
            lambda_value=cfg.lambda_value,
            seed=cfg.seed,
        )
        idx = compact.selected_indices
        V_b = V_s[idx] if exact_values else compact.values
        out.append((off, idx, compact.keys, V_b, compact.logit_bias))
    return out


def _merge_chunks(queries, chunks, chunk_size, world_size, causal, mask_by_position):
    """把各块的紧凑表示按 Ring 语义归并成输出。

    `mask_by_position=True` 时按原始位置做因果掩码（修正后语义）；
    否则按数组行序切片（原实现语义）。
    """
    L = queries.shape[0]
    scale = 1.0 / (queries.shape[-1] ** 0.5)
    outs = []
    for r in range(L):
        states = []
        for off, idx, K_b, V_b, bias_b in chunks:
            if causal and off > r:
                continue
            if causal:
                if mask_by_position:
                    keep = idx < (r + 1 - off)
                else:
                    keep = torch.arange(idx.shape[0]) < min(
                        r + 1 - off, idx.shape[0]
                    )
                if not bool(keep.any()):
                    continue
                K_v, V_v, b_v = K_b[keep], V_b[keep], bias_b[keep]
            else:
                K_v, V_v, b_v = K_b, V_b, bias_b
            logits = (queries[r] @ K_v.T) * scale + b_v
            states.append(online_softmax_from_attention(logits, V_v))
        merged = states[0]
        for st in states[1:]:
            merged = merge_softmax_states(merged, st)
        outs.append(merged.o / merged.l)
    return torch.stack(outs)


def _apb_chunks(queries, keys, values, chunk_size, world_size, budget):
    """按 APB 语义选 anchor（按质量），但保留**位置升序**的索引与原始行序。"""
    L = queries.shape[0]
    scale = 1.0 / (queries.shape[-1] ** 0.5)
    out = []
    for s in range(world_size):
        off = s * chunk_size
        end = (s + 1) * chunk_size if s < world_size - 1 else L
        K_s, V_s = keys[off:end], values[off:end]
        future = queries[off:] if len(queries[off:]) else queries
        mass = torch.softmax((future @ K_s.T) * scale, dim=-1).sum(dim=0)
        raw = torch.topk(mass, min(budget, len(K_s))).indices  # 质量降序
        out.append(
            {
                "off": off,
                "roworder": (K_s[raw], V_s[raw]),
                "position": (K_s[raw.sort().values], V_s[raw.sort().values], raw.sort().values),
            }
        )
    return out


def _apb_merge(queries, chunks, causal, use_mask):
    L = queries.shape[0]
    scale = 1.0 / (queries.shape[-1] ** 0.5)
    outs = []
    for r in range(L):
        states = []
        for ch in chunks:
            off = ch["off"]
            if causal and off > r:
                continue
            if use_mask:
                K_a, V_a, idx = ch["position"]
                if causal:
                    keep = idx < (r + 1 - off)
                    if not bool(keep.any()):
                        continue
                    K_a, V_a = K_a[keep], V_a[keep]
            else:
                K_a, V_a = ch["roworder"]
                if causal:
                    n = min(r + 1 - off, K_a.shape[0])
                    if n <= 0:
                        continue
                    K_a, V_a = K_a[:n], V_a[:n]
            states.append(
                online_softmax_from_attention((queries[r] @ K_a.T) * scale, V_a)
            )
        merged = states[0]
        for st in states[1:]:
            merged = merge_softmax_states(merged, st)
        outs.append(merged.o / merged.l)
    return torch.stack(outs)


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


# -----------------------------------------------------------------------------
# 三组对照
# -----------------------------------------------------------------------------
def run_d1(Q, K, V, chunk_size, world_size) -> Dict[str, Any]:
    print("=" * 100)
    print("D1：口径 —— 因果前缀切片 vs 完整块（误差中有多少来自压缩本身）")
    print("=" * 100)
    cfg = FastKVConfig(budget=64, num_repr_queries=32, projection_dim=16)
    apb_cfg = APBConfig(anchor_budget=32, num_anchor_queries=16)
    rows = []
    for causal in (True, False):
        dense = ring_attention_dense(Q, K, V, chunk_size, world_size, causal=causal)
        fkv = fast_kv_cpu(Q, K, V, chunk_size, world_size, config=cfg, causal=causal)
        apb = apb_cpu(Q, K, V, chunk_size, world_size, config=apb_cfg, causal=causal)
        rec = {
            "causal": causal,
            "dense_absmax": float(dense.abs().max().item()),
            "fastkv_maxdiff": _maxdiff(fkv, dense),
            "apb_maxdiff": _maxdiff(apb, dense),
        }
        rows.append(rec)
        print(
            f"  causal={str(causal):5s}  FastKV={rec['fastkv_maxdiff']:.4f}"
            f"   APB(b32)={rec['apb_maxdiff']:.4f}"
            f"   |Dense|max={rec['dense_absmax']:.4f}"
        )
    print("  ⇒ 非因果给出压缩本身的误差；因果额外叠加前缀不一致。")
    return {"rows": rows}


def run_d2(Q, K, V, chunk_size, world_size) -> Dict[str, Any]:
    print()
    print("=" * 100)
    print("D2：误差分解 —— FastKV 在 B = L_s（无选键压缩）下把回归 V 换回精确 V")
    print("=" * 100)
    cfg = FastKVConfig(budget=64, num_repr_queries=32, projection_dim=16)
    res: Dict[str, Any] = {}
    for causal in (True, False):
        dense = ring_attention_dense(Q, K, V, chunk_size, world_size, causal=causal)
        fitted = _merge_chunks(
            Q,
            _iter_chunks(Q, K, V, chunk_size, world_size, cfg, exact_values=False),
            chunk_size, world_size, causal, mask_by_position=True,
        )
        exact = _merge_chunks(
            Q,
            _iter_chunks(Q, K, V, chunk_size, world_size, cfg, exact_values=True),
            chunk_size, world_size, causal, mask_by_position=True,
        )
        src = fast_kv_cpu(Q, K, V, chunk_size, world_size, config=cfg, causal=causal)
        res[f"causal_{causal}"] = {
            "src_fast_kv_maxdiff": _maxdiff(src, dense),
            "ref_regressed_v_maxdiff": _maxdiff(fitted, dense),
            "ref_exact_v_maxdiff": _maxdiff(exact, dense),
        }
        print(
            f"  causal={str(causal):5s}  src(fast_kv_cpu)={res[f'causal_{causal}']['src_fast_kv_maxdiff']:.4f}"
            f"   参考实现/回归V={res[f'causal_{causal}']['ref_regressed_v_maxdiff']:.4f}"
            f"   参考实现/精确V={res[f'causal_{causal}']['ref_exact_v_maxdiff']:.4f}"
        )
    # 检查 B = L_s 时是否真的没有选键压缩
    idx_chk = _iter_chunks(Q, K, V, chunk_size, world_size, cfg, exact_values=False)
    all_identity = all(
        bool(torch.equal(idx, torch.arange(idx.shape[0]))) for _, idx, _, _, _ in idx_chk
    )
    res["selected_indices_is_identity"] = all_identity
    print(f"  selected_indices 恰为 0..L_s-1（即未发生选键压缩）: {all_identity}")
    res["fastkv_config"] = {
        "budget": cfg.budget,
        "num_repr_queries": cfg.num_repr_queries,
        "projection_dim": cfg.projection_dim,
        "lambda_beta": cfg.lambda_beta,
        "lambda_value": cfg.lambda_value,
        "seed": cfg.seed,
    }
    print(
        f"  FastKVConfig: budget={cfg.budget} M={cfg.num_repr_queries} "
        f"d_p={cfg.projection_dim} lambda_beta={cfg.lambda_beta:g} "
        f"lambda_value={cfg.lambda_value:g}"
    )
    print(
        "  ⇒ 精确 V 显著压低偏差 ⇒ 非因果下的误差主要来自 Value 回归"
        "（而不是 β 或选键）；注意该残差对 lambda_beta 敏感。"
    )
    return res


def run_d3(Q, K, V, chunk_size, world_size) -> Dict[str, Any]:
    print()
    print("=" * 100)
    print("D3：APB 位置序缺陷 —— 行序切片（原实现） vs 位置掩码（修正后）")
    print("=" * 100)
    res: Dict[str, Any] = {}
    for budget in (32, 64):
        chunks = _apb_chunks(Q, K, V, chunk_size, world_size, budget)
        raw0 = torch.topk(
            torch.softmax(
                (Q[0:chunk_size] @ K[0:chunk_size].T)
                / (Q.shape[-1] ** 0.5),
                dim=-1,
            ).sum(dim=0),
            budget,
        ).indices
        res[f"topk_head_b{budget}"] = raw0[:6].tolist()
        res[f"topk_is_position_sorted_b{budget}"] = bool(
            torch.all(raw0[1:] > raw0[:-1])
        )
        for causal in (True, False):
            dense = ring_attention_dense(Q, K, V, chunk_size, world_size, causal=causal)
            row = _apb_merge(Q, chunks, causal, use_mask=False)
            pos = _apb_merge(Q, chunks, causal, use_mask=True)
            res[f"b{budget}_causal_{causal}"] = {
                "roworder_maxdiff": _maxdiff(row, dense),
                "position_maxdiff": _maxdiff(pos, dense),
            }
            print(
                f"  budget={budget:3d} causal={str(causal):5s}"
                f"   行序切片={res[f'b{budget}_causal_{causal}']['roworder_maxdiff']:.4f}"
                f"   位置掩码={res[f'b{budget}_causal_{causal}']['position_maxdiff']:.4f}"
            )
    print(
        f"  topk(budget=32) 前 6 个索引 = {res['topk_head_b32']}"
        f"（按位置升序？{res['topk_is_position_sorted_b32']}）"
    )
    print("  ⇒ budget = 块长时不丢任何 Key，两种实现都应与 Dense 逐位一致（正确性锚点）。")
    print("  ⇒ budget < 块长时，只有位置掩码与因果语义自洽。")
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="C10 基线失败根因定位")
    parser.add_argument("--out", type=str, default="results/cpu/c10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--L", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    Q, K, V = make_scene(seed=args.seed, L=args.L)
    print("=" * 100)
    print(
        f"C10 基线失败根因定位：seed={args.seed} L={args.L}"
        f" chunk={args.chunk_size} world={args.world_size} dtype=FP64"
    )
    print("=" * 100)

    summary: Dict[str, Any] = {
        "config": {
            "seed": args.seed,
            "L": args.L,
            "chunk_size": args.chunk_size,
            "world_size": args.world_size,
            "dtype": "float64",
        },
        "note": (
            "只做定位：解释 test_fast_kv_vs_dense_within_tolerance 与 "
            "test_apb_vs_dense_within_tolerance 为何不成立，不改默认超参。"
        ),
    }
    summary["d1_convention"] = run_d1(Q, K, V, args.chunk_size, args.world_size)
    summary["d2_error_decomposition"] = run_d2(Q, K, V, args.chunk_size, args.world_size)
    summary["d3_apb_position_order"] = run_d3(Q, K, V, args.chunk_size, args.world_size)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print(f"落盘：{out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
