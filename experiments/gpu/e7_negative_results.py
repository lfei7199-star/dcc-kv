#!/usr/bin/env python
"""E7：负结果与适用边界 —— GPU 端（真实模型）。

论文 §6.4 的 E7 要求以下四种条件**无论结果是否符合预期都必须报告**：

    1. 短上下文（L < 4K）   —— 压缩的固定开销占比上升，预期收益消失甚至为负
    2. 低预算（B < 1%）     —— 确定可用压缩下界
    3. 强检索任务           —— 对 token 级定位敏感，是对"误差不随块数发散"的压力测试
    4. batch = 1            —— 计算侧难以饱和，异步收益受限

本脚本把这四条实现为可选的 `--conditions`，且**没有一条会因为结果不利而被
静默跳过**：每条都以 `verdict` 字段给出"预期是否成立"的判定，无论正负都写入结果。

一个必须说清的量纲问题
----------------------
条件 1 的"收益消失"有两个可能的口径：
    (a) 延迟口径 —— 压缩后 prefill 更快？
    (b) 精度口径 —— 同等预算下精度损失是否与 L 相关？
本脚本两个都报，但**不**把它们合成一个"是否值得用"的结论 ——
那取决于部署时对延迟与精度的相对定价，属于决策而非测量。

用法
----
    python experiments/gpu/e7_negative_results.py --plan
    python experiments/gpu/e7_negative_results.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --eval-file data/longbench_qa.jsonl --conditions short_ctx low_budget \
        --out results/gpu/e7
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import report as R      # noqa: E402
from experiments.gpu import _env, _hf            # noqa: E402

SCRIPT = "experiments/gpu/e7_negative_results.py"

CONDITIONS = ("short_ctx", "low_budget", "retrieval", "batch1")

RETRIEVAL_TASK_HINTS = ("retrieval", "niah", "needle", "kv_retrieval", "passkey")


# =============================================================================
# 条件 1：短上下文
# =============================================================================

def cond_short_ctx(a: argparse.Namespace, lm: _hf.LoadedModel,
                   samples: Sequence[_hf.EvalSample]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for L in a.short_lengths:
        base = _hf.measure_prefill(lm, seq_len=L, batch_size=a.batch_size,
                                   warmup=a.warmup, iters=a.iters)
        b = R.summarize(base["prefill_ms_samples"], "prefill", "ms", seed=a.seed)
        row: Dict[str, Any] = {
            "L": L,
            "L_over_4K": L / 4096.0,
            "budget_ratio": a.budget_ratio,
            "B": max(1, int(round(a.budget_ratio * L))),
            "B_over_L": a.budget_ratio,
            "dense_prefill_ms": b.median,
            "dense_tokens_per_s": base["tokens_per_s_median"],
            "dense_peak_gb": base["peak_memory_gb"],
            "compacted_prefill_ms": None,
            "compacted_tokens_per_s": None,
            "compacted_peak_gb": None,
            "latency_payoff": None,
            "dense_accuracy": None,
            "compacted_accuracy": None,
            "accuracy_delta": None,
        }

        c = _hf.measure_prefill(lm, seq_len=L, batch_size=a.batch_size,
                                warmup=a.warmup, iters=a.iters,
                                budget_ratio=a.budget_ratio,
                                compaction_mode=a.compaction_mode)
        cb = R.summarize(c["prefill_ms_samples"], "prefill", "ms", seed=a.seed)
        row["compacted_prefill_ms"] = cb.median
        row["compacted_tokens_per_s"] = c["tokens_per_s_median"]
        row["compacted_peak_gb"] = c["peak_memory_gb"]
        row["latency_payoff"] = (b.median / cb.median) if cb.median else float("nan")

        if samples:
            ed = _hf.evaluate(lm, samples, budget_ratio=None,
                              compaction_mode="identity",
                              max_prompt_tokens=L)
            ec = _hf.evaluate(lm, samples, budget_ratio=a.budget_ratio,
                              compaction_mode=a.compaction_mode,
                              max_prompt_tokens=L)
            row["dense_accuracy"] = ed["accuracy"]
            row["compacted_accuracy"] = ec["accuracy"]
            row["accuracy_delta"] = ec["accuracy"] - ed["accuracy"]
            row["by_task_dense"] = ed["by_task"]
            row["by_task_compacted"] = ec["by_task"]

        rows.append(row)
        dacc_s = ("n/a" if row["accuracy_delta"] is None
                  else f"{row['accuracy_delta']:+.4f}")
        print(f"  L={L:<7} ({row['L_over_4K']:.2f}×4K)  B={row['B']:<5}  "
              f"dense={b.median:8.2f}ms  compacted={cb.median:8.2f}ms  "
              f"延迟收益={row['latency_payoff']:.4f}×  Δacc={dacc_s}")

    under4k = [r for r in rows if r["L"] < 4096]
    verdict = None
    if under4k:
        verdict = all((r["latency_payoff"] is None or r["latency_payoff"] <= 1.0)
                      for r in under4k)
    return {
        "condition": "short_ctx",
        "rows": rows,
        "expected": "L < 4K 时压缩的固定开销占比上升，预期无收益甚至劣于精确注意力",
        "verdict": ("预期成立：所有 L<4K 的条件均未出现延迟收益"
                    if verdict else
                    "预期不成立或部分不成立：L<4K 下仍出现延迟收益，需重新界定边界"),
        "verdict_supported": verdict,
        "note": ("延迟收益 = dense_prefill_ms / compacted_prefill_ms，>1 才表示压缩更快。"
                 "注意本实现中压缩只在 prefill 之后作用于 cache，"
                 "构造开销未计入 —— 若把构造计入，短上下文下收益会进一步下降。"),
    }


# =============================================================================
# 条件 2：低预算
# =============================================================================

def cond_low_budget(a: argparse.Namespace, lm: _hf.LoadedModel,
                    samples: Sequence[_hf.EvalSample]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for ratio in a.low_budget_ratios:
        L = a.low_budget_L
        B = max(1, int(round(ratio * L)))
        row: Dict[str, Any] = {
            "L": L, "budget_ratio": ratio, "B": B,
            "budget_pct": ratio * 100.0,
            "accuracy": None, "accuracy_drop_vs_dense": None,
            "prefill_ms_median": None, "kv_bytes_kept": _hf.kv_cache_bytes(
                lm, L, keep_ratio=min(1.0, ratio), batch_size=a.batch_size),
        }
        pre = _hf.measure_prefill(lm, seq_len=L, batch_size=a.batch_size,
                                  warmup=a.warmup, iters=a.iters,
                                  budget_ratio=ratio,
                                  compaction_mode=a.compaction_mode)
        row["prefill_ms_median"] = R.summarize(
            pre["prefill_ms_samples"], "prefill", "ms", seed=a.seed).median

        if samples:
            ev = _hf.evaluate(lm, samples, budget_ratio=ratio,
                              compaction_mode=a.compaction_mode,
                              max_prompt_tokens=L)
            row["accuracy"] = ev["accuracy"]
            row["accuracy_by_task"] = ev["by_task"]
            row["n_eval"] = ev["n"]

        rows.append(row)
        acc_s = "n/a" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
        print(f"  B/L={ratio:.4f} ({row['budget_pct']:.2f}%)  B={B:<5}  "
              f"prefill={row['prefill_ms_median']:8.2f}ms  acc={acc_s}")

    dense_acc = None
    if samples:
        ed = _hf.evaluate(lm, samples, budget_ratio=None,
                          compaction_mode="identity", max_prompt_tokens=a.low_budget_L)
        dense_acc = ed["accuracy"]
        for r in rows:
            if r["accuracy"] is not None:
                r["accuracy_drop_vs_dense"] = r["accuracy"] - dense_acc
        print(f"  (dense 参照 acc={dense_acc:.4f})")

    sub1pct = [r for r in rows if r["budget_ratio"] < 0.01 and r["accuracy"] is not None]
    usable_floor = None
    if sub1pct and dense_acc is not None:
        ok = [r for r in sub1pct if r["accuracy_drop_vs_dense"] is not None
              and r["accuracy_drop_vs_dense"] >= -a.accuracy_tolerance]
        usable_floor = max((r["budget_ratio"] for r in ok), default=None)

    return {
        "condition": "low_budget",
        "rows": rows,
        "dense_accuracy": dense_acc,
        "expected": "B < 1% 时误差显著增大，用于确定可用压缩下界",
        "usable_budget_floor": usable_floor,
        "accuracy_tolerance": a.accuracy_tolerance,
        "verdict": (
            f"在容差 {a.accuracy_tolerance} 点内，B<1% 区间的最小可用预算比为 "
            f"{usable_floor}" if usable_floor is not None else
            "B<1% 区间内无任何档位落在精度容差内 —— 可用下界高于 1%"
        ),
        "note": ("usable_budget_floor 依赖 --accuracy-tolerance 这一人为设定的容差，"
                 "换一个容差就会得到不同的下界。报告中必须把这个容差一起写出，"
                 "否则「可用压缩下界」这个数字没有意义。"),
    }


# =============================================================================
# 条件 3：强检索任务
# =============================================================================

def cond_retrieval(a: argparse.Namespace, lm: _hf.LoadedModel,
                   samples: Sequence[_hf.EvalSample]) -> Dict[str, Any]:
    def _is_retrieval(s: _hf.EvalSample) -> bool:
        t = s.task.lower()
        return any(h in t for h in RETRIEVAL_TASK_HINTS)

    retr = [s for s in samples if _is_retrieval(s)]
    other = [s for s in samples if not _is_retrieval(s)]

    if not retr:
        return {
            "condition": "retrieval",
            "rows": [],
            "status": "no_data",
            "expected": "检索任务对 token 级定位敏感，压缩可能系统性丢失关键条目",
            "verdict": ("评测集中没有可识别的检索类任务（task 字段需包含 "
                        + "/".join(RETRIEVAL_TASK_HINTS) + " 之一）—— 本条件未被执行"),
            "verdict_supported": None,
            "note": ("这不是「结果不好」，而是「没有数据」。"
                     "按 §6.4 的要求，该条件必须在获得对应评测集后补测，"
                     "当前状态记为 no_data 而非 pass。"),
        }

    rows: List[Dict[str, Any]] = []
    for ratio in a.retrieval_ratios:
        ed = _hf.evaluate(lm, retr, budget_ratio=None, compaction_mode="identity",
                          max_prompt_tokens=a.max_prompt_tokens)
        ec = _hf.evaluate(lm, retr, budget_ratio=ratio,
                          compaction_mode=a.compaction_mode,
                          max_prompt_tokens=a.max_prompt_tokens)
        eo = None
        if other:
            eo = _hf.evaluate(lm, other, budget_ratio=ratio,
                              compaction_mode=a.compaction_mode,
                              max_prompt_tokens=a.max_prompt_tokens)
        rows.append({
            "budget_ratio": ratio,
            "n_retrieval": len(retr),
            "retrieval_accuracy_dense": ed["accuracy"],
            "retrieval_accuracy_compacted": ec["accuracy"],
            "retrieval_delta": ec["accuracy"] - ed["accuracy"],
            "other_accuracy_compacted": (eo["accuracy"] if eo else None),
            "differential_vs_other": (
                (ec["accuracy"] - eo["accuracy"]) if eo else None),
        })
        r = rows[-1]
        print(f"  B/L={ratio:.4f}  检索 acc {r['retrieval_accuracy_dense']:.4f} "
              f"→ {r['retrieval_accuracy_compacted']:.4f} "
              f"(Δ={r['retrieval_delta']:+.4f})  "
              f"非检索 acc={r['other_accuracy_compacted']}")

    worst = min(r["retrieval_delta"] for r in rows)
    return {
        "condition": "retrieval",
        "rows": rows,
        "n_retrieval_samples": len(retr),
        "n_other_samples": len(other),
        "expected": "检索任务对 token 级定位敏感，压缩可能系统性丢失关键条目",
        "verdict": (f"检索任务上的最大精度损失为 {worst:+.4f} 点；"
                    "与同预算下非检索任务的差值为 verification 提供差分对照"),
        "differential_note": (
            "关键不是一个绝对数字，而是**差分**：若检索类任务的精度损失显著大于"
            "同预算下的非检索任务，才说明压缩在系统性地伤及 token 级定位；"
            "若两者相当，则损失来自通用信息丢失而非检索特有的失效模式。"
        ),
    }


# =============================================================================
# 条件 4：batch = 1
# =============================================================================

def cond_batch1(a: argparse.Namespace, lm: _hf.LoadedModel,
                samples: Sequence[_hf.EvalSample]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    ref_tps: Optional[float] = None
    for bs in a.batch_sizes:
        pre = _hf.measure_prefill(lm, seq_len=a.batch1_L, batch_size=bs,
                                  warmup=a.warmup, iters=a.iters)
        s = R.summarize(pre["prefill_ms_samples"], "prefill", "ms", seed=a.seed)
        tps = pre["tokens_per_s_median"]
        if bs == 1:
            ref_tps = tps
        rows.append({
            "batch_size": bs,
            "seq_len": a.batch1_L,
            "prefill_ms_median": s.median,
            "prefill_ms_p95": s.p95,
            "tokens_per_s": tps,
            "peak_memory_gb": pre["peak_memory_gb"],
            "throughput_gain_vs_bs1": (tps / ref_tps) if ref_tps else None,
            "batch_gain": bs,
            # 计算侧饱和度的直接度量：吞吐增益 / batch 增益。
            # ≈1 说明线性扩展（计算未饱和）；<<1 说明已饱和。
            "saturation_ratio": (tps / ref_tps / bs) if ref_tps else None,
        })
        r = rows[-1]
        print(f"  batch={bs:<3} prefill={s.median:9.2f}ms  "
              f"{r['tokens_per_s']:10.1f} tok/s  "
              f"吞吐增益={r['throughput_gain_vs_bs1']:.4f}×  "
              f"饱和比={r['saturation_ratio']:.4f}  "
              f"peak={r['peak_memory_gb']:.2f}GB")

    b1 = rows[0] if rows else None
    large = rows[-1] if len(rows) > 1 else None
    saturated = None
    if large and large["saturation_ratio"] is not None:
        saturated = large["saturation_ratio"] < 0.5

    return {
        "condition": "batch1",
        "rows": rows,
        "expected": "batch=1 时计算侧难以饱和，T_comp 偏小，异步收益受限",
        "compute_saturated": saturated,
        "verdict": (
            "预期成立：batch=1 时 T_comp 显著低于大 batch，"
            "式(speedup-bound) 预测的异步收益上限更低"
            if saturated else
            "预期不成立或需更多档位：在所测档位内未观察到明显饱和，"
            "需把 batch 继续放大或换更小的模型以逼近饱和点"
        ),
        "verdict_supported": saturated,
        "note": ("saturation_ratio = 吞吐增益 / batch 增益，是「计算是否已被打满」"
                 "的直接度量；本条件测的是**计算侧**的饱和度，"
                 "用于解释 A5 的 overlap 上限，本身不是对异步实现的检验 —— "
                 "后者需要多卡（见 E5/A5）。"),
    }


# =============================================================================
# main
# =============================================================================

COND_FN = {
    "short_ctx": cond_short_ctx,
    "low_budget": cond_low_budget,
    "retrieval": cond_retrieval,
    "batch1": cond_batch1,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E7 负结果与适用边界（GPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--plan", action="store_true", help="只打印条件清单并退出，无需 GPU")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                   choices=list(CONDITIONS))
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--eval-file", dest="eval_file", type=str, default=None)
    p.add_argument("--eval-limit", dest="eval_limit", type=int, default=None)
    p.add_argument("--max-prompt-tokens", dest="max_prompt_tokens", type=int, default=None)
    p.add_argument("--precision", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-impl", dest="attn_impl", type=str, default=None)
    p.add_argument("--compaction-mode", dest="compaction_mode", type=str,
                   default="topk_rms",
                   choices=["identity", "topk_rms", "topk_norm", "stride", "random"])
    p.add_argument("--budget-ratio", dest="budget_ratio", type=float, default=0.05)

    p.add_argument("--short-lengths", dest="short_lengths", type=int, nargs="+",
                   default=[1024, 2048, 4096])
    p.add_argument("--low-budget-ratios", dest="low_budget_ratios", type=float, nargs="+",
                   default=[0.001, 0.0025, 0.005])
    p.add_argument("--low-budget-L", dest="low_budget_L", type=int, default=32768)
    p.add_argument("--accuracy-tolerance", dest="accuracy_tolerance", type=float,
                   default=1.0,
                   help="判定「可用压缩下界」时允许的精度损失（任务指标点数）")
    p.add_argument("--retrieval-ratios", dest="retrieval_ratios", type=float, nargs="+",
                   default=[0.01, 0.05])
    p.add_argument("--batch1-L", dest="batch1_L", type=int, default=32768)
    p.add_argument("--batch-sizes", dest="batch_sizes", type=int, nargs="+",
                   default=[1, 2, 4, 8])

    p.add_argument("--batch-size", dest="batch_size", type=int, default=1)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--interconnect", type=str, default="unknown")
    p.add_argument("--out", type=str, default="results/gpu/e7")
    p.add_argument("--print-env", action="store_true")
    return p


def print_plan(a: argparse.Namespace) -> int:
    print("=" * 78)
    print("E7 负结果与适用边界 —— 条件清单")
    print("=" * 78)
    print("  按论文 §6.4，以下条件无论结果是否符合预期都必须报告。")
    print("  每条都以 verdict / verdict_supported 字段落盘，不因结果不利而跳过。")
    print()
    print(f"  1. short_ctx   L ∈ {a.short_lengths}  (含 L<4K 的档位)")
    print(f"  2. low_budget  B/L ∈ {a.low_budget_ratios}  (全部 <1%)  L={a.low_budget_L}")
    print(f"  3. retrieval   task 字段含 {'/'.join(RETRIEVAL_TASK_HINTS)} 的样本")
    print(f"  4. batch1      batch ∈ {a.batch_sizes}  L={a.batch1_L}")
    print()
    print("  需要真实模型权重；条件 3 还需要一个带 task 标签的评测集。")
    print("  缺评测集时条件 3 记为 no_data（而非 pass）——「没有数据」不等于「通过」。")
    print("=" * 78)
    return 0


def main() -> int:
    a = build_parser().parse_args()

    if a.plan:
        return print_plan(a)
    if a.print_env:
        return _env.print_env_only("E7 负结果与适用边界", min_gpus=1, need_nccl=False)

    gate = _env.probe(min_gpus=1, need_nccl=False, hf_backend=True,
                      model_name=a.model)
    code = _env.enforce(gate, "E7 负结果与适用边界", SCRIPT)
    if code is not None:
        return code

    if a.iters < 10:
        print(f"[警告] --iters={a.iters} < 10，不满足 §6.1 的重复次数规范。")

    samples: List[_hf.EvalSample] = []
    if a.eval_file:
        samples = _hf.load_eval_file(a.eval_file, limit=a.eval_limit)
        print(f"评测集：{len(samples)} 条")
    else:
        print("[警告] 未提供 --eval-file：所有精度维度将为 n/a，"
              "只剩延迟维度。按 §6.4，精度维度不可省略。")

    print()
    lm = _hf.load_model(a.model, a.precision, attn_implementation=a.attn_impl)

    results: Dict[str, Any] = {}
    for cond in a.conditions:
        print()
        print(f"--- E7.{cond} ---")
        try:
            results[cond] = COND_FN[cond](a, lm, samples)
        except Exception as e:
            results[cond] = {
                "condition": cond, "status": "error",
                "error_type": type(e).__name__, "error": str(e),
                "traceback": traceback.format_exc().splitlines()[-10:],
            }
            print(f"  [错误] {type(e).__name__}: {e}")

    payload: Dict[str, Any] = {
        "experiment": "E7",
        "model": a.model,
        "precision": a.precision,
        "conditions_run": a.conditions,
        "results": results,
        "metadata": _env.build_metadata(
            run_id="e7-negative",
            model_name=a.model, context_length=max(a.short_lengths + [a.batch1_L]),
            seed=a.seed, budget_ratio=a.budget_ratio, sync_async="sync",
            task="negative-results", precision=a.precision,
            interconnect=a.interconnect,
        ).to_dict(),
        "reporting_rule": (
            "按 §6.4 与发布清单：E7 的任何一条结论，无论正负，都必须进入正文。"
            "若 H2/H3/H4/H5 中任一未达标，主张按预设阈值收敛，"
            "并在正文如实报告 —— 不得在结果不利时事后调整主张范围。"
        ),
    }

    out = REPO_ROOT / a.out
    R.save_json(str(out / "e7_results.json"), payload)
    for cond, res in results.items():
        if res.get("rows"):
            R.save_csv(str(out / f"e7_{cond}.csv"), res["rows"])
    print()
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
