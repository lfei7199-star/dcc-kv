#!/usr/bin/env python
"""E6：主表与可扩展性 —— GPU 端。

论文 §6.4 把 E6 描述为
    "3 模型 × 5 上下文长度 × 2 GPU 数 × 2 同步模式 × 3 基线 = 144 个数据点"

**这个等式不成立**：3 × 5 × 2 × 2 × 3 = 180，不是 144。反推可得
144 / (3 × 2 × 2 × 3) = 4，即上下文长度应为 **4** 档。这是论文里的一处
算术不一致，需要改成 4 档长度（或把数据点数改成 180）。本脚本按
`--context-lengths` 的实际长度计算数据点数并打印，不再复述一个对不上的总数。

本脚本另一个必须明确的点：**GPU 数不是所有方法都能调的自变量。**
    dense / kv_budget_shared —— 单卡测量。它们不涉及跨设备通信，
                               把它们的 "2 卡" 数字填进主表等于编造。
    dcc_kv / ring / apb / fastkv_official —— 才真正有 2 卡 / 4 卡 的维度，
                               而这四个当前都没有可用的 GPU 实现（见 --plan）。
因此本脚本把 gpu_count 记为**方法的要求**而非自由轴，并在结果里显式标注
`gpu_count_observed` 与 `gpu_count_required`。

用法
----
    # 任何机器上都能跑：打印完整网格与每个方法的前置条件
    python experiments/gpu/e6_main_table.py --plan

    # 真实测量（需要 GPU + 模型权重）
    python experiments/gpu/e6_main_table.py \
        --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct \
        --context-lengths 4096 8192 16384 32768 \
        --methods dense kv_budget_shared \
        --eval-file data/longbench_qa.jsonl --out results/gpu/e6
"""

from __future__ import annotations

import argparse
import json
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

SCRIPT = "experiments/gpu/e6_main_table.py"

# 每个方法的前置条件。blocked 的项在这里给出**具体缺什么**，
# 而不是一句"待实现"。
METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "dense": {
        "measurable": True,
        "gpu_required": 1,
        "desc": "精确注意力（不压缩）—— 精度上界与性能参照",
        "impl": "HF 模型 + full KV cache",
    },
    "kv_budget_shared": {
        "measurable": True,
        "gpu_required": 1,
        "desc": "共享预算裁剪（按 Key 能量保留 top-B 位置，所有 Query 共用）",
        "impl": "_hf.apply_kv_budget(mode='topk_rms')",
        "caveat": (
            "**这不是 FastKV**。FastKV 是「用共享目的端构造一份紧凑 KV」"
            "（含 β 与 Value 回归）；本项是「直接裁剪缓存」，没有构造链路。"
            "它给出的是目的端无关族在同等预算下的参考曲线，"
            "不能用它声称「已与 FastKV 对比」。"
        ),
    },
    "fastkv_official": {
        "measurable": False,
        "gpu_required": 2,
        "desc": "FastKV 官方实现（共享压缩）",
        "blockers": [
            "src/baselines/fast_kv_cpu.py 是 CPU 实现，无 GPU kernel。",
            "共享压缩的多设备语义需要 dcc_kv_sync 的 GPU 版本。",
        ],
    },
    "dcc_kv": {
        "measurable": False,
        "gpu_required": 2,
        "desc": "DCC-KV（逐边条件化 + 异步 All-to-Allv）",
        "blockers": [
            "CompactKV → GPU attention kernel 缺失"
            "（src/distributed/dcc_kv_sync_cpu.py 为 CPU 同步版）。",
            "构造链路的 CUDA 可用性未打通，见 _env.CUDA_CONSTRUCTION_DEFECTS。",
            "异步 All-to-Allv 流水尚无 GPU 实现入口。",
        ],
    },
    "ring": {
        "measurable": False,
        "gpu_required": 2,
        "desc": "Ring Attention（精确序列并行）",
        "blockers": [
            "src/baselines/ring_attention_cpu.py 是 CPU 实现；"
            "GPU 版需要 NCCL P2P ring 通信，仓库中无对应文件。",
        ],
    },
    "apb": {
        "measurable": False,
        "gpu_required": 2,
        "desc": "APB（全网共享 anchor）",
        "blockers": [
            "src/baselines/apb_cpu.py 是 CPU 实现，无 GPU kernel。",
            "APB 编号（2502.12085）本身尚待二次确认，实现前需先定稿引用。",
        ],
    },
}


# =============================================================================
# 计划打印（无需 GPU）
# =============================================================================

def print_plan(a: argparse.Namespace) -> int:
    n_models = len(a.models)
    n_ctx = len(a.context_lengths)
    n_sync = len(a.sync_modes)
    n_methods = len(a.methods)
    n_points = n_models * n_ctx * n_sync * n_methods

    print("=" * 78)
    print("E6 网格计划")
    print("=" * 78)
    print(f"  模型      ({n_models})：{a.models}")
    print(f"  上下文长度 ({n_ctx})：{a.context_lengths}")
    print(f"  同步模式   ({n_sync})：{a.sync_modes}")
    print(f"  方法      ({n_methods})：{a.methods}")
    print(f"  => 数据点 = {n_models} × {n_ctx} × {n_sync} × {n_methods} = {n_points}")
    print(f"  每点 ≥{a.iters} 次 run（报告规范要求）"
          f" => 总前向次数 ≥ {n_points * a.iters}")

    print()
    print("  论文 §6.4 写的「3 模型 × 5 长 × 2 GPU × 2 同步 × 3 基线 = 144」"
          "算不出来：")
    print(f"      3 × 5 × 2 × 2 × 3 = 180 ≠ 144")
    print(f"      144 / (3 × 2 × 2 × 3) = 4  → 上下文长度应为 4 档")
    print("  需要二选一：把「5 上下文长度」改为 4，或把「144 个数据点」改为 180。")

    print()
    print("  方法前置条件：")
    print("  " + "-" * 74)
    for m in a.methods:
        spec = METHOD_SPECS[m]
        tag = "可测量" if spec["measurable"] else "被阻断"
        print(f"  [{tag}] {m}  (需 {spec['gpu_required']} 卡)")
        print(f"           {spec['desc']}")
        if spec.get("impl"):
            print(f"           实现：{spec['impl']}")
        for b in spec.get("blockers", []):
            print(f"           ✗ {b}")
        if spec.get("caveat"):
            print(f"           ! {spec['caveat']}")
    print("  " + "-" * 74)
    blocked = [m for m in a.methods if not METHOD_SPECS[m]["measurable"]]
    print(f"  本次计划中 {len(blocked)}/{n_methods} 个方法会被阻断并如实记账：{blocked}")
    print("=" * 78)
    return 0


# =============================================================================
# 测量
# =============================================================================

def measure_point(
    a: argparse.Namespace,
    model: str,
    ctx_len: int,
    sync_mode: str,
    method: str,
    lm: _hf.LoadedModel,
    samples: Sequence[_hf.EvalSample],
) -> Dict[str, Any]:
    """测量主表的一个数据点。"""
    spec = METHOD_SPECS[method]
    budget_ratio = None if method == "dense" else a.budget_ratio
    mode = "identity" if method == "dense" else a.compaction_mode

    pre = _hf.measure_prefill(
        lm, seq_len=ctx_len, batch_size=a.batch_size,
        warmup=a.warmup, iters=a.iters,
        budget_ratio=budget_ratio, compaction_mode=mode,
    )
    ps = R.summarize(pre["prefill_ms_samples"], "prefill", "ms", seed=a.seed)

    acc: Optional[float] = None
    acc_by_task: Optional[Dict[str, float]] = None
    n_eval = 0
    if samples:
        ev = _hf.evaluate(lm, samples, budget_ratio=budget_ratio,
                          compaction_mode=mode,
                          max_prompt_tokens=a.max_prompt_tokens)
        acc = ev["accuracy"]
        acc_by_task = ev["by_task"]
        n_eval = ev["n"]

    kv_ratio = 1.0 if method == "dense" else (
        budget_ratio if budget_ratio is not None else 1.0)
    keep = min(1.0, max(0.0, a.budget_ratio if budget_ratio is not None else 1.0))

    return {
        "model": model,
        "context_length": ctx_len,
        "gpu_count_observed": 1,
        "gpu_count_required": spec["gpu_required"],
        "gpu_count_note": ("单卡测量（该方法不涉及跨设备通信）"
                           if spec["gpu_required"] == 1 else
                           f"该方法需要 {spec['gpu_required']} 卡，本次未满足"),
        "sync_async": sync_mode,
        "method": method,
        "method_measurable": spec["measurable"],
        "num_repr_queries": a.M,
        "projection_dim": a.d_p,
        "budget_ratio": budget_ratio,
        "compaction_mode": mode,
        "prefill_ms_median": ps.median,
        "prefill_ms_p5": ps.p5,
        "prefill_ms_p95": ps.p95,
        "prefill_ms_ci": [ps.ci_95_lower, ps.ci_95_upper],
        "tokens_per_s_median": pre["tokens_per_s_median"],
        "peak_memory_gb": pre["peak_memory_gb"],
        "kv_bytes_full": pre["kv_bytes_full"],
        "kv_bytes_kept": _hf.kv_cache_bytes(lm, ctx_len, keep_ratio=keep,
                                            batch_size=a.batch_size),
        "accuracy": acc,
        "accuracy_by_task": acc_by_task,
        "n_eval": n_eval,
        "status": "ok",
    }


def blocked_point(a: argparse.Namespace, model: str, ctx_len: int,
                  sync_mode: str, method: str) -> Dict[str, Any]:
    spec = METHOD_SPECS[method]
    return {
        "model": model,
        "context_length": ctx_len,
        "gpu_count_observed": 0,
        "gpu_count_required": spec["gpu_required"],
        "sync_async": sync_mode,
        "method": method,
        "method_measurable": False,
        "budget_ratio": a.budget_ratio,
        "status": "blocked",
        "blockers": spec.get("blockers", []),
        "accuracy": None,
        "prefill_ms_median": None,
    }


# =============================================================================
# main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E6 主表与可扩展性（GPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--plan", action="store_true",
                   help="只打印网格与前置条件并退出，无需 GPU")
    p.add_argument("--models", nargs="+", default=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ])
    p.add_argument("--context-lengths", dest="context_lengths", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768],
                   help="论文写 5 档但总数 144 对应 4 档；此处默认 4 档并与论文核对")
    p.add_argument("--sync-modes", dest="sync_modes", nargs="+",
                   default=["sync", "async"], choices=["sync", "async"])
    p.add_argument("--methods", nargs="+",
                   default=["dense", "kv_budget_shared", "fastkv_official",
                            "dcc_kv", "ring", "apb"],
                   choices=sorted(METHOD_SPECS))
    p.add_argument("--eval-file", dest="eval_file", type=str, default=None)
    p.add_argument("--eval-limit", dest="eval_limit", type=int, default=None)
    p.add_argument("--max-prompt-tokens", dest="max_prompt_tokens", type=int, default=None)
    p.add_argument("--budget-ratio", dest="budget_ratio", type=float, default=0.05)
    p.add_argument("--compaction-mode", dest="compaction_mode", type=str,
                   default="topk_rms",
                   choices=["identity", "topk_rms", "topk_norm", "stride", "random"])
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--d-p", dest="d_p", type=int, default=32)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=1)
    p.add_argument("--precision", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-impl", dest="attn_impl", type=str, default=None)
    p.add_argument("--interconnect", type=str, default="unknown")
    p.add_argument("--rope-extension-used", dest="rope_extension_used", type=str,
                   default=None)
    p.add_argument("--rope-extension-disclosed", dest="rope_extension_disclosed",
                   action="store_true")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/gpu/e6")
    p.add_argument("--print-env", action="store_true")
    p.add_argument("--allow-cpu-plan-only", action="store_true",
                   help="确认只跑 --plan 而不执行测量")
    return p


def main() -> int:
    a = build_parser().parse_args()

    if a.plan:
        return print_plan(a)
    if a.print_env:
        return _env.print_env_only("E6 主表与可扩展性", min_gpus=1, need_nccl=False)

    measurable = [m for m in a.methods if METHOD_SPECS[m]["measurable"]]
    if not measurable:
        print_plan(a)
        print("\n阻断：所选方法全部被前置缺口阻断，无可测量项。")
        print("可测量方法：" + str(sorted(k for k, v in METHOD_SPECS.items()
                                          if v["measurable"])))
        return _env.GATE_EXIT_CODE

    gate = _env.probe(min_gpus=1, need_nccl=False, hf_backend=True,
                      model_name=a.models[0])
    code = _env.enforce(gate, "E6 主表与可扩展性", SCRIPT)
    if code is not None:
        return code

    if a.iters < 10:
        print(f"[警告] --iters={a.iters} < 10，不满足 §6.1 的重复次数规范。")

    samples: List[_hf.EvalSample] = []
    if a.eval_file:
        samples = _hf.load_eval_file(a.eval_file, limit=a.eval_limit)
        print(f"评测集：{len(samples)} 条（{a.eval_file}）")

    rows: List[Dict[str, Any]] = []
    for model in a.models:
        print()
        print("=" * 78)
        lm = _hf.load_model(model, a.precision, attn_implementation=a.attn_impl)
        print("=" * 78)
        for ctx_len in a.context_lengths:
            for sync_mode in a.sync_modes:
                for method in a.methods:
                    if not METHOD_SPECS[method]["measurable"]:
                        rows.append(blocked_point(a, model, ctx_len, sync_mode, method))
                        continue
                    try:
                        r = measure_point(a, model, ctx_len, sync_mode, method, lm, samples)
                    except Exception as e:
                        r = {
                            "model": model, "context_length": ctx_len,
                            "sync_async": sync_mode, "method": method,
                            "status": "error", "error_type": type(e).__name__,
                            "error": str(e),
                            "traceback": traceback.format_exc().splitlines()[-8:],
                            "accuracy": None, "prefill_ms_median": None,
                        }
                    rows.append(r)
                    if r["status"] == "ok":
                        m = lm.model
                        print(f"  {model.split('/')[-1]:<28} L={ctx_len:<6} "
                              f"{sync_mode:<5} {method:<17} "
                              f"prefill={r['prefill_ms_median']:9.2f}ms "
                              f"(p95={r['prefill_ms_p95']:9.2f})  "
                              f"{r['tokens_per_s_median']:10.1f} tok/s  "
                              f"peak={r['peak_memory_gb']:5.2f}GB  "
                              f"acc={'n/a' if r['accuracy'] is None else f'{r[chr(97)+chr(99)+chr(99)+chr(117)+chr(114)+chr(97)+chr(99)+chr(121)]:.4f}'}")
                    else:
                        print(f"  {model.split('/')[-1]:<28} L={ctx_len:<6} "
                              f"{sync_mode:<5} {method:<17} [{r['status']}] "
                              f"{r.get('error_type', '')}")
        del lm
        torch.cuda.empty_cache()

    # 达标性判定
    meta = _env.build_metadata(
        run_id="e6-main-table",
        model_name=",".join(a.models),
        context_length=max(a.context_lengths),
        seed=a.seed, budget_ratio=a.budget_ratio,
        sync_async="both", num_repr_queries=a.M, projection_dim=a.d_p,
        task="main-table", precision=a.precision,
        interconnect=a.interconnect,
        rope_extension_used=a.rope_extension_used,
        rope_extension_disclosed=a.rope_extension_disclosed,
        notes=f"methods={a.methods}",
    )
    admissible = _env.gate_guard_for_report(meta)

    payload: Dict[str, Any] = {
        "experiment": "E6",
        "rows": rows,
        "metadata": meta.to_dict(),
        "report_admissibility": admissible,
        "grid": {
            "models": a.models, "context_lengths": a.context_lengths,
            "sync_modes": a.sync_modes, "methods": a.methods,
            "n_points_planned": (len(a.models) * len(a.context_lengths)
                                 * len(a.sync_modes) * len(a.methods)),
            "n_points_measured": sum(1 for r in rows if r["status"] == "ok"),
            "n_points_blocked": sum(1 for r in rows if r["status"] == "blocked"),
            "paper_claim": "3 模型 × 5 长度 × 2 GPU × 2 同步 × 3 基线 = 144",
            "paper_claim_check": "3×5×2×2×3 = 180 ≠ 144；144 对应 4 档长度",
        },
        "caveat": (
            "gpu_count 在可测量方法（dense / kv_budget_shared）上不是自由轴："
            "它们是单卡测量，不涉及跨设备通信。主表中的「2 GPU」维度只对"
            "需要多卡的四个方法有意义，而那四个当前均被阻断，"
            "因此本表**不能**用于支撑 H5（设备数翻倍加速 ≥1.5×）。"
            "H5 需要 dcc_kv 的 GPU 多卡实现就绪后单独测量。"
        ),
    }

    out = REPO_ROOT / a.out
    R.save_json(str(out / "e6_results.json"), payload)
    R.save_csv(str(out / "e6_main_table.csv"), rows)
    print()
    print(f"数据点：计划 {payload['grid']['n_points_planned']} / "
          f"实测 {payload['grid']['n_points_measured']} / "
          f"阻断 {payload['grid']['n_points_blocked']}")
    if not admissible["admissible"]:
        print(f"[不可入主表] {admissible['reason']}")
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
