#!/usr/bin/env python
"""E5：消融实验 A1 / A2 / A3 / A5 —— GPU 端（需要多卡 + NCCL）。

与论文 §6.4 的对应
------------------
    A1  通信集大小   —— 扫描有效边数 |E_s|，拆解 构造开销 : 通信开销 的比例
    A2  压缩预算     —— 5 档预算下的 精度–通信量 帕累托前沿
    A3  组件拆分     —— full / no_beta / no_value / no_both 的任务指标差
    A5  异步 vs 同步 —— p50 延迟、T_comm/T_comp、实测加速比与理论上界

三件事必须先说清楚，否则结果会被误读

**第一，A2/A3 的"精度"必须是任务指标，不能是重构误差。**
本脚本的 synthetic 后端只能给出**通信量曲线**（这是真实测量），
它给不出准确率。若用 synthetic 后端跑 A2 并把重构误差当"精度"填进
帕累托前沿，得到的就是一张无法与 FastKV/APB 对比的图。
因此 A2 在 synthetic 后端下会把 accuracy 字段显式置为 null 并警告；
A3 在 synthetic 后端下**直接拒绝执行**（请改用
`experiments/cpu/e5a_mechanism_ablation.py` 做机制级版本）。

**第二，A3 当前存在硬性前置缺口（不是环境问题，是实现缺口）。**
A3 要把四个变体的紧凑 KV 喂进真实模型的注意力，需要一条
"CompactKV → attention kernel"的 GPU 通路。仓库目前只有
`src/distributed/dcc_kv_sync_cpu.py`（CPU、同步、且只做数值等价性验证），
没有可用的 GPU kernel。因此本脚本的 A3 会在前置检查处停下并打印
缺失清单，而不是产出一个看起来像模像样的假数字。
另外，紧凑 KV 的构造链路曾有**三处 device 缺陷**（`_env.CUDA_CONSTRUCTION_DEFECTS`
保留的是历史清单），三者均已在 `5b5ce98` 修复；本机无 CUDA，所以"已修复"只是
静态审计结论，能否真跑必须由目标机上的 `_env.probe_gpu_construction()` 实测。
实际构造位置据此决定：探测通过走 GPU 构造，否则退化为 CPU 构造 + H2D 传输。

**第三，A1 的构造代价有两种口径，且它们不可互换。**
`--build-location gpu`  — 构造在 GPU 上（当前不可用，会如实报错）
`--build-location cpu`  — CPU 构造 + H2D 传输，**这是当前唯一可部署的路径**
`--build-location auto` — 先试 GPU，失败则退到 CPU，并在 payload 里
                          记录 `build_location_effective`
两种口径下 T_build 的含义不同：CPU 版把 PCIe 传输算进了构造，
在 NVLink 机器上会系统性高估构造开销。报告时必须写明用的是哪一种。

用法
----
    # 任何机器上都可以先体检环境（不执行实验）
    python experiments/gpu/e5_gpu_ablation.py --print-env

    # 单机 4 卡（torchrun 由 run_gpu.sh 封装）
    torchrun --nproc_per_node=4 experiments/gpu/e5_gpu_ablation.py \
        --parts a1 a2 a5 --out results/gpu/a1a2a5

    # 需要任务指标时才加载模型
    torchrun --nproc_per_node=1 experiments/gpu/e5_gpu_ablation.py \
        --parts a2 --backend hf --model meta-llama/Llama-3.1-8B-Instruct \
        --eval-file data/longbench_qa.jsonl
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

from experiments.common import report as R          # noqa: E402
from experiments.gpu import _comm, _env, _hf          # noqa: E402
from src.dcc_kv_ref import CompactKV, build_compact_kv  # noqa: E402

SCRIPT = "experiments/gpu/e5_gpu_ablation.py"

A3_MISSING_PREREQUISITES = [
    "CompactKV → GPU attention kernel：需要把紧凑 K/β/V 送进 SDPA，"
    "现有实现 src/distributed/dcc_kv_sync_cpu.py 是 CPU 同步版，只做数值等价性验证。",
    "构造链路的 CUDA 可用性：三处 device 缺陷（representative_query / "
    "value_regression / key_selection）已在 5b5ce98 修复，但本机无 CUDA 无法实测，"
    "须由 _env.probe_gpu_construction() 在目标机上确认。",
    "逐边条件化的多设备语义：单进程 harness 只有单一目的端，"
    "无法区分 DCC-KV 与 FastKV —— A3 的对照必须走多设备路径。",
]


# =============================================================================
# 紧凑 KV 构造（带 GPU/CPU 位置选择）
# =============================================================================

def _make_edge_inputs(
    n: int, L_s: int, d_h: int, d_v: int, L_r: int,
    device: torch.device, dtype: torch.dtype, seed: int,
) -> List[Dict[str, torch.Tensor]]:
    """预生成 n 组 (K, V, Q)，**在计时区间之外**，避免把 RNG 开销算进构造。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(n):
        out.append({
            "K": torch.randn(L_s, d_h, generator=g, dtype=torch.float32).to(dtype).to(device),
            "V": torch.randn(L_s, d_v, generator=g, dtype=torch.float32).to(dtype).to(device),
            "Q": torch.randn(L_r, d_h, generator=g, dtype=torch.float32).to(dtype).to(device),
        })
    return out


def _build_edge(
    inp: Dict[str, torch.Tensor],
    budget: int, M: int, d_p: int, seed: int, location: str,
) -> CompactKV:
    """构造一条边的紧凑 KV。location ∈ {gpu, cpu}。"""
    if location == "gpu":
        return build_compact_kv(
            source_keys=inp["K"], source_values=inp["V"], destination_queries=inp["Q"],
            budget=budget, num_representative_queries=M, projection_dim=d_p, seed=seed,
        )
    ck = build_compact_kv(
        source_keys=inp["K"].cpu(), source_values=inp["V"].cpu(),
        destination_queries=inp["Q"].cpu(),
        budget=budget, num_representative_queries=M, projection_dim=d_p, seed=seed,
    )
    dev = inp["K"].device
    return CompactKV(
        keys=ck.keys.to(dev), logit_bias=ck.logit_bias.to(dev),
        values=ck.values.to(dev), selected_indices=ck.selected_indices.to(dev),
    )


def _resolve_build_location(a: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """决定构造位置，并把探测结论记进 payload。"""
    probe = _env.probe_gpu_construction() if torch.cuda.is_available() else {
        "ok": False, "skipped": True, "reason": "无 CUDA 设备"}
    requested = a["build_location"]
    if requested == "cpu":
        effective = "cpu"
    elif requested == "gpu":
        effective = "gpu"
    else:  # auto
        effective = "gpu" if probe.get("ok") else "cpu"
    return {
        "requested": requested,
        "effective": effective,
        "probe": probe,
        "note": (
            "构造位置为 CPU：T_build 包含 H2D 传输，在 NVLink 机器上会系统性高估构造开销。"
            if effective == "cpu" else
            "构造位置为 GPU：T_build 为纯 GPU 构造耗时。"
        ),
    }


# =============================================================================
# A1 —— 通信集大小
# =============================================================================

def a1_comm_set_size(rank: int, world: int, a: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """扫描每个源设备持有的块数 × 目的端数（即 |E_s|），拆分构造/通信开销。"""
    dtype = _env.dtypes_for(a["precision"])
    itemsize = dtype.itemsize
    dev = _env.local_device(rank)
    loc = ctx["build_location"]["effective"]

    max_blocks = max(a["blocks_per_rank"])
    inputs = _make_edge_inputs(max_blocks, a["L_s"], a["d_h"], a["d_v"],
                               a["L_r"], dev, dtype, a["seed"])
    rows: List[Dict[str, Any]] = []

    for n_blocks in a["blocks_per_rank"]:
        # |E_s| = 每个源块要发给 world 个目的端
        n_edges = n_blocks * world
        per_dst_rows = n_blocks * a["budget"]

        # --- 构造：n_edges 次逐边构造 ---
        def _do_build() -> None:
            for j in range(n_edges):
                _build_edge(inputs[j % n_blocks], a["budget"], a["M"], a["d_p"],
                            a["seed"] + j, loc)

        t_build = _env.benchmark_ms(_do_build, warmup=a["warmup"], iters=a["iters"])

        # --- 通信：一次性把 n_blocks×world 条紧凑边 All-to-Allv 出去 ---
        F = a["d_h"] + 1 + a["d_v"]
        payload = torch.randn(per_dst_rows * world, F, device=dev, dtype=dtype)
        send_sizes = [per_dst_rows] * world
        recv_sizes = [per_dst_rows] * world

        t_comm = _env.benchmark_ms(
            lambda: _comm.all_to_all_v(payload, send_sizes, recv_sizes),
            warmup=a["warmup"], iters=a["iters"],
        )
        # 端到端（构造 + 通信串行）
        t_e2e = _env.benchmark_ms(
            lambda: (_do_build(),
                     _comm.all_to_all_v(payload, send_sizes, recv_sizes)),
            warmup=max(1, a["warmup"] - 1), iters=max(3, a["iters"] // 2),
        )

        b = R.summarize(t_build, "T_build", "ms", seed=a["seed"])
        c = R.summarize(t_comm, "T_comm", "ms", seed=a["seed"])
        e2e = R.summarize(t_e2e, "T_total", "ms", seed=a["seed"])
        denom = b.median + c.median

        rows.append({
            "world_size": world,
            "blocks_per_rank": n_blocks,
            "num_edges_per_rank": n_edges,
            "budget": a["budget"],
            "build_location": loc,
            "t_build_median_ms": b.median,
            "t_build_ci": [b.ci_95_lower, b.ci_95_upper],
            "t_comm_median_ms": c.median,
            "t_comm_ci": [c.ci_95_lower, c.ci_95_upper],
            "t_total_median_ms": e2e.median,
            "build_share": (b.median / denom) if denom else float("nan"),
            "comm_share": (c.median / denom) if denom else float("nan"),
            "outbound_bytes_per_step": per_dst_rows * F * itemsize * world,
            "serial_sum_ms": denom,
            "e2e_over_serial": (e2e.median / denom) if denom else float("nan"),
        })

        if rank == 0:
            r = rows[-1]
            print(f"  |E_s|={n_edges:<3} (块/rank={n_blocks})  "
                  f"T_build={r['t_build_median_ms']:8.3f}ms  "
                  f"T_comm={r['t_comm_median_ms']:8.3f}ms  "
                  f"占比 构造/通信 = {r['build_share']:.1%}/{r['comm_share']:.1%}  "
                  f"（构造位置={loc}）")

    return {
        "experiment": "A1",
        "rows": rows,
        "caveat": llm_caveat(a, ctx) + (
            f" | 构造位置={loc}："
            + ("T_build 含 H2D 传输，在 NVLink 机器上会系统性高估。"
               if loc == "cpu" else "T_build 为纯 GPU 构造。")
        ),
    }


def llm_caveat(a: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    if a["backend"] == "hf":
        return "使用真实模型权重，指标为任务指标。"
    return ("synthetic 后端：只测量通信量、延迟与开销拆分，"
            "不含任何任务指标；不得作为准确率结论引用。")


# =============================================================================
# A2 —— 压缩预算扫描（精度–通信量）
# =============================================================================

def a2_budget_sweep(rank: int, world: int, a: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """5 档预算下测通信量与任务指标，绘制帕累托前沿。"""
    dtype = _env.dtypes_for(a["precision"])
    itemsize = dtype.itemsize
    dev = _env.local_device(rank)

    lm = None
    samples: List[_hf.EvalSample] = []
    if a["backend"] == "hf":
        lm = ctx.get("lm") or ctx["load_lm"]()
        ctx["lm"] = lm
        if a["eval_file"]:
            samples = _hf.load_eval_file(a["eval_file"], limit=a["eval_limit"])

    rows: List[Dict[str, Any]] = []
    for ratio in a["budget_ratios"]:
        budget = max(1, int(round(ratio * a["L_s"])))
        plan = _comm.make_uniform_plan(world, budget, a["d_h"], a["d_v"],
                                       itemsize, full_kv_len=a["L_s"])
        F = a["d_h"] + 1 + a["d_v"]
        payload = torch.randn(budget * world, F, device=dev, dtype=dtype)

        t = _env.benchmark_ms(
            lambda: _comm.all_to_all_v(payload, [budget] * world, [budget] * world),
            warmup=a["warmup"], iters=a["iters"],
        )
        ts = R.summarize(t, "T_comm", "ms", seed=a["seed"])

        row: Dict[str, Any] = {
            "budget_ratio": ratio,
            "budget": budget,
            "world_size": world,
            "t_comm_median_ms": ts.median,
            "t_comm_ci": [ts.ci_95_lower, ts.ci_95_upper],
            "outbound_bytes_per_step": plan.outbound_bytes(rank),
            "inbound_bytes_per_step": plan.inbound_bytes(rank),
            "full_kv_outbound_bytes_per_step": plan.full_outbound_bytes(rank),
            "compression_ratio": plan.compression_ratio(rank),
            # 精度轴：只有真实模型才有
            "accuracy": None,
            "accuracy_by_task": None,
            "accuracy_by_length": None,
            "n_eval": 0,
        }

        if lm is not None and samples:
            ev = _hf.evaluate(lm, samples, budget_ratio=ratio,
                              compaction_mode=a["compaction_mode"],
                              max_prompt_tokens=a["max_prompt_tokens"])
            row["accuracy"] = ev["accuracy"]
            row["accuracy_by_task"] = ev["by_task"]
            row["accuracy_by_length"] = ev["by_length"]
            row["n_eval"] = ev["n"]
            row["rows_detail"] = ev["rows"]

        rows.append(row)
        if rank == 0:
            acc = "n/a(需 --backend hf)" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
            print(f"  B={budget:<5} ratio={ratio:<6} "
                  f"compression={row['compression_ratio']:.4%}  "
                  f"出站={_env.human_bytes(row['outbound_bytes_per_step']):>10}  "
                  f"T_comm={row['t_comm_median_ms']:7.3f}ms  acc={acc}")

    have_acc = any(r["accuracy"] is not None for r in rows)
    caveat = [
        "通信量口径：边 = K[B,d_h] + β[B] + V[B,d_v]，未 padding 到等长。",
        "compression_ratio 的分母取「不压缩时每边发完整 L_s 的 K 与 V」。",
    ]
    if not have_acc:
        caveat.append(
            "**精度轴缺失**：本次运行未使用真实模型，因此这不是帕累托前沿，"
            "只是通信量曲线。要得到可与 FastKV/APB 叠加对比的前沿，"
            "需以 --backend hf 重跑。"
        )
    return {"experiment": "A2", "rows": rows, "caveat": " ".join(caveat)}


# =============================================================================
# A3 —— 组件拆分（当前被前置缺口阻断）
# =============================================================================

def a3_component_ablation(rank: int, world: int, a: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """A3 任务级组件消融：前置缺口未补齐，在此停下并打印缺失清单。"""
    blockers = list(A3_MISSING_PREREQUISITES)
    probe = ctx["build_location"]["probe"]
    if not probe.get("ok"):
        blockers.append(
            "构造链路 CUDA 探测未通过："
            + str(probe.get("error_type", "")) + " " + str(probe.get("error", ""))[:160]
        )
    if a["backend"] != "hf":
        blockers.append("--backend 必须为 hf（任务指标需要真实模型权重）。")
    if not a["eval_file"]:
        blockers.append("--eval-file 未提供（任务指标需要一个评测集）。")

    if rank == 0:
        print()
        print("  A3 未执行 —— 前置条件未满足：")
        for i, b in enumerate(blockers, 1):
            print(f"    {i}. {b}")
        print("  → 机制级版本（可在 CPU 上直接跑）："
              "python experiments/cpu/e5a_mechanism_ablation.py")

    return {
        "experiment": "A3",
        "status": "blocked",
        "executed": False,
        "blockers": blockers,
        "cpu_alternative": "experiments/cpu/e5a_mechanism_ablation.py",
        "note": (
            "A3 未被跳过，而是被前置换缺阻断。此处不产出任何数值，"
            "以避免用重构误差冒充任务指标点数。"
        ),
    }


# =============================================================================
# A5 —— 异步 vs 同步
# =============================================================================

def a5_async_vs_sync(rank: int, world: int, a: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """固定其他条件，比较异步流水与同步实现的 p50 延迟。"""
    dtype = _env.dtypes_for(a["precision"])
    itemsize = dtype.itemsize
    dev = _env.local_device(rank)
    F = a["d_h"] + 1 + a["d_v"]
    B = a["budget"]

    payload = torch.randn(B * world, F, device=dev, dtype=dtype)
    send_sizes = [B] * world
    recv_sizes = [B] * world
    comp = _comm.make_comp_work(B * world, F, a["d_h"], a["d_v"],
                                flops_scale=a["comp_scale"])

    # 单次通信、单次计算的独立基线，用于算 T_comm / T_comp
    t_comm_only = _env.benchmark_ms(
        lambda: _comm.all_to_all_v(payload, send_sizes, recv_sizes),
        warmup=a["warmup"], iters=a["iters"])
    recv0 = _comm.all_to_all_v(payload, send_sizes, recv_sizes)
    t_comp_only = _env.benchmark_ms(lambda: comp(recv0),
                                    warmup=a["warmup"], iters=a["iters"])

    def _sync_once() -> None:
        r = _comm.run_sync_pipeline(payload, send_sizes, recv_sizes, comp,
                                    n_chunks=a["chunks"])
        assert r.total_ms >= 0

    def _async_once() -> None:
        r = _comm.run_async_pipeline(payload, send_sizes, recv_sizes, comp,
                                     n_chunks=a["chunks"])
        assert r.total_ms >= 0

    t_sync = _env.benchmark_ms(_sync_once, warmup=a["warmup"], iters=a["iters"])
    t_async = _env.benchmark_ms(_async_once, warmup=a["warmup"], iters=a["iters"])

    s = R.summarize(t_sync, "T_p50_sync", "ms", seed=a["seed"])
    c = R.summarize(t_async, "T_p50_async", "ms", seed=a["seed"])
    tc = R.summarize(t_comm_only, "T_comm", "ms", seed=a["seed"])
    tp = R.summarize(t_comp_only, "T_comp", "ms", seed=a["seed"])

    speedup = (s.median / c.median) if c.median else float("nan")
    ratio_comm_over_comp = (tc.median / tp.median) if tp.median else float("nan")
    # 论文式(speedup-bound)：T_comm/T_comp 越大，可重叠的收益上限越高
    theoretical = (1.0 + tc.median / max(tp.median, 1e-9))

    if rank == 0:
        print(f"  chunks={a['chunks']}  B={B}  world={world}")
        print(f"  p50 sync  = {s.median:8.3f} ms   [{s.ci_95_lower:.3f}, {s.ci_95_upper:.3f}]")
        print(f"  p50 async = {c.median:8.3f} ms   [{c.ci_95_lower:.3f}, {c.ci_95_upper:.3f}]")
        print(f"  T_comm/T_comp = {tc.median:.3f}/{tp.median:.3f} = {ratio_comm_over_comp:.3f}")
        print(f"  实测加速 = {speedup:.4f}x   理论上界 = {theoretical:.4f}x")
        print(f"  H4 判据（≥1.05x）：{'达标' if speedup >= 1.05 else '未达标'}")

    return {
        "experiment": "A5",
        "rows": [{
            "world_size": world,
            "budget": B,
            "chunks": a["chunks"],
            "comp_scale": a["comp_scale"],
            "p50_sync_ms": s.median,
            "p50_async_ms": c.median,
            "sync_ci": [s.ci_95_lower, s.ci_95_upper],
            "async_ci": [c.ci_95_lower, c.ci_95_upper],
            "speedup": speedup,
            "theoretical_bound": theoretical,
            "t_comm_ms": tc.median,
            "t_comp_ms": tp.median,
            "t_comm_over_t_comp": ratio_comm_over_comp,
            "h4_pass": bool(speedup >= 1.05),
        }],
        "caveat": (
            "计算侧由 make_comp_work 生成的稠密注意力式算子模拟，"
            "其规模与接收到的紧凑块严格对应，但**不是真实模型的前向**。"
            "因此本结果只回答「流水结构能否产生 overlap」这一问题，"
            "不构成端到端延迟结论；后者需要真实模型的 prefill 计时（见 A2/E6）。"
            "T_comm 与 T_comp 的比值必须如实报告 —— 若 T_comp 远大于 T_comm，"
            "异步的收益上限本身就很低，此时把「加速不足」归因于实现问题是错的。"
        ),
    }


# =============================================================================
# worker / main
# =============================================================================

PARTS = {
    "a1": a1_comm_set_size,
    "a2": a2_budget_sweep,
    "a3": a3_component_ablation,
    "a5": a5_async_vs_sync,
}


def worker(rank: int, world: int, a: Dict[str, Any]) -> None:
    _env.dist_init(rank, world, a["port"], backend="nccl")
    try:
        ctx: Dict[str, Any] = {
            "build_location": _resolve_build_location(a, rank),
            "load_lm": lambda: _hf.load_model(a["model"], a["precision"],
                                              device=str(_env.local_device(rank)),
                                              attn_implementation=a["attn_impl"]),
        }
        if rank == 0:
            print("=" * 78)
            print(f"E5 消融 {a['parts']}  world={world}  backend={a['backend']}")
            print(f"  构造位置：{ctx['build_location']['requested']}"
                  f" → {ctx['build_location']['effective']}")
            if not ctx["build_location"]["probe"].get("ok", True):
                print("  [警告] 构造链路 CUDA 探测未通过："
                      f"{ctx['build_location']['probe'].get('note','')}")
            print("=" * 78)

        payload: Dict[str, Any] = {
            "experiment": "E5",
            "world_size": world,
            "backend": a["backend"],
            "build_location": ctx["build_location"],
            "results": {},
        }

        for part in a["parts"]:
            if rank == 0:
                print(f"\n--- {part.upper()} ---")
            try:
                payload["results"][part] = PARTS[part](rank, world, a, ctx)
            except Exception as e:
                payload["results"][part] = {
                    "experiment": part.upper(),
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc().splitlines()[-12:],
                }
                if rank == 0:
                    print(f"  [错误] {type(e).__name__}: {e}")

        if rank == 0:
            meta = _env.build_metadata(
                run_id=f"e5-{'-'.join(a['parts'])}-w{world}-b{a['budget']}",
                model_name=a["model"] if a["backend"] == "hf" else "synthetic",
                context_length=a["L_s"], seed=a["seed"],
                budget_ratio=a["budget"] / max(1, a["L_s"]),
                sync_async="both", num_repr_queries=a["M"],
                projection_dim=a["d_p"], task="ablation",
                precision=a["precision"], interconnect=a["interconnect"],
                notes=f"parts={a['parts']} build_location={ctx['build_location']['effective']}",
            )
            payload["metadata"] = meta.to_dict()
            out = REPO_ROOT / a["out"]
            R.save_json(str(out / "e5_results.json"), payload)
            for part, res in payload["results"].items():
                rows = res.get("rows")
                if rows:
                    R.save_csv(str(out / f"e5_{part}.csv"), rows)
            print(f"\n结果已写入 {out}")
    finally:
        _env.dist_destroy()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E5 消融 A1/A2/A3/A5（GPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parts", nargs="+", default=["a1", "a2", "a5"],
                   choices=sorted(PARTS))
    p.add_argument("--backend", choices=["synthetic", "hf"], default="synthetic")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--eval-file", dest="eval_file", type=str, default=None,
                   help="JSONL 评测集（--backend hf 时必需才能出准确率）")
    p.add_argument("--eval-limit", dest="eval_limit", type=int, default=None)
    p.add_argument("--attn-impl", dest="attn_impl", type=str, default=None,
                   help="如 flash_attention_2 / eager / sdpa；留空由 transformers 自选")
    p.add_argument("--compaction-mode", dest="compaction_mode", type=str,
                   default="topk_rms",
                   choices=["identity", "topk_rms", "topk_norm", "stride", "random"])
    p.add_argument("--max-prompt-tokens", dest="max_prompt_tokens", type=int, default=None)
    p.add_argument("--precision", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    p.add_argument("--L-s", dest="L_s", type=int, default=32768, help="源块长度")
    p.add_argument("--d-h", dest="d_h", type=int, default=128)
    p.add_argument("--d-v", dest="d_v", type=int, default=128)
    p.add_argument("--L-r", dest="L_r", type=int, default=512, help="目的端 Query 数")
    p.add_argument("--budget", type=int, default=1024, help="A1/A5 的压缩预算 B")
    p.add_argument("--budget-ratios", dest="budget_ratios", type=float, nargs="+",
                   default=[0.005, 0.01, 0.02, 0.05, 0.10], help="A2 的 5 档预算比")
    p.add_argument("--blocks-per-rank", dest="blocks_per_rank", type=int, nargs="+",
                   default=[1, 2, 4, 8], help="A1 扫描的每 rank 源块数")
    p.add_argument("--M", type=int, default=64, help="代表 Query 数")
    p.add_argument("--d-p", dest="d_p", type=int, default=32, help="投影维度")
    p.add_argument("--chunks", type=int, default=4, help="A5 的流水分块数")
    p.add_argument("--comp-scale", dest="comp_scale", type=float, default=1.0,
                   help="A5 计算侧规模系数：调小可放大 T_comm/T_comp 比，用于探测 overlap 上限")
    p.add_argument("--build-location", dest="build_location", type=str, default="auto",
                   choices=["auto", "gpu", "cpu"],
                   help="紧凑 KV 的构造位置。auto=先用 probe_gpu_construction() 探测 CUDA "
                        "构造链路，通过则用 gpu，否则退化为 cpu（CPU 构造 + H2D 传输，"
                        "T_build 会含 PCIe 传输，必须在报告里声明）")
    p.add_argument("--interconnect", type=str, default="unknown",
                   help="如实填写：nvlink / pcie / ib；会写进元数据")

    p.add_argument("--iters", type=int, default=10, help="每点的重复次数（≥10 才满足报告规范）")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=29540)
    p.add_argument("--out", type=str, default="results/gpu/e5")
    p.add_argument("--print-env", action="store_true",
                   help="只打印环境探测结果并退出，可在无 GPU 的机器上运行")
    return p


def main() -> int:
    a = build_parser().parse_args()

    if a.print_env:
        return _env.print_env_only("E5 消融 A1/A2/A3/A5", min_gpus=2, need_nccl=True)

    hf = a.backend == "hf"
    gate = _env.probe(min_gpus=2, need_nccl=True, hf_backend=hf,
                      model_name=a.model if hf else None)
    code = _env.enforce(gate, "E5 消融 A1/A2/A3/A5", SCRIPT)
    if code is not None:
        return code

    if a.iters < 10:
        print(f"[警告] --iters={a.iters} < 10，不满足 §6.1 的重复次数规范；"
              "结果不应进入主表。")

    world = int(gate.facts["device_count"])
    args = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    print(f"启动 {world} 进程（world_size={world}）\n")
    _env.maybe_spawn(worker, world, (args,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
