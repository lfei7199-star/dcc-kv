#!/usr/bin/env python
"""E5：消融实验 A1 / A2 / A3 / A4 / A5 —— GPU 端（需要多卡 + NCCL）。

与论文 §6.4 的对应
------------------
    A1  通信集大小   —— 扫描有效边数 |E_s|，拆解 构造开销 : 通信开销 的比例
    A2  压缩预算     —— 预算–精度**参考曲线**（不是帕累托前沿：两轴不同源）
    A3  组件拆分     —— full / no_beta / no_value / no_both 的任务指标差
    A5  异步 vs 同步 —— p50 延迟、T_comm/T_comp、实测加速比与理论上界

    A4  压缩 × 异步交互 —— 预算 × 同步模式的二维格。
        **已实现**（2026-09-16，缺口 G4）：见 a4_interaction_grid。

四件事必须先说清楚，否则结果会被误读

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

**第四，A4 已实现，但它不是 A1--A3 那种任务级消融，边界要说清。**
原件（第 1--4 章）通篇没有「消融」字样，也没有 A1--A5 编号 —— 编号是本仓库引入的。
2026-09-16 之前全仓缺失 A4，编号从 A3 直跳 A5（由独立监督查出，见 `commit_log.md` §S.3）。
论文 §6.3 给出的设计已实现为 `a4_interaction_grid`：四档预算，每档用**同一轮内交错**
取得的配对样本算加速比，并用配对 bootstrap 给出 95% CI；区间是否两两重叠、
两条可证伪预测是否成立，全部交给 `experiments.common.hypotheses.a4_interaction_verdict`
—— 本脚本不写任何阈值。

计算侧刻意走 G1 算子核（`src/dcc_kv_ref/attention_kernel.py`），而**不是**
`_comm.make_comp_work` 那种"规模对应"的模拟算子。差别不是洁癖：后者让查询行数随
接收元素数换算，于是通信量与计算量按同一比例缩，压缩轴与异步轴在结构上就解耦了 ——
那种构造下无论怎么测都只会得到「无交互」，结论是造出来的而不是测出来的。

边界：A4 回答的是「overlap 结构是否随预算改变」，**不产出任何准确率**，
因而不构成端到端延迟结论。见 A4_MISSING_PREREQUISITES。

用法
----
    # 任何机器上都可以先体检环境（不执行实验）
    python experiments/gpu/e5_gpu_ablation.py --print-env

    # 单机 4 卡（torchrun 由 run_gpu.sh 封装）
    torchrun --nproc_per_node=4 experiments/gpu/e5_gpu_ablation.py \
        --parts a1 a2 a5 --out results/gpu/a1a2a5

    # 只跑 A4（预算 × 同步模式二维格，判据在 hypotheses.py）
    torchrun --nproc_per_node=4 experiments/gpu/e5_gpu_ablation.py \
        --parts a4 --a4-budget-ratios 0.01 0.02 0.05 0.10 --out results/gpu/a4

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
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import hypotheses as H        # noqa: E402
from experiments.common import report as R          # noqa: E402
from experiments.gpu import _comm, _env, _forward, _hf  # noqa: E402
from src.dcc_kv_ref import CompactKV, build_compact_kv  # noqa: E402
from src.dcc_kv_ref import attention_kernel as K       # noqa: E402

SCRIPT = "experiments/gpu/e5_gpu_ablation.py"

A3_MISSING_PREREQUISITES = [
    "注意力替换钩子：紧凑 K/β/V → 注意力的**算子核**已就绪"
    "（src/dcc_kv_ref/attention_kernel.py，2026-09-16），但还缺把它挂进真实模型 "
    "forward 的钩子（src/distributed/attention_hook.py）。没有钩子就无法把四个"
    "变体（full / no_beta / no_value / no_both）喂进真实模型的注意力。",
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
    """5 档预算下测通信量与任务指标，绘制「预算–精度参考曲线」。

    注意：本函数的两个轴**不同源** —— 横轴是 DCC-KV 的逐边预算，纵轴是共享
    top-B 裁剪下的任务准确率。因此画出的只是参考曲线，**不能**称作帕累托
    前沿（返回值里 `pareto_frontier_valid=False`）。同源前沿需要
    `CompactKV -> GPU attention kernel` 就绪。
    """
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
            # 精度轴：只有真实模型才有。
            # 且两轴**不同源**：x 轴（预算 / 通信量）来自 DCC-KV 的逐边预算
            # （每条边各自选 B 个 Key），y 轴（准确率）走的是
            # _hf.apply_kv_budget 的**共享** top-B 裁剪（所有 Query 共用同一组
            # 保留位置）。二者只共享"B 是同一个标量"这一层耦合，不是同一条
            # 曲线上的两点。在 CompactKV → GPU attention kernel 就绪前，
            # 这里得到的是「预算–精度参考曲线」，**不是**帕累托前沿。
            "accuracy": None,
            "accuracy_by_task": None,
            "accuracy_by_length": None,
            "n_eval": 0,
            "comm_axis_source": "dcc_kv(per-edge 逐边预算)",
            "accuracy_axis_source": "kv_budget_shared(proxy, 共享 top-B 裁剪)",
            "axes_are_same_method": False,
            "pareto_frontier_valid": False,
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
        "**两轴不同源**：x 轴（预算 / 通信量）来自 DCC-KV 的逐边预算，"
        "y 轴（准确率）来自 kv_budget_shared 的共享 top-B 裁剪 —— "
        "预算标量 B 在两轴上的语义不同（逐边各自选 vs 全体共用一组）。"
        "因此即使 --backend hf 跑通，得到的也只是「预算–精度参考曲线」，"
        "不能称作帕累托前沿；同源前沿需要 CompactKV → GPU attention kernel"
        "（当前缺失，见 A3_MISSING_PREREQUISITES）。",
    ]
    if not have_acc:
        caveat.append(
            "**精度轴缺失**：本次运行未使用真实模型，因此连参考曲线也算不上，"
            "只有一条通信量曲线。要画出参考曲线需以 --backend hf 重跑。"
        )
    return {
        "experiment": "A2",
        "rows": rows,
        "caveat": " ".join(caveat),
        "comm_axis_source": "dcc_kv(per-edge 逐边预算)",
        "accuracy_axis_source": "kv_budget_shared(proxy, 共享 top-B 裁剪)",
        "axes_are_same_method": False,
        "pareto_frontier_valid": False,
    }


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
# A4 —— 压缩 × 异步的交互（预算 × 同步模式二维格）
# =============================================================================

A4_MISSING_PREREQUISITES = [
    "真实模型前向：本函数量测的是「紧凑 KV 算子 + 变长 All-to-Allv」这一条链，"
    "不是模型前向。要给出端到端 prefill 延迟结论，需要注意力替换钩子把 "
    "compact_kv_attention 挂进 HF forward。",
    "任务指标：A4 只回答「overlap 结构是否随预算改变」，不产出准确率。"
    "要把压缩轴与准确率联系起来，须以 --backend hf + --eval-file 跑 E6/A2。",
]


def _upper_median(values: Sequence[float]) -> float:
    """上中位数 `sorted(v)[n // 2]`。

    与 `experiments/cpu/` 各表的 `_med` 同口径（论文 §6.2 已声明该口径）。
    刻意不用 `statistics.median`：n 为偶数时两者不同，混用会让同表数字差
    0.0036 量级（曾实测 0.4354 对 0.4318）。
    """
    v = sorted(float(x) for x in values)
    if not v:
        return float("nan")
    return v[len(v) // 2]


def speedup_bound(comm_ms: float, comp_ms: float) -> float:
    """论文式~(eq:speedup-bound) 的理想加速比上界。

        (T_comm + T_comp) / max(T_comm, T_comp)
        = 1 + min(T_comm, T_comp) / max(T_comm, T_comp)

    自查发现（2026-09-18）：A4 与 A5 原先各自把它写成 `1 + T_comm / T_comp`，
    这只在 **T_comp >= T_comm** 时与论文一致。通信主导时（T_comm > T_comp）
    该写法**高估**上界（如 comm=10、comp=1：论文给 1.1，错的写法给 11.0）——
    而论文 §5 与 §7 反复强调的恰是"任一侧主导则上界趋近 1"，
    也就是错的那一半正是论文特意讨论的那一半。

    为什么必须是函数：这个不一致本来就是因为**同一个公式在脚本里抄了两遍**
    （A4 一处、A5 一处），改一处漏一处不会被任何测试发现。上界是
    `gap_to_bound` 与可证伪预测 (i)/(ii) 的基准，抄错会放大"距上界还有多远"。
    """
    lo = min(float(comm_ms), float(comp_ms))
    hi = max(float(comm_ms), float(comp_ms))
    if hi <= 0.0:
        return float("nan")
    return 1.0 + lo / hi


def _bench_paired(
    sync_fn,
    async_fn,
    warmup: int,
    iters: int,
) -> Tuple[List[float], List[float]]:
    """**同轮交错**测量 sync / async，返回配对的两个逐次耗时列表（毫秒）。

    为什么必须交错：A4 的判据是比较两条边缘曲线的加速比**置信区间**是否重叠。
    若先测完 sync 再测完 async，一次降频或邻居噪声会整体落在其中一段里，把比值
    推离 1，于是 CI 不重叠、判成「有交互、必须报二维格」—— 那是测量顺序伪影。
    同轮内交错 + 配对 bootstrap（`R.bootstrap_ratio_ci(..., paired=True)`）
    能把轮次间的公共漂移消掉。

    计时窗口纪律与 `_env.benchmark_ms` 一致：窗口**之前**做 barrier 对齐，
    窗口**之内**只做 device_sync（不做集合通信）。
    """
    for _ in range(max(0, warmup)):
        sync_fn()
        async_fn()
    _env.barrier_and_sync()

    out_sync: List[float] = []
    out_async: List[float] = []
    for _ in range(iters):
        _env.barrier_and_sync()
        t0 = time.perf_counter()
        sync_fn()
        _env.device_sync()
        out_sync.append((time.perf_counter() - t0) * 1000.0)

        _env.barrier_and_sync()
        t0 = time.perf_counter()
        async_fn()
        _env.device_sync()
        out_async.append((time.perf_counter() - t0) * 1000.0)
    return out_sync, out_async


def _a4_comp_work(query: torch.Tensor, world: int, d_h: int, d_v: int):
    """A4 的计算侧：把接收到的紧凑边喂进 G1 算子核并归并。

    与 A5 的 `make_comp_work` 的关键差别，正是 A4 存在的理由：

    1. **计算量 ∝ B。** 算子核要对 B 个 key 做 softmax 与加权；而
       `make_comp_work` 的查询行数由**接收元素数**换算，压缩一变小计算量就
       同比例缩 —— 于是压缩轴与异步轴结构性解耦，"无交互"是被构造出来的。
    2. **走真实通路。** `compact_kv_attention` + `merge_partial_attention` 与
       A3/E6/钩子共用同一段代码，量的不是另一个算子。
    """
    def _work(recv: torch.Tensor) -> torch.Tensor:
        rows = int(recv.shape[0])
        if rows == 0:
            return recv
        per = rows // int(world)
        if per < 1:
            raise ValueError(
                f"接收块行数 {rows} 少于 world={world}，无法均分到各源边。"
                "预算小于卡数时这条轴无意义（应调大预算或调小卡数），"
                "此处报错而不是静默跳过计算 —— 跳过会让 T_comp 趋零、"
                "伪造出一个巨大的加速比。"
            )
        if rows % int(world) != 0:
            # 自查发现（2026-09-18）：整除去切会**静默丢掉余数行** ——
            # `seg` 只覆盖前 per*world 行，剩下的 key 对应的计算量不计入 T_comp，
            # 且没有任何提示（T_comp 偏小 ⇒ 加速比偏乐观）。
            # 均分是 `_split_chunks` 的前提；走到这里说明布局与块对不上。
            raise ValueError(
                f"接收块行数 {rows} 不能被 world={world} 整除：按整除去切会把"
                f"余下的 {rows % int(world)} 行静默丢掉，那部分计算量不计入 T_comp。"
                "等预算分块应保证块行数 = k×world；收到非整除说明布局与块对不上。"
            )
        partials = []
        for i in range(int(world)):
            seg = recv[i * per:(i + 1) * per]
            ck = CompactKV(
                keys=seg[:, :d_h],
                logit_bias=seg[:, d_h],
                values=seg[:, d_h + 1:],
                selected_indices=torch.arange(int(seg.shape[0]),
                                              device=seg.device, dtype=torch.long),
            )
            partials.append(K.compact_kv_attention(query, ck, return_lse=True))
        return K.merge_partial_attention(partials)

    return _work


def a4_interaction_grid(rank: int, world: int, a: Dict[str, Any],
                        ctx: Dict[str, Any]) -> Dict[str, Any]:
    """A4：预算 × {同步, 异步} 二维格（论文 §6.3）。

    每档预算都测同一件事：把**同一条**紧凑边发出去并算完注意力 —— 同步臂串行
    （发完再算），异步臂让第 i 块的通信与第 i-1 块的计算重叠。加速比取
    p50_sync / p50_async，区间由**配对** bootstrap 给出。

    判定（区间两两是否重叠、两条可证伪预测）一律交给
    `experiments.common.hypotheses.a4_interaction_verdict`；本函数只测量与搬运，
    不写任何阈值 —— 阈值写在脚本里就没人改得动、也没有测试锁得住。

    计时口径
    --------
    T_comm / T_comp 的拆解**取自同步臂**。异步臂的 `comp_ms` 会吸收尚未完成的
    下一块传输（见 `_comm.run_async_pipeline` 文档），是上界；同步与异步之间
    只有 `total_ms` 可比。若拿异步臂的拆解去算 T_comm/T_comp，会把 overlap
    的收益错记成计算变慢，进而低估理论上界。
    """
    dtype = _env.dtypes_for(a["precision"])
    dev = _env.local_device(rank)
    loc = ctx["build_location"]["effective"]

    ratios = list(a["a4_budget_ratios"])
    if len(ratios) < H.A4_MIN_BUDGET_LEVELS:
        raise ValueError(
            f"A4 需要至少 {H.A4_MIN_BUDGET_LEVELS} 档预算（论文 §6.3），"
            f"收到 {len(ratios)} 档：{ratios}。在租卡前失败好过在卡上跑完一轮"
            "才由 verdict 报 insufficient_data。"
        )

    inputs = _make_edge_inputs(1, a["L_s"], a["d_h"], a["d_v"], a["L_r"],
                               dev, dtype, a["seed"])
    query = torch.randn(a["L_r"], a["d_h"], device=dev, dtype=dtype)

    cells: List[H.A4Cell] = []
    points: List[H.A4BudgetPoint] = []
    rows: List[Dict[str, Any]] = []

    for ratio in ratios:
        budget = max(2, int(round(ratio * a["L_s"])))

        # --- T_build：逐边构造（口径同 A1）---
        t_build = _env.benchmark_ms(
            lambda: _build_edge(inputs[0], budget, a["M"], a["d_p"], a["seed"], loc),
            warmup=a["warmup"], iters=a["iters"])
        b_s = R.summarize(t_build, "T_build", "ms", seed=a["seed"])

        ck = _build_edge(inputs[0], budget, a["M"], a["d_p"], a["seed"], loc)
        edge = _comm.pack_compact_edge(ck.keys, ck.logit_bias, ck.values)
        payload = torch.cat([edge] * world, dim=0)
        send_sizes = [budget] * world
        recv_sizes = [budget] * world
        comp = _a4_comp_work(query, world, a["d_h"], a["d_v"])

        seen: Dict[str, List[Any]] = {"sync": [], "async": []}

        def _sync_once() -> None:
            seen["sync"].append(_comm.run_sync_pipeline(
                payload, send_sizes, recv_sizes, comp, n_chunks=a["chunks"]))

        def _async_once() -> None:
            seen["async"].append(_comm.run_async_pipeline(
                payload, send_sizes, recv_sizes, comp, n_chunks=a["chunks"]))

        t_sync, t_async = _bench_paired(_sync_once, _async_once,
                                        warmup=a["warmup"], iters=a["iters"])

        s = R.summarize(t_sync, "T_total_sync", "ms", seed=a["seed"])
        c = R.summarize(t_async, "T_total_async", "ms", seed=a["seed"])

        comm_ms = _upper_median([t.comm_ms for t in seen["sync"]])
        comp_ms = _upper_median([t.comp_ms for t in seen["sync"]])
        ratio_comm_comp = (comm_ms / comp_ms) if comp_ms > 0 else float("nan")
        bound = speedup_bound(comm_ms, comp_ms)

        rci = R.bootstrap_ratio_ci(
            t_sync, t_async, metric_name="speedup", unit="x",
            ci_level=H.A4_CI_LEVEL, seed=a["seed"],
        )

        async_comm_raw = _upper_median([t.comm_ms for t in seen["async"]])
        async_comp_ub = _upper_median([t.comp_ms for t in seen["async"]])
        cells.append(H.A4Cell(
            budget_ratio=ratio, budget=budget, sync_mode="sync",
            t_build_ms=b_s.median,
            t_comm_ms=comm_ms, t_comp_ms=comp_ms,
            t_total_ms=s.median, p50_ms=s.median,
            timing_decomposition_valid=True,
            t_comm_ms_raw=comm_ms,
            timing_decomposition_note="同步臂：通信段与计算段分开计时，可作拆解用"))
        # 异步格的 comm 置 NaN：它不是通信时间（device_sync 已把飞行中的下一块
        # 等掉，下一轮 wait 立即返回，只剩发起开销）。实测（单块传输 20ms、
        # 单块计算 5ms、4 块）：同步 85.8ms vs 异步 20.5ms，而真实通信总量 80ms。
        # 保留 NaN 而不是 0：0 会被读成"通信为零"。原值进 t_comm_ms_raw。
        cells.append(H.A4Cell(
            budget_ratio=ratio, budget=budget, sync_mode="async",
            t_build_ms=b_s.median,
            t_comm_ms=float("nan"),
            t_comp_ms=async_comp_ub,
            t_total_ms=c.median, p50_ms=c.median,
            timing_decomposition_valid=False,
            t_comm_ms_raw=async_comm_raw,
            timing_decomposition_note=(
                "异步臂：本次 comm_ms 只剩发起开销（%.3fms，已存入 raw），"
                "不是通信时间；t_comp_ms 是上界（%.3fms）。"
                "跨同步模式只比 t_total_ms。" % (async_comm_raw, async_comp_ub))))

        points.append(H.A4BudgetPoint(
            budget_ratio=ratio, budget=budget,
            p50_sync_ms=s.median, p50_async_ms=c.median,
            speedup=rci.point,
            speedup_ci_low=rci.ci_low, speedup_ci_high=rci.ci_high,
            t_comm_over_t_comp=ratio_comm_comp,
            theoretical_bound=bound,
            t_build_share=(b_s.median / s.median) if s.median > 0 else float("nan"),
        ))

        rows.append({
            "budget_ratio": ratio,
            "budget": budget,
            "world_size": world,
            "build_location": loc,
            "p50_sync_ms": s.median,
            "p50_async_ms": c.median,
            "speedup": rci.point,
            "speedup_ci_low": rci.ci_low,
            "speedup_ci_high": rci.ci_high,
            "speedup_ci_level": H.A4_CI_LEVEL,
            "ci_method": rci.method,
            "t_comm_over_t_comp": ratio_comm_comp,
            "theoretical_bound": bound,
            "t_build_ms": b_s.median,
            "t_build_share": (b_s.median / s.median) if s.median > 0 else float("nan"),
            "n_pairs": rci.n_numerator,
            "chunks": a["chunks"],
            "comp_source": "src/dcc_kv_ref/attention_kernel.py",
        })

        if rank == 0:
            print(f"  B={budget:<6} ratio={ratio:<6} "
                  f"p50 sync={s.median:8.3f}ms  async={c.median:8.3f}ms  "
                  f"speedup={rci.point:.4f}x CI[{rci.ci_low:.4f},{rci.ci_high:.4f}]  "
                  f"bound={bound:.3f}x  T_comm/T_comp={ratio_comm_comp:.3f}")

    verdict = H.a4_interaction_verdict(points)

    if rank == 0:
        print()
        print(f"  A4 判定：resolved={verdict.resolved}  "
              f"CI 重叠 {verdict.n_overlapping_pairs}/{verdict.n_pairs} 对  "
              f"→ {verdict.reporting_requirement}")
        print(f"    {verdict.reason}")
        print(f"    预测(i) 加速比随预算单调且最靠近上界档 = "
              f"{'成立' if verdict.prediction_i_holds else '不成立'}"
              f"（单调比例 {verdict.monotone_fraction:.2f}）；"
              f"预测(ii) 与上界之差随预算减小而扩大 = "
              f"{'成立' if verdict.prediction_ii_holds else '不成立'}"
              f"（单调比例 {verdict.prediction_ii_monotone_fraction:.2f}）")

    return {
        "experiment": "A4",
        "status": "done" if verdict.resolved else "insufficient_data",
        "executed": True,
        "grid": [cell.to_dict() for cell in cells],
        "points": [{
            "budget_ratio": p.budget_ratio,
            "budget": p.budget,
            "p50_sync_ms": p.p50_sync_ms,
            "p50_async_ms": p.p50_async_ms,
            "speedup": p.speedup,
            "speedup_ci_low": p.speedup_ci_low,
            "speedup_ci_high": p.speedup_ci_high,
            "t_comm_over_t_comp": p.t_comm_over_t_comp,
            "theoretical_bound": p.theoretical_bound,
            "t_build_share": p.t_build_share,
        } for p in points],
        "verdict": verdict.to_dict(),
        "rows": rows,
        "compute_source": "src/dcc_kv_ref/attention_kernel.py (G1 operator kernel)",
        "timing_convention": (
            "T_comm/T_comp 拆解取自同步臂；异步臂 comp_ms 为上界，"
            "同步/异步之间只有 total_ms 可比（见 _comm.run_async_pipeline 文档）。"
        ),
        "caveat": (
            "计算侧走真实算子核（compact_kv_attention + merge_partial_attention），"
            "不是规模模拟 —— 因此预算轴确实同时改变通信量与计算量，交互是可测的。"
            "但本结果仍**不是端到端延迟结论**：没有真实模型前向（见 "
            "A4_MISSING_PREREQUISITES）。判定用的 CI 由同轮交错样本配对 bootstrap 得出；"
            "「无交互」是有效结论（A2 与 A5 的边缘结果可分开引用），"
            "不是失败。"
        ),
        "missing_prerequisites": list(A4_MISSING_PREREQUISITES),
    }


# =============================================================================
# A5 —— 异步 vs 同步
# =============================================================================

A5_ARMS = ("sim", "real")

A5_MISSING_PREREQUISITES = [
    "真实模型前向：A5 的两个臂**都**不是模型前向 —— sim 臂是规模匹配的模拟算子，"
    "real 臂走 G1 算子核但输入是合成的紧凑 KV。要给出端到端 prefill 延迟结论，"
    "需要注意力替换钩子把 compact_kv_attention 挂进 HF forward。",
]


def _a5_baselines_sim(a, payload, send_sizes, recv_sizes, comp):
    """sim 臂的独立基线：只通信一次 / 只计算一次（不含流水结构）。"""
    t_comm_only = _env.benchmark_ms(
        lambda: _comm.all_to_all_v(payload, send_sizes, recv_sizes),
        warmup=a["warmup"], iters=a["iters"])
    recv0 = _comm.all_to_all_v(payload, send_sizes, recv_sizes)
    t_comp_only = _env.benchmark_ms(lambda: comp(recv0),
                                    warmup=a["warmup"], iters=a["iters"])
    return (R.summarize(t_comm_only, "T_comm", "ms", seed=a["seed"]),
            R.summarize(t_comp_only, "T_comp", "ms", seed=a["seed"]))


def a5_async_vs_sync(rank: int, world: int, a: Dict[str, Any],
                     ctx: Dict[str, Any]) -> Dict[str, Any]:
    """固定其他条件，比较异步流水与同步实现的 p50 延迟。

    两个臂，问的是两个不同的问题
    ---------------------------
    ``sim``  —— 计算侧是 ``make_comp_work``，其规模与接收到的紧凑块严格对应
        （查询行数由接收元素数换算）。它回答「当计算量与通信量可比时，流水结构
        能不能产生 overlap」，对应真实 prefill 里本地长上下文注意力占主导的情形。
    ``real`` —— 计算侧走 G2 的桥接层（``_forward.pipelined_attention``），算的是
        **真算子核**（`compact_kv_attention` + lse 归并）。它回答「只算紧凑块的
        注意力时 T_comp 有多大、值不值得重叠」。

    **两个臂的加速比不可混报。** 同一份硬件上 sim 臂可能给出远大于 1 的加速比、
    real 臂给出约 1.0 —— 这不是矛盾，是"计算量是多少"这个前提不同。故每行都带
    ``arm`` 字段，且顶层 ``h4`` 显式声明判据落在哪个臂上（改口径要改论文与
    release_checklist，不是改这个脚本）。

    另一条运行时自检：real 臂每次测量后比较同步与异步两路的输出是否**逐位**相同
    （``_forward.assert_same_answer`` 的判据）。若这条不成立，本臂的加速比就建立在
    两套数值上，不可用 —— 此时该行的 ``h4_pass`` 置 None（未判定）、``h4_judged=False``，
    顶层 ``status`` 变 ``numeric_invariance_broken``。

    为什么**不**在这里抛错：计时数据本身仍然有效，失效的只是"加速比可比"这一层。
    直接抛错会把诊断与整段计时一并带走，还会让人以为流水实现坏了 —— 而真实原因
    可能是传输不确定性或构造用的随机种子在两次测量间变了。落盘 + 拒绝下判定，
    比抛错更能说明问题。

    边界：**两个臂都不是真实模型前向**，因此都不构成端到端延迟结论。
    """
    dtype = _env.dtypes_for(a["precision"])
    dev = _env.local_device(rank)
    F = a["d_h"] + 1 + a["d_v"]
    B = a["budget"]
    chunks_list = [int(c) for c in a["a5_chunks"]]
    arms = A5_ARMS if a["a5_comp_source"] == "both" else (a["a5_comp_source"],)

    rows: List[Dict[str, Any]] = []
    invariance_ok = True

    for arm in arms:
        if arm == "sim":
            payload = torch.randn(B * world, F, device=dev, dtype=dtype)
            send_sizes = [B] * world
            recv_sizes = [B] * world
            comp = _comm.make_comp_work(B * world, F, a["d_h"], a["d_v"],
                                        flops_scale=a["comp_scale"])
            tc, tp = _a5_baselines_sim(a, payload, send_sizes, recv_sizes, comp)
            bound_basis = "standalone_baselines"
        else:
            inputs = _make_edge_inputs(world, a["L_s"], a["d_h"], a["d_v"],
                                       a["L_r"], dev, dtype, a["seed"])
            loc = ctx["build_location"]["effective"]
            edges = [_build_edge(inputs[j], B, a["M"], a["d_p"],
                                 a["seed"] + j, loc) for j in range(world)]
            layout = _forward.make_uniform_layout(world, B, a["d_h"], a["d_v"])
            payload = _forward.pack_edges(edges, layout)
            send_sizes = list(layout.send_sizes)
            recv_sizes = list(layout.recv_sizes)
            query = torch.randn(a["L_r"], a["d_h"], device=dev, dtype=dtype)
            bound_basis = "sync_pipeline_decomposition"

        for chunks in chunks_list:
            if arm == "sim":
                # ⚠️ 不能用 `assert f(...).total_ms >= 0` 来"触发测量"：
                # `python -O` 会把 assert 整条删掉，函数体随即变空 ⇒ 什么都不测、
                # 计时趋零、加速比变成垃圾，而且**没有任何报错**。
                # 这里与 real 臂一样用 sink 接住返回值，保证调用真发生。
                sim_sink: Dict[str, Any] = {}

                def _sync_once(p=payload, ss=send_sizes, rs=recv_sizes, cw=comp,
                               n=chunks, s=sim_sink):
                    s["sync"] = _comm.run_sync_pipeline(p, ss, rs, cw, n_chunks=n)

                def _async_once(p=payload, ss=send_sizes, rs=recv_sizes, cw=comp,
                                n=chunks, s=sim_sink):
                    s["async"] = _comm.run_async_pipeline(p, ss, rs, cw, n_chunks=n)

                t_sync, t_async = _bench_paired(_sync_once, _async_once,
                                                a["warmup"], a["iters"])
                sync_comm_ms = float("nan")
                sync_comp_ms = float("nan")
                same_answer: Optional[bool] = None
                max_diff: Optional[float] = None
                n_partials = 0
            else:
                sink: Dict[str, Any] = {}

                def _sync_once(q=query, p=payload, lay=layout, n=chunks, s=sink):
                    s["sync"] = _forward.pipelined_attention(
                        q, p, lay, mode="sync", n_chunks=n)

                def _async_once(q=query, p=payload, lay=layout, n=chunks, s=sink):
                    s["async"] = _forward.pipelined_attention(
                        q, p, lay, mode="async", n_chunks=n)

                t_sync, t_async = _bench_paired(_sync_once, _async_once,
                                                a["warmup"], a["iters"])
                # 运行时自检：调度不得改变答案（逐位）。这里只取判据、不抛错
                # —— 见 docstring「为什么不在这里抛错」。
                o_sync = sink["sync"].out
                o_async = sink["async"].out
                same_answer = bool(torch.equal(o_sync, o_async))
                max_diff = float((o_sync - o_async).abs().max())
                if not same_answer:
                    invariance_ok = False
                n_partials = int(sink["sync"].n_partials)
                # 拆解取自同步臂：异步臂的 comp_ms 会吸收尚未完成的下一块传输，
                # 是上界，只有 total_ms 在同步/异步间可比。
                sync_comm_ms = float(sink["sync"].timing.comm_ms)
                sync_comp_ms = float(sink["sync"].timing.comp_ms)

            s = R.summarize(t_sync, "T_p50_sync", "ms", seed=a["seed"])
            c = R.summarize(t_async, "T_p50_async", "ms", seed=a["seed"])
            rci = R.bootstrap_ratio_ci(t_sync, t_async, metric_name="speedup",
                                       unit="x", ci_level=H.A4_CI_LEVEL,
                                       seed=a["seed"])

            if arm == "sim":
                t_comm_ms, t_comp_ms = float(tc.median), float(tp.median)
            else:
                t_comm_ms, t_comp_ms = sync_comm_ms, sync_comp_ms
            ratio_comm_over_comp = (t_comm_ms / t_comp_ms) if t_comp_ms > 0 else float("nan")
            theoretical = speedup_bound(t_comm_ms, t_comp_ms)

            chunks_effective = _comm.effective_chunk_count(send_sizes, recv_sizes, chunks)
            row: Dict[str, Any] = {
                "arm": arm,
                "comp_source": ("make_comp_work" if arm == "sim"
                                else "src/dcc_kv_ref/attention_kernel.py"),
                "world_size": world,
                "budget": B,
                "chunks": chunks,
                "chunks_effective": chunks_effective,
                "overlap_window_available": chunks_effective > 1,
                "comp_scale": a["comp_scale"] if arm == "sim" else None,
                "p50_sync_ms": s.median,
                "p50_async_ms": c.median,
                "sync_ci": [s.ci_95_lower, s.ci_95_upper],
                "async_ci": [c.ci_95_lower, c.ci_95_upper],
                "speedup": rci.point,
                "speedup_ci_low": rci.ci_low,
                "speedup_ci_high": rci.ci_high,
                "speedup_ci_level": H.A4_CI_LEVEL,
                "ci_method": rci.method,
                "theoretical_bound": theoretical,
                "bound_basis": bound_basis,
                "t_comm_ms": t_comm_ms,
                "t_comp_ms": t_comp_ms,
                "t_comm_over_t_comp": ratio_comm_over_comp,
                "n_pairs": rci.n_numerator,
                "sync_equals_async": same_answer,
                "sync_async_max_abs_diff": max_diff,
                "n_partials": n_partials,
                # real 臂只有在"调度不改变答案"成立时才允许下判据；否则加速比
                # 是拿两套数值比出来的。None = 未判定，不是"未达标"。
                "h4_judged": bool(arm != "real" or same_answer),
                "h4_pass": (H.h4_pass(rci.point)
                            if (arm != "real" or same_answer) else None),
                "h4_threshold_used": H.H4_MIN_P50_SPEEDUP,
            }
            rows.append(row)

            if rank == 0:
                print(f"  [{arm:4s}] chunks={chunks:<2} (实际 {chunks_effective:<2}) "
                      f"B={B}  world={world}")
                print(f"    p50 sync  = {s.median:9.3f} ms  [{s.ci_95_lower:.3f}, {s.ci_95_upper:.3f}]")
                print(f"    p50 async = {c.median:9.3f} ms  [{c.ci_95_lower:.3f}, {c.ci_95_upper:.3f}]")
                print(f"    T_comm/T_comp = {t_comm_ms:.3f}/{t_comp_ms:.3f} "
                      f"= {ratio_comm_over_comp:.3f}  ({bound_basis})")
                print(f"    实测加速 = {rci.point:.4f}x "
                      f"CI[{rci.ci_low:.4f},{rci.ci_high:.4f}]  理论上界 = {theoretical:.4f}x")
                if same_answer is not None:
                    print(f"    同步/异步输出逐位相同 = {same_answer}"
                          f"（{n_partials} 个 partial，最大绝对差 {max_diff:.3e}）")
                    if not same_answer:
                        print("    [严重] 调度改变了答案 ⇒ 本行的加速比不可用，"
                              "h4_pass 置 None（见 _forward 模块文档的两条不变式）")
                print(f"    H4 判据（>={H.H4_MIN_P50_SPEEDUP}x）："
                      f"{'达标' if H.h4_pass(rci.point) else '未达标'}")

    passes_by_arm: Dict[str, Any] = {}
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        # 行级纪律：h4_pass 为 None 表示**未判定**（real 臂的同步/异步不逐位相同），
        # 不是未达标。自查发现（2026-09-18）：原先写 `bool(r["h4_pass"])`，
        # 把 None 折成 False 混进 n_rows_passing —— 聚合这一步恰好抹掉了行级注释
        # 明确写出的那条区分（「no-judge ≠ 结论为假」是本仓库反复踩的坑）。
        judged = [r for r in sub if r.get("h4_pass") is not None]
        vals = [bool(r["h4_pass"]) for r in judged]
        passes_by_arm[arm] = {
            "n_rows": len(sub),
            "n_rows_judged": len(judged),
            "n_rows_passing": int(sum(1 for v in vals if v)),
            "n_rows_unjudged": len(sub) - len(judged),
            "any_row_passes": (bool(any(vals)) if judged else None),
        }

    h4 = {
        "valid": bool(invariance_ok),
        "threshold": H.H4_MIN_P50_SPEEDUP,
        "judged_on_arm": "sim",
        "rationale": (
            "沿用接线前的口径（sim 臂）。两个臂的加速比**不可混报** —— "
            "同一份硬件上 sim 臂可能远大于 1、real 臂约 1.0，"
            "差别只在「计算量是多少」这个前提。改口径要改论文 §6.4 与 "
            "docs/release_checklist.md，不是改这个脚本。"
        ),
        "passes_by_arm": passes_by_arm,
        "invalid_reason": (
            None if invariance_ok else
            "real 臂的同步/异步输出不逐位相同 ⇒ 加速比建立在两套数值上，"
            "本次运行的 A5 判据整体不可用。先查传输确定性与构造种子是否稳定。"
        ),
    }

    if rank == 0:
        print()
        print(f"  H4（阈值 {H.H4_MIN_P50_SPEEDUP}x）判据落在 {h4['judged_on_arm']} 臂上；"
              "两臂数值不可混报。")
        for arm, info in passes_by_arm.items():
            print(f"    {arm:4s}: {info['n_rows_passing']}/{info['n_rows_judged']} 行达标"
                  f"（未判定 {info['n_rows_unjudged']} 行）"
                  f"（任一行达标 = {info['any_row_passes']}）")

    if rank == 0 and not invariance_ok:
        print()
        print("  !! 本次运行的 real 臂破坏了「调度不改变答案」⇒ "
              "A5 判据整体不可用（计时仍已落盘，但 h4_pass 一律为 None）")

    return {
        "experiment": "A5",
        "status": "ok" if invariance_ok else "numeric_invariance_broken",
        "rows": rows,
        "arms": list(arms),
        "h4": h4,
        "analyzed": True,
        "caveat": (
            "两个臂的计算侧都不是真实模型前向：sim 臂是 make_comp_work 的规模模拟，"
            "real 臂走 G1 算子核但输入是合成的紧凑 KV。因此本结果只回答"
            "「流水结构能否产生 overlap」与「只算紧凑块的注意力时 T_comp 有多大」，"
            "不构成端到端延迟结论（后者需要真实模型的前向计时，见 A2/E6）。"
            "T_comm 与 T_comp 的比值必须如实报告 —— 若 T_comp 远小于 T_comm，"
            "异步的收益上限本身就很低，此时把「加速不足」归因于实现问题是错的。"
            "bound_basis 注明上界是用独立基线算的（sim，理想单次口径）还是用"
            "同步臂的流水拆解算的（real）—— 两者不可互换。"
            "另注意 n_chunks 是时延/显存旋钮而非数值旋钮：分块会在 ULP 级改变"
            "输出，需要逐位复现时须固定它（见 _forward 模块文档）。"
        ),
        "missing_prerequisites": list(A5_MISSING_PREREQUISITES),
    }


# =============================================================================
# worker / main
# =============================================================================

PARTS = {
    "a1": a1_comm_set_size,
    "a2": a2_budget_sweep,
    "a3": a3_component_ablation,
    "a4": a4_interaction_grid,
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
                warmup=a["warmup"], iters=a["iters"],
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
        description="E5 消融 A1/A2/A3/A4/A5（GPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parts", nargs="+", default=["a1", "a2", "a4", "a5"],
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
    p.add_argument("--a4-budget-ratios", dest="a4_budget_ratios", type=float,
                   nargs="+", default=[0.01, 0.02, 0.05, 0.10],
                   help="A4 二维格的预算档（至少 4 档，论文 §6.3）")
    p.add_argument("--blocks-per-rank", dest="blocks_per_rank", type=int, nargs="+",
                   default=[1, 2, 4, 8], help="A1 扫描的每 rank 源块数")
    p.add_argument("--M", type=int, default=64, help="代表 Query 数")
    p.add_argument("--d-p", dest="d_p", type=int, default=32, help="投影维度")
    p.add_argument("--chunks", type=int, default=4, help="A4 的流水分块数")
    p.add_argument("--a5-chunks", dest="a5_chunks", type=int, nargs="+",
                   default=[1, 2, 4, 8],
                   help="A5 扫描的流水分块数（对齐执行计划的 4 档网格）")
    p.add_argument("--a5-comp-source", dest="a5_comp_source", type=str,
                   default="both", choices=["sim", "real", "both"],
                   help="A5 的计算侧：sim=规模模拟（make_comp_work）、"
                        "real=G1 算子核（_forward.pipelined_attention）。"
                        "两臂的加速比不可混报，故默认都跑并分别标注")
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
        return _env.print_env_only("E5 消融 A1/A2/A3/A4/A5", min_gpus=2, need_nccl=True)

    hf = a.backend == "hf"
    gate = _env.probe(min_gpus=2, need_nccl=True, hf_backend=hf,
                      model_name=a.model if hf else None)
    code = _env.enforce(gate, "E5 消融 A1/A2/A3/A4/A5", SCRIPT)
    if code is not None:
        return code

    if a.iters < 10:
        print(f"[警告] --iters={a.iters} < 10，不满足 §6.1 的重复次数规范；"
              "结果不应进入主表。")

    # A4 的档数在校验期就卡住：档数不足时 verdict 会返回 resolved=False，
    # 而那要等整轮跑完才看得到 —— 在租来的卡上白烧一轮。这里提前失败。
    if "a4" in a.parts and len(a.a4_budget_ratios) < H.A4_MIN_BUDGET_LEVELS:
        print(f"[错误] A4 需要至少 {H.A4_MIN_BUDGET_LEVELS} 档预算（论文 §6.3），"
              f"收到 {len(a.a4_budget_ratios)} 档：{a.a4_budget_ratios}")
        return 2

    world = int(gate.facts["device_count"])
    args = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    print(f"启动 {world} 进程（world_size={world}）\n")
    _env.maybe_spawn(worker, world, (args,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
