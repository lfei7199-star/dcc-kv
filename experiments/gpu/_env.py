"""GPU 实验的公共环境闸门、计量与元数据工具。

设计原则
--------
本目录下的脚本**必须**在真实 GPU（且多数需要 NCCL）上运行。为避免误在
CPU 机器上跑出一堆无意义的数字并污染结果表，所有脚本在进入实验逻辑前
先过 `probe()` + `enforce()` 两道闸门：

    gate = probe(min_gpus=2, need_nccl=True)
    code = enforce(gate, "E5/A5")
    if code is not None:
        return code          # 环境不达标，直接退出，不产生任何结果文件

闸门失败时 **不写结果文件**，退出码为 3（便于 CI 区分"没跑"与"跑失败"）。

`--print-env` 可以在**任何**机器上运行（包括纯 CPU），只打印环境探测
结果，用于在正式申请 GPU 前确认目标机器是否满足要求。

计量口径
--------
所有延迟都走 `benchmark_ms()`：先 warmup 再重复 iters 次，每次前后
`torch.cuda.synchronize()`，返回**逐次** wall-clock（ms）列表而非均值。
理由与论文 §6.1 的统计规范一致：报告必须来自多次 run 的分布
（median / p5 / p95 / bootstrap CI），单次运行值不可用。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment_metadata import ExperimentMetadata  # noqa: E402

GATE_EXIT_CODE = 3


# =============================================================================
# 环境探测
# =============================================================================

@dataclass
class Gate:
    """环境探测结果。

    ok          —— 满足实验的最低要求
    blockers    —— 硬性不满足项（会让实验无意义或直接报错）
    warnings    —— 软性不满足项（能跑，但结果的可解释性受限，需在报告里注明）
    facts       —— 供元数据落盘的实测环境事实
    """
    ok: bool
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "facts": self.facts,
        }


def nccl_version() -> str:
    """NCCL 版本。未编译或不支持时返回 'unavailable'。"""
    try:
        import torch.distributed as dist
        if dist.is_nccl_available():
            v = torch.cuda.nccl.version()
            if isinstance(v, (tuple, list)):
                return ".".join(str(x) for x in v)
            return str(v)
    except Exception:
        pass
    return "unavailable"


def gpu_facts() -> Dict[str, Any]:
    """采集 GPU 环境事实，不抛异常（CPU 机器上返回 available=False）。"""
    facts: Dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda or "unavailable",
        "nccl_version": nccl_version(),
        "torch_version": torch.__version__,
        "gpu_model": "unknown",
        "gpu_memory_gb": 0,
        "gpu_names": [],
    }
    if facts["cuda_available"]:
        names = []
        for i in range(facts["device_count"]):
            try:
                names.append(torch.cuda.get_device_name(i))
            except Exception:
                names.append("unknown")
        facts["gpu_names"] = names
        facts["gpu_model"] = names[0] if names else "unknown"
        try:
            props = torch.cuda.get_device_properties(0)
            facts["gpu_memory_gb"] = int(props.total_memory // (1 << 30))
        except Exception:
            pass
    return facts


def probe(
    min_gpus: int = 1,
    need_nccl: bool = True,
    min_memory_gb: int = 0,
    model_name: Optional[str] = None,
    hf_backend: bool = False,
) -> Gate:
    """探测当前环境能否支撑某个 GPU 实验。

    Args:
        min_gpus:      最低 GPU 数（论文 §6.1 的 M2/M3/M4 档位）
        need_nccl:     是否需要 NCCL 后端（多进程通信实验需要）
        min_memory_gb: 单卡最低显存
        model_name:    HF 模型名/路径；给定时检查能否加载
        hf_backend:    是否使用真实模型权重后端
    """
    facts = gpu_facts()
    blockers: List[str] = []
    warnings: List[str] = []

    if not facts["cuda_available"]:
        blockers.append(
            "torch.cuda.is_available() == False：当前机器无可用 GPU。"
            "本脚本不会退化为 CPU 运行——机制级版本见 experiments/cpu/ 下同名实验。"
        )
    else:
        n = facts["device_count"]
        if n < min_gpus:
            blockers.append(
                f"可用 GPU 数 {n} < 要求的 {min_gpus}。"
                f"请调整 --nproc-per-node 或申请更多卡。"
            )
        if min_memory_gb and facts["gpu_memory_gb"] < min_memory_gb:
            blockers.append(
                f"单卡显存 {facts['gpu_memory_gb']}GB < 要求的 {min_memory_gb}GB。"
            )
        if need_nccl and facts["nccl_version"] == "unavailable":
            blockers.append(
                "NCCL 不可用（torch.distributed.is_nccl_available() == False）。"
                "多卡通信实验必须有 NCCL；gloo 后端不支持 CUDA tensor 的集合通信。"
            )
        if need_nccl and facts["device_count"] > 1 and "NVLink" not in facts["gpu_model"]:
            warnings.append(
                "未检测到型号中的 NVLink 字样；跨卡带宽可能是 PCIe 而非 NVLink。"
                "M3 异步实验的 overlap 收益对带宽敏感，需在元数据 interconnect 字段如实记录。"
            )

    if hf_backend:
        try:
            import transformers  # noqa: F401
        except ImportError:
            blockers.append(
                "使用 --backend hf 需要 transformers，当前未安装。"
                "请先 pip install -r requirements.txt（GPU 环境）。"
            )
        if model_name:
            local = pathlib.Path(model_name).expanduser()
            looks_like_path = (os.sep in model_name) or model_name.startswith(".")
            if looks_like_path and not local.exists():
                blockers.append(f"模型路径不存在：{local}")
            elif not looks_like_path:
                warnings.append(
                    f"模型 {model_name} 将走 HF 在线拉取路径；"
                    "离线集群请改为传入本地权重目录的绝对路径。"
                )
        else:
            blockers.append("--backend hf 必须同时给出 --model。")

    return Gate(ok=not blockers, blockers=blockers, warnings=warnings, facts=facts)


def enforce(gate: Gate, experiment: str, script_path: str) -> Optional[int]:
    """闸门不通过时打印阻断说明并返回退出码；通过时返回 None。"""
    print("=" * 78)
    print(f"{experiment} —— 环境闸门检查")
    print("=" * 78)
    print(f"  torch          : {gate.facts.get('torch_version')}")
    print(f"  CUDA           : {gate.facts.get('cuda_version')}"
          f"  (available={gate.facts.get('cuda_available')})")
    print(f"  GPU 数         : {gate.facts.get('device_count')}")
    print(f"  GPU 型号       : {gate.facts.get('gpu_model')}"
          f"  ({gate.facts.get('gpu_memory_gb')} GB)")
    print(f"  NCCL           : {gate.facts.get('nccl_version')}")
    for w in gate.warnings:
        print(f"  [警告] {w}")

    if gate.ok:
        print("  => 闸门通过\n")
        return None

    print()
    print("-" * 78)
    print(f"阻断：{experiment} 无法在当前环境执行。")
    for b in gate.blockers:
        print(f"  - {b}")
    print("-" * 78)
    print(f"本次未产生任何结果文件（退出码 {GATE_EXIT_CODE}）。")
    print("如需确认目标机器的环境是否达标，可在该机器上运行：")
    print(f"    python {script_path} --print-env")
    print()
    print("本实验的 CPU 可复现部分见 experiments/cpu/ —— 那里给出的是机制级证据，")
    print("二者不能互相替代：CPU 侧测重构精度，GPU 侧测任务指标与通信性能。")
    print("-" * 78)
    return GATE_EXIT_CODE


def print_env_only(experiment: str, min_gpus: int = 1, need_nccl: bool = True) -> int:
    """`--print-env`：只打印探测结果，不执行实验，可在任何机器上运行。"""
    gate = probe(min_gpus=min_gpus, need_nccl=need_nccl)
    print(json.dumps({"experiment": experiment, **gate.to_dict()},
                     indent=2, ensure_ascii=False))
    return 0 if gate.ok else GATE_EXIT_CODE


# =============================================================================
# 元数据
# =============================================================================

def build_metadata(
    run_id: str,
    *,
    model_name: str = "unknown",
    context_length: int = 0,
    seed: int = 42,
    budget_ratio: float = 0.05,
    sync_async: str = "async",
    baseline: str = "dcc_kv",
    num_repr_queries: int = 64,
    projection_dim: int = 32,
    lambda_beta: float = 3e-2,   # 与 src.dcc_kv_ref.DEFAULT_LAMBDA_BETA 一致（E11）
    lambda_value: float = 1e-3,
    task: str = "unknown",
    precision: str = "bfloat16",
    interconnect: str = "unknown",
    rope_extension_used: Optional[str] = None,
    rope_extension_disclosed: bool = False,
    notes: str = "",
) -> ExperimentMetadata:
    """按论文 §6.2 的规范填充元数据，CUDA/NCCL/GPU 字段从环境实测取。"""
    facts = gpu_facts()
    return ExperimentMetadata(
        run_id=run_id,
        cuda_version=str(facts["cuda_version"]),
        nccl_version=str(facts["nccl_version"]),
        model_name=model_name,
        precision=precision,
        context_length=context_length,
        gpu_count=int(facts["device_count"]),
        gpu_model=str(facts["gpu_model"]),
        gpu_memory_gb=int(facts["gpu_memory_gb"]),
        interconnect=interconnect,
        seed=seed,
        budget_ratio=budget_ratio,
        sync_async=sync_async,
        baseline=baseline,
        num_repr_queries=num_repr_queries,
        projection_dim=projection_dim,
        lambda_beta=lambda_beta,
        lambda_value=lambda_value,
        rope_extension_used=rope_extension_used,
        rope_extension_disclosed=rope_extension_disclosed,
        task=task,
        notes=notes,
    )


def gate_guard_for_report(meta: ExperimentMetadata) -> Dict[str, Any]:
    """论文 §6.2 的硬规则：用了 RoPE 外推却未披露的 run 不进主表。

    返回 {"admissible": bool, "reason": str}，供聚合阶段直接过滤。
    """
    used = meta.rope_extension_used not in (None, "", "none", "None", "false")
    if used and not meta.rope_extension_disclosed:
        return {
            "admissible": False,
            "reason": "rope_extension_used 为真但 rope_extension_disclosed 为假，按 §6.2 不进主表",
        }
    return {"admissible": True, "reason": ""}


# =============================================================================
# 分布式
# =============================================================================

def dist_init(rank: int, world_size: int, port: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )


def dist_destroy() -> None:
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass


def maybe_spawn(worker: Callable[..., None], nprocs: int, args: Tuple[Any, ...]) -> None:
    """入口分发：torchrun 下直接跑，否则单机 spawn。

    - 由 torchrun 启动：环境里有 RANK / LOCAL_RANK / WORLD_SIZE → 直接调用 worker
    - 单进程（nprocs == 1）：直接调用 worker(0, args)
    - 多进程但无 torchrun：用 torch.multiprocessing.spawn 兜底（仅限单机）
    """
    if os.environ.get("RANK") is not None:
        rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        worker(rank, world, *args)
        return
    if nprocs <= 1:
        worker(0, 1, *args)
        return
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    ctx.spawn(worker, args=(nprocs,) + args, nprocs=nprocs)


def barrier_and_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# =============================================================================
# 计量
# =============================================================================

def benchmark_ms(
    fn: Callable[[], Any],
    warmup: int = 3,
    iters: int = 10,
    sync: bool = True,
) -> List[float]:
    """逐次计时（毫秒），返回长度 iters 的列表。

    刻意不返回均值 —— 统计规范要求从分布出发（median + p5/p95 + bootstrap CI），
    把原始逐次值交回调用方，由 experiments.common.report.summarize 汇总。
    """
    for _ in range(max(0, warmup)):
        fn()
    if sync:
        barrier_and_sync()

    samples: List[float] = []
    for _ in range(iters):
        if sync:
            barrier_and_sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            barrier_and_sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def dtypes_for(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float64": torch.float64, "fp64": torch.float64,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"未知 dtype：{name}（可选：{sorted(set(table))}）")
    return table[key]


def memory_report() -> Dict[str, Any]:
    """显存占用快照，用于在报告里佐证"没爆显存"这一前提。"""
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "free_gb": round(free / (1 << 30), 2),
        "total_gb": round(total / (1 << 30), 2),
        "allocated_gb": round(torch.cuda.memory_allocated() / (1 << 30), 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1 << 30), 2),
    }


# =============================================================================
# 构造链路在 CUDA 上的可用性探测（当前已知为不可用）
# =============================================================================

CUDA_CONSTRUCTION_DEFECTS: List[Dict[str, str]] = [
    {
        "file": "src/dcc_kv_ref/representative_query.py",
        "line": "27, 30",
        "symptom": "torch.Generator() 与 torch.randint(...) 默认在 CPU 上创建投影矩阵，"
                   "随后 queries @ proj.T 触发 CUDA 与 CPU 张量的 device mismatch。",
        "fix": "g = torch.Generator(device=queries.device); "
               "proj = proj.to(device=queries.device, dtype=queries.dtype)",
    },
    {
        "file": "src/dcc_kv_ref/value_regression.py",
        "line": "32",
        "symptom": "torch.eye(B, dtype=X.dtype) 创建在 CPU 上，与 XTX（CUDA）相加时 device mismatch。",
        "fix": "torch.eye(B, dtype=X.dtype, device=X.device)",
    },
    {
        "file": "src/dcc_kv_ref/representative_query.py",
        "line": "57, 79",
        "symptom": "torch.arange(N) / torch.tensor(selected) 返回 CPU 索引张量。"
                   "CUDA 张量用 CPU 索引当前可用，但会在 --device cpu 与 cuda 混跑时引入隐式同步。",
        "fix": "显式 .to(features.device)",
    },
]


def probe_gpu_construction(device: str = "cuda") -> Dict[str, Any]:
    """在 GPU 张量上试跑 `build_compact_kv`，报告能否通过。

    这是 GPU 实验的**前置条件**：A1/A2/A5 需要把"逐边构造"的代价计入
    总延迟，A3 需要构造四个消融变体。若构造链路在 CUDA 上跑不通，
    这些实验只能退化为"CPU 构造 + 传输"，而那会改变延迟的归因
    （构造开销被 PCIe 传输掩盖或放大），必须在报告里显式声明。

    本函数只做探测，不修改任何源码。
    """
    if not torch.cuda.is_available():
        return {"ok": False, "skipped": True, "reason": "无可用 CUDA 设备"}
    try:
        from src.dcc_kv_ref import build_compact_kv
    except Exception as e:  # pragma: no cover
        return {"ok": False, "skipped": True, "reason": f"无法导入 build_compact_kv: {e}"}

    dev = torch.device(device)
    try:
        K = torch.randn(64, 16, device=dev)
        V = torch.randn(64, 16, device=dev)
        Q = torch.randn(24, 16, device=dev)
        ck = build_compact_kv(
            source_keys=K, source_values=V, destination_queries=Q,
            budget=8, num_representative_queries=4, projection_dim=8, seed=0,
        )
        return {
            "ok": True,
            "note": "构造链路在 CUDA 上可用",
            "device_of_logit_bias": str(ck.logit_bias.device),
        }
    except Exception as e:
        return {
            "ok": False,
            "skipped": False,
            "error_type": type(e).__name__,
            "error": str(e),
            "known_defects": CUDA_CONSTRUCTION_DEFECTS,
            "note": (
                "构造链路当前在 CUDA 张量上不可用（CPU-only 实现）。"
                "GPU 实验中需要构造的环节将退化为 CPU 构造 + H2D/D2H 传输，"
                "该退化会改变延迟归因，必须在结果 payload 的 caveat 中声明。"
            ),
        }


__all__ = [
    "GATE_EXIT_CODE",
    "Gate",
    "probe",
    "enforce",
    "print_env_only",
    "gpu_facts",
    "nccl_version",
    "build_metadata",
    "gate_guard_for_report",
    "dist_init",
    "dist_destroy",
    "maybe_spawn",
    "barrier_and_sync",
    "benchmark_ms",
    "human_bytes",
    "tensor_bytes",
    "dtypes_for",
    "memory_report",
    "CUDA_CONSTRUCTION_DEFECTS",
    "probe_gpu_construction",
    "REPO_ROOT",
]
