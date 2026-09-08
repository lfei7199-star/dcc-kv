"""实验元数据 + 结果 schema（blueprint §4.1 末尾要求）。

每次实验 run 必填的元数据。
"""
from __future__ import annotations

import os
import json
import subprocess
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from datetime import datetime


def get_git_commit() -> str:
    """获取当前 git commit hash（如果存在）。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_version(pkg: str) -> str:
    """获取 Python 包的版本。"""
    try:
        return __import__(pkg).__version__
    except (ImportError, AttributeError):
        return "unknown"


@dataclass
class ExperimentMetadata:
    """每次实验 run 必填的元数据。

    blueprint §4.1 末尾要求：模型版本、Transformers/PyTorch/CUDA/NCCL 驱动、
    权重 commit 与精度必须写入每次实验元数据。
    """
    # 标识
    run_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # 代码 / 软件
    git_commit: str = field(default_factory=get_git_commit)
    python_version: str = field(default_factory=lambda: get_version("sys").split()[0])
    pytorch_version: str = field(default_factory=lambda: get_version("torch"))
    transformers_version: str = field(default_factory=lambda: get_version("transformers"))
    cuda_version: str = "unknown"  # GPU 上才有
    nccl_version: str = "unknown"  # GPU 上才有
    flash_attn_version: str = field(default_factory=lambda: get_version("flash_attn"))

    # 模型
    model_name: str = "unknown"
    model_commit_or_path: str = "unknown"
    tokenizer_commit_or_path: str = "unknown"
    precision: str = "bfloat16"

    # 上下文 / GPU
    context_length: int = 0
    gpu_count: int = 0
    gpu_model: str = "unknown"
    gpu_memory_gb: int = 0
    interconnect: str = "unknown"

    # 实验配置
    seed: int = 42
    budget_ratio: float = 0.05
    sync_async: str = "async"
    baseline: str = "unknown"
    num_repr_queries: int = 64
    projection_dim: int = 32
    lambda_beta: float = 1e-3
    lambda_value: float = 1e-3
    rope_extension_used: Optional[str] = None
    rope_extension_disclosed: bool = False

    # 元
    task: str = "unknown"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


@dataclass
class RunResult:
    """每次 run 的结果（对齐 blueprint §4.2 的统计规范）。

    blueprint §4.2 要求：报告中位数、p5/p95 或 bootstrap 95% CI。
    """
    median: float
    p5: float
    p95: float
    ci_95_lower: float
    ci_95_upper: float
    raw_values: List[float] = field(default_factory=list)
    n_runs: int = 0
    metric_name: str = "unknown"
    unit: str = "unknown"  # "ms" / "GB" / "%" / "accuracy" / "ppl"

    @classmethod
    def from_values(
        cls,
        values: List[float],
        metric_name: str = "unknown",
        unit: str = "unknown",
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> "RunResult":
        """从原始值构造（计算中位数、分位、bootstrap CI）。"""
        if not values:
            return cls(0, 0, 0, 0, 0, [], 0, metric_name, unit)
        import numpy as np
        arr = np.array(values)
        median = float(np.median(arr))
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))

        # bootstrap 95% CI
        g = np.random.default_rng(seed)
        boot_medians = []
        for _ in range(n_bootstrap):
            sample = g.choice(arr, size=len(arr), replace=True)
            boot_medians.append(np.median(sample))
        ci_lower = float(np.percentile(boot_medians, 2.5))
        ci_upper = float(np.percentile(boot_medians, 97.5))

        return cls(
            median=median,
            p5=p5,
            p95=p95,
            ci_95_lower=ci_lower,
            ci_95_upper=ci_upper,
            raw_values=list(values),
            n_runs=len(values),
            metric_name=metric_name,
            unit=unit,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def __str__(self) -> str:
        return (
            f"{self.metric_name}: median={self.median:.4f} "
            f"[{self.ci_95_lower:.4f}, {self.ci_95_upper:.4f}] "
            f"(p5={self.p5:.4f}, p95={self.p95:.4f}, n={self.n_runs}, unit={self.unit})"
        )
