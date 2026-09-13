"""结果聚合与落盘 —— 对齐 `docs/reproducibility.md` 的统计规范。

规范要求（§4.2）：所有报告数字必须来自多次 run 的分布，给出 median / p5 / p95
与 bootstrap 95% CI，而非单次运行值。这里复用仓库已有的
`src.experiment_metadata.RunResult`，不另起一套。

另外提供**配对** bootstrap —— 用于把 E3 从"两者各自有界"升级为
"DCC-KV 显著优于共享压缩"的配对比较。这是本文核心主张（H2）唯一
不依赖 GPU 的检验路径。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.experiment_metadata import RunResult


# =============================================================================
# 单组数值的汇总
# =============================================================================

def summarize(
    values: Sequence[float],
    metric_name: str,
    unit: str = "unknown",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> RunResult:
    """把一组重复测量的原始值汇总为 RunResult（median + p5/p95 + bootstrap CI）。"""
    return RunResult.from_values(
        [float(v) for v in values],
        metric_name=metric_name,
        unit=unit,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


# =============================================================================
# 配对比较（H2 的关键工具）
# =============================================================================

@dataclass
class PairedResult:
    """两个方法在同一批配对样本上的差异检验结果。

    配对的意义：对同一个（目的端, 预算）组合，DCC-KV 与共享压缩面对的是
    **同一份源块、同一组探针 Query**，因此误差差异是同源配对量，
    可以直接做配对 bootstrap，检验力远高于独立样本比较。
    """
    n_pairs: int
    mean_a: float                 # 方法 A（如 DCC-KV）的均值
    mean_b: float                 # 方法 B（如共享压缩）的均值
    mean_diff: float              # mean(A - B)，负值表示 A 更优（误差更小）
    ci_95_lower: float            # 配对差异的 bootstrap 95% CI
    ci_95_upper: float
    p_value_one_sided: float      # 单侧置换检验 p 值（H1: mean_diff < 0，即 A 更优）
    a_wins: int                   # A 严格更优的配对数
    b_wins: int
    ties: int
    metric_name: str
    unit: str

    def verdict(self, alpha: float = 0.05) -> str:
        """按预设显著性水平给出结论字符串（不做任何超出数据的断言）。"""
        if self.p_value_one_sided < alpha and self.mean_diff < 0:
            return f"A 显著更优（p={self.p_value_one_sided:.4f} < {alpha}）"
        if self.p_value_one_sided < alpha and self.mean_diff > 0:
            return f"B 显著更优（p={self.p_value_one_sided:.4f} < {alpha}）"
        return "差异不显著，不能拒绝 H0（按发布清单须转为负结果报告）"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    metric_name: str = "unknown",
    unit: str = "unknown",
    higher_is_better: bool = False,
    n_bootstrap: int = 10000,
    n_permutation: int = 10000,
    seed: int = 42,
) -> PairedResult:
    """配对差异检验：bootstrap 置信区间 + 置换检验 p 值。

    不依赖 scipy —— 只用 numpy，避免给"基础设备"引入额外依赖。

    Args:
        a, b: 等长的配对样本（a[i] 与 b[i] 必须来自同一实验条件）
        metric_name: 指标名
        unit: 单位
        higher_is_better: 该指标是否越大越好。
            误差类指标为 False（小者优），准确率类为 True。
        n_bootstrap: bootstrap 重采样次数
        n_permutation: 置换检验次数（用于 p 值）
        seed: 随机种子

    Returns:
        PairedResult

    Raises:
        ValueError: 两个序列长度不等或为空
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"配对样本长度必须相等，得到 {arr_a.shape} 与 {arr_b.shape}")
    if arr_a.size == 0:
        raise ValueError("配对样本为空")

    diff = arr_a - arr_b
    if higher_is_better:
        diff = -diff  # 统一成"越小越好"的方向

    n = diff.size
    rng = np.random.default_rng(seed)

    # --- bootstrap 95% CI（对配对差异重采样） ---
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diff[idx].mean(axis=1)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # --- 置换检验：H0 为"配对差异关于 0 对称"（即无效应）---
    # 在 H0 下每个配对差异的符号可交换，因此随机翻转符号得到零分布。
    # 注意 diff 已统一为"越小越好"的方向，故单侧备择假设恒为 mean_diff < 0，
    # p 值即 P(置换均值 <= 观测均值)。
    signs = rng.choice([-1.0, 1.0], size=(n_permutation, n))
    perm_means = (signs * diff).mean(axis=1)
    observed = float(diff.mean())
    p_one_sided = float((perm_means <= observed).mean())

    # --- 逐对胜负（方向分数，不含显著性判断） ---
    eps = 1e-12
    a_wins = int((diff < -eps).sum())
    b_wins = int((diff > eps).sum())
    ties = int(n - a_wins - b_wins)

    return PairedResult(
        n_pairs=n,
        mean_a=float(arr_a.mean()),
        mean_b=float(arr_b.mean()),
        mean_diff=float((arr_a - arr_b).mean()),
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        p_value_one_sided=p_one_sided,
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        metric_name=metric_name,
        unit=unit,
    )


# =============================================================================
# 落盘
# =============================================================================

def save_json(path: str, payload: Any) -> None:
    """写出 JSON（自动建目录，保留中文）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """写出 CSV。rows 为空时不写文件并返回。"""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_run_result(r: RunResult, indent: str = "  ") -> str:
    """把 RunResult 格式化为一行人类可读文本。"""
    return (
        f"{indent}{r.metric_name:34s} "
        f"median={r.median:.6g}  p5={r.p5:.6g}  p95={r.p95:.6g}  "
        f"CI95=[{r.ci_95_lower:.6g}, {r.ci_95_upper:.6g}]  n={r.n_runs}"
    )


__all__ = [
    "summarize",
    "PairedResult",
    "paired_bootstrap",
    "save_json",
    "save_csv",
    "format_run_result",
]
