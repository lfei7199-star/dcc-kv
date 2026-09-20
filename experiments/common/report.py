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
import math
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
    mean_diff: float              # mean(A - B)，**原始方向**，仅供报告
    ci_95_lower: float            # 配对差异的 bootstrap 95% CI（原始方向）
    ci_95_upper: float
    p_value_one_sided: float      # 单侧置换检验 p 值
    diff_unified: float           # mean(A - B) 已按"越小越好"统一方向后的值
    higher_is_better: bool        # 该指标的原始方向（True = 大者优，如准确率）
    a_wins: int                   # A 严格更优的配对数
    b_wins: int
    ties: int
    metric_name: str
    unit: str

    def verdict(self, alpha: float = 0.05) -> str:
        """按预设显著性水平给出结论字符串（不做任何超出数据的断言）。

        ⚠️ 方向必须用 `diff_unified`（已统一成"越小越好"），**不能**用
        `mean_diff`。自查发现（2026-09-18）：p 值是在统一方向的 `diff` 上算的，
        而本函数原先拿**原始方向**的 `mean_diff` 判方向 —— `higher_is_better=True`
        时（准确率类指标）两者符号相反。H2 的质量比较正是准确率（大者优），
        恰好落在这条被写反的路径上。

        ⚠️ 本检验是**单侧**的，方向固定为"越小越好"（H1/H2 问的都是
        「A(DCC-KV) 是否更优」）。因此：
        - `diff_unified < 0` 时 p 值才有判别力（小 ⇒ A 显著更优）；
        -           反过来（B 更优）**不能**由这个 p 值宣称：零分布是围绕 0 的符号翻转
          分布，观测到 `diff_unified > 0` 时 p 恒接近 1（实测：20 个配对差全为
          +0.4 时 p = 1.0000），于是 `p < alpha` 与 `diff_unified > 0` 事实上
          不会同时成立。所以本函数**没有**"B 显著更优"这一档 ——
          原先那一档是不可达的死分支，而且措辞会让人以为"我们做了双向检验"。
          现在改为：走到不显著那一支时，若观测方向指向 B，就**明说方向**并声明
          本检验不判别它（要判别须另做反向检验）。
        """
        if self.p_value_one_sided < alpha:
            return (f"A 显著更优（单侧 p={self.p_value_one_sided:.4f} < {alpha}）")
        if self.diff_unified > 0:
            return ("未达显著：观测方向为 B 更优，但本检验是**单侧**的、备择假设"
                    "固定为 A 更优（diff_unified < 0），故该 p 值不能用来宣称 B 更优；"
                    "要宣称 B 更优须另做反向检验。（不显著的严格含义仍是"
                    "「不能拒绝 H0」。）")
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
        # 判定方向一律走这个量：它已经和 p 值同一方向（越小越好）
        diff_unified=observed,
        higher_is_better=bool(higher_is_better),
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        metric_name=metric_name,
        unit=unit,
    )



# =============================================================================
# 比值的不确定度（A4 的加速比 CI）
# =============================================================================

@dataclass
class RatioCI:
    """一个比值（如加速比 t_sync / t_async）的点估计与 bootstrap 置信区间。

    为什么不复用 `paired_bootstrap`：它检验的是**差值**的方向与显著性
    （H0: mean(a-b)=0）。A4 要的是**比值**的区间，且判定方式是区间重叠，
    两者不可互换 —— 差值 CI 不包含 0 与比值 CI 不包含 1 在数学上等价，
    但「区间是否重叠」这个判据只在比值口径下有直接意义
    （它对应「两个预算档的加速比是否可分辨」）。
    """

    point: float
    ci_low: float
    ci_high: float
    n_numerator: int
    n_denominator: int
    ci_level: float
    n_bootstrap: int
    seed: int
    paired: bool
    method: str
    metric_name: str
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bootstrap_ratio_ci(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    metric_name: str = "ratio",
    unit: str = "x",
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
    paired: bool = True,
    require_min_runs: int = 2,
) -> RatioCI:
    """比值的 percentile bootstrap 置信区间。

    Args:
        numerator: 分子样本（如 T_sync 的重复测量）。
        denominator: 分母样本（如 T_async 的重复测量）。必须恒正。
        paired: True 表示两组样本**配对**（同一轮 run 内的 sync/async 各测一次），
            重采样时按**同一索引**同时取两边的值 —— 配对能消掉轮次间的
            公共漂移（GPU 时钟、机器负载），区间更窄且更可信。
            False 则两边独立重采样（仅当两组来自不同 run 时使用）。
        require_min_runs: 少于该样本数直接报错（默认 2：1 个样本的 CI 宽度恒为
            0 却看起来像"很确定"，是典型的假精确）。

    Raises:
        ValueError: 长度不匹配 / 样本不足 / 含非有限值 / 分母出现非正值。
            **这里一律抛错而不返回 NaN**：A4 的区间重叠判定把 NaN 当作
            "不重叠"，于是 NaN 会被读成"有交互、必须报二维格" ——
            一个静默的统计失败会伪装成一条更强的结论。宁可在这里炸掉。

    Returns:
        RatioCI（point = mean(numerator)/mean(denominator)，percentile 区间）。
    """
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("numerator/denominator 必须是一维序列")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level 必须落在 (0,1)，得到 {ci_level}")
    if paired and a.shape != b.shape:
        raise ValueError(
            f"paired=True 要求两组等长（同一轮 run 的配对测量），得到 {a.shape} 与 {b.shape}"
        )
    if a.size < require_min_runs or b.size < require_min_runs:
        raise ValueError(
            f"样本数不足（要求 >= {require_min_runs}）：分子 {a.size}、分母 {b.size}"
        )
    for name, arr in (("numerator", a), ("denominator", b)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} 含非有限值（NaN/inf），无法给出区间")
    if not np.all(b > 0.0):
        raise ValueError(
            "denominator 必须恒正（本仓库的分母是耗时，非正值说明测量本身有问题）"
        )

    n_a, n_b = int(a.size), int(b.size)
    point = float(a.mean() / b.mean())
    rng = np.random.default_rng(seed)

    if paired:
        n = n_a
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        ratios = a[idx].mean(axis=1) / b[idx].mean(axis=1)
        method = "paired_percentile_bootstrap"
    else:
        ia = rng.integers(0, n_a, size=(n_bootstrap, n_a))
        ib = rng.integers(0, n_b, size=(n_bootstrap, n_b))
        ratios = a[ia].mean(axis=1) / b[ib].mean(axis=1)
        method = "independent_percentile_bootstrap"

    lo_q = (1.0 - ci_level) / 2.0 * 100.0
    hi_q = (1.0 + ci_level) / 2.0 * 100.0
    return RatioCI(
        point=point,
        ci_low=float(np.percentile(ratios, lo_q)),
        ci_high=float(np.percentile(ratios, hi_q)),
        n_numerator=n_a,
        n_denominator=n_b,
        ci_level=float(ci_level),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        paired=bool(paired),
        method=method,
        metric_name=metric_name,
        unit=unit,
    )

# =============================================================================
# 落盘
# =============================================================================

def json_safe(obj: Any) -> Any:
    """把非有限浮点递归换成 ``None``，使产物成为**合法 JSON**。

    为什么必须做：`json.dump` 默认 `allow_nan=True`，会把 `nan`/`inf` 写成裸
    `NaN` / `Infinity`。RFC 8259 里没有这三个字面量 —— JS 的 `JSON.parse`、
    Go 的 `encoding/json`、`jq` 的默认模式都会直接报错，而那正是别人拿到
    `results/*.json` 时最先做的事。实测（2026-09-20）：`results/` 下已有 2 个
    产物因此不是合法 JSON。

    换 `None` 而不是字符串 `"nan"`：本仓库已统一用 `None` 表示「未测得/未定」
    （见 E6 的 `accuracy`），字符串会被下游误当成一个合法取值。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_json(path: str, payload: Any) -> None:
    """写出 JSON（自动建目录，保留中文，非有限浮点转 None）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """写出 CSV。rows 为空时不写文件并返回。

    表头取**全体行的键的并集**（按首次出现顺序），而不是 `rows[0]` 的键。
    自查发现（2026-09-18）：`csv.DictWriter` 默认 `extrasaction="raise"`，
    只要某一行的键不在表头里就抛 `ValueError`。而 E6 主表把**三类键集不同的行**
    混在同一个 rows 里（measured 行有 `gpu_count_note`/`n_compaction_mode` 等、
    blocked 行有 `blockers`、error 行有 `error_type`）⇒ 这一行代码会让
    **整轮 GPU 测量跑完之后**在落盘这一步失败，租卡时间白花。

    取并集同时避免另一种更隐蔽的错：若改用 `extrasaction="ignore"`，
    blocked 行的 `blockers` 与 error 行的 `error_type` 会被**静默丢列**，
    读者会以为「从来没有这些字段」，而不是「这一行没有值」。
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
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
    "json_safe",
    "summarize",
    "PairedResult",
    "paired_bootstrap",
    "RatioCI",
    "bootstrap_ratio_ci",
    "save_json",
    "save_csv",
    "format_run_result",
]
