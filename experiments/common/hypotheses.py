"""可检验假设 H1–H5 的阈值：**单一事实源**（blueprint v1.1 §3）。

为什么单独立一个模块
--------------------
在本模块出现之前，H1–H5 的阈值只活在文档里（`docs/reproducibility.md` §6、
`docs/release_checklist.md` §4），代码侧只剩 A5 里一处硬编码的 ``1.05``。
后果是「论文声称某假设成立」这条链中间**没有任何机器可读的判定**：
阈值一旦漂移不会有人发现，判定只能靠人工目测，也无法被测试锁住。

约定
----
- 本模块**只登记数字与出处，不解释语义**。语义有争议时以 blueprint 原件为准
  （该原件不在仓库内，见 `docs/commit_log.md` 的 C11）。
- 判定一律**按字面取边界**：原文写 ``>`` 就用严格大于，写 ``≥`` 就用闭区间。
  这个区别不是学究 —— H1 的原文是「KL > 0.5」，它在 0.5 处**不算达标**。
- 实验脚本要判定假设是否达标，一律调用本模块，不得再写魔法数字。

H2 的口径与它的未决点
---------------------
以 `docs/release_checklist.md` §4 为准（2026-09-15 定稿，两条**同时**满足）::

    质量提升 ≥ 1.5 pp   且   在质量相近前提下 prefill 加速 ≥ 1.10×

「质量相近」这一**前提**的判定主体与阈值，仓库内没有任何定义。因此
:func:`h2_pass` 把它做成**必填的关键字参数**，强迫调用方显式表态，
而不是替它猜一个默认值 —— 默认值会让这个缺口悄悄消失。

代码侧的落地状态
----------------
``HYPOTHESES[h].code_status`` 如实记录每个假设的判定有没有脚本产出：

- ``judged``   已经有脚本产出该字段
- ``no-judge`` 尚无

截至今日本仓库**H1 与 H4 是 ``judged``**：H1 由
``experiments/cpu/e3_edge_conditioning.py`` 的 ``h1_criterion_met`` 字段产出，
H4 由 ``experiments/gpu/e5_gpu_ablation.py`` 的 A5 产出；H2 / H3 / H5 为
``no-judge``。这个分布由 :data:`UNJUDGED` 暴露出来，避免它被忘记。

（2026-09-15 更正：此前 H1 被登记为 ``no-judge``，note 称「E3 测的是配对显著
不同，不是与 0.5 比较」—— 该说法**与代码不符**。E3 的 ``h1_criterion_met``
恰恰就是拿 KL 的 CI 下界与 0.5 比较。这个错位由独立监督查出，见
`docs/commit_log.md` 第 22 条。）

``no-judge`` **不等于**「结论未定」—— 它表示**判据未接**。E6 能产出 H2 所需的
两个原始量，但「按什么聚合 4 个上下文长度」属于方法学决定；未定之前本模块
不代调用方猜测，也不生成一个看似权威的布尔值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    # 阈值本体
    "H1_MIN_KL",
    "H2_MIN_QUALITY_GAIN_PP",
    "H2_MIN_PREFILL_SPEEDUP",
    "H3_MIN_DROP_BETA_PP",
    "H3_MIN_DROP_VALUE_PP",
    "H4_MIN_P50_SPEEDUP",
    "H5_MIN_SPEEDUP_AT_2X",
    # 注册表
    "HypothesisSpec",
    "HYPOTHESES",
    "UNJUDGED",
    # 判定函数
    "h1_pass",
    "h2_pass",
    "h3_pass",
    "h4_pass",
    "h5_pass",
]


# ---------------------------------------------------------------------------
# 阈值本体（数字只写在这里）
# ---------------------------------------------------------------------------
H1_MIN_KL = 0.5
"""H1：同一源块对不同目的端的紧凑 KV 的 KL 散度下限（**严格**大于）。"""

H2_MIN_QUALITY_GAIN_PP = 1.5
"""H2 质量侧：相对共享压缩的质量提升下限（百分点）。"""

H2_MIN_PREFILL_SPEEDUP = 1.10
"""H2 性能侧：质量相近前提下的 prefill 加速比下限。"""

H3_MIN_DROP_BETA_PP = 0.5
"""H3：移除 β 后允许的最大质量退化（百分点，即"至少退化这么多"）。"""

H3_MIN_DROP_VALUE_PP = 1.0
"""H3：移除 V 回归后允许的最大质量退化（百分点）。"""

H4_MIN_P50_SPEEDUP = 1.05
"""H4：异步 vs 同步的 p50 加速比下限。"""

H5_MIN_SPEEDUP_AT_2X = 1.5
"""H5：设备数翻倍后的加速比下限。"""


# ---------------------------------------------------------------------------
# 判定函数
# ---------------------------------------------------------------------------
# 每个函数只做一件事：把原始量按**字面边界**与阈值比较。不做单位换算、
# 不做插值、不做聚合 —— 那些属于调用方的口径，不是阈值的一部分。

def h1_pass(kl_divergence: float) -> bool:
    """H1：同一源块对不同目的端的紧凑 KV **显著不同**。

    原文写的是「KL 散度 > 0.5」，故此处为**严格大于**：
    ``h1_pass(0.5) is False``。
    """
    return bool(kl_divergence > H1_MIN_KL)


def h2_pass(
    quality_gain_pp: float,
    prefill_speedup: float,
    *,
    quality_comparable: bool,
) -> bool:
    """H2：质量提升 ≥ 1.5 pp **且**（质量相近前提下）prefill 加速 ≥ 1.10×。

    Args:
        quality_gain_pp: 相对共享压缩的质量提升，单位百分点。
        prefill_speedup: prefill 加速比（如 1.10 表示快 10%）。
        quality_comparable: **必填**。调用方要判定"本次比较是否满足
            「质量相近」这一前提"。仓库内没有给出该前提的判定主体与阈值
            （见模块 docstring），所以这里不设默认值 —— 默认值会掩盖缺口。

    Returns:
        三条同时成立才是 True。
    """
    return bool(
        quality_comparable
        and quality_gain_pp >= H2_MIN_QUALITY_GAIN_PP
        and prefill_speedup >= H2_MIN_PREFILL_SPEEDUP
    )


def h3_pass(drop_without_beta_pp: float, drop_without_value_pp: float) -> bool:
    """H3：移除 β 后质量退化 ≥ 0.5 点；移除 V 回归后 ≥ 1.0 点。

    两个组件**各自**都要达到对应下限（缺一不算）—— 原文用分号并列两组阈值，
    语义是"两个组件都不可缺"。
    """
    return bool(
        drop_without_beta_pp >= H3_MIN_DROP_BETA_PP
        and drop_without_value_pp >= H3_MIN_DROP_VALUE_PP
    )


def h4_pass(p50_speedup: float) -> bool:
    """H4：异步 vs 同步的 p50 加速 ≥ 1.05×。"""
    return bool(p50_speedup >= H4_MIN_P50_SPEEDUP)


def h5_pass(speedup_at_2x_devices: float) -> bool:
    """H5：设备数翻倍后的加速 ≥ 1.5×。"""
    return bool(speedup_at_2x_devices >= H5_MIN_SPEEDUP_AT_2X)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HypothesisSpec:
    """一个可检验假设的登记项。

    Attributes:
        hid: 假设编号（"H1"…"H5"）。
        statement: **原文照抄**的表述，不做改写。
        thresholds: 机器可读的阈值，键名与判定函数的入参对应。
        boundary: "closed"（≥）或 "strict"（>）。与判定函数的行为必须一致。
        source: 出处——仓库内的哪份文档承接了它。
        judge: 判定函数名，必须能在本模块里找到。
        code_status: "judged"（已有脚本产出该字段）/ "no-judge"（尚无）。
        note: 需要提醒调用方的补充信息，可为空。
    """

    hid: str
    statement: str
    thresholds: Dict[str, float]
    boundary: str
    source: str
    judge: str
    code_status: str
    note: str = ""


_SNAPSHOT = "docs/reproducibility.md §6（blueprint v1.1 §3 的下游快照）"

HYPOTHESES: Dict[str, HypothesisSpec] = {
    "H1": HypothesisSpec(
        hid="H1",
        statement="同一源块对不同目的端的紧凑 KV 显著不同（KL 散度 > 0.5）",
        thresholds={"min_kl_divergence": H1_MIN_KL},
        boundary="strict",
        source=_SNAPSHOT,
        judge="h1_pass",
        code_status="judged",
        note=(
            "E3 的 `h1_criterion_met` 用 KL 的 CI 下界与 H1_MIN_KL 比较（严格大于），"
            "已在 e3_edge_conditioning.py 中改为调用 h1_pass，不再写裸 0.5。"
            "但该判据建在 E3 的 H1 网格上，与 H2 一样尚未按留出集重跑。"
        ),
    ),
    "H2": HypothesisSpec(
        hid="H2",
        statement=(
            "DCC-KV 相对共享压缩，质量提升 ≥ 1.5 pp，且在质量相近前提下 "
            "prefill 加速 ≥ 1.10×"
        ),
        thresholds={
            "min_quality_gain_pp": H2_MIN_QUALITY_GAIN_PP,
            "min_prefill_speedup": H2_MIN_PREFILL_SPEEDUP,
        },
        boundary="closed",
        source="docs/release_checklist.md §4（2026-09-15 定稿为逻辑与）",
        judge="h2_pass",
        code_status="no-judge",
        note=(
            "两个原始量由 E6 产出，但「按什么聚合 4 个上下文长度」未定，"
            "故判据未接；「质量相近」这一前提的判定主体与阈值也未定义。"
        ),
    ),
    "H3": HypothesisSpec(
        hid="H3",
        statement="移除 β 后质量退化 ≥ 0.5 点；移除 V 回归后 ≥ 1.0 点",
        thresholds={
            "min_drop_beta_pp": H3_MIN_DROP_BETA_PP,
            "min_drop_value_pp": H3_MIN_DROP_VALUE_PP,
        },
        boundary="closed",
        source=_SNAPSHOT,
        judge="h3_pass",
        code_status="no-judge",
        note="A3 是它的落点，但 A3 因缺 CompactKV → GPU attention kernel 而硬阻断。",
    ),
    "H4": HypothesisSpec(
        hid="H4",
        statement="异步 vs 同步 p50 加速 ≥ 1.05×",
        thresholds={"min_p50_speedup": H4_MIN_P50_SPEEDUP},
        boundary="closed",
        source=_SNAPSHOT,
        judge="h4_pass",
        code_status="judged",
        note="experiments/gpu/e5_gpu_ablation.py 的 A5 产出 h4_pass 字段。",
    ),
    "H5": HypothesisSpec(
        hid="H5",
        statement="设备数翻倍，加速 ≥ 1.5×",
        thresholds={"min_speedup_at_2x": H5_MIN_SPEEDUP_AT_2X},
        boundary="closed",
        source=_SNAPSHOT,
        judge="h5_pass",
        code_status="no-judge",
        note="E6 可扫 gpu_count，但「翻倍」的基准点与聚合方式尚未定义。",
    ),
}

UNJUDGED: Tuple[str, ...] = tuple(
    hid for hid, spec in HYPOTHESES.items() if spec.code_status != "judged"
)
"""尚未在代码侧接上判据的假设编号（顺序同 :data:`HYPOTHESES` 的定义序）。"""
