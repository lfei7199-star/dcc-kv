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

「质量相近」这一**前提**原先在仓库内外都没有定义。2026-09-16 已给出定义
（完整论证见论文 §7「「质量相近」的判定」，摘要见 `docs/reproducibility.md` §6）：
参照物取**精确注意力**而非共享压缩基线，判据为**单侧非劣检验**
``CI_low(Q_DCC - Q_dense) > -delta``，非劣边界满足
``噪声底线 <= delta < min(1.5, delta_bad)``。机器可读的判定程序是
:func:`quality_comparable_non_inferior`。

:func:`h2_pass` 仍把 ``quality_comparable`` 做成**必填的关键字参数** ——
定义解决的是「按什么程序判」，而 ``delta`` 与噪声底线两个**参数**依赖
尚未取得的测量（E6 的重复 run），在此之前任何默认值都是编造。

代码侧的落地状态
----------------
``HYPOTHESES[h].code_status`` 如实记录每个假设的判定有没有脚本产出：

- ``judged``   已经有脚本产出该字段
- ``no-judge`` 尚无

截至今日本仓库**H1 与 H4 是 ``judged``**：H1 由
``experiments/cpu/e3_edge_conditioning.py`` 的 ``h1_criterion_met`` 字段产出
（2026-09-16 起该脚本默认走留出协议），H4 由
``experiments/gpu/e5_gpu_ablation.py`` 的 A5 产出；H2 / H3 / H5 为
``no-judge``。这个分布由 :data:`UNJUDGED` 暴露出来，避免它被忘记。

（2026-09-15 更正：此前 H1 被登记为 ``no-judge``，note 称「E3 测的是配对显著
不同，不是与 0.5 比较」—— 该说法**与代码不符**。E3 的 ``h1_criterion_met``
恰恰就是拿 KL 的 CI 下界与 0.5 比较。这个错位由独立监督查出，见
`docs/commit_log.md` 第 22 条。）

``no-judge`` **不等于**「结论未定」—— 它表示**判据未接**。

H2 的跨长度聚合规则已于 2026-09-16 定下（见 :data:`H2_LENGTH_AGGREGATION`
与 :func:`h2_pass_across_lengths`）。判定链路已于 2026-09-21 **完整闭合**：
E6 把逐样本判对错与该样本的配对身份键随聚合准确率**同行落盘**，配对 95% CI
因此算得出来，:func:`quality_comparable_non_inferior` 有了可消费的输入。

判定仍为 ``no-judge``，但**唯一剩下的原因是参数取值**：非劣边界 ``delta``
与噪声底线按设计**无默认值**（有默认值等于假称前提永远成立），必须由实验者
显式声明。**这不是「结论未定」** —— 链路、口径、阈值都已就位，缺的是一次
带参数的运行，以及一次真实的重复 run 来把噪声底线测出来。

**规则为什么是「每个长度分别判定、全通过才算成立」而不是把 4 个长度池化**

池化（对 4 个长度求平均后与阈值比较）会让某个长度上的大额提升**掩盖**另一个
长度上的退化。H2 的表述是「质量提升 ≥ 1.5 pp」，它是一个可在外推区间上被
证伪的普遍主张；若它在 4 个长度里有 1 个不成立，「平均成立」并不等于该主张
成立。这与 E11 定超参数默认值时"必须报最差单格退化"是同一条纪律：
**先看最差的那一格，再看平均。**
"""
from __future__ import annotations

import math
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
    # H2 前提「质量相近」的判定（2026-09-16）
    "QUALITY_COMPARABLE_REFERENCE",
    "QUALITY_COMPARABLE_MAX_DELTA_PP",
    # H2 跨上下文长度的聚合规则（2026-09-16）
    "H2_LENGTH_AGGREGATION",
    "H2PerLength",
    "H2AcrossLengths",
    "h2_pass_across_lengths",
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
    "quality_comparable_non_inferior",
    # A4 压缩 × 异步交互的判据（2026-09-16，缺口 G4）
    "A4_CI_LEVEL",
    "A4_MIN_BUDGET_LEVELS",
    "A4Cell",
    "A4BudgetPoint",
    "A4Interaction",
    "a4_interaction_verdict",
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


QUALITY_COMPARABLE_REFERENCE = "dense"
"""H2 前提「质量相近」的参照物：**精确注意力**（未经压缩的全局注意力）。

不取共享压缩基线：那样「相近」与「相对基线提升 >= 1.5 pp」只有在非劣边界
大于 1.5 pp 时才可能同时成立，而那时前半句已不表达任何主张 —— 自我矛盾。
"""

QUALITY_COMPARABLE_MAX_DELTA_PP = H2_MIN_QUALITY_GAIN_PP
"""H2 前提的**先验上界**：非劣边界必须严格小于它（单位：百分点）。

理由：等价/非劣边界不得大于所要检测的效应量，否则前提会把 H2 前半句
声称的 1.5 pp 提升一并吞没。另一个上界 ``delta_bad``（共享压缩相对精确
注意力的退化量）需要数据，故不在此登记为常量。
"""


def quality_comparable_non_inferior(
    ci_low_pp: float,
    *,
    delta_pp: float,
    noise_floor_pp: float,
    delta_bad_pp: float | None = None,
) -> bool:
    """H2 前提「质量相近」的判定程序：**单侧非劣检验**。

    Args:
        ci_low_pp: ``Q_DCC - Q_dense`` 的 95% 置信区间**下界**，单位百分点。
        delta_pp: 非劣边界（允许比精确注意力差多少），单位百分点。**无默认值**。
        noise_floor_pp: 同一配置重复 run 的噪声底线（可用 run 间质量差的
            p95-p50 或 bootstrap 区间半宽估计），单位百分点。**无默认值**。
        delta_bad_pp: 共享压缩相对精确注意力的退化量。给出时才校验
            ``delta_pp < delta_bad_pp`` —— 否则共享压缩本身也会被判为「相近」。

    Returns:
        下界严格大于 ``-delta_pp`` 时为 True。

    Raises:
        ValueError: ``delta_pp`` 越出可行区间时。此时**拒绝执行**而不是返回
            False —— 容差取值不当意味着本次实验不具备判定该前提的分辨率，
            与「质量确实不相近」是两个不同的结论，前者必须报为分辨率不足。

    取单侧而非双侧等价的依据：该前提的**功能**是排除「以质量换速度」这一
    混淆，只要没有显著更差，加速比的比较就是公平的。若 ``ci_low_pp > 0``
    （显著更优），前提以更强的形式成立，但调用方必须在结果中**显式披露
    方向**，不得笼统写作「相近」。
    """
    if delta_pp <= 0:
        raise ValueError(f"delta_pp 必须为正，收到 {delta_pp}")
    if delta_pp < noise_floor_pp:
        raise ValueError(
            f"非劣边界 delta_pp={delta_pp} 小于噪声底线 noise_floor_pp={noise_floor_pp}"
            "：等价性无法建立（测不出差异不能与确实无差异区分），应报告「分辨率不足」"
        )
    if not delta_pp < QUALITY_COMPARABLE_MAX_DELTA_PP:
        raise ValueError(
            f"非劣边界 delta_pp={delta_pp} 未严格小于 QUALITY_COMPARABLE_MAX_DELTA_PP="
            f"{QUALITY_COMPARABLE_MAX_DELTA_PP}：边界不得大于所要检测的效应量，"
            "否则前提会把 H2 前半句声称的提升一并吞没"
        )
    if delta_bad_pp is not None and not delta_pp < delta_bad_pp:
        raise ValueError(
            f"非劣边界 delta_pp={delta_pp} 未严格小于共享压缩的退化量 "
            f"delta_bad_pp={delta_bad_pp}：否则共享压缩本身也落入「相近」"
        )
    return bool(ci_low_pp > -delta_pp)


# ---------------------------------------------------------------------------
# H2 跨上下文长度的聚合（2026-09-16）
# ---------------------------------------------------------------------------
H2_LENGTH_AGGREGATION = "all"
"""H2 在多个上下文长度上的聚合规则：**每个长度分别判定、全部通过才算成立**。

取值只有 ``"all"``。之所以不做成可配置的开关：另一条路（池化）会改变命题的
真值条件，那是**方法学决定**而不是运行参数 —— 把它做成参数只会让两种不同的
主张共用一个假设编号。真要改用池化，应当先改 H2 的表述本身。
"""


@dataclass(frozen=True)
class H2PerLength:
    """H2 在**单个**上下文长度上的三个原始量。

    Attributes:
        length: 上下文长度（token 数）。
        quality_gain_pp: 相对共享压缩的质量提升，单位百分点。
        prefill_speedup: prefill 加速比（1.10 表示快 10%）。
        quality_comparable: 该长度上是否满足「质量相近」前提
            （判定程序见 :func:`quality_comparable_non_inferior`）。
        prefill_instrument_valid: 该长度的 prefill 加速比是否由**能体现压缩收益**
            的量具测得。当前 E6 的 prefill 计时窗口里只有全长前向、裁剪后的 KV
            从未被使用（`_hf.PREFILL_TIMING_CONSUMES_COMPACT_KV`），因此这个量具
            下 ``prefill_speedup`` 结构上恒 <= 1 < 1.10x。为 False 时该长度记
            ``unresolved``（判不了），**不记 failed** —— 否则会把「量具测不出收益」
            报成「方法没有收益」。钩子接好后置 True，判定自动恢复。
    """

    length: int
    quality_gain_pp: float
    prefill_speedup: float
    quality_comparable: bool
    prefill_instrument_valid: bool = True


@dataclass(frozen=True)
class H2AcrossLengths:
    """H2 跨长度的聚合结果。

    Attributes:
        passed: 全部长度都通过 :func:`h2_pass` 时为 True。
        rule: 固化为 ``H2_LENGTH_AGGREGATION``，写进落盘产物以便复现判定口径。
        n_lengths / n_passed / n_failed: 计数。
        failing_lengths: 未通过的长度列表（**按原顺序**，便于直接定位）。
        worst_quality_gain_pp: 最差的一个长度的质量提升（忽略未定长度的 ``nan``）。
            **必须与 passed 同报** —— 只报均值会让"某个长度明显退化、其余长度
            大幅提升"读起来像全面胜利。
        worst_prefill_speedup: 最差的一个长度的 prefill 加速比。
        quality_comparable_all: 是否**每个**长度都满足「质量相近」前提。
        unresolved: 无法判定的长度（``quality_comparable`` 为 None 时）。这些既不算
            通过也不算失败 —— 记为「分辨率不足」而不是「未达标」。
    """

    passed: bool
    rule: str
    n_lengths: int
    n_passed: int
    n_failed: int
    failing_lengths: Tuple[int, ...]
    worst_quality_gain_pp: float
    worst_prefill_speedup: float
    quality_comparable_all: bool
    unresolved: Tuple[int, ...] = ()
    unresolved_reasons: Tuple[Tuple[int, str], ...] = ()

    def to_dict(self) -> Dict[str, object]:
        """摊平为可 JSON 落盘的字典（供 E6 结果结构使用）。"""
        return {
            "h2_passed": self.passed,
            "h2_aggregation_rule": self.rule,
            "h2_n_lengths": self.n_lengths,
            "h2_n_passed": self.n_passed,
            "h2_n_failed": self.n_failed,
            "h2_failing_lengths": list(self.failing_lengths),
            "h2_worst_quality_gain_pp": self.worst_quality_gain_pp,
            "h2_worst_prefill_speedup": self.worst_prefill_speedup,
            "h2_quality_comparable_all": self.quality_comparable_all,
            "h2_unresolved_lengths": list(self.unresolved),
            # 为什么判不了，逐长度写下来：'resolution'（非劣边界未定）
            # 与 'instrument'（量具测不出收益）是两件完全不同的事，
            # 只报 unresolved 会让人以为是数据不够。
            "h2_unresolved_reasons": {str(k): v
                                      for k, v in self.unresolved_reasons},
        }


def _finite_min(values):
    """忽略 non-finite 的最小值；全为 non-finite 时返回 ``nan``。

    为什么不能用内置 ``min``：未定长度的原始量被记为 ``nan``，而 ``nan`` 在
    比较中不可传递（``min(nan, 1.0)`` 的结果取决于顺序）。若直接 ``min``，
    「最差质量提升」这一栏会随机地变成 ``nan``，读者会以为数据坏了。
    最差值的用途是**防掩盖**（某个长度明显退化不能被其余长度的大幅提升掩盖），
    因此宁可省略不可比项，也不能让它变成无意义的 nan 而不作说明。
    """
    fin = [v for v in values if v is not None and math.isfinite(v)]
    return min(fin) if fin else float("nan")


def h2_pass_across_lengths(points) -> H2AcrossLengths:
    """把各上下文长度的 H2 原始量聚成一个判定（规则见 :data:`H2_LENGTH_AGGREGATION`）。

    Args:
        points: :class:`H2PerLength` 的可迭代对象。

    Returns:
        :class:`H2AcrossLengths`。

    Raises:
        ValueError: ``points`` 为空，或某个 ``quality_comparable`` 既不是
            ``bool`` 也不是 ``None``。

    ``quality_comparable`` 允许取 ``None``，表示该长度上**无法判定**该前提
    （例如重复 run 的噪声底线尚未取得 ⇒ 容差区间为空）。这种情况下该长度被
    记入 ``unresolved`` 而**不计入** ``n_failed``，且整体 ``passed`` 为 False ——
    「分辨率不足」与「确实没达标」是两个不同的结论，不能合并。

    同理，``prefill_instrument_valid`` 为 False 的长度也记 ``unresolved``（原因
    记为 ``instrument``）：量具本身体现不了压缩收益时，算出来的加速比无论多少
    都不构成对 H2 的证据。两种原因在 ``unresolved_reasons`` 里分开记录。
    """
    items = list(points)
    if not items:
        raise ValueError("points 为空：没有任何上下文长度可供判定")

    n_passed = 0
    failing: list = []
    unresolved: list = []
    reasons: list = []
    for pt in items:
        if not pt.prefill_instrument_valid:
            unresolved.append(pt.length)
            reasons.append((pt.length, "instrument"))
            continue
        if pt.quality_comparable is None:
            unresolved.append(pt.length)
            reasons.append((pt.length, "resolution"))
            continue
        if not isinstance(pt.quality_comparable, bool):
            raise ValueError(
                f"quality_comparable 必须是 bool 或 None，"
                f"length={pt.length} 收到 {pt.quality_comparable!r}"
            )
        if h2_pass(
            pt.quality_gain_pp,
            pt.prefill_speedup,
            quality_comparable=pt.quality_comparable,
        ):
            n_passed += 1
        else:
            failing.append(pt.length)

    n_failed = len(failing)
    return H2AcrossLengths(
        passed=(n_passed == len(items)),
        rule=H2_LENGTH_AGGREGATION,
        n_lengths=len(items),
        n_passed=n_passed,
        n_failed=n_failed,
        failing_lengths=tuple(failing),
        worst_quality_gain_pp=_finite_min(pt.quality_gain_pp for pt in items),
        worst_prefill_speedup=_finite_min(pt.prefill_speedup for pt in items),
        quality_comparable_all=all(
            pt.quality_comparable is True for pt in items
        ),
        unresolved=tuple(unresolved),
        unresolved_reasons=tuple(reasons),
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
            "2026-09-16 起 E3 默认走 --protocol heldout（缺口 M9），"
            "H1 的探针也改为留出集；in-sample 仅供复现旧数字。"
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
        judge="h2_pass_across_lengths",
        code_status="no-judge",
        note=(
            "跨长度的聚合规则已于 2026-09-16 定为"
            " `H2_LENGTH_AGGREGATION='all'`（每个长度分别判定、全通过才算成立），"
            "落点 h2_pass_across_lengths 已接线并在 E6 结果结构中产出字段。"
            "judge 指聚合入口；单长度原语仍是 h2_pass。"
            "「质量相近」的判定程序已给出（quality_comparable_non_inferior，"
            "论证见论文 §7、摘要见 reproducibility.md §6）。"
            "2026-09-21：**判定链路已完整闭合** —— 该检验要的 (Q_DCC − Q_dense) "
            "配对 95% CI 由 report.paired_bootstrap 计算，其输入（逐样本判对错 + "
            "配对身份键 eval_sample_keys）已由 E6 的 measure_point 与该行的聚合"
            "准确率**同行落盘**；配对前**先逐位比对键**，不同源即拒绝配对"
            "（记 unresolved），不产出一个「数值正常、实际无意义」的 CI。"
            "仍为 no-judge 的唯一原因只剩**参数取值**：--h2-delta-pp 与 "
            "--h2-noise-floor-pp 按设计**无默认值**，未声明时该长度记 unresolved，"
            "并在 payload 的 h2_comparable_diagnostics 里写明缺的是哪一个。"
            "缺口沿革：「定义缺失」→「聚合未定」→「参数待测」→「质量侧通路未独立」"
            "→「判定所需的逐样本数据未落盘」→ 此。"
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


# ---------------------------------------------------------------------------
# A4：压缩与异步的交互（**判据**，不属于 H1–H5）
# ---------------------------------------------------------------------------
#
# A4 不是假设，而是一个**引用前提的检验**：A2 在固定同步模式下扫预算、A5 在固定
# 预算下扫同步模式，两者各自把另一条轴边缘化。只有当两轴无显著交互时，它们的
# 结果才可以分开引用。判据写在论文 §6.3，本段是它的机器可读形式。
#
# 为什么放在本模块而不是脚本里：这与 H1–H5 是同一条纪律 —— **判定程序只能有
# 一份**。若把「区间是否重叠」写进 e5 脚本，日后改动（如换成重叠系数的阈值）
# 不会有人发现，且没有测试能锁住它。

A4_CI_LEVEL = 0.95
"""A4 判定所用的置信水平（论文 §6.3 原文写 95% 置信区间）。"""

A4_MIN_BUDGET_LEVELS = 4
"""A4 二维格的最少预算档数（论文 §6.3 原文写「至少 4 档 B/L_s × {同步, 异步}」）。"""


@dataclass(frozen=True)
class A4Cell:
    """二维格里的**一个格子**：某预算 × 某同步模式下的四项计时与 p50。

    四项计时必须齐全。论文 §6.3 特意强调「同步」臂的所指是式~(serial-time) 的
    纯串行实现（"等待全部通信完成后统一计算"），**不是**在设备间额外插一次屏障 ——
    因此本结构只承载测量值，不对同步语义做任何解释。

    异步格的 `t_comm_ms` 为 NaN，且 `timing_decomposition_valid=False`
    --------------------------------------------------------------
    「齐全」指四项都出现在结构里，**不**指四项都可用。异步臂的 `comm_ms`
    只剩发起开销（原因见 `_forward.ForwardResult.to_dict` 与
    `_comm.run_async_pipeline` 的同步纪律），拿它当通信时间会得出
    「异步消掉了通信」这种度量产物式结论。故置 NaN（**不是 0** —— 0 会被读成
    「通信为零」），原值保留在 `t_comm_ms_raw`。
    `t_comp_ms` 在异步臂是**上界**（吸收了未完成的下一块传输）。
    跨同步模式只有 `t_total_ms` 可比；要用拆解值请取同步格。
    """

    budget_ratio: float
    budget: int
    sync_mode: str              # "sync" | "async"
    t_build_ms: float
    t_comm_ms: float
    t_comp_ms: float
    t_total_ms: float
    p50_ms: float
    timing_decomposition_valid: bool = True
    t_comm_ms_raw: float = float("nan")
    timing_decomposition_note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "budget_ratio": self.budget_ratio,
            "budget": self.budget,
            "sync_mode": self.sync_mode,
            "t_build_ms": self.t_build_ms,
            "t_comm_ms": self.t_comm_ms,
            "t_comp_ms": self.t_comp_ms,
            "t_total_ms": self.t_total_ms,
            "p50_ms": self.p50_ms,
            "timing_decomposition_valid": self.timing_decomposition_valid,
            "t_comm_ms_raw": self.t_comm_ms_raw,
            "timing_decomposition_note": self.timing_decomposition_note,
        }


@dataclass(frozen=True)
class A4BudgetPoint:
    """某个预算档上的**边缘**量：加速比的点估计与 95% CI、以及式(speedup-bound) 的上界。

    加速比的 CI 由 `experiments.common.report.bootstrap_ratio_ci` 算出后传入 ——
    本模块只做判定，不做统计（统计口径统一在 report.py）。
    """

    budget_ratio: float
    budget: int
    p50_sync_ms: float
    p50_async_ms: float
    speedup: float
    speedup_ci_low: float
    speedup_ci_high: float
    t_comm_over_t_comp: float
    theoretical_bound: float
    t_build_share: float


@dataclass(frozen=True)
class A4Interaction:
    """A4 的判定结果。

    Attributes:
        resolved: 数据是否足以判定（预算档数够不够、配对是否齐全）。
            **False 表示"判不了"，不是"没有交互"** —— 这两者绝不能混。
        all_ci_overlap: 全部预算档的加速比 95% CI 两两重叠。
        interaction_observed: = not all_ci_overlap。
        reporting_requirement: 报告义务。
            ``"edge_results_may_be_cited_separately"`` —— 未观测到显著交互，
            A2/A5 的边缘结果可独立引用（**这是有效结论，不是失败**）；
            ``"must_report_2d_grid"`` —— 出现了可分辨结构，必须报二维格。
        prediction_i_*: 论文 §6.3 的两条可证伪预测之一：S_obs(B) 随 B 上升，
            并在 T_comm ≈ T_comp 的预算档附近接近上界。
        prediction_ii_*: 预测之二：S_obs(B) 与上界之差随 B 减小而扩大。
        monotone_fraction: 相邻预算档上单调一致的**比例**。刻意与布尔量同报：
            严格单调在测量数据上常常差一格，只报布尔量会把"部分成立"读成"被证伪"。
    """

    resolved: bool
    reason: str
    n_budgets: int
    budgets: Tuple[int, ...]
    all_ci_overlap: bool
    n_overlapping_pairs: int
    n_pairs: int
    interaction_observed: bool
    reporting_requirement: str
    monotone_speedup_in_budget: bool
    monotone_fraction: float
    prediction_i_holds: bool
    prediction_ii_holds: bool
    prediction_ii_monotone_fraction: float
    gap_to_bound_at_min_budget: float
    gap_to_bound_at_max_budget: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "a4_resolved": self.resolved,
            "a4_reason": self.reason,
            "a4_n_budgets": self.n_budgets,
            "a4_budgets": list(self.budgets),
            "a4_ci_level": A4_CI_LEVEL,
            "a4_min_budget_levels": A4_MIN_BUDGET_LEVELS,
            "a4_all_ci_overlap": self.all_ci_overlap,
            "a4_n_overlapping_pairs": self.n_overlapping_pairs,
            "a4_n_pairs": self.n_pairs,
            "a4_interaction_observed": self.interaction_observed,
            "a4_reporting_requirement": self.reporting_requirement,
            "a4_monotone_speedup_in_budget": self.monotone_speedup_in_budget,
            "a4_monotone_fraction": self.monotone_fraction,
            "a4_prediction_i_holds": self.prediction_i_holds,
            "a4_prediction_ii_holds": self.prediction_ii_holds,
            "a4_prediction_ii_monotone_fraction": self.prediction_ii_monotone_fraction,
            "a4_gap_to_bound_at_min_budget": self.gap_to_bound_at_min_budget,
            "a4_gap_to_bound_at_max_budget": self.gap_to_bound_at_max_budget,
            "a4_note": (
                "「无交互」是有效结论而非失败 —— 它恰好是允许 A2 与 A5 的边缘结果"
                "分开引用的前提。resolved=False 表示数据不足以判定，"
                "不是「未观测到交互」。"
            ),
        }


def _ci_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    """两个闭区间是否重叠。NaN 一律视为不重叠（信息缺失不能当成重叠）。"""
    for v in (lo1, hi1, lo2, hi2):
        if math.isnan(v):
            return False
    return (lo1 <= hi2) and (lo2 <= hi1)


def a4_interaction_verdict(points) -> A4Interaction:
    """A4 的判定程序（论文 §6.3）。

    主判据：对四档预算的**加速比 95% CI** 做两两比较。
    全部重叠 ⇒ 报告「未观测到显著交互」；出现可分辨结构 ⇒ 必须报二维格。

    判据的方向性（要写进论文，否则会被读成一次检验）
    ----------------------------------------------
    「CI 不重叠」**不等于**「差异显著」，「CI 重叠」也**不等于**「无差异」——
    区间重叠只是对显著性的一次粗略筛查。本判据取「**全部**两两重叠才算无交互」，
    因此倾向于**多报**交互：档数越多越难全部重叠。这是有意选的方向 ——
    假「有交互」的代价只是多报一张二维格，假「无交互」的代价是错称两条边缘
    结果可以分开引用。故本判据只能当**筛查**用，不能当结论性检验引用；
    真要下「无显著交互」的结论，须另做正式的交互效应检验。

    两条可证伪预测同时评估，且都**同时给出比例**：
        (i)  S_obs(B) 随 B 增大而上升，并在 T_comm ≈ T_comp 的预算档附近接近上界；
        (ii) S_obs(B) 与上界之差随 B 减小而扩大。

    Raises:
        ValueError: 预算档数为 0（无数据时不该走到这里；空输入是调用方的 bug）。
    """
    pts = list(points)
    if not pts:
        raise ValueError("a4_interaction_verdict 需要至少一个预算档；空输入属调用方错误")

    pts = sorted(pts, key=lambda p: p.budget)
    budgets = tuple(int(p.budget) for p in pts)

    if len(pts) < A4_MIN_BUDGET_LEVELS:
        return A4Interaction(
            resolved=False,
            reason=(f"预算档数 {len(pts)} < 要求的最少 {A4_MIN_BUDGET_LEVELS} 档；"
                    "数据不足以判定交互，这不等于「未观测到交互」"),
            n_budgets=len(pts), budgets=budgets,
            all_ci_overlap=False, n_overlapping_pairs=0, n_pairs=0,
            interaction_observed=False,
            reporting_requirement="insufficient_data",
            monotone_speedup_in_budget=False, monotone_fraction=float("nan"),
            prediction_i_holds=False, prediction_ii_holds=False,
            prediction_ii_monotone_fraction=float("nan"),
            gap_to_bound_at_min_budget=float("nan"),
            gap_to_bound_at_max_budget=float("nan"),
        )

    # --- 主判据：两两 CI 重叠 ---
    n_pairs = 0
    n_overlap = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            n_pairs += 1
            if _ci_overlap(pts[i].speedup_ci_low, pts[i].speedup_ci_high,
                           pts[j].speedup_ci_low, pts[j].speedup_ci_high):
                n_overlap += 1
    all_overlap = (n_pairs > 0 and n_overlap == n_pairs)

    # --- 预测 (i)：加速比随预算单调不减 ---
    diffs = [pts[k + 1].speedup - pts[k].speedup for k in range(len(pts) - 1)]
    n_adj = len(diffs)
    n_consistent = sum(1 for d in diffs if d >= 0.0)
    monotone = (n_adj > 0 and n_consistent == n_adj)
    frac = (n_consistent / n_adj) if n_adj else float("nan")

    # 「接近上界」：取 |log(T_comm/T_comp)| 最小的档，看它是不是也是 gap 最小的档
    def _balance(p: A4BudgetPoint) -> float:
        r = p.t_comm_over_t_comp
        if r <= 0 or math.isnan(r):
            return float("inf")
        return abs(math.log(r))

    def _gap(p: A4BudgetPoint) -> float:
        return p.theoretical_bound - p.speedup

    balanced = min(pts, key=_balance)
    closest = min(pts, key=_gap)
    prediction_i = bool(monotone and balanced.budget == closest.budget)

    # --- 预测 (ii)：gap 随预算减小而扩大 ---
    gaps = [_gap(p) for p in pts]
    gdiffs = [gaps[k + 1] - gaps[k] for k in range(len(gaps) - 1)]
    n_ii = len(gdiffs)
    n_ii_consistent = sum(1 for d in gdiffs if d <= 0.0)   # 预算增大 ⇒ gap 不增
    prediction_ii = bool(n_ii > 0 and n_ii_consistent == n_ii
                         and gaps[0] > gaps[-1])

    return A4Interaction(
        resolved=True,
        reason=("全部预算档的加速比 95% CI 两两重叠 ⇒ 未观测到显著交互"
                if all_overlap else
                "出现可分辨结构 ⇒ 必须报二维格，不得只报两条边缘曲线"),
        n_budgets=len(pts), budgets=budgets,
        all_ci_overlap=all_overlap,
        n_overlapping_pairs=n_overlap, n_pairs=n_pairs,
        interaction_observed=not all_overlap,
        reporting_requirement=("edge_results_may_be_cited_separately" if all_overlap
                               else "must_report_2d_grid"),
        monotone_speedup_in_budget=monotone,
        monotone_fraction=frac,
        prediction_i_holds=prediction_i,
        prediction_ii_holds=prediction_ii,
        prediction_ii_monotone_fraction=(n_ii_consistent / n_ii) if n_ii else float("nan"),
        gap_to_bound_at_min_budget=gaps[0],
        gap_to_bound_at_max_budget=gaps[-1],
    )
