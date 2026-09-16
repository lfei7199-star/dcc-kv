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
与 :func:`h2_pass_across_lengths`），因此 H2 的缺口再降一级：从「按什么聚合
未定」降为「**参数待测**」（非劣边界 ``delta`` 与噪声底线依赖 E6 的重复 run）。
判定仍为 ``no-judge`` —— 规则接好了，但 E6 还没产出可供它消费的数字。

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
    """

    length: int
    quality_gain_pp: float
    prefill_speedup: float
    quality_comparable: bool


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
    """
    items = list(points)
    if not items:
        raise ValueError("points 为空：没有任何上下文长度可供判定")

    n_passed = 0
    failing: list = []
    unresolved: list = []
    for pt in items:
        if pt.quality_comparable is None:
            unresolved.append(pt.length)
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
            "仍为 no-judge 的唯一原因：非劣边界 delta 与噪声底线依赖 E6 的重复 run，"
            "**参数待测** —— 缺口已由「定义缺失」→「聚合未定」→ 此。"
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
