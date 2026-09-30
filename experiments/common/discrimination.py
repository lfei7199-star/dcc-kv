"""评测集「本条对照可否用于比较」的判据：**单一事实源**（2026-09-30 用户裁定）。

为什么单独立一个模块，而不是并进 `hypotheses.py`
------------------------------------------------
`hypotheses.py` 登记的是**关于系统本身的假设**（H1–H5 阈值、A4 交互判据）——
它回答「我的方法成立了没有」。

本模块登记的是**关于量具的问题**——它回答「这支评测集有没有资格给上面那个问题
作证」。两者是不同层级：量具不合格时，假设的判定结果不是「不成立」，而是
**无从谈起**。把这两类阈值塞进同一个模块，会让「方法失败」与「仪器无效」
这两种完全不同的结论共用一个编号空间。

`docs/writing_scope_and_metrics.md` §5 第 4 条原写「阈值一律走单一事实源
（`hypotheses.py`）」。该条的**目的**是「不得在脚本里写魔法数字」，本模块
以同样方式满足该目的；§5 已同步修订为「判据模块」，并点名两个模块的分工。
**这是一次显式修订，不是对既有硬约束的绕过。**

判据本体（用户 2026-09-30 裁定，逐字）
--------------------------------------
    不一致对 > 0   且   （方向不对称 或 参照臂两端都有余量）

三项术语的机器可读定义：

- **不一致对** ``b + c``：``b`` = 参照臂错、对照臂对；``c`` = 参照臂对、对照臂错。
  ``b + c == 0`` 时本条对照与随机噪声不可区分，``p`` **无定义**。
- **方向不对称**：精确 McNemar 双尾 ``p < DISCRIMINATION_ALPHA``。
  注意它比「``b`` 或 ``c`` 为 0」更严 —— ``b=0, c=5`` 的 ``p = 0.0625`` **不算**
  方向可分辨（这正是 n=8 那轮的处境：方向看着一致，但功效不够，不得当结论）。
- **参照臂两端都有余量**：``REF_HEADROOM_MIN_ACC <= ref_acc <= 1 - REF_HEADROOM_MIN_ACC``。
  参照臂若有 20 条全对（``ref_acc = 1.0``），``b`` 必然恒为 0，此时「方向不对称」
  是**自动成立**的（假绿灯），故必须另有一条判据兜住端点情形。

判据之外还必须看 `grade`
------------------------
``passes_discrimination_gate`` 是**必要条件**，不是许可证。同一份 ``b=3, c=3``
在「参照臂 0.30」处通过判据，却依然**下不了任何结论**：改进与退化各 3 条、
方向不可分辨。故本模块同时给出 :attr:`DiscriminationVerdict.grade` 与
:attr:`DiscriminationVerdict.claim_licensed`，把「过了门槛」与「能说话」分开。

分辨率（``resolution_halfwidth_pp``）为什么要显式声明
----------------------------------------------------
「未检出差异」是一句**有内容**的话，它默认了一个分辨率：能排除的效应量得够大。
同一个 ``b=3, c=3``、``n=20``，分辨率读数约 24 pp —— 它容得下一个 20 pp 的真实
退化，所以「未检出」在 1.5 pp 尺度上是**空话**。

⚠️ 该读数**不是可用于推断的置信区间**：推断一律以 `p_value` 为准。
构造理由与实测反例见 :func:`_resolution_halfwidth_pp`。

因此:func:`discrimination_verdict` 把 ``required_resolution_pp`` 做成**必填、
无默认值**的关键字参数，理由与 `hypotheses.quality_comparable_non_inferior`
的 ``delta_pp`` 相同：**有默认值等于替调用方假称某个分辨率够用**，
而这件事只有知道要断言什么的人才能定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from statistics import NormalDist
from typing import Dict, Literal, Mapping, Sequence, Tuple

__all__ = [
    # 阈值本体
    "DISCRIMINATION_ALPHA",
    "DISCRIMINATION_CI_LEVEL",
    "REF_HEADROOM_MIN_ACC",
    "GRADES",
    "CLAIM_LICENSING_GRADES",
    # 判定
    "DiscriminationVerdict",
    "mcnemar_exact_p",
    "discrimination_verdict",
    "discrimination_verdict_from_arms",
]


# ---------------------------------------------------------------------------
# 阈值本体（数字只写在这里）
# ---------------------------------------------------------------------------
DISCRIMINATION_ALPHA = 0.05
"""方向不对称的显著性水平（精确 McNemar 双尾，**严格**小于）。

取 0.05 是惯例而非本项目的自拟量；它在此的作用只有一个：
「方向可分辨」需要一个事先说死的门槛，否则事后挑一个能过线的门槛就没有约束力。
"""

DISCRIMINATION_CI_LEVEL = 0.95
"""报告**分辨率读数**（`resolution_halfwidth_pp`）所用的置信水平。

⚠️ 它不产生可用于推断的置信区间（推断以 `p_value` 为准），只用来定分辨率读数的宽度。
与 `hypotheses.A4_CI_LEVEL` **数值相同但语义不同**（那是 A4 交互的判定水平）：
两者各自登记、各自引用；将来若要统一，应写成显式引用，不得靠「反正都是 0.95」口头对齐。
"""

REF_HEADROOM_MIN_ACC = 0.10
"""参照臂「两端都有余量」的下限（对称地，上限为 ``1 - 本值``）。

**这是一条声明性约定，不是推导出来的量。** 理由：``ref_acc`` 若为 1.0（或 0.0），
``b``（或 ``c``）在数学上必然恒为 0 ⇒「方向不对称」自动成立，判据会给出假绿灯。
约定 0.10 的含义是「参照臂距两端都还留有 10% 的准确率余量，两个方向都动得了」。

改动本值必须同步 `docs/writing_scope_and_metrics.md` §5，并重跑锚点测试 ——
它直接改变哪些历史对照会被判为「不能用于比较」。
"""

GRADES: Tuple[str, ...] = (
    "blind",
    "directional",
    "not_detected",
    "underresolved",
    "inconclusive",
)
"""全部判定档位。语义见 :attr:`DiscriminationVerdict.licence`。"""

CLAIM_LICENSING_GRADES: Tuple[str, ...] = ("directional", "not_detected")
"""真正**允许下结论**的档位（``passes_discrimination_gate`` 只是必要条件）。"""


# ---------------------------------------------------------------------------
# 统计量
# ---------------------------------------------------------------------------
def mcnemar_exact_p(b: int, c: int) -> float:
    """精确 McNemar 双尾 p 值（二项精确检验，无卡方近似）。

    Args:
        b: 参照臂错、对照臂对的不一致对数（非负整数）。
        c: 参照臂对、对照臂错的不一致对数（非负整数）。

    Returns:
        双尾 p 值，上限截断为 1.0。

    Raises:
        ValueError: ``b + c == 0`` 时。此时不一致对为空集，**检验不适用**；
            返回 1.0 会让「未定义」冒充「不显著」，故拒绝执行。

    与 `chi2` 近似的区别在小样本上是决定性的：``b=0, c=5`` 时精确双尾
    ``p = 0.0625``（卡方近似会给 0.025，把不显著读成显著）。
    """
    if b < 0 or c < 0:
        raise ValueError(f"不一致对数不得为负：b={b}, c={c}")
    n = b + c
    if n == 0:
        raise ValueError("不一致对为 0：精确 McNemar 检验无定义，不得当作 p=1")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail * 0.5 ** n)


def _resolution_halfwidth_pp(b: int, c: int, n_items: int) -> float:
    """本条对照的**分辨率读数**（百分点）：Wald 半宽，按零假设方差算。

    ``z * sqrt(b + c) / n``。它的用途**只有一个**：给「未检出差异」一句话
    一个可比较的量级——即「这条对照能把多大的效应排除掉」。

    ⚠️ **它不是可用于推断的置信区间。** 两条理由：

    1. 它按**零假设方差**取宽（配对差在 ``H0`` 下的方差正是 ``(b+c)/n**2``），
       而区间的构造应当用**无约束**方差；
    2. 小样本下 Wald 型区间本身不可靠。实测一个反例：``b=0, c=4, n=20`` 时
       半宽 ``19.6 pp < |Δ| = 20 pp``（看着显著），而**精确 McNemar p = 0.125**
       （不显著）。两者矛盾 —— **推断一律以 `p_value` 为准**。

    保留这个量是因为它做**分辨率闸门**是合适的（它就该表示「宽度」而不是「结论」）。
    需要真区间时，得另用与精确检验同源的 Clopper–Pearson / 精确 McNemar 区间，
    本模块**不提供**。
    """
    if n_items <= 0:
        raise ValueError(f"n_items 必须为正，收到 {n_items}")
    z = NormalDist().inv_cdf(1.0 - (1.0 - DISCRIMINATION_CI_LEVEL) / 2.0)
    return z * math.sqrt(b + c) / n_items * 100.0


def _as_binary(values: Sequence[object], name: str) -> Tuple[int, ...]:
    out = []
    for i, v in enumerate(values):
        if isinstance(v, bool):
            out.append(int(v))
        elif isinstance(v, int) and v in (0, 1):
            out.append(v)
        else:
            raise ValueError(
                f"{name}[{i}] = {v!r} 不是 0/1 判对错值；"
                "逐样本结果必须是 0/1（None/漏答不得静默当错）"
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DiscriminationVerdict:
    """单条对照（参照臂 vs 一个对照臂）的量具判定结果。

    字段分三层：**原始统计量** → **三条判据分量** → **最终结论与许可证**。
    分层是为了让调用方永远看得见「结论是从哪个分量来的」，
    而不是只拿到一个布尔值。
    """

    # --- 原始统计量 ---
    n_items: int
    b: int
    c: int
    n_discordant: int
    ref_acc: float
    arm_acc: float
    delta_pp: float
    p_value: float | None
    resolution_halfwidth_pp: float

    # --- 判据分量 ---
    required_resolution_pp: float
    alpha: float
    headroom_min_acc: float
    not_blind: bool
    directional: bool
    ref_headroom: bool
    resolution_ok: bool

    # --- 结论 ---
    passes_discrimination_gate: bool
    claim_licensed: bool
    grade: str
    licence: str

    def as_dict(self) -> Dict[str, object]:
        """转成 JSON 可序列化的普通 dict（供落盘/报告复用）。"""
        return asdict(self)


def discrimination_verdict(
    ref_per_sample: Sequence[object],
    arm_per_sample: Sequence[object],
    *,
    required_resolution_pp: float,
    alpha: float = DISCRIMINATION_ALPHA,
    headroom_min_acc: float = REF_HEADROOM_MIN_ACC,
) -> DiscriminationVerdict:
    """判定「这一条对照能不能用来比较」。

    判据（用户 2026-09-30 裁定）::

        passes_discrimination_gate =
            不一致对 > 0  且  （方向不对称  或  参照臂两端都有余量）

    Args:
        ref_per_sample: 参照臂（如 ``dense``）的逐样本对错，元素为 0/1。
        arm_per_sample: 对照臂的逐样本对错，**与参照臂逐项同序、同一批题目**。
        required_resolution_pp: 本对照要支撑的断言所需的分辨率（百分点）。
            必填。**无默认值是刻意的** —— 「未检出」是否算一句话，取决于
            你想排除多大的效应量；替调用方假定它，等于假称前提永远成立。
        alpha: 方向不对称的显著性水平，默认取登记值。
        headroom_min_acc: 「两端有余量」的余量下限，默认取登记值。

    Returns:
        :class:`DiscriminationVerdict`。⚠️ ``passes_discrimination_gate`` 为 True
        **不等于**可以下结论 —— 还须看 ``claim_licensed`` / ``grade``。

    Raises:
        ValueError: 逐样本长度不等、存在非 0/1 值、``required_resolution_pp``
            非正、``alpha`` 越出 ``(0,1)``、``headroom_min_acc`` 越出 ``[0,0.5)``。
            这些都意味着**输入不支持判定**，故拒绝执行（``SystemExit``/``False``
            都不能表达「输入无效」这一层）。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha 必须落在 (0,1)，收到 {alpha}")
    if not 0.0 <= headroom_min_acc < 0.5:
        raise ValueError(
            f"headroom_min_acc 必须落在 [0,0.5)，收到 {headroom_min_acc}"
            "（余量下限 >= 0.5 会让区间 [m, 1-m] 为空，判据自相矛盾）"
        )
    if not required_resolution_pp > 0:
        raise ValueError(
            f"required_resolution_pp 必须为正，收到 {required_resolution_pp}"
        )

    ref = _as_binary(ref_per_sample, "ref_per_sample")
    arm = _as_binary(arm_per_sample, "arm_per_sample")
    if len(ref) != len(arm):
        raise ValueError(
            f"逐样本长度不等：ref {len(ref)} vs arm {len(arm)}；"
            "配对检验要求同一批题目逐项对齐，长度不等说明对齐已失效"
        )
    n_items = len(ref)
    if n_items == 0:
        raise ValueError("逐样本为空：无可判定的对照")

    b = sum(1 for r, a in zip(ref, arm) if r == 0 and a == 1)
    c = sum(1 for r, a in zip(ref, arm) if r == 1 and a == 0)
    n_discordant = b + c
    ref_acc = sum(ref) / n_items
    arm_acc = sum(arm) / n_items
    delta_pp = (arm_acc - ref_acc) * 100.0

    not_blind = n_discordant > 0
    p_value = mcnemar_exact_p(b, c) if not_blind else None
    directional = bool(p_value is not None and p_value < alpha)
    ref_headroom = bool(headroom_min_acc <= ref_acc <= 1.0 - headroom_min_acc)
    resolution_halfwidth_pp = _resolution_halfwidth_pp(b, c, n_items)
    resolution_ok = bool(resolution_halfwidth_pp <= required_resolution_pp)

    passes = bool(not_blind and (directional or ref_headroom))

    if not not_blind:
        grade: Literal[
            "blind", "directional", "not_detected", "underresolved", "inconclusive"
        ] = "blind"
        licence = (
            f"不一致对为 0（{n_items} 条上两臂逐条相同）：本条对照与噪声不可区分，"
            "p 无定义。**任何比较结论都不成立**，不是「差异不显著」。"
        )
    elif directional:
        grade = "directional"
        licence = (
            f"方向可分辨（精确 McNemar p={p_value:.4g} < {alpha}）。"
            "可断言方向；**量值须同报分辨率读数**"
            f"（本条 {resolution_halfwidth_pp:.1f} pp），不得只报点估计。"
            + (
                ""
                if resolution_ok
                else f"⚠️ 半宽 {resolution_halfwidth_pp:.1f} pp > 所需分辨率 "
                f"{required_resolution_pp:.1f} pp ⇒ **只能报方向，不能报量值**。"
            )
        )
    elif ref_headroom and resolution_ok:
        grade = "not_detected"
        licence = (
            f"方向不可分辨（p={p_value:.4g} >= {alpha}），但参照臂 "
            f"{ref_acc:.3f} 两端留有余量，且分辨率读数 {resolution_halfwidth_pp:.1f} pp "
            f"<= 所需分辨率 {required_resolution_pp:.1f} pp。"
            "**仅可报「未检出差异」**；不得读作「无效应」或「方法无损」。"
        )
    elif ref_headroom:
        grade = "underresolved"
        licence = (
            f"方向不可分辨（p={p_value:.4g} >= {alpha}），参照臂两端虽有余量，"
            f"但分辨率读数 {resolution_halfwidth_pp:.1f} pp > 所需分辨率 "
            f"{required_resolution_pp:.1f} pp ⇒ 该区间容得下关注量级的真实效应，"
            "**连「未检出」也不成立**。需扩样或换量具。"
        )
    else:
        grade = "inconclusive"
        licence = (
            f"方向不可分辨（p={p_value:.4g} >= {alpha}），且参照臂 {ref_acc:.3f} "
            f"落在端点带 [0,{headroom_min_acc}) ∪ ({1.0 - headroom_min_acc},1] 内，"
            "该方向的检测被端点截断（参照臂自己已无余量）⇒ 本条对照不得用于比较。"
        )

    return DiscriminationVerdict(
        n_items=n_items,
        b=b,
        c=c,
        n_discordant=n_discordant,
        ref_acc=ref_acc,
        arm_acc=arm_acc,
        delta_pp=delta_pp,
        p_value=p_value,
        resolution_halfwidth_pp=resolution_halfwidth_pp,
        required_resolution_pp=required_resolution_pp,
        alpha=alpha,
        headroom_min_acc=headroom_min_acc,
        not_blind=not_blind,
        directional=directional,
        ref_headroom=ref_headroom,
        resolution_ok=resolution_ok,
        passes_discrimination_gate=passes,
        claim_licensed=bool(grade in CLAIM_LICENSING_GRADES),
        grade=grade,
        licence=licence,
    )


def discrimination_verdict_from_arms(
    arms: Mapping[str, Mapping[str, object]],
    *,
    reference: str,
    required_resolution_pp: float,
    alpha: float = DISCRIMINATION_ALPHA,
    headroom_min_acc: float = REF_HEADROOM_MIN_ACC,
) -> Dict[str, DiscriminationVerdict]:
    """对一份 `niah_eval*.json` 的 ``arms`` 逐臂判定，参照臂由 ``reference`` 指定。

    Args:
        arms: ``{"dense": {"per_sample": [...], ...}, "topk_0.25": {...}}``。
        reference: 参照臂名，必须存在于 ``arms`` 且带 ``per_sample``。
        required_resolution_pp: 见 :func:`discrimination_verdict`。
        alpha / headroom_min_acc: 同 :func:`discrimination_verdict`。

    Returns:
        ``{臂名: DiscriminationVerdict}``，**不含参照臂自身**。

    Raises:
        KeyError / ValueError: 参照臂缺失或无 ``per_sample`` 时。缺 ``per_sample``
            的历史产物**不得空跑** —— 那种情况下无法判定，必须显式报出来。
    """
    if reference not in arms:
        raise KeyError(f"参照臂 {reference!r} 不在 arms 中：{sorted(arms)}")
    ref_entry = arms[reference]
    if "per_sample" not in ref_entry:
        raise ValueError(
            f"参照臂 {reference!r} 缺 per_sample 字段（该产物未逐样本落盘），无法判定"
        )
    ref_ps = ref_entry["per_sample"]
    out: Dict[str, DiscriminationVerdict] = {}
    for name, entry in arms.items():
        if name == reference:
            continue
        if "per_sample" not in entry:
            raise ValueError(
                f"臂 {name!r} 缺 per_sample 字段（该产物未逐样本落盘），无法判定"
            )
        out[name] = discrimination_verdict(
            ref_ps,
            entry["per_sample"],
            required_resolution_pp=required_resolution_pp,
            alpha=alpha,
            headroom_min_acc=headroom_min_acc,
        )
    return out
