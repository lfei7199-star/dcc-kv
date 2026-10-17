"""守卫：论文「假设接线状态」表述与代码登记表一致（2026-10-14 修 :102 自相矛盾）。

背景（2026-09-23 深度审查记账，见 docs/commit_log.md 与该日审查报告）：
  06-experiment.tex 曾写「五条假设中**只有 H1 与 H4** 在代码侧接了判据」，
  而同一份文件稍后的「假设判定的接线状态」段、以及代码事实都表明
  **H2 的判定入口 ``h2_pass_across_lengths`` 已接进 E6**
  （``e6_main_table.py`` 调用它并产出 ``payload["h2"]``）——
  论文自相矛盾，且错的是前面那句（:102）。

修法：:102 改为三态表述 —— 已判定 ``judged``（H1/H4）／入口已接·参数待测（H2）／
尚未接线（H3/H5）。

**2026-10-17 升级为四态**：H2 那一态拆成两条 —— 「入口已接·参数缺」与
「结构性不可判（单卡模拟）」。后者不是措辞调整：单卡模拟下所有源段看到同一批
目的端 query，`dcc_kv` 与共享压缩数值上不可区分，H2 第（i）项恒不成立，
补 δ 与噪声底线无用。论文据此改写，守卫同步钉住「不许退回参数待测」。

本守卫钉住两侧不再各自漂移：
  ① 那句 stale 二元表述不许回来；
  ② 论文现状表述与 ``hypotheses.py`` 登记（judged 集合、H2 的 judge 名与
     code_status、E6 调用点）逐项对得上 —— 登记若变而论文没跟上，这里红。
"""
from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import hypotheses as H  # noqa: E402

TEX = REPO_ROOT / "paper" / "sections" / "06-experiment.tex"
E6 = REPO_ROOT / "experiments" / "gpu" / "e6_main_table.py"

STALE_CLAIM = "只有 H1 与 H4 在代码侧接了判据"


def _tex_code() -> str:
    """剥掉 LaTeX 行注释（行首 % 起头）后的正文。"""
    lines = []
    for line in TEX.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("%"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_stale_binary_wiring_claim_is_gone():
    """stale 的二元表述（只有 H1/H4 接了判据）不许回来。"""
    code = _tex_code()
    assert STALE_CLAIM not in code, (
        "06-experiment.tex 回到了「只有 H1 与 H4 接了判据」的二元表述 —— "
        "它与同文件「假设判定的接线状态」段及代码登记（H2 入口已接 E6）矛盾"
    )


def test_paper_states_four_way_wiring_status():
    """论文必须陈述四态（2026-10-17 由三态升级）：

        judged        —— H1/H4 已判定
        入口已接·参数缺 —— H2 的两个参数（δ、噪声底线）无默认值
        结构性不可判   —— H2 第（i）项在单卡模拟下恒不成立（`structural`）
        尚未接线       —— H3/H5

    第四态是这次修的重点：旧稿把 H2 的缺口写成「仅剩参数待测」，暗示再跑一轮
    就能判；代码事实是 `attention_hook.conditionalization_marginal_available`
    恒 False（所有源段共享同一批目的端 query），补参数无济于事。论文若退回
    「参数待测」，本测试红。
    """
    code = _tex_code()
    assert "已判定" in code and "judged" in code, "论文未陈述 judged 一侧（H1/H4）"
    assert "h2\\_pass\\_across\\_lengths" in code, (
        "论文未提到 H2 的聚合入口 —— 「入口已接」的表述失去锚点"
    )
    assert "结构性不可判（单卡模拟）" in code, (
        "论文未陈述 H2 的第四态：单卡模拟下第（i）项结构上恒不成立。"
        "这正是 2026-10-17 修正的口径 —— 它不是「参数待测」。"
    )
    assert "structural" in code, (
        "论文未把该状态与落盘产物的 `structural` 原因对应起来，"
        "读者无法在产物里复核"
    )
    assert "尚未接线" in code, "论文未陈述 H3/H5 尚未接线"
    # 回退守卫：把 H2 说成「仅剩参数待测」就等于宣称一条不存在的补测路径
    assert "仅剩两个" not in code and "降为「参数待测」" not in code, (
        "论文退回「H2 仅剩参数待测」—— 与单卡模拟的结构性限制矛盾"
    )


def test_registry_matches_paper():
    """代码登记与论文三态逐项对应；登记变了而论文没跟 => 本测试红。"""
    judged = sorted(h for h, s in H.HYPOTHESES.items() if s.code_status == "judged")
    assert judged == ["H1", "H4"], (
        f"judged 集合实为 {judged}，论文写的是「H1 与 H4 已判定」—— "
        "若登记已变，请同步更新论文表述与本断言"
    )
    h2 = H.HYPOTHESES["H2"]
    assert h2.judge == "h2_pass_across_lengths"
    assert h2.code_status == "no-judge"
    e6 = E6.read_text(encoding="utf-8")
    assert "H.h2_pass_across_lengths(" in e6, (
        "E6 不再调用 H2 聚合入口 —— 论文「已接进 E6」的表述失去依据"
    )
    for hid in ("H3", "H5"):
        assert H.HYPOTHESES[hid].code_status == "no-judge", (
            f"{hid} 的登记变了（论文写「尚未接线」）—— 同步更新论文与本断言"
        )
