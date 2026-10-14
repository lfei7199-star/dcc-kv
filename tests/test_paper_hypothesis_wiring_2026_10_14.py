"""守卫：论文「假设接线状态」表述与代码登记表一致（2026-10-14 修 :102 自相矛盾）。

背景（2026-09-23 深度审查记账，见 docs/commit_log.md 与该日审查报告）：
  06-experiment.tex 曾写「五条假设中**只有 H1 与 H4** 在代码侧接了判据」，
  而同一份文件稍后的「假设判定的接线状态」段、以及代码事实都表明
  **H2 的判定入口 ``h2_pass_across_lengths`` 已接进 E6**
  （``e6_main_table.py`` 调用它并产出 ``payload["h2"]``）——
  论文自相矛盾，且错的是前面那句（:102）。

修法：:102 改为三态表述 —— 已判定 ``judged``（H1/H4）／入口已接·参数待测（H2）／
尚未接线（H3/H5）。

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


def test_paper_states_three_way_wiring_status():
    """论文必须陈述三态：judged / 入口已接·参数待测 / 尚未接线。"""
    code = _tex_code()
    assert "已判定" in code and "judged" in code, "论文未陈述 judged 一侧（H1/H4）"
    assert "H2 的判定入口已接进 E6" in code, (
        "论文未陈述「H2 入口已接进 E6、仅剩参数待测」—— "
        "这正是 2026-09-23 审查记账的自相矛盾点"
    )
    assert "尚未接线" in code, "论文未陈述 H3/H5 尚未接线"


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
