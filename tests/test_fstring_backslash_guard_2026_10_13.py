r"""守卫：f-string 表达式内不得含反斜杠 + e9 表头常量渲染值不变。

背景（2026-10-13）
----------------
``experiments/cpu/e9_knob_localization.py`` 的表头写作
``f"{'M\B':>6} ..."`` —— f-string 的**表达式部分**内含反斜杠。
Python 3.12 之前（PEP 701 之前）这是 SyntaxError，整个模块导入即失败；
3.12+ 放开了该限制，所以 ``repo-tests``（py3.13）下这条错误**一直隐身**，
只在 py3.11 环境暴露。该写法在 HEAD 态已存在多轮，直到本轮才修掉
（常量 ``_MB_LABEL = "M\B"``，渲染值逐字节不变）。

本守卫两道：

1. ``test_no_backslash_in_fstring_expressions`` —— 全仓 AST 扫描，
   任何 .py 的 f-string 表达式里出现反斜杠即失败。按解释器精度分流：
   - **py3.12+**（PEP 701，f-string 内部 AST 位置精确）：直接查
     ``FormattedValue`` 的源码段，完备无假阳性；
   - **py3.11**（内部位置不精确，源码段会溢出到字面部分 —— 实测会把
     ``f"...\n...{x}"`` 的literal 段算进来，造成假阳性）：改查
     「表达式内字符串常量的**解码值**含反斜杠」。解码值里的反斜杠
     只可能来自源码表达式里的反斜杠，故无假阳性；漏报 only 逃逸类
     （``f"{'\n'}"``，解码成换行）—— 而那种形态在 py3.11 上会当场
     SyntaxError，本就暴露，不需扫描兜底。
   扫描排除 ``venv/``（第三方代码）、``__pycache__/``、
   ``_audit_tmp/``（``_e9_head.py`` 是 HEAD 态旧版的刻意留档，
   含旧写法是它的职责所在）。
2. ``test_e9_importable_and_header_label`` —— e9 模块必须可导入，
   且表头常量的渲染值必须逐字节等于 ``M\B``（修复没有改变任何输出）。
"""

import ast
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EXCLUDE_DIRS = {"venv", "__pycache__", "_audit_tmp", ".git", "node_modules"}
BACKSLASH = chr(92)  # 一个字面反斜杠；避免在测试源码里写转义对
_PRECISE_FSTRING_POSITIONS = sys.version_info >= (3, 12)  # PEP 701


def _iter_repo_py():
    for p in sorted(REPO_ROOT.rglob("*.py")):
        if _EXCLUDE_DIRS & set(p.parts):
            continue
        yield p


def _segment_violations(src, tree, rel):
    """py3.12+：FormattedValue 的源码段内含反斜杠 ⇒ 违规（位置精确，无假阳性）。"""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FormattedValue):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if BACKSLASH in seg:
            out.append("%s:%d：f-string 表达式部分含反斜杠" % (rel, node.lineno))
    return out


def _constant_violations(src, tree, rel):
    """py3.11：表达式内字符串常量的解码值含反斜杠 ⇒ 违规（不依赖位置）。"""
    out = []

    def visit(node, in_expr):
        if isinstance(node, ast.FormattedValue):
            # value 与 format_spec 整体都是 py3.11 口径的「表达式部分」
            visit(node.value, True)
            if node.format_spec is not None:
                visit(node.format_spec, True)
            return
        if (in_expr and isinstance(node, ast.Constant)
                and isinstance(node.value, str) and BACKSLASH in node.value):
            out.append("%s:%d：f-string 表达式内的字符串常量含反斜杠"
                       % (rel, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, in_expr)

    visit(tree, False)
    return out


def test_no_backslash_in_fstring_expressions():
    """全仓扫描：f-string 表达式部分不得含反斜杠（py3.11 SyntaxError 之源）。"""
    strategy = ("源码段（PEP 701，精确）" if _PRECISE_FSTRING_POSITIONS
                else "常量解码值（位置不精确，保守）")
    offenders = []
    for p in _iter_repo_py():
        src = p.read_text(encoding="utf-8")
        rel = p.relative_to(REPO_ROOT)
        try:
            tree = ast.parse(src, filename=str(rel))
        except SyntaxError as exc:
            # 当前解释器都解析失败 ⇒ 更老的解释器更不行，同视为违规
            offenders.append("%s：解析失败（%s）" % (rel, exc.msg))
            continue
        if _PRECISE_FSTRING_POSITIONS:
            offenders.extend(_segment_violations(src, tree, rel))
        else:
            offenders.extend(_constant_violations(src, tree, rel))
    assert not offenders, (
        "以下位置的 f-string 表达式内含反斜杠（本次扫描策略：%s）—— "
        "Python 3.11 及以下会直接 SyntaxError（模块无法导入，且 py3.13 的"
        "测试全绿也发现不了）；请把字面量提为模块级常量再引用：\n  "
        % strategy + "\n  ".join(offenders))


def test_e9_importable_and_header_label():
    """e9 必须可导入；表头常量 _MB_LABEL 的渲染值必须逐字节是 M\\B。"""
    pytest.importorskip("torch")
    from experiments.cpu import e9_knob_localization as E9

    expected = "M" + BACKSLASH + "B"
    assert E9._MB_LABEL == expected, (
        "表头标签渲染值被改动：%r != %r —— 修复不得改变任何输出"
        % (E9._MB_LABEL, expected))
