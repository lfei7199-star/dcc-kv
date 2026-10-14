"""守卫：experiments/run_gpu.sh 退出码透传与环境自举（2026-10-14 修缺陷①②）。

背景（详见 docs/commit_log.md 第 47 条）：
  缺陷① 八个调用处写 `|| FAILED=1`，内层 gate 的退出码 3 被无条件压成 1，
         README 承诺的「退出码 3 = 环境闸门未通过，不是实验失败」经本入口
         **永远不可达**，CI 无法区分「没显卡」与「实验失败」；
  缺陷② `PY="${PYTHON:-python}"` 不自举环境：PATH 里没 torch 的 python 让
         内层脚本在 import 阶段就 ModuleNotFoundError，gate 根本够不着。

修法：
  - 汇总规则改为「取所有实验里最大的非零退出码」track()；
  - 入口处探测解释器：不存在 -> 127；存在但缺 torch -> 3（均带指引）。

本文件全部用例 CPU 可跑：真跑 bash 脚本、用退出码断言，不 mock。
"""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "run_gpu.sh"

BASH = shutil.which("bash")
HAVE_TORCH = importlib.util.find_spec("torch") is not None

needs_bash = pytest.mark.skipif(BASH is None, reason="本机找不到 bash")
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="当前解释器无 torch")


def _run(args, env_python=None, timeout=300):
    """真跑 run_gpu.sh，返回 (退出码, 合并输出)。退出码以进程为准，不靠管道。"""
    env = None
    if env_python is not None:
        env = dict(os.environ)
        env["PYTHON"] = str(env_python)
    proc = subprocess.run(
        [BASH, str(SCRIPT)] + list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        env=env,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _find_no_torch_python():
    """找一个存在但 import 不了 torch 的解释器（缺陷②的行为断言用）。"""
    candidates = [
        Path("C:/Program Files/Python311/python.exe"),
        Path("C:/Program Files/Python312/python.exe"),
        Path("C:/Program Files/Python313/python.exe"),
        Path("C:/Python311/python.exe"),
        Path("/usr/bin/python3"),
        Path("/usr/bin/python"),
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            r = subprocess.run(
                [str(cand), "-c", "import torch"],
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode != 0:
            return cand
    return None


# ---------------------------------------------------------------------------
# 一、源码级守卫（防回归，任何环境可跑）
# ---------------------------------------------------------------------------

@needs_bash
def test_source_has_no_failed_swallow_pattern():
    """旧实现用 `|| FAILED=1` 吞掉内层退出码 —— 这种形态不许回来。

    注意：必须先剥掉**整行注释**再匹配。脚本里留了一句说明旧缺陷的注释，
    原文就含 `|| FAILED=1` 字样；不剥注释会把史实说明误判成缺陷回归
    （2026-10-14 首版守卫实测踩到，故判据建在「代码」上而非「文本」上）。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "|| FAILED=1" not in code, (
        "run_gpu.sh 回到了 `|| FAILED=1` 形态：内层 gate 的退出码 3 会被压成 1"
    )
    # 新机制的两个锚点：汇总变量 + track 函数
    assert "FINAL=0" in code, "缺少退出码汇总变量 FINAL"
    assert "track()" in code, "缺少 track() 汇总函数"
    # 环境自举的两个锚点
    assert 'import torch' in code, "缺少 torch 自举探测"


# ---------------------------------------------------------------------------
# 二、缺陷①：退出码透传（核心回归）
# ---------------------------------------------------------------------------

@needs_bash
@needs_torch
def test_gate_exit_3_passthrough():
    """有 torch 无 CUDA：e8 必须经入口透传退出码 3（修复前是 1）。"""
    rc, out = _run(["e8"], env_python=sys_executable())
    assert rc == 3, f"期望 gate 透传 3，实际 {rc}。输出尾部：\n{out[-800:]}"
    assert "环境闸门" in out, "汇总行未说明这是环境闸门未通过"


@needs_bash
@needs_torch
def test_param_reject_exit_2_passthrough():
    """e6 缺 --dcc-world：参数层拒绝的 2 也必须透传（不被压成 1）。"""
    rc, out = _run(["e6"], env_python=sys_executable())
    assert rc == 2, f"期望参数层拒绝透传 2，实际 {rc}。输出尾部：\n{out[-800:]}"
    assert "dcc-world" in out


@needs_bash
@needs_torch
def test_plan_mode_still_exit_zero():
    """正常路径不被破坏：e6 --plan 任何机器可跑，必须 exit 0。"""
    rc, out = _run(["e6", "--plan"], env_python=sys_executable())
    assert rc == 0, f"期望 --plan 为 0，实际 {rc}。输出尾部：\n{out[-800:]}"


# ---------------------------------------------------------------------------
# 三、缺陷②：环境自举
# ---------------------------------------------------------------------------

@needs_bash
def test_missing_torch_interpreter_exits_3_with_guidance():
    """解释器在但缺 torch：必须 exit 3 + 明确指引（修复前是 import 崩掉 exit 1）。"""
    cand = _find_no_torch_python()
    if cand is None:
        pytest.skip("本机找不到「存在但无 torch」的解释器")
    rc, out = _run(["e8"], env_python=cand)
    assert rc == 3, f"期望缺 torch 判 3，实际 {rc}。输出尾部：\n{out[-800:]}"
    assert "无法 import torch" in out, "未给出缺 torch 的明确指引"
    assert "PYTHON=" in out, "指引里未说明如何用 PYTHON= 覆盖"


@needs_bash
def test_nonexistent_interpreter_exits_127():
    """解释器不存在：必须 exit 127 + 指引。"""
    rc, out = _run(["e8"], env_python="/nonexistent/python_xyz_definitely_not_here")
    assert rc == 127, f"期望 127，实际 {rc}。输出尾部：\n{out[-800:]}"
    assert "找不到 Python 解释器" in out


def sys_executable():
    import sys

    return sys.executable
