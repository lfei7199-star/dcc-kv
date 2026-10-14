"""守卫：experiments/run_gpu.sh 的退出码语义、环境自举与 torchrun 探测。

2026-10-14 一天内修了四批缺陷（详见 docs/commit_log.md 第 47/49 条）：
  缺陷① 八个调用处写 `|| FAILED=1`，内层 gate 的退出码 3 被无条件压成 1，
         README 承诺的「退出码 3 = 环境闸门未通过，不是实验失败」经本入口
         **永远不可达**，CI 无法区分「没显卡」与「实验失败」；
  缺陷② `PY="${PYTHON:-python}"` 不自举环境：PATH 里没 torch 的 python 让
         内层脚本在 import 阶段就 ModuleNotFoundError，gate 根本够不着；
  缺陷③ `usage()` 恒 exit 0：无参数、未知实验名（含把 --plan 当实验名的
         位置陷阱）在 CI 里看起来像**成功**；且 -h 作为第一个参数会被当成
         实验名而 exit 2（与「-h 是要看帮助」的普遍预期相反）；
  缺陷④ 多卡分支找 torchrun 只查 PATH：Windows venv 布局里 torchrun 与解释器
         同在 Scripts/，未激活 venv 时不在 PATH 上 ⇒ 装好了 PyTorch 也 127。

修法：
  - 汇总规则改为「取所有实验里最大的非零退出码」track()；
  - 入口处探测解释器：不存在 -> 127；存在但缺 torch -> 3（均带指引）；
  - usage() 带退出码参数（缺省 0；无参/未知实验显式传 2；首参 -h/--help 认帮助）；
  - torchrun 探测：PATH 优先，找不到回落到 $PY 同目录。

另修一个**测试基建自身的坑**：本机 PATH 里第一个 `bash` 可能是 WSL 启动器
（system32\\bash.EXE）—— 它收下 Windows 路径也找不到脚本文件，直接用它跑
本文件会得到与环境无关的假失败。故 BASH 解析加一道「能跑 Windows 路径脚本」
的实弹验证，候选全部不可用才 skip（是 skip 不是 fail：换台没有 bash 的机器
不该把守卫染红）。

本文件全部用例 CPU 可跑：真跑 bash 脚本、用退出码断言，不 mock。
"""

import importlib.util
import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "run_gpu.sh"

HAVE_TORCH = importlib.util.find_spec("torch") is not None

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="当前解释器无 torch")


# ---------------------------------------------------------------------------
# bash 解析：要能跑 Windows 路径脚本的真 bash，不是 WSL 启动器
# ---------------------------------------------------------------------------

def _bash_can_run_windows_path_script(cand: str, scratch: Path) -> bool:
    """候选 bash 必须能执行「以 Windows 路径传入的脚本文件」。

    WSL 启动器会把 `C:/...` 当相对路径，报 no such file ⇒ 拒收。
    Linux 上 which("bash") 就是真 bash，天然通过。
    """
    probe = scratch / "_bash_probe.sh"
    probe.write_text("echo PROBE_OK\n", encoding="utf-8")
    try:
        r = subprocess.run(
            [cand, str(probe)], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and "PROBE_OK" in (r.stdout or "")


def _resolve_bash() -> Optional[str]:
    """按优先级找可用 bash：PATH 里的 → PortableGit → Program Files\\Git。"""
    cands = []
    w = shutil.which("bash")
    if w:
        cands.append(w)
    home = Path.home()
    # 注意用 glob.glob 而非 Path.glob：后者只收相对模式，绝对模式会
    # NotImplementedError（2026-10-14 实测踩到）。
    for pat in (
        f"{home}/.workbuddy/binaries/PortableGit/versions/*/usr/bin/bash.exe",
        f"{home}/.workbuddy/binaries/PortableGit/versions/*/bin/bash.exe",
        "C:/Program Files/Git/usr/bin/bash.exe",
        "C:/Program Files/Git/bin/bash.exe",
        "C:/Program Files (x86)/Git/usr/bin/bash.exe",
    ):
        cands.extend(sorted(glob.glob(pat), reverse=True))
    seen = set()
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        for c in cands:
            if c in seen or not Path(c).exists():
                continue
            seen.add(c)
            if _bash_can_run_windows_path_script(c, scratch):
                return c
    return None


BASH = _resolve_bash()

needs_bash = pytest.mark.skipif(
    BASH is None, reason="找不到可用的 bash（WSL 启动器不算：它跑不了 Windows 路径脚本）"
)


# ---------------------------------------------------------------------------
# 公共助手
# ---------------------------------------------------------------------------

def _run(args, env_python=None, env_path=None, timeout=300):
    """真跑 run_gpu.sh，返回 (退出码, 合并输出)。退出码以进程为准，不靠管道。"""
    env = None
    if env_python is not None or env_path is not None:
        env = dict(os.environ)
        if env_python is not None:
            env["PYTHON"] = str(env_python)
        if env_path is not None:
            env["PATH"] = env_path
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


def sys_executable():
    import sys

    return sys.executable


def _minimal_path_for_bash(bash_exe: str) -> str:
    """收窄 PATH 到「bash 所在目录 + usr/bin + System32」，确保其中没有 torchrun。

    本机 torchrun 装在 ``C:\\Python313\\Scripts``（在 PATH 上）—— 不收窄的话
    PATH 分支会赢，回落分支根本走不到，测试将失去意义。
    """
    bdir = Path(bash_exe).parent
    dirs = [bdir, bdir.parent / "usr" / "bin"]
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    dirs.append(Path(sysroot) / "System32")
    parts = []
    for d in dirs:
        if d.is_dir():
            s = str(d).replace("\\", "/")
            if s not in parts:
                parts.append(s)
    return ":".join(parts)


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


@needs_bash
def test_source_has_usage_exit_code_and_torchrun_fallback():
    """缺陷③④ 的源码锚点：usage 带退出码、torchrun 有同目录回落。"""
    src = SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    # 缺陷③：usage() 不再恒 exit 0
    assert 'exit "${1:-0}"' in code, "usage() 又回到了恒 exit 0（用法错误会假装成功）"
    assert "usage 2" in code, "无参 / 未知实验分支没有显式传退出码 2"
    assert re.search(r"-h\|--help\)\s*usage", code), (
        "首参 -h/--help 没有接入 usage（会被当成实验名 exit 2）"
    )
    # 缺陷④：torchrun 回落
    assert "_pybin/torchrun" in code, "缺少 $PY 同目录的 torchrun 回落探测"
    assert '"$TORCHRUN"' in code, "多卡分支没有改用解析出的 $TORCHRUN"


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


# ---------------------------------------------------------------------------
# 四、缺陷③：usage 退出码（用法错误不再假装成功）
# ---------------------------------------------------------------------------

@needs_bash
@needs_torch
def test_unknown_experiment_exits_2():
    """未知实验名：必须 exit 2（修复前 usage 恒 0，CI 会读成成功）。"""
    rc, out = _run(["bogus"], env_python=sys_executable())
    assert rc == 2, f"期望未知实验 exit 2，实际 {rc}。输出尾部：\n{out[-800:]}"
    assert "未知实验" in out


@needs_bash
@needs_torch
def test_no_args_exits_2():
    """无参数：打印用法但 exit 2（不是 0）。"""
    rc, out = _run([], env_python=sys_executable())
    assert rc == 2, f"期望无参 exit 2，实际 {rc}。输出尾部：\n{out[-800:]}"
    assert "可用实验" in out


@needs_bash
@needs_torch
@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_flag_exits_zero(flag):
    """-h/--help：无论放第一个参数还是实验名后，都是 exit 0（帮助请求）。

    放第一个参数曾被打成「未知实验」exit 2（缺陷③ 的附带症状）。
    """
    rc, out = _run([flag], env_python=sys_executable())
    assert rc == 0, f"`run_gpu.sh {flag}` 期望 0，实际 {rc}。输出尾部：\n{out[-800:]}"
    assert "可用实验" in out
    rc2, out2 = _run(["e6", flag], env_python=sys_executable())
    assert rc2 == 0, f"`run_gpu.sh e6 {flag}` 期望 0，实际 {rc2}"
    assert "可用实验" in out2


@needs_bash
@needs_torch
def test_plan_as_first_arg_exits_2():
    """位置陷阱的收口：`run_gpu.sh --plan`（--plan 当实验名）必须 exit 2。

    这是 2026-09-16 审计记账的用法与实现不符之处；旧实现经 usage 恒 0，
    这条路径在 CI 里静悄悄成功，故钉住非零。
    """
    rc, out = _run(["--plan"], env_python=sys_executable())
    assert rc == 2, f"期望 --plan 位置陷阱 exit 2，实际 {rc}。输出尾部：\n{out[-800:]}"
    assert "未知实验" in out


# ---------------------------------------------------------------------------
# 五、缺陷④：torchrun 同目录回落（Windows venv 布局）
# ---------------------------------------------------------------------------

@needs_bash
def test_torchrun_falls_back_to_interpreter_sibling(tmp_path):
    """PATH 没有 torchrun、但 $PY 同目录有时，必须用同目录那个。

    场景搭建（全部是真文件、真执行，不 mock）：
      - ``bin/python.sh``  解释器替身：只对 `import torch` 探测放行，
        其余调用一律 exit 42（本测试只走多卡分支，不会再用解释器跑脚本；
        真被误用会以 42 响亮失败，而不是静默换真解释器）；
      - ``bin/torchrun``   假 torchrun：打标记、打印实收参数、exit 3
        （顺便验退出码透传：FINAL 应汇总成 3）；
      - PATH 收窄到不含 torchrun 的最小集（本机 torchrun 在
        ``C:\\Python313\\Scripts``，不收窄则 PATH 分支赢、回落走不到）。
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "python.sh"
    shim.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ] && [ "$2" = "import torch" ]; then exit 0; fi\n'
        'echo "shim: 非预期调用: $*" >&2\n'
        "exit 42\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    fake = bindir / "torchrun"
    fake.write_text(
        "#!/bin/sh\n"
        'echo "FAKE_TORCHRUN_ARGS: $@"\n'
        "exit 3\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    scrubbed = _minimal_path_for_bash(BASH)
    probe = subprocess.run(
        [BASH, "-c", "command -v torchrun"],
        env={**os.environ, "PATH": scrubbed},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode != 0, (
        f"收窄后的 PATH 里仍能找到 torchrun（{probe.stdout.strip()}）—— "
        "回落分支走不到，本测试失去意义"
    )

    rc, out = _run(["e5", "--nproc", "2"], env_python=shim, env_path=scrubbed)
    assert rc == 3, (
        f"期望假 torchrun 的退出码 3 被透传，实际 {rc}。输出尾部：\n{out[-1500:]}"
    )
    assert "FAKE_TORCHRUN_ARGS" in out, "同目录的 torchrun 没有被调用（回落失效）"
    assert "--nproc_per_node=2" in out, "nproc 参数没有正确传给 torchrun"
    assert "--standalone" in out, "缺少 --standalone"
    assert "e5_gpu_ablation.py" in out, "实验脚本路径没有传给 torchrun"
