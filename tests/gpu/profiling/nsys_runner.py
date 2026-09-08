"""M3 GPU profiling：Nsight Systems trace 收集。

⚠️ 必须有 GPU + nsys 工具。

使用：
    nsys profile -o my_trace --trace=cuda,nvtx --force-overwrite=true \\
        python -m pytest tests/gpu/test_async_overlap.py -v

或直接：
    nsys profile -o my_trace --trace=cuda,nvtx python my_script.py

    nsys-ui my_trace.nsys-rep
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def test_nsys_installation():
    """检查 nsys 是否安装。"""
    import shutil
    nsys = shutil.which("nsys")
    if nsys is None:
        pytest.skip("nsys not installed; please install NVIDIA Nsight Systems")
    # 检查版本
    result = subprocess.run([nsys, "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Nsight Systems" in result.stdout or "nsys" in result.stdout.lower()


def test_nsys_profile_help():
    """nsys profile --help 能跑（基础功能验证）。"""
    import shutil
    nsys = shutil.which("nsys")
    if nsys is None:
        pytest.skip("nsys not installed")
    result = subprocess.run(
        [nsys, "profile", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "--trace" in result.stdout


# 提供给 nsys 调用的真实脚本（手动运行，不在 pytest 里）
def collect_trace_for_dcc_kv_sync(
    output_dir: str = "./traces",
    duration_sec: int = 30,
    port: int = 29550,
):
    """收集 DCC-KV 同步版本的 Nsight trace。

    用法：
        nsys profile -o trace_sync --trace=cuda,nvtx --duration=$duration_sec \\
            python -m tests.gpu.profiling.nsys_runner collect_trace_for_dcc_kv_sync
    """
    os.makedirs(output_dir, exist_ok=True)
    # 真实实现：调 DCC-KV 同步版 DCC-KV-sync 主循环
    # 留作占位
    raise NotImplementedError(
        "Will be implemented when DCC-KV 同步版 GPU 实现完成（M3 阶段）"
    )


def collect_trace_for_dcc_kv_async(
    output_dir: str = "./traces",
    duration_sec: int = 30,
    port: int = 29551,
):
    """收集 DCC-KV 异步版本的 Nsight trace。

    用法：
        nsys profile -o trace_async --trace=cuda,nvtx --duration=$duration_sec \\
            python -m tests.gpu.profiling.nsys_runner collect_trace_for_dcc_kv_async
    """
    os.makedirs(output_dir, exist_ok=True)
    raise NotImplementedError(
        "Will be implemented when DCC-KV 异步版 GPU 实现完成（M3 阶段）"
    )
