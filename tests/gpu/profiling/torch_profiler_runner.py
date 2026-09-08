"""M3 GPU profiling：PyTorch Profiler trace 收集。

⚠️ 必须有 GPU。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

pytestmark = pytest.mark.gpu


def test_torch_profiler_basic(gpu_available):
    """PyTorch Profiler 基础功能（CUDA activity）。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU")

    from torch.profiler import profile, ProfilerActivity

    # 简单 CUDA 操作
    x = torch.randn(1024, 1024, device="cuda:0")
    y = torch.randn(1024, 1024, device="cuda:0")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            z = torch.matmul(x, y)
            torch.cuda.synchronize()

    # 验证 trace 被记录
    assert prof is not None
    output = prof.key_averages().table(sort_by="cuda_time_total", row_limit=5)
    assert "matmul" in output.lower() or "gemm" in output.lower()
