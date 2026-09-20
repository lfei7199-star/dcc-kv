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

# ---------------------------------------------------------------------------
# 设备绑定：必须取自 LOCAL_RANK，不能用全局 rank
# ---------------------------------------------------------------------------
# 多节点下全局 rank 8 在第 2 个 8 卡节点上是 cuda:0，而 `cuda:{rank}` 单节点上
# 恰好是对的 —— 所以这个错会一直藏着，直到上多节点才以"设备不存在"暴露。
# 定义**只有一处**（experiments/gpu/_env.local_rank_of），这里不再自己拼。
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.gpu._env import local_rank_of as _local_rank  # noqa: E402


pytestmark = pytest.mark.gpu


def test_torch_profiler_basic(gpu_available):
    """PyTorch Profiler 基础功能（CUDA activity）。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 GPU")

    from torch.profiler import profile, ProfilerActivity

    # 简单 CUDA 操作。设备序号同样取自 LOCAL_RANK：写死 "cuda:0" 在多节点上
    # 会让非首节点的进程在别人的卡上跑（或直接报设备不可用）。
    _dev = f"cuda:{_local_rank(0)}"
    x = torch.randn(1024, 1024, device=_dev)
    y = torch.randn(1024, 1024, device=_dev)

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
