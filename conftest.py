"""Pytest 全局配置 + 共享 fixtures。

放项目根目录的 conftest.py 让所有 tests/ 下的文件都能用。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 把 src 加入 sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def pytest_collection_modifyitems(config, items):
    """自动给测试打 marker。

    规则：
    - 测试函数名含 'distributed' 或 '2proc' / '4proc' → distributed
    - 测试函数名含 'gpu' / 'nccl' / 'async_overlap' / 'end_to_end' → gpu
    - 测试在 tests/gpu/ 目录下 → gpu
    """
    for item in items:
        # 路径判断
        if "tests/gpu/" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)
            continue

        # 名字判断
        name = item.name.lower()
        if "gpu" in name or "nccl" in name or "async_overlap" in name or "end_to_end" in name:
            item.add_marker(pytest.mark.gpu)
        elif "distributed" in name or "2proc" in name or "4proc" in name or "all_to_all" in name:
            item.add_marker(pytest.mark.distributed)


@pytest.fixture(scope="session")
def project_root():
    """项目根目录。"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def torch_double_seeded():
    """设置双精度 + 固定种子。"""
    import torch
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    return torch
