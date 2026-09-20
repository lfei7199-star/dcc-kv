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

    ⚠️ 判定依据必须取**函数名本身**（`item.originalname`），不能用 `item.name`
    ------------------------------------------------------------------
    `item.name` 是节点名，**包含 parametrize id**。于是
    `@pytest.mark.parametrize("sub", ["experiments/gpu", "tests/gpu"])`
    会被判成 gpu 用例而默认排除 —— 一条以「gpu」为审查对象的 CPU 锚点
    就这样被静默关掉了。实测（2026-09-20）：新写的三条设备索引锚点
    `test_d1_...[.../gpu]`、`test_d2_gpu_...` 全部没跑，而全量测试仍报全绿。
    反向的坑同样存在：名字里带 'gpu' 的纯 CPU 守卫也会被关掉。

    这一改是**放宽**（参数 id 里的关键词不再参与判定），因此要确认没有
    真正需要 GPU 的用例靠参数 id 被排除 —— `tests/test_adversarial_2026_09_20.py`
    的 `test_h_...` 用子进程收集结果做元守卫，把这件事变成可测的。
    """
    for item in items:
        # 路径判断
        if "tests/gpu/" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)
            continue

        # 名字判断：取函数名（不含 parametrize id）
        name = str(getattr(item, "originalname", item.name)).lower()
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
