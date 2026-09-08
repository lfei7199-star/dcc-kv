"""DCC-KV 实验基线（CPU mock 实现）。

所有基线都是 reference-quality 的 CPU 实现，用于：
- M2 启动前验证接口形状
- M3+ 真实 GPU 跑性能时的对照（算法逻辑相同，kernel 优化不同）

⚠️ 警告：这些 mock 不做性能优化，只保证接口和数值正确。
真实性能基线需要 GPU 上的高度优化版本。
"""
from .ring_attention_cpu import (
    ring_attention_cpu,
    ring_attention_dense,
)
from .fast_kv_cpu import (
    fast_kv_cpu,
    FastKVConfig,
)
from .apb_cpu import (
    apb_cpu,
    APBConfig,
)

__all__ = [
    "ring_attention_cpu",
    "ring_attention_dense",
    "fast_kv_cpu",
    "FastKVConfig",
    "apb_cpu",
    "APBConfig",
]
