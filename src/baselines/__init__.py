"""DCC-KV 实验基线。

两层，**别混用**：

``*_cpu``（reference）
    reference-quality 的**逐 query Python 循环**实现。语义权威（因果掩码按
    原始位置、APB 的锚点用后续 queries 选、FastKV 的共享压缩用全体 queries 选），
    但**不是可上机的实现** —— L=32k 时是 3 万多次核启动，测出来的 T_comp 是
    启动开销而非算子代价。用于对照与语义检查。

``operators``（G3，可上机）
    `ring_attention` / `fastkv_attention` / `apb_attention`：device-agnostic 的
    向量化版本（整块 query 一次算 + lse 归并），语义与 `*_cpu` 逐条对齐，
    等价强度为 **ULP 级**而非逐位（求和次序不同，见模块文档）。
    E6 主表按方法计时与算指标时必须用这一层。

⚠️ 历史教训：曾把 `*_cpu` 的逐 query 循环当作"基线性能"，并因此写出
「Ring Attention 比 DCC-KV 快 1.99×」这类比较 —— 那比的是 Python 循环的
开销，不是方法。任何跨方法的计时/准确率对照都必须确认两侧走的是同一层。
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
from .operators import (
    ring_attention,
    fastkv_attention,
    apb_attention,
)

__all__ = [
    # reference（逐 query，语义权威，不可用于计时）
    "ring_attention_cpu",
    "ring_attention_dense",
    "fast_kv_cpu",
    "apb_cpu",
    # 可上机（向量化，G3）
    "ring_attention",
    "fastkv_attention",
    "apb_attention",
    "FastKVConfig",
    "APBConfig",
]
