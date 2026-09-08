"""DCC-KV 分布式通信子包。

设计目标：
- 接口统一：gloo (CPU) / nccl (GPU) 切换对上层代码透明
- 变长 All-to-Allv：M2 阶段核心通信原语
- mock 模式：单元测试不需要真实 distributed 环境
"""
from .comm import DistributedComm, VarLenMessage
from .launch_dist import launch_dist, setup_distributed, cleanup_distributed

__all__ = [
    "DistributedComm",
    "VarLenMessage",
    "launch_dist",
    "setup_distributed",
    "cleanup_distributed",
]
