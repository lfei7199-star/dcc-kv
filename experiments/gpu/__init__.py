"""GPU 实验（E5–E8）—— 需要真实 GPU 与 NCCL，不接受 CPU 退化。

与 `experiments/cpu/` 的分工：

    experiments/cpu/  机制级证据 —— 重构精度、数值等价性、顺序无关性
                      任意多核机器可复现，不需模型权重，不需 NCCL
    experiments/gpu/  任务级与系统级证据 —— 任务指标、通信量、延迟、可扩展性
                      需要 A100/H100 + NCCL + 真实模型权重

两侧的对应关系不是"简化版 vs 完整版"，而是**不同量纲的证据**：

    E3（CPU, KL/JS/配对检验）  vs  A2（GPU, 预算–精度参考曲线，非帕累托前沿）
    A3 机制级（CPU, 重构误差） vs  A3 任务级（GPU, 任务指标点数）
    E0（CPU, FP32/FP64）       vs  E8（GPU, FP16/BF16 + 低精度失效边界）

因此 GPU 结果不能由 CPU 结果外推得到，反之亦然。
"""
