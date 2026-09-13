"""DCC-KV 实验代码的公共模块。

- `synthetic`：合成场景生成、完整注意力参考、误差与分布度量
- `report`：结果汇总（median/p5/p95/bootstrap CI）、配对检验、落盘

仅依赖 torch + numpy，可在纯 CPU 环境运行。
"""
