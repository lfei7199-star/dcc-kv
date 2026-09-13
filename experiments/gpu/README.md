# experiments/gpu/ —— GPU 侧实验（E5–E8）

⚠️ **这些脚本需要真实 GPU。** 未达标时不会退化为 CPU 运行，而是退出码 3 并
**不产出任何结果文件**。机制级版本在 `experiments/cpu/`，两侧不可互替。

## 快速自检（任何机器可跑）

```bash
python experiments/gpu/e5_gpu_ablation.py --print-env   # 环境探测
python experiments/gpu/e6_main_table.py   --plan        # 主表网格与前置条件
python experiments/gpu/e7_negative_results.py --plan    # 负结果条件清单
```

退出码：`0` = 环境达标 / `3` = 环境闸门未通过。

## 资源需求

| 脚本 | 最小 GPU | 最小显存 | 推荐 | 备注 |
|---|---|---|---|---|
| `e8_low_precision.py` | 1 卡 | 4 GB | 1 卡 A100 | 只做归并算子，无通信、无模型 |
| `e7_negative_results.py` | 1 卡 | 40 GB | 1 卡 A100 80G | 32K 上下文 + 8B 模型 |
| `e6_main_table.py` | 1 卡 | 40 GB | 1 卡 A100 80G | 只有 `dense` / `kv_budget_shared` 可测 |
| `e5_gpu_ablation.py` | **2 卡** | 16 GB | 4 卡 A100 80G NVLink | A1/A2/A5 需要 NCCL |

70B/72B 与跨节点实验需要 8×H100 或双节点，见 `docs/reproducibility.md` §4。

## 启动顺序

1. `--print-env` 确认环境达标
2. `e8` —— 最省资源，验证脚本与数值链路
3. `e7` —— 单卡 + 模型，先跑出负结果边界
4. `e6` —— 单卡主表的可测部分
5. `e5` —— 多卡 + NCCL，核心系统性能

```bash
bash experiments/run_gpu.sh e8
bash experiments/run_gpu.sh e5 --nproc 4 --parts a1 a2 a5
bash experiments/run_gpu.sh all --nproc 4
```

## 三个模块的分工

| 模块 | 职责 | 不要放什么 |
|---|---|---|
| `_env.py` | 环境闸门、元数据、逐次计时、CUDA 构造探测 | 实验逻辑 |
| `_comm.py` | 变长 All-to-Allv、同步/异步流水、体积口径 | 模型相关的东西 |
| `_hf.py` | 模型加载、KV 预算、打分、prefill 计时 | 多卡通信 |

## 三条口径纪律

**1. 通信量口径固定。** 一条紧凑边 = `K[B,d_h] + β[B] + V[B,d_v]`，
不 padding 到等长。β 必须计入，否则压缩比被系统性高估。

**2. 计时拆成四项，不许混。** `T_build` / `T_comm` / `T_comp` / `T_total`。
异步的收益只来自 `T_comm` 与 `T_comp` 的重叠；把构造塞进任何一项都会
让归因失真。

**3. 延迟必须报分布。** 所有计时走 `_env.benchmark_ms`，返回**逐次**样本，
由 `experiments/common/report.py` 汇总成 median / p5 / p95 / bootstrap CI。
`--iters` 低于 10 时会打印警告，因为 §6.1 要求每点 ≥10 次重复。

## 已知前置缺口

详见 `experiments/README.md` 的「当前被阻断的部分」。摘要：

- **A3 与 E6 的 `dcc_kv` 不可测**：没有 `CompactKV → GPU attention kernel`。
  脚本会停下并打印缺失清单，不产出假数字。
- **构造链路 CUDA 不可用**：`src/dcc_kv_ref` 下三处 device 缺陷
  （`representative_query.py:27,30`、`value_regression.py:32`），
  见 `_env.CUDA_CONSTRUCTION_DEFECTS`。A1 的 `--build-location cpu`
  是当前唯一可部署路径，其 `T_build` 含 H2D 传输。
- **`ring` / `apb` / `fastkv` 无 GPU 实现**，E6 主表里这三个以
  `status: "blocked"` 记账。

## `--device cpu` 只用于 E8

`e8_low_precision.py` 支持 `--device cpu --allow-cpu`，理由是它测的归并算子
只有逐元素运算与归约、没有矩阵乘也没有通信，结构上可完整跑通。
但 CPU 与 GPU 的 fp16/bf16 归约路径不同，**误差常数不可迁移**：
CPU 运行只用于验证脚本逻辑与观察趋势（误差随 K 与树深的增长阶），
不得作为 E8 结论。没有 `--allow-cpu` 时会被拒绝执行。
