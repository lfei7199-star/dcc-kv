# experiments/ —— 实验代码：CPU 侧与 GPU 侧

本目录把论文 §6 的实验方案拆成**两套不可互相替代**的实现。
拆分的依据不是"简化版 / 完整版"，而是**证据的量纲不同**：

| | `experiments/cpu/` | `experiments/gpu/` |
|---|---|---|
| 证据层级 | **机制级** —— 分布重构精度、数值等价性、顺序无关性 | **任务级 / 系统级** —— 准确率、通信量、延迟、可扩展性 |
| 硬件 | 任意多核 CPU | A100 / H100 + NCCL |
| 模型权重 | 不需要（合成数据） | 必需（Llama / Qwen / Mistral） |
| 依赖 | torch(CPU) + numpy | torch(CUDA) + NCCL + transformers |
| 单次时长 | 秒 ~ 分钟 | 小时 ~ 天 |
| 能否被对方外推 | ✗ | ✗ |

**为什么不能互相外推。** CPU 侧测的是"紧凑块对注意力分布的重构有多准"，
这是压缩机制的**上界**——即使重构完美，任务指标仍可能退化（检索任务丢失关键
token 就是这样）。反过来，GPU 侧的任务指标退化无法定位到具体环节：可能是
Key 选择、可能是 β、可能是 Value 回归、可能是归约精度。两侧各自回答不同问题，
缺一不可。

---

## 一、CPU 侧（`experiments/cpu/`）

| 文件 | 对应 | 测什么 | 状态 |
|---|---|---|---|
| `e0_order_invariance.py` | E0 | 顺序 / 平衡树 / 随机树 × FP32/FP64 的归并误差 | ✅ 可运行 |
| `e1_interface_shapes.py` | E1 | 形状、索引合法性、可复现性等 11 项不变量 | ✅ 可运行 |
| `e2_fidelity_curve.py` | E2 | ε_mass(B)、ε_out(B) 曲线 + 有符号质量误差 | ✅ 可运行 |
| `e3_edge_conditioning.py` | E3 | H1（KL/JS/Jaccard）+ H2（配对 bootstrap + 置换检验） | ✅ 可运行 |
| `e4_dist_equivalence.py` | E4 | 同步 DCC-KV vs dense 的三档对照 + 多进程 gloo | ✅ 可运行 |
| `e5a_mechanism_ablation.py` | A3（机制级） | 移除 β / 移除 Value 回归各自的重构误差贡献 | ✅ 可运行 |

公共模块：

- `common/synthetic.py` —— 合成场景（多目的端各有关注带，使 H1 可检验）、
  注意力分布度量（KL / JS / 饱和上限 / 有符号质量误差）、dtype 可用性探测
- `common/report.py` —— 统计汇总（median/p5/p95/bootstrap CI）与**配对**检验

```bash
bash experiments/run_cpu.sh            # 全部跑一遍
bash experiments/run_cpu.sh --quick    # 快速冒烟
# 或单独跑
python experiments/cpu/e3_edge_conditioning.py --out results/cpu/e3
```

### 跑 CPU 代码时发现的仓库问题

这些是**运行代码反查出来的**，不是读代码猜的。均未修改 `src/`，
只在实验侧记录并设计绕过/检验方式。

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | `src/dcc_kv_ref/compact_kv.py:101` | `block_mass = A_orig.sum(dim=-1)`，softmax 沿最后一维求和**恒等于 1** | β 的拟合目标无信息量，β 机制近乎空转 |
| 2 | `value_regression.py:72` ÷ M，而 `distributed/dcc_kv_sync_cpu.py:73/151` 用全量 β | **β 的训练/推理口径不一致，相差 M 倍**（默认 M=64） | E3 的 β 模式扫描显示 `full` 模式下 H2 方向在 `strength=8` 时**翻转** |
| 3 | `value_regression.py:57,72` | 设计矩阵 X 只有 M 行，`rank(X) ≤ M`；B > M 时严重欠定 | M=16、B=128 时回归出的 Value 与真值差 94% |
| 4 | `representative_query.py:27,30` | `torch.Generator()` / `torch.randint` 默认在 CPU 建矩阵 | **构造链路在 CUDA 张量上直接报 device mismatch** |
| 5 | `value_regression.py:32` | `torch.eye(B, dtype=X.dtype)` 建在 CPU | 同上，GPU 路径不可用 |

关于第 1–3 条的一个可检验推论：**紧凑 KV 的真正瓶颈是 M（代表 Query 数），
不是 B（预算）。** 这与论文当前把 B 当作主要旋钮的叙述有出入，值得先复核。
第 4–5 条是 GPU 侧 A1/A2/A3 的**硬前置**（见下）。

---

## 二、GPU 侧（`experiments/gpu/`）

| 文件 | 对应 | 测什么 | 前置 |
|---|---|---|---|
| `e5_gpu_ablation.py` | A1/A2/A3/A5 | 通信集大小、预算扫描、组件拆分、异步 vs 同步 | ≥2 卡 + NCCL；A3 另有实现缺口 |
| `e6_main_table.py` | E6 | 主表（模型 × 长度 × 同步 × 方法）+ 可扩展性 | 1 卡 + 权重（dense / kv_budget 可测） |
| `e7_negative_results.py` | E7 | 四个负结果条件的强制报告 | 1 卡 + 权重 + 带 task 标签的评测集 |
| `e8_low_precision.py` | E8 | FP16/BF16 下顺序无关性的失效边界 | GPU 权威；`--device cpu --allow-cpu` 可 reduced 复现 |

公共模块：

- `_env.py` —— 环境闸门（不达标**不产出结果**，退出码 3）、元数据填充、
  逐次计时、CUDA 构造可用性探测
- `_comm.py` —— 变长 All-to-Allv、同步/异步流水、消息体积口径
- `_hf.py` —— 模型加载、KV 预算约束、多项选择打分、prefill 计时

```bash
bash experiments/run_gpu.sh --print-env      # 体检，任何机器可跑
bash experiments/run_gpu.sh --plan           # 打印网格与前置条件，任何机器可跑
bash experiments/run_gpu.sh e8               # E8
bash experiments/run_gpu.sh e5 --nproc 4     # E5
```

### 当前被阻断的部分（不是环境问题，是实现缺口）

| 缺口 | 阻塞了 | 补什么 |
|---|---|---|
| `CompactKV` → GPU attention kernel | A3、E6 的 `dcc_kv` | 一个能把紧凑 K/β/V 喂进 SDPA 的 GPU 通路 |
| 构造链路 CUDA 可用性（上表 #4/#5） | A1/A2 的 GPU 构造口径 | 三处 `device=` 补全 |
| `ring` / `apb` / `fastkv` 的 GPU 实现 | E6 的 3 个基线 | 各自从 CPU 版迁移到 GPU |
| 异步 All-to-Allv 的 GPU 入口 | A5 的真实模型版 | 基于 `_comm.run_async_pipeline` 接真实前向 |

这些项在结果文件里以 `status: "blocked"` 落盘并附 `blockers` 清单，
**不会**以 `null` 或省略的方式静默消失。

### 顺带发现的一处论文算术错误

论文 §6.4 的 E6 写「3 模型 × 5 上下文长度 × 2 GPU 数 × 2 同步模式 × 3 基线
= 144 个数据点」。算不出来：`3×5×2×2×3 = 180`。
反推 `144 / (3×2×2×3) = 4`，即**上下文长度应为 4 档**。
需要二选一：把长度改为 4 档，或把数据点数改为 180。
（`e6_main_table.py --plan` 会把这个检查打印出来。）

---

## 三、目录约定

```
experiments/
├── README.md          本文件：CPU/GPU 分工与现状
├── run_cpu.sh         一键跑全部 CPU 实验
├── run_gpu.sh         一键跑 GPU 实验（torchrun 封装）
├── common/            CPU 侧公共模块
│   ├── synthetic.py
│   └── report.py
├── cpu/               E0–E4 + A3 机制级
└── gpu/               E5–E8（含 _env / _comm / _hf）
```

结果统一落在 `results/`（`results/cpu/**`、`results/gpu/**`），
不纳入版本控制。每次运行同时写 JSON（完整 payload）与 CSV（长表，便于画图）。
