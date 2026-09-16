# DCC-KV 代码与文件说明

> 面向第一次接触本仓库的人（合作者、reviewer、后续接手者）。
> 目标：读完能**知道每一块是什么、能不能跑、跑出来什么**，以及**哪些结论还没有证据**。

| 项 | 内容 |
|---|---|
| 仓库 | `github.com/lfei7199-star/dcc-kv` |
| 说明对应版本 | `cc582ae`（分支 `paper/sections-5-8`） |
| 更新日期 | 2026-09-16 |

---

## 0. 这份文档怎么用

| 你的目的 | 直接跳到 |
|---|---|
| 第一次接触项目，想搞清它做什么 | §1、§2 |
| 想马上把代码跑起来 | §2 |
| 想找某个文件是干什么的 | §3 + §4 |
| 想跑实验、找实验脚本 | §5、§8 |
| 想看论文本身 | §7 |
| **想知道哪些还没做、哪些结论没证据** | **§10（最该读的一节）** |
| 想查每个受控文件的逐项说明 | `docs/FILE_MAP.md` |

---

## 1. 项目做什么

**DCC-KV** 是一套**训练无关**的长上下文协同推理框架，用于多设备（多卡 / 多节点）场景：

把每台设备自己持有的 KV cache 压缩成一份「**目的端条件化**」的紧凑表示——由三个机制构成：**Key 选择**（RMS 准则 TopK）、**质量偏置 β**、**Value 回归**——再封装成**变长消息**，通过**异步 All-to-Allv** 与其他设备交换，从而以更低的通信量逼近完整注意力的输出。

**原创边界（重要，不要在汇报或写作中越界）**：

| 属于本文的贡献 | 不属于本文（源自他人工作） |
|---|---|
| 目的端条件化的 KV 通信抽象 | Key 选择、质量偏置 β、Value 回归这三个**压缩机制本身** |
| 变长紧凑 KV 的异步 All-to-Allv 执行路径 | |
| 块级误差界与复杂度分析 | |

压缩机制源自 MIT CSAIL 的 Attention Matching 工作，**非本文原创，论文正文已显式声明**。

**论文中不得主张的内容**（写在 `README.md` 里作为硬约束）：任务质量、通信性能、多卡可扩展性、与基线的对比——这四项都要等 GPU 实验，目前**没有任何一项有任务级证据**。

---

## 2. 五分钟上手

### 2.1 环境

```bash
git clone https://github.com/lfei7199-star/dcc-kv.git
cd dcc-kv
pip install -r requirements.txt
```

本机（无 CUDA）即可完成下面全部步骤。**GPU 相关的一切都需要真实 NVIDIA GPU + NCCL，不接受 CPU 退化。**

### 2.2 跑测试

```bash
pytest -q
```

期望输出：`226 passed / 7 deselected / 2 xfailed`。

- `7 deselected` = 被 `pytest.ini` 按 marker 排除的 **GPU 需求测试**（正常现象）；
- `2 xfailed` = 已知预期失败（记录在案的边界），**不是回归**。

### 2.3 跑一个 CPU 实验

```bash
python experiments/cpu/e13_bound_tightness.py --out results/cpu/e13
```

产出落在 `results/cpu/e13/`：一个 `.json`（完整数据 + 元数据）和若干 `.csv`（论文引用的是这些）。

### 2.4 查看 GPU 实验为什么还不能跑

```bash
python experiments/gpu/e6_main_table.py --plan
```

`--plan` 是**只规划、不执行**：它会打印出哪些方法可用、哪些被阻断，以及阻断原因。这是"现在不能租卡开跑"的直接证据，不需要 GPU 就能复现。

### 2.5 编译论文

```bash
bash paper/build.sh
```

需要 TeX Live（含 `xelatex` + `bibtex`，中文需 `xeCJK`）。

---

## 3. 目录总览

| 路径 | 职责 | 本机能否跑 |
|---|---|---|
| `src/` | 方法与基线实现 | ✅ |
| `experiments/` | 实验脚本（CPU / GPU 分开） | CPU ✅ / GPU ❌ |
| `results/` | 实验产物（**已入库，随仓库走**） | — |
| `tests/` | 测试（CPU + GPU 两组） | CPU ✅ / GPU ❌ |
| `paper/` | 论文源文件 | ✅（需 TeX Live） |
| `docs/` | 项目文档 | — |
| `reports/` | 对外汇报与本说明 | — |
| `scripts/` | 少量 shell 启动脚本 | 视内容 |
| `traces/` | profiling 输出目录（当前为空） | — |
| `.github/workflows/` | CI | — |

**两个关键设计约定**：

1. **`results/` 随仓库提交**（不只是 `.gitignore` 掉的本地产物）——租卡买来的数据一旦丢失就无法找回，因此实验产物必须是受版本控制的。
2. **CPU 与 GPU 两套证据不可互推**。CPU 侧是机制级、合成数据；GPU 侧是任务级、真实模型。量纲与结论层级都不同，不能把 CPU 的数字当成 GPU 结论的替代。

---

## 4. 核心代码：`src/`

### 4.1 `src/dcc_kv_ref/` —— 方法参考实现

这是论文方法的核心，7 个模块：

| 文件 | 对应方法步骤 | 内容 |
|---|---|---|
| `representative_query.py` | M1.1 | 代表 Query 选择：Rademacher 投影 + 最远点采样 |
| `key_selection.py` | M1.2 | Key 选择：基于 RMS 的 TopK |
| `calibration.py` | M1.3 | 质量偏置 β 的拟合：非负岭回归 |
| `value_regression.py` | M1.4 | Value 回归：ridge regression |
| `compact_kv.py` | M1.5 | `CompactKV` 数据结构 + 顶层构造入口 `build_compact_kv` |
| `online_softmax.py` | M0 | Online Softmax 状态与归并算子 |

**入口就是 `build_compact_kv`**：给定源端 KV、目的端 Query、预算 B，产出一份紧凑 KV。想复现论文的任意一条保真度曲线，从这里进。

### 4.2 `src/distributed/` —— 分布式通信

| 文件 | 内容 |
|---|---|
| `comm.py` | 通信抽象层 |
| `full_attention_cpu.py` | 完整分布式注意力的 CPU 参考实现（精度上界参照） |
| `dcc_kv_sync_cpu.py` | DCC-KV 同步版的 CPU 参考实现 |
| `var_len_msg.py` | **变长消息**序列化的 CPU 实现（本框架的关键设计之一） |
| `launch_dist.py` | 统一启动器：单进程直接调用 / 多进程走 `mp.spawn` |

> ⚠️ **一个踩过的坑**：多进程路径不能用局部闭包作为 spawn 目标（`mp.spawn` 要求目标函数可 pickle）。曾有一次失败被误归因为「Windows 环境限制」，真实异常是 `Can't get local object 'launch_dist.<locals>.wrapped_fn'`。**归因"环境限制"之前必须先读到真实异常文本。**

### 4.3 `src/baselines/` —— 三个基线的 CPU mock

| 文件 | 基线 |
|---|---|
| `fast_kv_cpu.py` | FastKV + Ring Attention |
| `ring_attention_cpu.py` | Ring Attention 参考实现 |
| `apb_cpu.py` | APB |

**注意**：这三个是 **CPU mock**，用于机制级对照。它们在 GPU 上**没有**对应实现——这正是主表被阻断的原因之一（见 §10）。

### 4.4 `src/experiment_metadata.py`

实验元数据与结果 schema。**每个实验产物都应带完整元数据**（commit hash、硬件、软件版本、随机种子），否则数字无法追溯。

---

## 5. 实验代码：`experiments/`

### 5.1 `experiments/common/` —— 公共层（改实验前先读这里）

| 文件 | 职责 | 重要度 |
|---|---|---|
| `hypotheses.py` | **H1–H5 的阈值与判定：单一事实源** | ★★★ |
| `synthetic.py` | 合成数据生成 + 全部误差度量 | ★★★ |
| `beta_variants.py` | β 链路的多种口径变体，用于判定哪种约定合理 | ★★ |
| `report.py` | 结果聚合与落盘（对齐统计规范：中位数 + 分位 + 置信区间） | ★★ |

**关于 `hypotheses.py`（必须遵守）**：

- H1–H5 的**阈值与判定逻辑只写在这一个文件里**，任何脚本都必须调 `h1_pass` / `h2_pass` … 这类函数；
- 脚本内**不得出现魔法数字**（例如凭空写一个 `0.5` 当阈值）；
- 这条约定有**源码级断言测试**守着（`tests/test_hypotheses.py`）。

之所以定得这么死：曾出现过脚本绕过统一入口、自己写了个裸阈值的情况——**"阈值只写一处"靠自觉是守不住的，必须靠测试。**

### 5.2 `experiments/cpu/` —— 15 个 CPU 实验（本机可跑）

| 脚本 | 实验 | 一句话结论 / 用途 |
|---|---|---|
| `e0_order_invariance.py` | E0 | Online Softmax 归并的**顺序无关性**（含置换次数收敛阶梯 + 归并树形状对照） |
| `e1_interface_shapes.py` | E1 | 紧凑 KV 构造的接口与形状不变量 |
| `e2_fidelity_curve.py` | E2 | 压缩保真度曲线 ε_mass(B) 与 ε_out(B) |
| `e2b_beta_convention.py` | E2b | β 链路的**口径判定**（哪一种约定最合理） |
| `e3_edge_conditioning.py` | E3 | **边级条件化 vs 共享压缩**（默认留出协议；支持 `--protocol in-sample` 复现旧口径） |
| `e4_dist_equivalence.py` | E4 | 分布式等价性：同步 DCC-KV vs 完整注意力（gloo） |
| `e5a_mechanism_ablation.py` | A3 机制级 | 组件消融 |
| `e9_knob_localization.py` | E9 | 旋钮定位：信息瓶颈在预算 B 还是代表 Query 数 M |
| `e10_beta_stability.py` | E10 | β 的**稀疏塌缩**：判定、根因与修复 |
| `e11_lambda_tuning.py` | E11 | β 岭正则 `lambda_beta` 的扫描与默认值确定 |
| `e12_representative_query.py` | E12 | 投影维度 `d_p` 的扫描 |
| `e13_bound_tightness.py` | E13 | **误差界的数值紧致度检验** |
| `c10_baseline_diagnosis.py` | C10 | 两项压缩基线测试失败的**根因定位** |
| `probe_e3_heldout.py` | — | 一次性判定探针（保留以便复现） |

运行方式统一：

```bash
python experiments/cpu/e3_edge_conditioning.py --out results/cpu/e3
```

**读实验结论时必看的两件事**：

1. **中位数是哪种**：本仓库的 `_med` 取 `vals[n//2]`，是**上中位数**，与 `statistics.median` 在偶数样本下**不一致**（实测差 0.0036）。论文表格已按此口径声明。
2. **误差是哪个口径**：`synthetic.py` 里有多个度量并存——`absolute_mass_error`（公共偏移）、`relative_output_error`（单块归一）、`mixture_relative_error`（归并侧），以及一个**已废弃**的 `mass_error`。**引用任何误差数字前先确认口径**。

### 5.3 `experiments/gpu/` —— GPU 实验（本机不能跑）

| 文件 | 内容 |
|---|---|
| `_env.py` | 环境闸门、计量与元数据工具。**闸门不达标就不产结果**（退出码 3），不会偷偷用 CPU 退化凑数 |
| `_comm.py` | 通信原语与 bench：变长 All-to-Allv 及其同步/异步流水 |
| `_hf.py` | 真实模型后端（HF）：任务指标与 prefill 性能测量 |
| `e5_gpu_ablation.py` | E5：消融 A1 / A2 / A3 / A5（A4 设计已定、代码待写） |
| `e6_main_table.py` | E6：主表与可扩展性 |
| `e7_negative_results.py` | E7：负结果与适用边界 |
| `e8_low_precision.py` | E8：低精度下的顺序无关性失效边界 |

**`_env.py` 的两条关键纪律**（都是踩过坑换来的）：

1. **全局 rank 绝不能当设备序号用**。多节点下第 2 个 8 卡节点的全局 rank 是 8..15，对应本地 `cuda:0..7`。所有设备索引必须走统一入口，不能各自算。
2. **计时窗口里不允许出现集合通信，也不允许 device-wide 同步**。异步路径如果在 `wait(handle)` 之后再做一次全设备同步，会把正在飞的下一块一起等掉 ⇒ overlap 恒为 0、**实测加速比恒为 1.0**，这种 run 会白花一整天。相应地：异步路径的 `comp_ms` 是**上界**，只有 `total_ms` 在同步/异步之间可比。

### 5.4 计时必须拆分

所有计时都要拆成四段分别报：`T_build`（构造）/ `T_comm`（通信）/ `T_comp`（计算）/ `T_total`。其中 `T_build` 若走 CPU 构造，会包含 PCIe 传输时间，**这种 run 不可与 GPU 构造的 run 混报**。

---

## 6. 测试：`tests/`

| 文件 | 内容 |
|---|---|
| `test_hypotheses.py` | H1–H5 阈值表的锚点（37 项）：把口径冻成可执行断言 |
| `test_gpu_pipeline.py` | E5/A5 流水线分块口径的单元测试（**纯 CPU**，121 项，不需要 GPU） |
| `test_smoke.py` | 接口验证 |
| `test_dist_equivalence.py` | 分布式注意力数值等价性 |
| `test_e0_order_invariance.py` | E0 置换次数轴的守卫（13 项）：**这条轴被删掉就必须失败** |
| `test_var_len_msg.py` | 变长消息序列化 |
| `test_phase_a.py` | Phase A 骨架验收 |
| `test_experiment_metadata.py` | 元数据与结果 schema |
| `tests/gpu/**` | GPU 需求测试（默认被排除，需真实 GPU） |

**测试的定位不只是"防回归"**：本仓库用测试来**冻结口径**——阈值、"质量相近"的定义、哪个轴必须存在、哪些参数不许有默认值，都写成断言。因为「口径」这类东西一旦只写在文档里，代码就会漂移。

---

## 7. 论文：`paper/`

| 路径 | 内容 |
|---|---|
| `main.tex` | 主文件；标题/摘要 + 前 4 章的入口 |
| `sections/05-analysis.tex` | 第 5 章：复杂度与误差分析（含误差界式(37)、三条性质） |
| `sections/06-experiment.tex` | 第 6 章：实验方案与结果（含度量口径表） |
| `sections/07-discussion.tex` | 第 7 章：讨论（含「质量相近」判定小节）+ 第 8 章结论 |
| `refs.bib` | 参考文献 17 条，全部被引用 |
| `figures/fig1..fig5*.tex` | 5 张图，全部为 **TikZ 源**（只提交 `.tex`，PDF 是构建产物） |
| `build.sh` | 编译脚本（xelatex ×3 + bibtex） |

**排版注意事项（改完必须复核）**：

- 栏右边界实测众数为 **558.2 pt**（双栏）。判断是否溢出**不能**只看整行 x1：以中文标点（。，、；：）】》）结尾的行是**中文标点悬挂**，属正常排版，不算溢出。**正确判据是看"最后一个非标点 span 的 x1"**——曾有一行整行 x1 = 566.08 pt 看着超了，但文本 span 止于 557.11，只有逗号字形外挂。
- 长 `\texttt` 标识符不可断行，会造成真实溢出，需插 `\allowbreak`。

---

## 8. 结果产物：`results/`

```
results/
├── cpu/           # 25 个子目录/文件，每个实验一个
│   ├── e0/ ... e13/       # 各实验的 .json + .csv
│   ├── e3_in_sample/      # E3 的**样本内**旧口径产物（保留作对照，勿当结论引用）
│   ├── a3/  c10/          # 机制级消融与基线诊断
│   ├── *_run.log          # 运行日志
│   └── smoke/             # 快速冒烟产物
└── gpu/
    └── e8_cpu_reduced/    # E8 的 CPU 缩减版
```

**约定**：

- 每个实验目录里，`.json` 是完整数据 + 元数据，`.csv` 是论文引用的表格；
- `results/` 下的 `.csv` / `.json` / `.log` **全部**受版本控制；
- `e3_in_sample/` 是**对照产物**（旧口径），引用时必须声明它是样本内评估，**不可与留出结果混报**。

---

## 9. 文档：`docs/`

| 文件 | 作用 |
|---|---|
| `FILE_MAP.md` | **每个受控文件的逐项说明**（增删 / 改名文件必须同步它） |
| `commit_log.md` | 按提交倒序的变更日志：动机 / 改动 / 验证 / 遗留。被推翻的表述**保留原文 + 加更正标注**，不抹除 |
| `writing_scope_and_metrics.md` | 写作范围与误差度量口径的裁定 |
| `reproducibility.md` | 硬件要求、复现命令、统计规范 |
| `gpu_execution_plan.md` | **GPU 实验执行计划与预算控制**（含四级梯队与终止判据） |
| `release_checklist.md` | 投稿 / Camera Ready 前的逐项清单 |
| `git_strategy.md` | 仓库管理与提交规范 |
| `ssh_setup.md` | 目标机 SSH 配置 |

**两条台账纪律**：

1. `commit_log.md` **每次提交都要追加**一条；
2. 一次性脚本若承载过结论，**必须补记进 `commit_log.md`**——曾有一个 432 行的一次性补丁用完即删、从未进 git 历史，导致结论无法追溯。

---

## 10. 已知缺口与诚实边界

> **这一节最该读。** 它决定你能引用什么、不能引用什么。

### 10.1 结论层面：哪些已经成立、哪些还没有

| 结论 | 状态 | 证据层级 |
|---|---|---|
| 归并顺序无关性（E0） | ✅ 已量化 | CPU 机制级 |
| 误差界式(37) 成立且紧致度已测（E13） | ✅ 已测 | CPU 机制级 |
| 目的端条件化的优势（E3） | ⚠️ **有条件成立**：只在目的端确有可分关注区域时（强度 0 时无优势） | CPU 合成数据 |
| `d_p ≥ 8` 饱和（E12） | ✅ 已测 | CPU 机制级 |
| λ_β 默认值 3e-2（E11） | ✅ 有公开判据支撑 | CPU 机制级 |
| **任务质量 / 通信性能 / 多卡可扩展性 / 与基线对比** | ❌ **全部无任务级证据** | 待 GPU |

### 10.2 代码缺口（租卡前必须补完）

| # | 缺口 | 阻塞了 |
|---|---|---|
| G1 | `CompactKV` → GPU attention kernel 的通路 | E5-A3、E6 的本文方法 |
| G2 | 异步 All-to-Allv 接真实前向的 GPU 入口 | E5-A5（真实模型版） |
| G3 | `ring` / `apb` / `fastkv` 的 GPU 实现 | E6 的三个基线 |
| G4 | A4 的代码（设计已定） | E5-A4 |
| G5 | 评测集 JSONL 转换器 | E5-A2/A3、E6、E7 的**准确率** |
| G6 | CUDA 构造可用性实测 | **只能上机才知道**（其余五项都能本地写完并静态验证） |

### 10.3 假设判定状态

H1–H5 中：**H1 与 H4 已判定**；**H2 / H3 / H5 为 `no-judge`**——意思是**判据尚未接通**，而不是"结论未定"：

- H2 缺 GPU 上才能测得的噪声底线与退化量（缺口已从「定义缺失」降为「参数待测」）；
- H3 待 A3 的 GPU 通路；
- H5 待"设备数翻倍"基准点的定义。

### 10.4 引用数字时的三条禁令

1. **CPU 的数字不能当作 GPU 结论**。两套证据量纲不同（机制级 vs 任务级、合成 vs 真实模型）。
2. **确认误差口径再引用**。单块版与跨块版不可互比、不可互相代入误差界。
3. **不加条件的优势陈述不成立**。E3 的结论必须带限定语「当目的端确有可分关注区域时」。

---

## 附：最短路径速查

```bash
pytest -q                                       # 226 passed / 7 deselected / 2 xfailed
python experiments/gpu/e6_main_table.py --plan  # 看 GPU 实验阻断清单
python experiments/cpu/e3_edge_conditioning.py --out results/cpu/e3
bash paper/build.sh                             # 编译论文
```

配套：`reports/2026-09-16-progress-report.md`（对外汇报）、`docs/FILE_MAP.md`（逐文件明细）。
