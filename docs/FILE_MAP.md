# 仓库文件说明（FILE MAP）

> 本文逐项说明本仓库每个受控文件的**作用**与**内容**，供接手者、协作者与未来的自己定位代码。
> 统计口径：`git ls-tree -r HEAD`（94 个受控文件 = 58 `.py` / 10 `.tex` / 14 `.md` / 6 `.sh` / 1 `.bib`），
> 不含编译产物、实验产物与本地素材（见 §9）。
> 与 `README.md` 的关系：README 讲"这是什么、怎么跑"，本文讲"每个文件是什么"。

---

## 0. 一分钟导航

| 我想…… | 去哪里 |
|---|---|
| 了解项目主张与边界（哪些**不能**声明） | `README.md` §不主张的内容；`docs/commit_log.md` §P |
| 跑 CPU 可复现实验（机制级） | `experiments/cpu/`，入口 `experiments/run_cpu.sh` |
| 跑 GPU 实验（任务级 / 系统级） | `experiments/gpu/`，入口 `experiments/run_gpu.sh` |
| 看算法参考实现 | `src/dcc_kv_ref/` |
| 看多进程与通信原语 | `src/distributed/` |
| 看基线（Ring / FastKV / APB） | `src/baselines/` |
| 读/改论文 | `paper/main.tex` + `paper/sections/` |
| 查"某个决定为什么这么定" | `docs/commit_log.md`（逐次提交报告） |
| 查投稿前要做什么 | `docs/release_checklist.md` |
| 查当前还有什么没解决 | `docs/commit_log.md` §P.2 |

三条贯穿全仓库的硬约束（读任何文件前先知道）：

1. **不主张**：任务质量、通信性能、多卡可扩展性、与基线的对比 —— 四条都不在本文主张范围内（`README.md`）。
2. **CPU 与 GPU 是两套证据量纲，不可互推**（`experiments/gpu/__init__.py`）。
3. **禁止直接 commit 到 `main`**（`CONTRIBUTING.md`）。

---

## 1. 顶层

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `README.md` | 145 | 项目门面。不主张清单（4 条 ❌）、项目结构树、当前状态（M0/M1 完成、M2 待 GPU）、快速开始、文档导航。末尾"上级文档"指向**仓库外**的 5 份文件（当前缺失，见 §8） |
| `CONTRIBUTING.md` | 178 | 协作规范。分支模型、commit 体例、**禁止直接 commit 到 main**、M2 验收口径（引 blueprint v1.1 §6） |
| `conftest.py` | 52 | pytest 根配置：注册自定义 marker（`gpu` / `distributed` 等）、把仓库根注入 `sys.path` |
| `pytest.ini` | 30 | 测试发现与默认行为。`addopts = -m "not gpu"` ⇒ **GPU 测试默认跳过**；`testpaths = tests` |
| `requirements.txt` | 23 | 依赖清单。⚠️ torch 写 `>=2.6.0`，而 README / CONTRIBUTING 写 `2.3.0+cpu`，**三处不一致**（待决 C5） |
| `.gitignore` | 137 | 忽略规则。Python / pytest / IDE 常规项 + 项目特定：`models/`、`data/`、`results/`、`traces/`、LaTeX 中间产物；末尾有 `!` 白名单例外 |

---

## 2. `src/` —— 参考实现与分布式原语

### 2.1 `src/` 根

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 1 | 仅 docstring，声明包身份 |
| `experiment_metadata.py` | 161 | **实验元数据与结果 schema**。`ExperimentMetadata` 记录模型版本、Transformers / PyTorch / CUDA / NCCL 驱动、硬件、实验 commit hash；结果侧规定报告 median + p5/p95 + bootstrap 95% CI。依据 blueprint §4.1 / §4.2 |

### 2.2 `src/dcc_kv_ref/` —— M0–M1 CPU 参考实现（本文的核心算法）

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 63 | 汇总导出 19 个公开符号（见各模块） |
| `online_softmax.py` | 176 | Online Softmax 状态机与归并算子 ⊕。`OnlineSoftmaxState`、`online_softmax_from_attention`、`merge_softmax_states(_list)`、`attention_output_from_state`、`verify_order_invariance`。**归并要求两侧 `d_v` 相同** |
| `representative_query.py` | 124 | 目的端代表 Query 选取。最远点采样（`farthest_point_sampling`）、Rademacher 投影（`rademacher_projection`）、`select_representative_queries` |
| `key_selection.py` | 64 | 选键。per-token RMS 打分（`rms_per_token_score`）+ `select_topk_keys`。`budget >= L_s` 时短路返回全部键与索引 |
| `calibration.py` | 232 | **β（质量偏置）拟合**。`DEFAULT_BETA_BOUND = 3.0`（箱约束，默认生效）、`DEFAULT_LAMBDA_BETA`、`nonneg_least_squares`（箱约束求解器，非旧版纯 NNLS）、`fit_logit_bias` |
| `value_regression.py` | 98 | 紧凑 V 的岭回归拟合。`ridge_regression_value`、`fit_compact_value` |
| `compact_kv.py` | 134 | **构造链路总入口**。`CompactKV` 数据类（keys / logit_bias / values / selected_indices）+ `build_compact_kv`，串起代表 Query → 选键 → β 拟合 → V 回归 |

> 原创边界：`key_selection` / `calibration` / `value_regression` 三个压缩机制源自 MIT CSAIL 的
> Attention Matching（arXiv 2602.16284），**非本文原创**，正文须显式声明（见 `paper/sections/06-experiment.tex`）。

### 2.3 `src/distributed/` —— 多进程与通信原语

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 17 | 导出 `DistributedComm`、`VarLenMessage`、`launch_dist`、`setup_distributed`、`cleanup_distributed` |
| `comm.py` | 339 | `DistributedComm`：gloo(CPU) / nccl(GPU) 切换对上层透明的统一接口。`VarLenMessage`：变长消息封装 |
| `launch_dist.py` | 280 | 多进程启动器。⚠️ `import datetime` 位于文件末尾（约 L264）但 L100 已使用（待决 C7） |
| `full_attention_cpu.py` | 195 | 精确注意力的 CPU 参考实现（精度上界） |
| `dcc_kv_sync_cpu.py` | 210 | DCC-KV 的 CPU **同步**版。只做数值等价性验证，**没有 GPU attention kernel** —— 这是 A3、E6 的关键阻断点 |
| `var_len_msg.py` | 139 | 变长消息的编码/解码协议 |

### 2.4 `src/baselines/` —— 基线（CPU mock）

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 30 | 导出 `ring_attention_cpu/_dense`、`fast_kv_cpu`、`FastKVConfig`、`apb_cpu`、`APBConfig` |
| `ring_attention_cpu.py` | 115 | Ring Attention（arXiv 2310.01889）CPU mock + 稠密对照 |
| `fast_kv_cpu.py` | 137 | FastKV（共享目的端压缩）CPU mock |
| `apb_cpu.py` | 160 | APB（全网共享 anchor，arXiv 2502.12085）CPU mock |

> ⚠️ 三个基线都是 reference-quality 实现，**只保证接口与数值正确，不做任何性能优化**。
> 真实性能对照需要 GPU 上的高度优化版本；当前**尚无**（E6 的四个多卡方法因此全部被阻断）。

---

## 3. `experiments/` —— 实验代码

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `README.md` | 136 | 实验总览：CPU/GPU 分工、编号体系（E0–E11 / A1–A5）、如何跑 |
| `__init__.py` | 15 | 声明 `experiments` 包 |
| `run_cpu.sh` | 65 | CPU 实验入口 |
| `run_gpu.sh` | 124 | GPU 实验入口（封装 `torchrun`）。⚠️ 数组展开必须用 `"${ARR[@]}"`，**不能**写 `"${ARR[@]:-}"`（后者会退化成空字符串参数） |

### 3.1 `experiments/common/` —— 公共模块（纯 CPU，只依赖 torch + numpy）

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 7 | 声明 `common` 包 |
| `synthetic.py` | 826 | 合成场景生成、完整注意力参考、误差与分布度量（含 KL / JS 等） |
| `report.py` | 203 | 结果汇总：`summarize`（median / p5 / p95 / bootstrap CI）、配对检验、`save_json` / `save_csv` |
| `beta_variants.py` | 652 | β 的各种口径变体与对照（per-key / 标量 / 关闭 β 分解） |

### 3.2 `experiments/cpu/` —— 机制级实验（任意多核机器可复现，无需权重/NCCL）

| 文件 | 行数 | 内容 |
|---|---|---|
| `e0_order_invariance.py` | 277 | 顺序无关性（归并 ⊕ 的交换律/结合律实证） |
| `e1_interface_shapes.py` | 181 | 构造接口的形状与 dtype 契约 |
| `e2_fidelity_curve.py` | 211 | 保真度随预算变化的曲线 |
| `e2b_beta_convention.py` | 435 | **β 口径判定**：系数应为 1；度量必须建在归并侧（源块 + 固定上下文块拼接后再与 Dense 比） |
| `e3_edge_conditioning.py` | 584 | **边级条件化**：20 组配对检验（17 显著 / 3 不显著 / 0 反向）—— H2 机制级证据 |
| `e4_dist_equivalence.py` | 283 | 分布式等价性（单进程 mock） |
| `e5a_mechanism_ablation.py` | 229 | A3 的机制级版本（重构误差口径，对应 GPU 版的任务指标口径） |
| `e9_knob_localization.py` | 389 | **M–B 旋钮定位**：375 格扫描，分离"可达上界（B）"与"能否触及上界（M）" |
| `e10_beta_stability.py` | 656 | **β 稀疏塌缩**的定位与修复；把 β 分解为"块级常数分量（收益）"与"per-key 离散分量（代价）" |
| `e11_lambda_tuning.py` | 774 | λ_β 调参（尺度无关的相对正则强度） |
| `c10_baseline_diagnosis.py` | 379 | 基线诊断：为何 `test_fast_kv_*` / `test_apb_*` 从未被 pytest 收集（包导出缺失 + `budget == L_s` 根本没压缩） |

### 3.3 `experiments/gpu/` —— 任务级与系统级实验（需 A100/H100 + NCCL + 真实权重）

> ⚠️ **本机无 CUDA ⇒ E5–E8 全部未执行**。环境闸门不达标时脚本**不写结果文件**、以退出码 3 结束。
> 因此"代码写完"不等于"实验做了"；仓库内**没有任何 GPU 数值**。

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `README.md` | 97 | GPU 实验说明、入口脚本、环境闸门与 device 修复记录 |
| `__init__.py` | 17 | **声明 CPU/GPU 两套证据的量纲关系**（E3↔A2、A3 机制↔A3 任务、E0↔E8），并明确不可互推 |
| `_env.py` | 618 | 环境闸门（`probe` / `enforce`，不达标退出码 3）、计时（`benchmark_ms`；窗口内只做 `device_sync()`，集合 barrier 留窗口外）、设备绑定（`local_rank_of` / `local_device`，取 `LOCAL_RANK`）、`CUDA_CONSTRUCTION_DEFECTS` 历史清单、元数据构造 |
| `_comm.py` | 437 | 变长 All-to-Allv（阻塞 / 异步）、同步与异步流水线（`run_sync_pipeline` / `run_async_pipeline`）、分块 `_split_chunks`（逐 dst 取片，尺寸自洽）、体积口径（`make_uniform_plan`） |
| `_hf.py` | 436 | HF 模型加载、KV 预算裁剪（`apply_kv_budget`：identity / topk_rms / topk_norm / stride / random）、多项选择打分（`score_choices`，含独立 cache 副本 + 显式 position_ids）、评测（`evaluate`）、prefill 计时、KV 字节数 |
| `e5_gpu_ablation.py` | 609 | **A1** 通信集大小、**A2** 压缩预算扫描、**A3** 组件拆分（被阻断，拒绝产假数字）、**A5** 异步 vs 同步 |
| `e6_main_table.py` | 441 | 主表与可扩展性。`gpu_count` 记为**方法要求**而非自由轴；sync/async 轴对单卡方法**折叠**（不生成两行相同的数） |
| `e7_negative_results.py` | 485 | 负结果四条件：短上下文 / 低预算 / 强 retrieval / batch=1 |
| `e8_low_precision.py` | 335 | 低精度（FP16/BF16）归并算子与失效边界 |

---

## 4. `tests/` —— 测试

`pytest.ini` 默认 `-m "not gpu"`，因此本节的非 `gpu/` 部分应全部通过；`gpu/` 下默认跳过。

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 2 | 声明测试包 |
| `test_smoke.py` | 416 | 冒烟测试：核心 API 的形状与数值 sanity |
| `test_phase_a.py` | 239 | Phase A（多进程 + 通信原语）验收 |
| `test_dist_equivalence.py` | 402 | 分布式等价性（含 2 进程 gloo 真实测试，标记 `distributed`） |
| `test_var_len_msg.py` | 97 | 变长消息协议 |
| `test_experiment_metadata.py` | 100 | 元数据与结果 schema 的字段契约 |
| `test_gpu_pipeline.py` | 400+ | **纯 CPU 的 GPU 侧口径锚点**：分块尺寸自洽、逐 dst 恰好覆盖一次、两条流水线的 sizes 接线、计时窗口不含集合 barrier、`local_device` 用 LOCAL_RANK、E6 的轴折叠规则、`apply_kv_budget` 分块累加的数值等价性。**不需要 GPU，也不需要进程组** |

`tests/gpu/`（默认跳过）：

| 文件 | 行数 | 内容 |
|---|---|---|
| `README.md` | 50 | 如何在目标机上启用这些测试 |
| `test_nccl_basic.py` | 151 | NCCL 基础连通性 |
| `test_async_overlap.py` | 120 | 异步变长通信的 overlap（对应 H4） |
| `test_end_to_end_8b.py` | 109 | 8B 端到端 |
| `profiling/nsys_runner.py` | 86 | Nsight Systems 采集封装 |
| `profiling/torch_profiler_runner.py` | 43 | torch.profiler 采集封装 |

---

## 5. `paper/` —— 论文

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `README.md` | 72 | 编译说明与章节目录 |
| `main.tex` | 445 | 主文件。第 1–4 章正文（引言 / 相关工作 / 问题定义 / 方法）+ 算法伪代码 + 三个 `\input`。⚠️ 作者 / 单位 / 邮箱仍是**占位符**，勿代填 |
| `refs.bib` | 242 | 参考文献 **17 条，全部被 `\cite`**。注释字段只用 `annote`（`note` 会被 bst 排版进参考文献表，中文注释会印进正文） |
| `build.sh` | 70 | 编译入口（**必须 xelatex**：`ctex` 与 `acmart` 冲突，改用 `xeCJK`）。⚠️ `set -e`：出错会**静默中断**，看起来像成功 |
| `fetch_refs.sh` | 74 | 拉取参考文献 PDF 到 `literature/` |
| `.gitignore` | 22 | 忽略 `literature/` 与 LaTeX 中间产物 |
| `sections/05-analysis.tex` | 260 | **第 5 章 复杂度与误差分析**。通信量、计算复杂度、误差分解与误差界、异步流水的理论加速上限 |
| `sections/06-experiment.tex` | 842 | **第 6 章 实验方案**。设置、评测协议与元数据规范、**已验证的 CPU 实验**（E0–E11 / A3 机制级）、**待执行的 GPU 实验**（E5–E8）、结果报告规范 |
| `sections/07-discussion.tex` | 132 | **第 7 章 讨论 + 第 8 章 结论**（两章写在同一文件，`\section{结论}` 在 L88） |
| `figures/preamble.tex` | 36 | 图的公共样式（颜色、字体、tikzlibrary） |
| `figures/fig1_architecture.tex` | 57 | 图 1 架构总览 |
| `figures/fig2_async_pipeline.tex` | 76 | 图 2 异步流水时序 |
| `figures/fig3_error_decomposition.tex` | 47 | 图 3 误差分解 |
| `figures/fig4_comm_scaling.tex` | 48 | 图 4 通信量随设备数变化 |
| `figures/fig5_lambda_curve.tex` | 62 | 图 5 λ 曲线（E11） |

> 图只提交 `.tex`（TikZ 源码）；`.pdf` / `.aux` / `.log` 是编译产物，不入库。

---

## 6. `docs/` —— 过程与规范文档

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `commit_log.md` | 1396+ | **逐次提交报告**，本仓库最重要的过程文档。体例：每条含动机 / 改动清单 / 验证 / 遗留；被推翻的结论**保留原文**并加 `⚠️ 已被 <hash> 修正`，不抹除历史。§P.1 是"已解决的问题"表，§P.2 是"仍未解决的问题"表，§Q 记录易被误读的坑 |
| `git_strategy.md` | 410 | Git 管理策略：分支、commit 体例、tag、实验可追溯 |
| `release_checklist.md` | 155 | 投稿前 / Camera Ready 清单。§4 是 failure_thresholds 校核（H2/H3/H4/H5） |
| `reproducibility.md` | 118 | 可复现性说明。§6 抄录 blueprint §3 的 H1–H5 及其阈值 |
| `ssh_setup.md` | 141 | SSH / 远程机器配置 |
| `FILE_MAP.md` | 本文 | 文件说明（你正在读的这份） |

---

## 7. `scripts/` 与 `.github/`

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `scripts/run_m2_real.sh` | 61 | M2 真实多卡运行脚本 |
| `scripts/run_m3_async.sh` | 70 | M3 异步运行脚本 |
| `.github/workflows/test.yml` | 90 | CI 工作流 |
| `.github/PULL_REQUEST_TEMPLATE.md` | 55 | PR 模板 |
| `.github/ISSUE_TEMPLATE/bug_report.md` | 49 | Bug 报告模板 |
| `.github/ISSUE_TEMPLATE/feature_request.md` | 40 | 功能请求模板 |

---

## 8. 仓库外依赖的文档（当前**缺失**）

`README.md` §上级文档 与 `docs/` 多份文件都引用以下**不在本仓库内**的规划文档。
全盘搜索确认它们既不在仓库内、也不在相邻目录，**需要外部提供**（记账为 C11）：

| 文档 | 被引用处 | 缺失的实际影响 |
|---|---|---|
| `../dcc_kv_plan/research_execution_blueprint_v1.md` | `CONTRIBUTING.md`、`docs/git_strategy.md`、`docs/release_checklist.md`、`docs/reproducibility.md`、`README.md`、`src/experiment_metadata.py`、`tests/gpu/test_async_overlap.py` — 共 9 处 | 它是 §3 failure_thresholds、§4 元数据 schema 的**权威原件**。内容已大体抄入仓库，但**无法做一致性核对**，且已出现下游表述不一致（见下） |
| `../dcc_kv_plan/experiment_matrix.yaml` v1.2.0 | `docs/git_strategy.md`、`docs/release_checklist.md`、`docs/reproducibility.md`、`README.md` | 实验格点的权威清单（哪几档预算、哪几种模型、哪些组合是"规划内"）。当前格点由需求反推，无法核对是否覆盖 |
| `../dcc_kv_plan/references.bib`（README 称 26 条） | `README.md` | 与 `paper/refs.bib`（实际 **17** 条）口径差 9 条。17 条均已引用，不阻塞；但"26"这一数字暂无法证实 |
| `../dcc_kv_plan/contribution_boundary_section.md` | `README.md` | 原创边界声明。其内容已吸收进 `README.md` 的"不主张的内容"节与论文 §6，风险较低 |
| `../dcc_kv_plan/M2_pre_launch_checklist.md` | `README.md` | M2 启动前检查项（**C11 原先漏记的一份**） |

**当前已因此显现的具体危害**：H2 的判据在两处下游文档里写法不同 ——
`docs/reproducibility.md` §6 写"质量提升 ≥ 1.5 个百分点"，`docs/release_checklist.md` §4 写
"prefill 加速 ≥ 1.10×"。二者应是同一假设的两个侧面，但**合并方式无法证实**（两处均已加注说明）。

---

## 9. 不入库的内容

| 路径 | 为什么不在版本控制里 |
|---|---|
| `paper/literature/` | 17 篇参考文献 PDF + `_download_log.txt`，由 `paper/.gitignore` 忽略。**与 `paper/refs.bib` 的 17 条一一对应**（含源论文 `zweiger2026attentionmatching_2602.16284`）。用 `paper/fetch_refs.sh` 重建 |
| `results/` | 实验产物（`results/cpu/`、`results/gpu/`），由根 `.gitignore` 忽略 —— 量大且可重跑 |
| `paper/main.pdf`、`figures/*.pdf`、`*.aux`、`*.log`、`*.bbl` | LaTeX 编译产物，由 `paper/.gitignore` 与根 `.gitignore` 忽略 |
| `.pytest_cache/`、`__pycache__/` | 工具缓存 |
| `models/`、`data/`、`traces/` | 权重、数据集、profiling trace —— 由根 `.gitignore` 显式排除（体积大、可能涉许可） |

---

## 10. 维护约定

- 新增/删除/改名任何文件，**同步更新本文**，并在 `docs/commit_log.md` 追加一条。
- 本文的统计数字以 `git ls-tree -r HEAD | wc -l` 为准，不手写估算。
- 引用 hash 时写完整 7 位；被后续提交推翻的表述不要删，加 `⚠️ 已被 <hash> 修正`。
