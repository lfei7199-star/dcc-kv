# 仓库文件说明（FILE MAP）

> 本文逐项说明本仓库每个受控文件的**作用**与**内容**，供接手者、协作者与未来的自己定位代码。
> 统计口径：`git ls-tree -r HEAD`（**184** 个受控文件 = 77 `.py` / 31 `.csv` / 26 `.json` /
> 21 `.md` / 10 `.tex` / 6 `.log` / 6 `.sh` / 1 `.bib`，另 6 个为 `.gitattributes` /
> `.gitignore` ×2 / `pytest.ini` / `requirements.txt` / `.github/workflows/test.yml`），
> 不含编译产物与本地素材（见 §9）。其中 **63 个是 `results/` 实验产物 —— 自 2026-09-16 起入库**
> （决定与体积策略见 §9 与 `docs/git_strategy.md` §8.1）。
> 与 `README.md` 的关系：README 讲"这是什么、怎么跑"，本文讲"每个文件是什么"。
> **2026-09-20 新增 11 个受控文件**：G1 算子核、G3 基线向量化算子、G2 桥接层、G5 评测集转换器，以及对应的 7 个锚点文件（含 2026-09-18 自查与 2026-09-20 对抗性审查各一份）。
> **2026-09-21 新增 2 个受控文件**：`src/distributed/attention_hook.py`（H0 的方法侧 —— 把 G1 的算子核挂进真实模型 attention 的钩子）与其锚点 `tests/test_attention_hook.py`。同日一并修了两处实现缺陷（见下）并把 I 组守卫的扫描面从 `tests/test_*.py` 扩到**仓库内任意 `.py` 引用**。

> ⚠️ **2026-09-21 第三次更新（`b44a7b3`）**：**未新增任何受控文件**（仓库仍是 180 个），
> 只改既有 9 个。内容是「把 H0 的方法侧钩子**接进** E6 的 `dcc_kv` 行」+ 三个接线才
> 看得见的口径修正。同轮**修掉本文档自身的一处遗留缺陷**：§2.2 / §2.4 / §2.5 曾有
> **4 行表格只写了 ``| `xxx.py` | 483 | **G1`` 就断掉**（`attention_kernel.py` /
> `operators.py` / `_forward.py` / `build_eval_set.py`），自 `c9ca6b0` 引入起一直
> 如此 —— 而本文档的职责恰是**逐项说明**每个受控文件，一行什么都不说明的记录比没有
> 更坏（读者会以为已经写过了）。已补齐，并加了一条守卫
> （`tests/test_adversarial_2026_09_20.py::test_i4_*`：`docs/*.md` 里不得有未闭合的表格行）。
> 另：本表行数字段**普遍陈旧**，本轮只把被改动的那些按实测刷新（见下），其余登记备查。
> ⚠️ **2026-09-21 第四次更新（`12ece9d`）**：行数字段按实测刷新 11 处
> （`hypotheses.py` / `_hf.py` / `e6_main_table.py` /
> `test_adversarial_2026_09_20.py` / `commit_log.md` / `reproducibility.md` /
> `gpu_execution_plan.md` / `writing_scope_and_metrics.md` / `release_checklist.md` /
> `code-and-files-guide.md` / `experiments/README.md`）。
> **本文件的「行数」字段在历次局部更新中会累积陈旧值** —— 每轮只刷新当时被改动的
> 那些，其余保持旧值不变，故引用前请以文件实际内容为准。
> ⚠️ **2026-09-22 第五次更新（`39063e7`）**：**新增 4 个受控文件**（仓库 180 → 184）——
> `reports/2026-09-22-theory-code-consistency-audit.md`、
> `reports/2026-09-22-lambda-zero-and-interval-verdicts.md` 与
> `results/cpu/e9_L2048/{rows.csv,summary.json}`（故 `results/` 计数 61 → 63）。
> 同轮按实测刷新 3 处行数（`representative_query.py` / `beta_variants.py` /
> `test_dist_equivalence.py`，见 §2.2 与 §4），并修正该轮报告自身的一处行数误记
> —— 原稿把 `git diff --stat` 的**总变更行数**（54 / 45 / 53）当成了「+行数」，
> 且 `representative_query.py` 计数含了一个随后删掉的空行（PEP8 E303）。
> 三处均已改成与 `git diff --numstat` 逐项对齐的实测值。


---

## 0. 一分钟导航

| 我想…… | 去哪里 |
|---|---|
| 了解项目主张与边界（哪些**不能**声明） | `README.md` §不主张的内容；`docs/commit_log.md` §P |
| 跑 CPU 可复现实验（机制级） | `experiments/cpu/`，入口 `experiments/run_cpu.sh` |
| 跑 GPU 实验（任务级 / 系统级） | `experiments/gpu/`，入口 `experiments/run_gpu.sh` |
| 规划租卡跑实验、控制预算 | **`docs/gpu_execution_plan.md`**（结论：当前**仍**不能开跑 —— 缺的不是卡，是代码。G1–G5 已提交，**H0 已于 2026-09-21 接上**；剩 **G6** 与「`dcc_kv` 构造路径独立」；含 S0–S3 梯队） |
| 看算法参考实现 | `src/dcc_kv_ref/` |
| 看多进程与通信原语 | `src/distributed/` |
| 看基线（Ring / FastKV / APB） | `src/baselines/` |
| 读/改论文 | `paper/main.tex` + `paper/sections/` |
| 查"某个决定为什么这么定" | `docs/commit_log.md`（逐次提交报告） |
| 查投稿前要做什么 | `docs/release_checklist.md` |
| 查当前还有什么没解决 | `docs/commit_log.md` §P.2 |
| **给合作者/师姐的进展汇报与代码导航** | **`reports/`**：`2026-09-16-progress-report.md`（进展 + 待办）、`code-and-files-guide.md`（代码与文件说明） |

三条贯穿全仓库的硬约束（读任何文件前先知道）：

1. **不主张**：任务质量、通信性能、多卡可扩展性、与基线的对比 —— 四条都不在本文主张范围内（`README.md`）。
2. **CPU 与 GPU 是两套证据量纲，不可互推**（`experiments/gpu/__init__.py`）。
3. **禁止直接 commit 到 `main`**（`CONTRIBUTING.md`）。

---

## 1. 顶层

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `README.md` | 148 | 项目门面。不主张清单（4 条 ❌）、项目结构树、当前状态（M0/M1 完成、M2 待 GPU）、快速开始、文档导航。末尾"上级文档"指向**仓库外**的 5 份文件（当前缺失，见 §8） |
| `CONTRIBUTING.md` | 178 | 协作规范。分支模型、commit 体例、**禁止直接 commit 到 main**、M2 验收口径（引 blueprint v1.1 §6） |
| `conftest.py` | 65 | pytest 根配置：注册自定义 marker（`gpu` / `distributed` 等）、把仓库根注入 `sys.path`。**按名自动打 marker 时取函数名本体**（`item.originalname`）而非 `item.name` —— 后者**含 parametrize id**，会把`@parametrize("sub", [..., "tests/gpu"])` 这种“以 gpu 为审查对象”的 CPU 守卫**静默排除**，而全量测试仍报全绿（2026-09-20 对抗性审查 D11） |
| `pytest.ini` | 30 | 测试发现与默认行为。`addopts = -m "not gpu"` ⇒ **GPU 测试默认跳过**；`testpaths = tests` |
| `requirements.txt` | 23 | 依赖清单。⚠️ torch 写 `>=2.6.0`，而 README / CONTRIBUTING 写 `2.3.0+cpu`，**三处不一致**（待决 C5） |
| `.gitattributes` | 19 | **行尾策略**（2026-09-16 新增）：固定 `*.sh` / `*.py` 为 LF。本机 `core.autocrlf=true`，索引侧虽始终是 `i/lf`（Linux clone 不受影响），但工作区有 36 个 `.py` + 2 个 `.sh` 是 CRLF —— 直接拷到 Linux 会 `bad interpreter`，而这种失败本地看不出来 |
| `.gitignore` | 142 | 忽略规则。Python / pytest / IDE 常规项 + 项目特定：`models/`、`data/`、`traces/`、LaTeX 中间产物。**`results/` 自 2026-09-16 起不再忽略**（改为 `!results/**` 显式放行；该规则必须排在 LaTeX 段的 `*.log` 之后，否则 `results/cpu/*_run.log` 仍会被挡掉）；末尾有 `!` 白名单例外 |

---

## 2. `src/` —— 参考实现与分布式原语

### 2.1 `src/` 根

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 1 | 仅 docstring，声明包身份 |
| `experiment_metadata.py` | 173 | **实验元数据与结果 schema**。`ExperimentMetadata` 记录模型版本、Transformers / PyTorch / CUDA / NCCL 驱动、硬件、实验 commit hash；结果侧规定报告 median + p5/p95 + bootstrap 95% CI。依据 blueprint §4.1 / §4.2。2026-09-20：`lambda_beta` 默认值改为引 `DEFAULT_LAMBDA_BETA`（原硬编码 1e-3 与项目默认 3e-2 **不同源**），并补 `warmup` / `iters` 字段使重复次数可追溯 |

### 2.2 `src/dcc_kv_ref/` —— M0–M1 CPU 参考实现（本文的核心算法）

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 82 | 汇总导出 **27** 个公开符号（见各模块） |
| `online_softmax.py` | 176 | Online Softmax 状态机与归并算子 ⊕。`OnlineSoftmaxState`、`online_softmax_from_attention`、`merge_softmax_states(_list)`、`attention_output_from_state`、`verify_order_invariance`。**归并要求两侧 `d_v` 相同** |
| `representative_query.py` | 142 | 目的端代表 Query 选取。最远点采样（`farthest_point_sampling`）、Rademacher 投影（`rademacher_projection`）、`select_representative_queries`。**2026-09-22（`39063e7`）**：FPS 起点默认 `newest`（服从论文 eq.(11)「以最新 Query 为初始锚点」，0-based 的 `N-1`），旧行为留在 `start="random"`；复杂度改为**增量维护 `min_dist`**，不再预计算 `N×N` 相似度矩阵（`L_r=10^5` 时需 37 GB），与 §4.2 声明的 `O(M·L_r·d_p)` 一致 |
| `key_selection.py` | 64 | 选键。per-token RMS 打分（`rms_per_token_score`）+ `select_topk_keys`。`budget >= L_s` 时短路返回全部键与索引 |
| `calibration.py` | 232 | **β（质量偏置）拟合**。`DEFAULT_BETA_BOUND = 3.0`（箱约束，默认生效）、`DEFAULT_LAMBDA_BETA`、`nonneg_least_squares`（箱约束求解器，非旧版纯 NNLS）、`fit_logit_bias` |
| `value_regression.py` | 98 | 紧凑 V 的岭回归拟合。`ridge_regression_value`、`fit_compact_value` |
| `compact_kv.py` | 134 | **构造链路总入口**。`CompactKV` 数据类（keys / logit_bias / values / selected_indices）+ `build_compact_kv`，串起代表 Query → 选键 → β 拟合 → V 回归 |
| `attention_kernel.py` | 514 | **G1**：把 `CompactKV` 送进注意力的算子核（device-agnostic）。这是补齐 GPU 侧缺口的**第一块** —— 此前"紧凑 KV 算 attention"只以逐 query 的 Python 循环存在于 `src/distributed/dcc_kv_sync_cpu.py`（CPU、单 head、只做数值等价性验证），既不能批量化也跑不了 GPU。接口：`causal_visibility`（只认 `selected_indices`，**不**按块/锚点数组的行序切片）、`identity_compact`（B = L_s 的参照块、β≡0）、`compact_kv_attention`、`dense_attention`、`merge_partial_attention`、`dcc_kv_attention`（G2 与 H0 钩子共用的前向入口）。三条"形状对、量级对、只算错"的约束：SDPA 加性掩码的 ndim ≤ query 的 ndim；跨块归并必须等于"拼接后一次算完"（这条不成立则 A3/A4/E6 的 `dcc_kv` 行全错、而且错得很安静）；`query_chunk` 只到 ULP 级、非逐位。**2026-09-21 补（`b44a7b3`）**：`dcc_kv_attention` 的因果策略拆成**两个独立开关**（`local_causal` / `remote_causal`），使「源块全可见 + 本地因果」这一组合可表达（H0 的钩子要的正是它）；不给开关时按 `query_positions` 推导，旧调用点行为**逐位不变** |

> 原创边界：`key_selection` / `calibration` / `value_regression` 三个压缩机制源自 MIT CSAIL 的
> Attention Matching（arXiv 2602.16284），**非本文原创**，正文须显式声明（见 `paper/sections/06-experiment.tex`）。

### 2.3 `src/distributed/` —— 多进程与通信原语

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 17 | 导出 `DistributedComm`、`VarLenMessage`、`launch_dist`、`setup_distributed`、`cleanup_distributed` |
| `comm.py` | 367 | `DistributedComm`：gloo(CPU) / nccl(GPU) 切换对上层透明的统一接口。`VarLenMessage`：变长消息封装。`all_to_all_v` 的 `recv_sizes` **声明与对端实际发送量不符即抛 `ValueError`**（不静默纠正） |
| `launch_dist.py` | 280 | 多进程启动器。⚠️ `import datetime` 位于文件末尾（约 L264）但 L100 已使用（待决 C7） |
| `full_attention_cpu.py` | 195 | 精确注意力的 CPU 参考实现（精度上界） |
| `dcc_kv_sync_cpu.py` | 210 | DCC-KV 的 CPU **同步**版。只做数值等价性验证，**没有 GPU attention kernel** —— 这是 A3、E6 的关键阻断点 |
| `var_len_msg.py` | 139 | 变长消息的编码/解码协议 |
| `attention_hook.py` | 844 | **H0 的方法侧**（2026-09-21）：把 G1 的算子核挂进真实模型 attention 的钩子。要点：注册**专用键**而非覆盖 `eager`/`sdpa`（后者一旦用 `pop` 还原就是把 transformers 自带实现删掉，不是还原）；source / destination 两阶段由调用方**显式声明**（不许靠「past 是不是 None」去猜 —— 猜错时产物看着完全正常）；目的端用 state 里的源端 K/V，**不是**传进来的 key（`cache.update` 在钩子之前，传进来的是「完整 cache + 本次新 token」）；返回值必须转置回 `[B, Lq, H_q*D_v]`；GQA 分组按 `repeat_interleave` 连续切；**不吃 `attention_mask`**（整列被遮蔽即抛错，否则 padding 会被当真 key）；`dcc_world` 单卡模拟**不体现代价化**（summary 里如实记 False 并附原因）。本机用本地构造的 tiny Llama 端到端验证：dense 参照臂与原生前向差 9.7e-08 | **2026-09-21 同日重写（`b44a7b3`，597→844 行）**：修掉两个被接线撞出来的契约漏洞 —— ① 本地块契约改为「传入的 cache **只含目的端本地 KV**」（旧实现取尾部 Lq，会丢掉评测路径的目的端本地上下文），并加**指纹守卫** `_assert_local_cache_excludes_source`（传入 cache 的前缀与 state 里源端 K 的 4 行采样逐位相同即抛错）；② `destination_phase(compacts=...)` 的 `compacts` 改为**必填**（`build` / `reuse`），杜绝「拿被打分的 token 去条件化」。另加 `budget_ratio`（评测源长逐样本变化）、`query_chunk`、`resolve_budget` / `edge_budget` / `describe`，并把因果策略交给 `K.dcc_kv_attention(local_causal=True, remote_causal=False)` |

### 2.4 `src/baselines/` —— 基线（CPU mock）

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 51 | 两组导出：**参考实现**（逐 query，语义权威，**不可用于计时**）`ring_attention_cpu/_dense` / `fast_kv_cpu` / `apb_cpu`；**可上机**（向量化，G3）`ring_attention` / `fastkv_attention` / `apb_attention`；另 `FastKVConfig` / `APBConfig` |
| `ring_attention_cpu.py` | 115 | Ring Attention（arXiv 2310.01889）CPU mock + 稠密对照 |
| `fast_kv_cpu.py` | 137 | FastKV（共享目的端压缩）CPU mock |
| `apb_cpu.py` | 160 | APB（全网共享 anchor，arXiv 2502.12085）CPU mock |
| `operators.py` | 334 | **G3**：三个基线的**算子化**实现（device-agnostic，可 GPU）：`ring_attention` / `fastkv_attention` / `apb_attention` + `FastKVConfig` / `APBConfig`。改写的理由不是"慢一点"：逐 query 循环在 L=32k 时是 3 万多次核启动，测出来的是**启动开销**而不是算子代价，拿它和 DCC-KV 的向量化路径比就是比实现质量；而且归并次序由人手写，基线之间的差异里会混进归并伪影。语义与 CPU 参考**逐条对齐**，含三处踩过的坑：因果掩码按**原始位置**（走 `attention_kernel.causal_visibility`，它只认 `selected_indices`）、APB 的锚点用**后续卡的 queries**（`queries[offset:]`）而不是全体、FastKV 的共享压缩用全体 queries 且 λ_β 与主方法**同源**。等价强度是 **ULP 级、不是逐位**（与 G1 的 `query_chunk`、G2 的 `n_chunks` 是同一现象），测试用相对容差。⚠️ **ring 是单进程模拟，不是 NCCL P2P ring** |

> ⚠️ 三个基线都是 reference-quality 实现，**只保证接口与数值正确，不做任何性能优化**。
> 真实性能对照需要 GPU 上的高度优化版本；当前**尚无**（E6 的四个多卡方法因此全部被阻断）。

---

## 3. `experiments/` —— 实验代码

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `README.md` | 171 | 实验总览：CPU/GPU 分工、编号体系（E0–E13 / A1–A5）、如何跑 |
| `__init__.py` | 15 | 声明 `experiments` 包 |
| `run_cpu.sh` | 65 | CPU 实验入口 |
| `run_gpu.sh` | 135 | GPU 实验入口（封装 `torchrun`）。⚠️ 数组展开必须用 `"${ARR[@]}"`，**不能**写 `"${ARR[@]:-}"`（后者会退化成空字符串参数） |

### 3.1 `experiments/common/` —— 公共模块（纯 CPU，只依赖 torch + numpy）

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 9 | 声明 `common` 包 |
| `synthetic.py` | 854 | 合成场景生成、完整注意力参考、误差与分布度量（含 KL / JS 等）。`mass_error` 已标废弃，公开偏移版本为 `absolute_mass_error`（列名 `eps_mass_abscommon_*`） |
| `report.py` | 390 | 结果汇总：`summarize`（median / p5 / p95 / bootstrap CI）、配对检验、`save_json` / `save_csv` |
| `hypotheses.py` | 825 | **H1–H5 阈值的单一事实源**（blueprint §3 的机器可读版）：阈值本体 + `h1_pass`…`h5_pass` + `HYPOTHESES` 注册表 + `UNJUDGED`（尚未接判据的假设，现为 `{H2,H3,H5}`）。2026-09-16 新增 H2 前提「质量相近」的判定程序 `quality_comparable_non_inferior`（入参显式含 `delta_pp` / `noise_floor_pp`，越出可行区间即 `ValueError` 而**拒绝执行**）与两个常量（`QUALITY_COMPARABLE_REFERENCE = "dense"`、非劣边界上界 = 1.5 pp）。2026-09-16 补 H2 的**跨长度聚合入口** `h2_pass_across_lengths`（逐长度分别判定、全通过才成立；前提未定记 `unresolved` 而非 `failed`），已接入 `e6_main_table.py` |
| `beta_variants.py` | 689 | β 的各种口径变体与对照（per-key / 标量 / 关闭 β 分解）。**2026-09-22（`39063e7`，上一轮写出）**：`nonneg_ridge_pgd` 增 `lambda_disp`，把岭惩罚正交分解为「块级常数」与「per-key 离散度」两项；默认 `None` ⇒ **逐位兼容旧实现** |

### 3.2 `experiments/cpu/` —— 机制级实验（任意多核机器可复现，无需权重/NCCL）

| 文件 | 行数 | 内容 |
|---|---|---|
| `e0_order_invariance.py` | 429 | 顺序无关性（归并 ⊕ 的交换律/结合律实证）。**含置换次数轴**：`--perm-ladder` 给出「累计最大相对误差随试验次数 n」的收敛阶梯（默认 1…1000），并汇总「平衡树不劣于顺序归并」的符号分布 |
| `e1_interface_shapes.py` | 181 | 构造接口的形状与 dtype 契约 |
| `e2_fidelity_curve.py` | 249 | 保真度随预算变化的曲线（**FP64**，$B$ 含 $B=L_s$ 边界） |
| `e2b_beta_convention.py` | 435 | **β 口径判定**：系数应为 1；度量必须建在归并侧（源块 + 固定上下文块拼接后再与 Dense 比） |
| `e3_edge_conditioning.py` | 701 | **边级条件化**：H1（KL/JS/Jaccard）+ H2 配对检验。**默认留出协议**（fit / eval 两池互不相交，`--eval-fraction`）；`--protocol in-sample` 仅用于复现旧数字。留出后 H2 优势是目的端分离度的函数（强度 0 无优势） |
| `probe_e3_heldout.py` | 226 | **E3 留出集判定探针**（一次性，保留以便复现）：同场景并排跑 in-sample 与 heldout 两套口径，只打印不写结果。结论：负对照上样本内口径造出 +0.156 的假优势；主线 \|Δ\| 缩水约 31%；方向稳定性 20/20 → 18/20 |
| `e4_dist_equivalence.py` | 283 | 分布式等价性（单进程 mock） |
| `e5a_mechanism_ablation.py` | 234 | A3 的机制级版本（重构误差口径，对应 GPU 版的任务指标口径）；落 `results/cpu/a3/`，用 `absolute_mass_error` |
| `e9_knob_localization.py` | 389 | **M–B 旋钮定位**：375 格扫描，分离"可达上界（B）"与"能否触及上界（M）" |
| `e10_beta_stability.py` | 656 | **β 稀疏塌缩**的定位与修复；把 β 分解为"块级常数分量（收益）"与"per-key 离散分量（代价）" |
| `e11_lambda_tuning.py` | 774 | λ_β 调参（尺度无关的相对正则强度） |
| `e12_representative_query.py` | 421 | **代表 Query 投影维度 $d_p$** 的扫描（`$d_p\in\{4,8,16,32,64\}$` + 无投影参照），报 JL 畸变 / 覆盖率 / 下游误差；5 种子 + 留出 |
| `e13_bound_tightness.py` | 657 | **误差界式 (37) 的紧致度检验**：逐 Query 比对 RHS/LHS（`rel_tol=1e-9`）与前置条件违反率，含块数轴（性质 3）；5760 个 Query、零反例 |
| `c10_baseline_diagnosis.py` | 379 | 基线诊断：为何 `test_fast_kv_*` / `test_apb_*` 从未被 pytest 收集（包导出缺失 + `budget == L_s` 根本没压缩） |

### 3.3 `experiments/gpu/` —— 任务级与系统级实验（需 A100/H100 + NCCL + 真实权重）

> ⚠️ **本机无 CUDA ⇒ E5–E8 全部未执行**。环境闸门不达标时脚本**不写结果文件**、以退出码 3 结束。
> 因此"代码写完"不等于"实验做了"；仓库内**没有任何 GPU 数值**。

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `README.md` | 97 | GPU 实验说明、入口脚本、环境闸门与 device 修复记录 |
| `__init__.py` | 17 | **声明 CPU/GPU 两套证据的量纲关系**（E3↔A2、A3 机制↔A3 任务、E0↔E8），并明确不可互推 |
| `_env.py` | 622 | 环境闸门（`probe` / `enforce`，不达标退出码 3）、计时（`benchmark_ms`；窗口内只做 `device_sync()`，集合 barrier 留窗口外）、设备绑定（`local_rank_of` / `local_device`，取 `LOCAL_RANK`）、`CUDA_CONSTRUCTION_DEFECTS` 历史清单、元数据构造 |
| `_comm.py` | 469 | 变长 All-to-Allv（阻塞 / 异步）、同步与异步流水线（`run_sync_pipeline` / `run_async_pipeline`）、分块 `_split_chunks`（逐 dst 取片，尺寸自洽）、体积口径（`make_uniform_plan`） |
| `_hf.py` | 897 | HF 模型加载、KV 预算裁剪（`apply_kv_budget`：identity / topk_rms / topk_norm / stride / random）、多项选择打分（`score_choices`，含独立 cache 副本 + 显式 position_ids）、评测（`evaluate`）、prefill 计时（**2026-09-21 起三段化**：源端构造 → 压缩 → 目的端前向，由 `PREFILL_TIMING_CONSUMES_COMPACT_KV=True` 声明且压缩臂缺 `dest_len` 即抛错）、KV 字节数 | **2026-09-21 补（`b44a7b3`）**：`attn_hook` 参数打通三个入口（`measure_prefill` / `score_choices` / `evaluate`）；新增 `prompt_split`，成为源段 / 目的端本地段切分的**唯一出处**（此前计时侧与评测侧各算一遍，实测导致同一行的预算读数对不上质量侧）；钩子路径下 `measure_prefill` 开**两个**计时窗口（含构造的 `prefill_ms_*` + 只目的端的 `dest_ms_*`，后者才是 kernel-matched 的量）；`_token_bound` 从**真实词表**取随机 token 上界（写死 1000 会让词表 < 1000 的模型死在 `F.embedding`，症状指向 embedding、真因在输入生成） |
| `_forward.py` | 426 | **G2**：异步 All-to-Allv 接**真实前向**的桥接层。`make_uniform_layout`（统一布局，供等价性验证）/ `pack_edges`（变长 edge 打包）/ `chunk_source_sizes`（源端分段约定，与 `attention_hook.source_partition` 同款 —— 两处必须一致，否则"每段长度"在两套代码里不同值）/ `decode_edges` / `pipelined_attention`（同步、异步两条流水）/ `assert_same_answer`。计时纪律：`T_build / T_comm / T_comp / T_total` 必须**拆开报**；异步的 `comp_ms` 是**上界**（`wait(handle)` 之后不允许任何 device-wide 同步，否则把正在飞的集合通信也等掉 ⇒ overlap 恒为 0）；同步与异步之间**只有 `total_ms` 可比**。⚠️ 单卡下真 `dist.all_to_all_single` 退化成一次本地拷贝、`chunks_effective=1`、**没有可重叠窗口** |
| `build_eval_set.py` | 557 | **G5**：评测集转换器 —— 把 LongBench 转成 `_hf.EvalSample` 的 JSONL。为什么需要它：E5-A2/A3、E6、E7 的**准确率列全部依赖它**，而 `_hf.score_choices` 只认 `{prompt, choices, answer, task, length_tag}` 这种多项选择形态。含 `convert_records` / `validate_samples`（转换后自检）/ `write_jsonl` / `write_report` 与 `run_selftest`（无数据也能跑）。⚠️ **评测集本体仍需生成** —— 本脚本是纯转换器，需要外部的 LongBench 输入数据 |
| `e5_gpu_ablation.py` | 1211 | **A1** 通信集大小、**A2** 压缩预算扫描、**A3** 组件拆分（被阻断，拒绝产假数字）、**A5** 异步 vs 同步（H4 判据走 `hypotheses.h4_pass`）。**A4**（压缩 × 异步交互，2026-09-20 已实现）。**A3 仍被阻断** —— 但**理由已换**（2026-09-21）：G1 算子核与 H0 的钩子都已就绪（`src/distributed/attention_hook.py`），真正剩下的是**逐边条件化的多设备语义**：单进程 harness 只有单一目的端，无法区分 DCC-KV 与 FastKV，故 A3 的对照必须走多设备路径。在此之前拒绝产假数字 |
| `e6_main_table.py` | 1279 | 主表与可扩展性。`gpu_count` 记为**方法要求**而非自由轴；sync/async 轴对单卡方法**折叠**（不生成两行相同的数）。⚠️ **2026-09-21 更新**：H0 已接（量具灵敏了），但 `dcc_kv` 等 4 个方法**仍** `measurable: False` —— 真实阻塞换了：`dcc_kv` 与 `kv_budget_shared` 仍走同一条 `apply_kv_budget`（共享裁剪）⇒ 质量差恒为 0。此时翻 `measurable` 会得到**假阳性**（拿一条不是本文方法的通路去判质量臂）。prefill 加速比分子已改判为 **dense**。**同日补**：H0 的**方法侧**钩子已就绪（`src/distributed/attention_hook.py`，本机 tiny Llama 端到端验证），但它**尚未接进 `measure_point` 的方法行** —— E6 的 `--dcc-world` **尚未加入 CLI**（钩子侧已有 `HookConfig.dcc_world`），接线完成前 `dcc_kv` 行仍与 `kv_budget_shared` 同路 | ⚠️ **已被 `b44a7b3` 修正**（上句保留）：`dcc_kv` 的 **`measurable` 已翻为 `True`**，其方法行改走 `attention_hook`；E6 加 `--dcc-world`（**无默认值**）/ `--dcc-budget-mode` / `--ranks`（默认 1）/ `--lambda-beta`。同轮另三处口径修正：折叠判据推广为「**本次运行真的没有多卡**」（`--ranks 1` 时 `dcc_kv` 的 sync 轴也折叠）；prefill 加速比落**两列**（`prefill_speedup_kernel_matched` 给 H2 用，`prefill_speedup_native` 只作端到端参考；**缺同核列时记 unresolved 而非 failed**）；计时与评测共用同一份 `_hf.prompt_split` |
| `e7_negative_results.py` | 486 | 负结果四条件：短上下文 / 低预算 / 强 retrieval / batch=1 |
| `e8_low_precision.py` | 348 | 低精度（FP16/BF16）归并算子与失效边界 |

---

## 4. `tests/` —— 测试

`pytest.ini` 默认 `-m "not gpu"`，因此本节的非 `gpu/` 部分应全部通过；`gpu/` 下默认跳过。

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 2 | 声明测试包 |
| `test_smoke.py` | 477 | 冒烟测试：核心 API 的形状与数值 sanity（含 `recv_sizes` 声明不符必须报错的 2 进程 gloo 锚点） |
| `test_phase_a.py` | 239 | Phase A（多进程 + 通信原语）验收 |
| `test_dist_equivalence.py` | 448 | 分布式等价性（含 2 进程 gloo 真实测试，标记 `distributed`）。**2026-09-22（`39063e7`）**：新增 `test_fps_start_follows_paper`（结构性断言：首点 = `N-1`、与 seed 无关、不重复、非法模式抛 `ValueError`）；FastKV 越界阈值 0.30 → 0.49（先重跑 `c10_baseline_diagnosis.py` 更新落盘，再按该测试自身约定改阈值） |
| `test_var_len_msg.py` | 97 | 变长消息协议 |
| `test_experiment_metadata.py` | 100 | 元数据与结果 schema 的字段契约 |
| `test_hypotheses.py` | 254 | **H1–H5 阈值表的锚点**：H2 的双条件口径（2026-09-15 裁决）、H1 用严格大于而其余用闭区间、`UNJUDGED` 缺口记录、以及"e5 不得硬编码 H4 阈值"与"e3 不得硬编码 H1 阈值"的源码级断言。2026-09-16 补 7 个锚点覆盖「质量相近」判定程序（参照物、两个上界、数据依赖上界、单侧严格性、正数校验、返回原生 bool） |
| `test_gpu_pipeline.py` | 605 | **纯 CPU 的 GPU 侧口径锚点**：分块尺寸自洽、逐 dst 恰好覆盖一次、两条流水线的 sizes 接线、计时窗口不含集合 barrier、`local_device` 用 LOCAL_RANK、E6 的轴折叠规则、`apply_kv_budget` 分块累加的数值等价性。**不需要 GPU，也不需要进程组** | **2026-09-21 补（`b44a7b3`）**：E6 段新增四类锚点 —— 折叠判据是「既有跨设备通信 **且** 本次真的起了多卡」的合取、`--dcc-world` 无默认值、**`dcc_kv` 行真的把 `HookConfig` 传下去**（AST 检查两个调用的关键字**取值**，不是「有没有这个词」）、kernel-matched 加速比在缺任一端或分母非正时返回 `None`（不是 0/1） |
| `test_e0_order_invariance.py` | 172 | **E0 置换次数轴的锚点**：阶梯按声明点升序返回、累计单调、**端点必须等于同一置换序列前 n 次的最大值**（防「非累计」写错）、固定种子可复现、空/非正输入抛错、`summary` 缺档位时返回 `None`；另有源码级断言保证 `--perm-ladder` 与 `e0_perm_ladder.csv` 不被整体删掉 |
| `test_attention_kernel.py` | 453 | **G1 算子核的锚点**（纯 CPU）。冻住三件“看起来对但实测不对”的事：SDPA 加性掩码的形状规则（少于 2 维抛 `IndexError`、多于 query 维数抛**指向输出形状**的 `RuntimeError`）、跨块归并必须等于“拼接后一次算完”（**这条不成立则 A3/A4/E6 的 dcc_kv 行全错，而且错得很安静**）、`query_chunk` 只到 ULP 级而非逐位。**2026-09-21 新增一条**：`dense_attention` 的因果掩码必须支持带前导维的 query/keys —— 旧实现把「最后一维是 key 维」写成「第 0 维」（`keys.shape[0]`），H ≠ Lk 时直接报错、**H == Lk 时不报错只算错** | **2026-09-21 同日再补**：`dcc_kv_attention` 的因果策略必须可解耦（远端全可见 + 本地因果），且**不给开关时按 `query_positions` 推导的结果与旧版逐位一致** —— 旧接口把两种可见性绑在同一个开关上，H0 的钩子因此表达不出它需要的那一组 |
| `test_attention_hook.py` | 556 | **H0 方法侧的锚点**（纯 CPU）：用**本地构造的 tiny Llama**（`LlamaConfig` + 随机权重，不下载任何东西）跑完整前向，于是本机能问出 GPU 机上要花很久才问得清的事 —— 接线到底对不对。最强的一条是 dense 参照臂必须复现原生前向（一条盖住形状、转置、GQA 分组、位置、因果、还原六个维度）；另有「钩子不许空转」（压缩臂的偏离必须 > 1e-4）、阶段与嵌套的响亮失败、注册表不覆盖 `eager`/`sdpa`、整列遮蔽守卫、切分与预算口径、以及 `measure_prefill` 的目的端位置必须是**绝对位置且两臂一致**。三条实现变异（回退 position_ids / 去掉 transpose / GQA 改 repeat）实测全部被抓 | **2026-09-21 同日扩充（`b44a7b3`，382→556 行）**：新增 4 条覆盖重写后的契约 —— 本地块契约（三段式 dense 与原生一次前向等价）、`reuse` 逐位相同且源变了必须拒、**源端不得留在传入的 cache 里**（指纹守卫）、`summary` 必须携带随数字一起走的声明 |
| `test_baseline_operators.py` | 246 | **G3 的锚点**：三个向量化算子与 CPU 参考 **ULP 级**一致（容差写成可解释的数，非随手 allclose）。只有 `budget < L_s` 的用例才能抓住“因果掩码按行序而不按位置”这类只在压缩时才显形的错 |
| `test_build_eval_set.py` | 168 | **G5 的锚点**（守**口径**而非功能）：选项只能来自数据自身；跳过必须记账且 E7 的「强检索」条件缺数据要被显式识别；非官方提示模板必须如实标记 |
| `test_forward_pipeline.py` | 532 | **G2 的锚点**（stub 顶掉集合通信后可纯 CPU 跑）。核心不变式只有一条：**流水只改变“什么时候算”，不改变“算什么”**。另有打包/解码互逆、块内源切分、退化为单块时如实上报。⚠️ 为躲开 conftest 的按名自动打标（含 `gpu` / `nccl` / `end_to_end` / `async_overlap` 会被默认排除），本文件**刻意避开这些词** |
| `test_a4_interaction.py` | 409 | **G4（A4 压缩 × 异步交互）的锚点**。A4 是**唯一**把测量与判定都放在设备无关路径上的消融，故本文件用 stub **真跑一遍** `a4_interaction_grid` 而非只查字符串（字符串检查挡不住“接线接反”）。三类锚点：判定语义、接线纪律（判据不得被重新实现、CI 水平不得硬编码）、测量正确性（A4 计算侧必须是真实算子核，T_comm/T_comp 取自同步臂） |
| `test_selfcheck_2026_09_18.py` | 362 | **2026-09-18 自查的回归锚点**：每条对应一个已修缺陷，且都是“形状/量级/文件名都正常、不报错但结论错或永远出不来”的那一类（如 `paired_bootstrap` 方向反了、`save_csv` 遇混合行整表 `ValueError`、`h4_pass=None` 被折进“未达标”） | **2026-09-21 补（`b44a7b3`）**：T6 新增一条 —— **单卡下表里写出的 `dcc_kv` 行 `sync_async` 也是 `"n/a"`**，H2 配对必须仍能找到它（旧实现只给基线做了轴无关查找，dcc 那一侧仍按 `sync_modes[0]` 查 ⇒ 折叠后配对恒为空；这是 `bee3388` 给基线修过的同一个坑在 `dcc_kv` 身上的第二次） |
| `test_adversarial_2026_09_20.py` | 989 | **2026-09-20 对抗性审查 + 2026-09-21 H0 接线的回归锚点**：12 处已修缺陷（D1–D12）各一条，每条先证明「它本可以不被发现」。**A 组已于 2026-09-21 改写** —— 原锁「量具失敏 ⇒ 加速比结构上恒不超 1」，现锁「目的端看到的 KV 必须等于 B、两臂唯一差别就是那个长度、漏给 `dest_len` 必须抛错、分子必须是 dense」；B 组异步臂 `comm_ms` 是上界（落 `nan` 而非 0）；C 组重复次数与 `lambda_beta` 默认值必须进产物且与项目默认**同源**；D 组用 AST 扫设备索引（`set_device(rank)` / `f"cuda:{rank}"`）；E 组双边同步次数断言；F/G 组 E8 元数据与 `json_safe`；**H 组元守卫**保证本文件自身的用例默认全部运行；**I 组（2026-09-21 新增、同日扩展）**断言源码与 `docs/*.md` 里引用的**仓库内任意 `.py` 路径**真实存在（原先只守 `tests/test_*.py`，于是 `src/distributed/attention_hook.py` 被两处源码引用却不存在这件事它看不见） | **2026-09-21 第三次更新（`b44a7b3`）**：A 组的分子由 dense 改为 **kernel-matched**（同核），并新增「缺列时不得回落 native、且不得把 nan 比成未达标」；另新增 **I4**：`docs/*.md` 里不得有**未闭合的表格行**（`FILE_MAP.md` 曾有 4 行以 ``| **G1`` 这种半截形态存在，而它的职责恰是逐项说明每个受控文件） |
`tests/gpu/`（默认跳过）：

| 文件 | 行数 | 内容 |
|---|---|---|
| `README.md` | 50 | 如何在目标机上启用这些测试 |
| `test_nccl_basic.py` | 161 | NCCL 基础连通性 |
| `test_async_overlap.py` | 130 | 异步变长通信的 overlap（对应 H4） |
| `test_end_to_end_8b.py` | 119 | 8B 端到端 |
| `profiling/nsys_runner.py` | 86 | Nsight Systems 采集封装 |
| `profiling/torch_profiler_runner.py` | 55 | torch.profiler 采集封装 |

---

## 5. `paper/` —— 论文

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `README.md` | 72 | 编译说明与章节目录 |
| `main.tex` | 480 | 主文件。第 1–4 章正文（引言 / 相关工作 / 问题定义 / 方法）+ 算法伪代码 + 三个 `\input`。**第 1–4 章以本条重建版定稿**（2026-09-16 决定，原件即桌面 `AuthorKit27 (1).pdf`）。⚠️ 作者 / 单位 / 邮箱是**有意的占位符**（暂不写入正文，勿代填） |
| `refs.bib` | 242 | 参考文献 **17 条，全部被 `\cite`**。注释字段只用 `annote`（`note` 会被 bst 排版进参考文献表，中文注释会印进正文） |
| `build.sh` | 70 | 编译入口（**必须 xelatex**：`ctex` 与 `acmart` 冲突，改用 `xeCJK`）。⚠️ `set -e`：出错会**静默中断**，看起来像成功 |
| `fetch_refs.sh` | 74 | 拉取参考文献 PDF 到 `literature/` |
| `.gitignore` | 23 | 忽略 `literature/` 与 LaTeX 中间产物 |
| `sections/05-analysis.tex` | 303 | **第 5 章 复杂度与误差分析**。通信量、计算复杂度、误差分解与误差界、异步流水的理论加速上限 |
| `sections/06-experiment.tex` | 1050 | **第 6 章 实验方案**。设置、评测协议与元数据规范、**已验证的 CPU 实验**（E0–E11 / A3 机制级）、**待执行的 GPU 实验**（E5–E8，含 2026-09-16 新设的 **A4** 压缩 × 异步交互）、结果报告规范。§6.2 的 H1–H5 阈值已显式标注为**待验假设** |
| `sections/07-discussion.tex` | 216 | **第 7 章 讨论 + 第 8 章 结论**（两章写在同一文件，`\section{结论}` 在 L153）。2026-09-16 新增小节「「质量相近」的判定」（标签 `sec:quality-comparable`） |
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
| `commit_log.md` | 3199 | **逐次提交报告**，本仓库最重要的过程文档。体例：每条含动机 / 改动清单 / 验证 / 遗留；被推翻的结论**保留原文**并加 `⚠️ 已被 <hash> 修正`，不抹除历史。§P.1 是"已解决的问题"表，§P.2 是"仍未解决的问题"表，§Q 记录易被误读的坑 |
| `git_strategy.md` | 418 | Git 管理策略：分支、commit 体例、tag、实验可追溯。§8.1 记录 **`results/` 入库**的决定与体积策略 |
| `release_checklist.md` | 177 | 投稿前 / Camera Ready 清单。§4 是 failure_thresholds 校核（H2/H3/H4/H5）；2026-09-16 起该节附「质量相近」的定义摘要 |
| `reproducibility.md` | 293 | 可复现性说明。§6 的 H1–H5 及其阈值：2026-09-21 起口径来源改为**论文原件（`AuthorKit27` 第 7 章）+ 仓库内机器可读实现 `hypotheses.py`**（原先声明「以 blueprint §3 为准」，该原件不在库内、已裁决不再作为权威锚点，原 C11 关闭）；2026-09-16 起附「质量相近」的完整定义摘要；H2 的 prefill 加速比分子于 2026-09-21 改判为 **dense** |
| `writing_scope_and_metrics.md` | 181 | **撰写范围与指标口径**。以第 1–4 章原始建模（桌面 `AuthorKit27 (1).pdf`）为标尺的「建模条款 → 应报指标 → 口径 → 现状 → 缺口」对照表：六个误差术语 ↔ 四个代码度量函数 ↔ §5 理论量的三方映射；缺口 M1–M9 及各自判定标准；7 条口径硬约束。§6 记录历次裁决的落实（D1–D6）与仍开放项。**2026-09-16 第八轮：M1/M2/M3/M5/M9 全部关闭**（见 §4 后附注）。第九轮补上 E0 的「置换次数」轴，论文 §6 里最后一条非 GPU 待补项关闭 |
| `gpu_execution_plan.md` | 460 | **GPU 实验执行计划与预算控制**（2026-09-16 新增）。结论先行：现在不能开跑 —— 缺的不是卡是代码，依据 `e6_main_table.py --plan` 实测 6 方法中 4 个 `blocked`（含本文方法 `dcc_kv`）。含 G1–G6 前置缺口、S0–S3 四级梯队、数据量与时长估算、预算纪律十条、租卡前 checklist、论文完整性核对。**§0/§5/§6/§7 已于 2026-09-21 第二次同步**：G1–G5 已**提交**（`c9ca6b0`），**H0 已接**（`measure_prefill` 三段化 + `PREFILL_TIMING_CONSUMES_COMPACT_KV` 翻正 + E6 加 `--dest-fraction`）。**未变的是结论，变了的是理由**：现在阻的是「`dcc_kv` 的构造路径还没独立」（仍与 `kv_budget_shared` 共用共享裁剪 ⇒ 质量差恒为 0）与 **G6 未做**；翻 `measurable` 仍不可以（否则得到假阳性） | ⚠️ **已被 `b44a7b3` 修正**（上句保留）：**接线已完成、`measurable` 已翻**；§0/§5/§7 已第三次同步，口径变化三条（sync 轴折叠判据推广、prefill 加速比两列、计时与评测同一份切分） |
| `ssh_setup.md` | 141 | SSH / 远程机器配置 |
| `FILE_MAP.md` | 本文 | 文件说明（你正在读的这份） |

---

## 7. `scripts/` / `.github/` / `reports/`

### 7.1 `scripts/` 与 `.github/`

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `scripts/run_m2_real.sh` | 61 | M2 真实多卡运行脚本 |
| `scripts/run_m3_async.sh` | 70 | M3 异步运行脚本 |
| `.github/workflows/test.yml` | 90 | CI 工作流 |
| `.github/PULL_REQUEST_TEMPLATE.md` | 55 | PR 模板 |
| `.github/ISSUE_TEMPLATE/bug_report.md` | 49 | Bug 报告模板 |
| `.github/ISSUE_TEMPLATE/feature_request.md` | 40 | 功能请求模板 |

### 7.2 `reports/` —— 对外汇报与代码导航（2026-09-16 新增）

面向**合作者**（而非维护者）的材料。与 `docs/` 的分工：`docs/` 讲过程与规范
（给自己看），`reports/` 讲进展与导览（给别人看）。**文件名用英文、内容用中文** ——
与仓库其余文件命名体例一致，且避免中文名在 git 终端输出里被转义成八进制、
以及在 Linux / CI 上的潜在编码问题。

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `2026-09-16-progress-report.md` | 200 | **进展汇报**（面向师姐）。一句话结论（论文非 GPU 内容已收口 + 当前仍不能租卡）、今日 6 项完成项（附证据）、论文/代码/证据三层状态快照、4 条风险、待完成事项分「CPU 可做 / 需租卡 / 需外部输入」、下一步计划、自查命令附录 |
| `code-and-files-guide.md` | 370 | **代码与文件说明**。面向第一次接触仓库的人：怎么用本文档、项目做什么与**原创边界**、五分钟上手、目录总览、`src/` 逐模块、`experiments/` 逐脚本（含 15 个 CPU 实验一览）、测试的定位、论文与排版注意、`results/` 产物约定、`docs/` 各文件作用、**§10 已知缺口与诚实边界**（含引用数字的三条禁令） |
| `2026-09-22-theory-code-consistency-audit.md` | 246 | **理论—实现一致性审计**（2026-09-22）。以 AM 原文（`2602.16284`）＋桌面 `AuthorKit27 (1).pdf` 为标尺，逐节对齐机制 / 口径 / 证据区间，查出 **8 处不一致（F1–F8）** 并列**三项待裁（Q1–Q3）** |
| `2026-09-22-lambda-zero-and-interval-verdicts.md` | 302 | **Q1–Q3 裁决的落地与证据**（2026-09-22）。Q1：`λ=0` 非绝对限制但实测更差（超定域也差 4.3%）⇒ 维持 `3e-2`；Q2b：E9 搬进声明区间（`results/cpu/e9_L2048/`），M 主导性 1.83× → **3.23×**；Q3：FPS 起点 / 复杂度按论文改代码、箱约束实测 0.038% 保留。文末列**三项待裁** |

---

## 8. 仓库外依赖的文档（**作废**：2026-09-21 裁决不再作为权威锚点）

> ⚠️ **2026-09-21 裁决**：本节登记的这批仓库外文档（原记账为 **C11**）
> **不再作为权威锚点，不予追索**。口径一律以**论文原件（`AuthorKit27`）**
> **+ 仓库内机器可读实现（`experiments/common/hypotheses.py`）** 为准。
> 下表保留原文，只作历史记录 —— 它记下了「长期依赖一份拿不到的原件」这件事
> 如何制造了下游表述分歧（本节的 H2 例是最好的标本）。

`README.md` §上级文档 与 `docs/` 多份文件都引用以下**不在本仓库内**的规划文档。
全盘搜索确认它们既不在仓库内、也不在相邻目录，**需要外部提供**（记账为 C11）：

| 文档 | 被引用处 | 缺失的实际影响 |
|---|---|---|
| `../dcc_kv_plan/research_execution_blueprint_v1.md` | `CONTRIBUTING.md`、`docs/git_strategy.md`、`docs/release_checklist.md`、`docs/reproducibility.md`、`README.md`、`src/experiment_metadata.py`、`tests/gpu/test_async_overlap.py` — 共 9 处 | 它是 §3 failure_thresholds、§4 元数据 schema 的**权威原件**。内容已大体抄入仓库，但**无法做一致性核对**，且已出现下游表述不一致（见下） |
| `../dcc_kv_plan/experiment_matrix.yaml` v1.2.0 | `docs/git_strategy.md`、`docs/release_checklist.md`、`docs/reproducibility.md`、`README.md` | 实验格点的权威清单（哪几档预算、哪几种模型、哪些组合是"规划内"）。当前格点由需求反推，无法核对是否覆盖 |
| `../dcc_kv_plan/references.bib`（README 称 26 条） | `README.md` | 与 `paper/refs.bib`（实际 **17** 条）口径差 9 条。17 条均已引用，不阻塞；但"26"这一数字暂无法证实 |
| `../dcc_kv_plan/contribution_boundary_section.md` | `README.md` | 原创边界声明。其内容已吸收进 `README.md` 的"不主张的内容"节与论文 §6，风险较低 |
| `../dcc_kv_plan/M2_pre_launch_checklist.md` | `README.md` | M2 启动前检查项（**C11 原先漏记的一份**） |

**当前已因此显现的具体危害**：H2 的判据一度在两处下游文档里写法不同 ——
`docs/reproducibility.md` §6 只写了质量侧，`docs/release_checklist.md` §4 写全了两个侧面。
2026-09-15 按后者（较完整的一版）**定稿为逻辑与**（两条同时满足），两处已对齐，
阈值的机器可读版本落在 `experiments/common/hypotheses.py`。
**2026-09-16 更新**：「质量相近」这一前提**已给出定义** —— 参照物取精确注意力、判据为单侧非劣检验、非劣边界满足 `噪声底线 <= delta < min(1.5, delta_bad)`（完整论证见论文 §7「「质量相近」的判定」，机器可读实现在 `hypotheses.py` 的 `quality_comparable_non_inferior`）。仍未闭合的只剩**两个参数**（`delta` 与噪声底线），二者依赖 E6 的重复 run。

**2026-09-15 独立监督的补充**：本条危害不止上述一处。监督者另发现 **A4 消融编号
在全仓（论文 + `docs` + `experiments` + `src`）完全不存在**，编号从 A3 直跳 A5 ——
即实验格点的缺口比本条原记载**多一格**。详见 `commit_log.md` §S.3。

**2026-09-16 处置**：A4 已补齐到**论文与文档**，并确认原件（第 1–4 章）**本就没有消融编号** ——`消融` / `ablation` / `A1`–`A5` 在原件中各 0 次命中，所以 A4 是**纯设计项**而非"找回"项。**代码侧仍未实现**：`e5_gpu_ablation.py` 的注册表仍是 A1/A2/A3/A5，其 docstring 已显式登记该落差。详见 `commit_log.md` 第 25 条。

---

## 9. 不入库的内容

| 路径 | 为什么不在版本控制里 |
|---|---|
| `paper/literature/` | 17 篇参考文献 PDF + `_download_log.txt`，由 `paper/.gitignore` 忽略。**与 `paper/refs.bib` 的 17 条一一对应**（含源论文 `zweiger2026attentionmatching_2602.16284`）。用 `paper/fetch_refs.sh` 重建 |
| `paper/main.pdf`、`figures/*.pdf`、`*.aux`、`*.log`、`*.bbl` | LaTeX 编译产物，由 `paper/.gitignore` 与根 `.gitignore` 忽略 |
| `.pytest_cache/`、`__pycache__/` | 工具缓存 |
| `scripts/.patch_*.py` | 一次性补丁脚本（2026-09-17 的 G2/G4 补丁 5 个）。按 G2 体例**不入库** —— 它们承载的结论已补记进 `docs/commit_log.md`。留在工作区仅供溯源，**任何一次 `git add -A` 都会把它们误收**（见 §10） |
| `models/`、`data/`、`traces/` | 权重、数据集、profiling trace —— 由根 `.gitignore` 显式排除（体积大、可能涉许可） |

> **`results/` 已不再属于本节。** 自 2026-09-16 起它**入库**（63 个文件；`results/cpu/**` 全部 + `results/gpu/e8_cpu_reduced/**`），由根 `.gitignore` 末尾的 `!results/**` 显式放行 —— 目的是让论文中每个 CPU 数字都能随仓库复现。
> 两条须知：① 该规则的**位置有约束**，必须排在 LaTeX 段的 `*.log` 之后，否则 `results/cpu/*_run.log` 仍会被挡掉；② `results/gpu/e8_cpu_reduced/` 是 **CPU 缩规模下的通路验证**，不是 GPU 机器上的实测。体积策略见 `docs/git_strategy.md` §8.1。

---

## 10. 维护约定

- 新增/删除/改名任何文件，**同步更新本文**，并在 `docs/commit_log.md` 追加一条。
- 本文的统计数字以 `git ls-tree -r HEAD | wc -l` 为准，不手写估算。
- 引用 hash 时写完整 7 位；被后续提交推翻的表述不要删，加 `⚠️ 已被 <hash> 修正`。
- **行数列的时效性**：2026-09-21 校准了本轮真涉及的 7 个文件（`_hf.py`、`e6_main_table.py`、两个测试文件、三份 docs）；2026-09-20 曾校准 20 余个（含新增的 11 个）；本列在多轮提交后未逐次校准 —— 2026-09-16 实测发现约 20 处偏小（1 至 90 行不等，例如 `_hf.py` 436→468、`e6_main_table.py` 441→499、`tests/test_gpu_pipeline.py` 400+→447、`e0_order_invariance.py` 277→429（第九轮校准））。本次只校准了**本轮真涉及的文件**，其余保留原值以免制造假精确。需要准确值时以
  `git ls-tree -r HEAD --name-only | xargs wc -l` 为准，不要相信本列。
