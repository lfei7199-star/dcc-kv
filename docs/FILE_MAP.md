# 仓库文件说明（FILE MAP）

> 本文逐项说明本仓库每个受控文件的**作用**与**内容**，供接手者、协作者与未来的自己定位代码。
> 统计口径：`git ls-tree -r HEAD`（**215** 个受控文件 = 88 `.py` / 32 `.csv` / 31 `.json` /
> 33 `.md` / 10 `.tex` / 8 `.log` / 6 `.sh` / 1 `.bib`，另 6 个为 `.gitattributes` /
> `.gitignore` ×2 / `pytest.ini` / `requirements.txt` / `.github/workflows/test.yml`），
> 不含编译产物与本地素材（见 §9）。其中 **72 个在 `results/` 下（32 `.csv` + 31 `.json` +
> 8 `.log` + 1 `.md`）—— 自 2026-09-16 起入库**（决定与体积策略见 §9 与 `docs/git_strategy.md` §8.1）。
> ⚠️ **2026-10-13 按提交后实测刷新（`cb7f0fb`）**：新增 2 个文件
> （`tests/test_audit_2026_09_23.py` + `results/diagnostics/lamv0_summary.json`）
> ⇒ `198` → `200`；分布由 `82 / 32 / 30 / 23 / 10 / 8 / 6 / 1` 变为
> `83 / 32 / 31 / 23 / 10 / 8 / 6 / 1`（`.py` 与 `.json` 各 +1）。
> ⚠️ **2026-10-14 按提交后实测刷新**：上一轮（第 47 条，run_gpu.sh 修复）提交后
> HEAD 实为 **203 / 85**，而本段写的仍是 **202 / 84** —— 只同步了 `commit_log.md`
> 的「仓库当前规模」一处，漏了本段（两处副本必须同改，见该文件维护约定）。
> 本轮（第 48 条，显存风险修复）新增 2 个文件 ⇒ `203 → 205`、分布
> `85 / 32 / 31 / 24 / 10 / 8 / 6 / 1` → `86 / 32 / 31 / 25 / 10 / 8 / 6 / 1`
> （`.py` +1 = 显存守卫测试；`.md` +1 = GPU 显存预估报告）。
> ⚠️ **2026-10-14 按提交后实测刷新（第 49 条，遗留问题清理）**：新增 **9** 个文件
> ⇒ `205 → 214`、分布 `86 / 32 / 31 / 25 / 10 / 8 / 6 / 1` →
> `87 / 32 / 31 / 33 / 10 / 8 / 6 / 1`（`.py` +1 = 论文接线状态守卫；
> `.md` +8 = `docs/reports/` 剩余 8 个报告**全部入库**，该目录自此 10 个报告
> 全部受版本控制）。同轮刷新多处行数实测：`run_gpu.sh` 196→231、
> `06-experiment.tex` 1050→1119、`git_strategy.md` 418→419、
> `reproducibility.md` 293→314、`requirements.txt` 26→29 —— 其中
> `06-experiment.tex` 与 `reproducibility.md` 两项**在 HEAD 态即已陈旧**
> （漂移 69 / 21 行，源自 2026-10-13 第 45 条 e13 误差界修正未同步本表），
> 本轮按实测登记；另补上第 47 条**漏建的一行**：
> `tests/test_run_gpu_entry_2026_10_14.py`（该文件自第 47 条起受控却无说明行）。
> ⚠️ **2026-10-14 按提交后实测刷新（第 46 条 docs 提交）**：新增 1 个文件
> （`docs/reports/2026-09-30-实验结果复核操作手册.md`，`.md`）—— 该手册此前
> **从未入过库**（`docs/reports/` 下 9 个对外文档当时全部未跟踪），本轮因
> 内容被改动（三处版本口径同步）而首次入库；其余 8 个报告仍未跟踪。
> ⇒ `201` → `202`、`.md` `23` → `24`。
> ⚠️ **2026-10-13 按提交后实测刷新（`152e5ac`）**：**新增 1 个受控文件**
> （`tests/test_fstring_backslash_guard_2026_10_13.py`）⇒ `200` → `201`；
> 分布由 `83 / 32 / 31 / 23 / 10 / 8 / 6 / 1` 变为
> `84 / 32 / 31 / 23 / 10 / 8 / 6 / 1`（`.py` +1）。同轮刷新 §3.2 的
> `e9_knob_localization.py` 行数（`389` → `438`）—— 该行在 HEAD 态即已陈旧
> （HEAD 实为 432，非 389；漂移 43 行，原因未考据、不擅归因），本轮改动使其
> +6 行，按实测登记。
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
> `results/cpu/e9/{rows.csv,summary.json}`（故 `results/` 计数 61 → 63）——
> **该目录 `e9_L2048/` 已于 `ad2f091` 删除并归档，见第六次更新。**
> 同轮按实测刷新 3 处行数（`representative_query.py` / `beta_variants.py` /
> `test_dist_equivalence.py`，见 §2.2 与 §4），并修正该轮报告自身的一处行数误记
> —— 原稿把 `git diff --stat` 的**总变更行数**（54 / 45 / 53）当成了「+行数」，
> 且 `representative_query.py` 计数含了一个随后删掉的空行（PEP8 E303）。
> 三处均已改成与 `git diff --numstat` 逐项对齐的实测值。
> ⚠️ **2026-09-22 第六次更新（`ad2f091`）**：**删除 2 个、新增 1 个受控文件**
> （仓库 184 → **183**）—— 删除 `results/cpu/e9_L2048/{rows.csv,summary.json}`
> （FPS 起点改动**之前**的产物、不可复现，归档于工作区外
> `_archive/results_cpu_e9_L2048_stale_2026-09-22/`；同轮 `results/cpu/e9/` 被
> 声明的 `L_s=2048` 区间**替换**，旧值归档于 `_archive/results_cpu_e9_L256_2026-09-22/`），
> 新增 `docs/author_kit27_deviation_log.md`（故 `results/` 计数 63 → **61**、
> `docs/` 8 → **9**）。同轮把 §2.2 的 `e9_knob_localization.py` 说明由「375 格」
> 改为「525 格（`L_s=2048`）」，并刷新被本轮改动文件的实测行数。
>
> ⚠️ **2026-09-30 第七次更新**：**新增 2 个受控文件**（`experiments/common/discrimination.py`
> 与 `tests/test_discrimination.py`）—— 落地用户裁定的**量具判据**（`writing_scope_and_metrics.md`
> §5 第 8 条），并同步 §5 第 4 条的显式修订。§3.1 / §4 已登记其条目与实测行数；
> 顶部「统计口径」与 `commit_log.md` 的「仓库当前规模」两处 **HEAD 口径**数字
> **已按本次提交后实测刷新为 `194`**（明细见顶部 ⚠️ 注记）。同轮刷新
> `writing_scope_and_metrics.md` 的实测行数（181 → **210**）。
>
> ⚠️ **2026-10-13 第八次更新**：**新增 4 个受控文件**（194 → **198**）——
> `tests/test_errbound_fix_2026_10_13.py`（11 项，见 §4）与
> `results/cpu/e13/` 下三件**旧界产物留档**
> （`e13_results_oldbound_20260922.json` / `e13_rows_oldbound_20260922.csv` /
> `e13_block_rows_oldbound_20260922.csv`），故 `.py` 81 → 82、`.csv` 30 → 32、
> `.json` 29 → 30，`results/` 计数 68 → **71**。
> 内容：**式(37) 误差界的修正与重验**。2026-10-12 全面核查发现论文 §5 误差界初版
> 可被**构造反例**击穿（两块归并 O₁=0、O₂=e₁、δM₁=0.1、δO₁=6e₁，前置条件
> Σ|δM|=0.1<S/2=1 满足而 LHS=3.119 > RHS=3.100）——根因是分子扰动
> ΔN = Σ_s(M_s·δO_s + **δM_s·Ô_s**) 的第二项（质量误差作用在紧凑输出上）
> 从未被计入；E13 网格「零反例」成立只是因为**合成压缩器不产生该构型**，
> 「零反例」≠「命题成立」在此再度应验。修正界改为
> `[ε_out + ε_mass·max_s(‖Ô_s‖+‖O_s‖)]/Ŝ ≤ 2·[同左]/S`，20 万次多块对抗搜索
> （初版界被击穿 11337 次作对照）零真反例。**E13 全网格重跑**：紧致度中位
> 2.10 → **2.23**、95 分位 3.51 → **7.37**、最大 3.51 → **9.56**（新界更保守），
> 零真反例；190 例纯浮点「<」全部位于取等格子（B=L_s 且不加 β 的 10 格）。
> ⚠️ 本轮曾把前置条件「最高违反率」误写成 **6.2%** —— 那是 aggregated 的
> **跨种子中位**，行级最大值是 **14.6%**；「超半数格子」误写 4/60，行级实为
> **22/60**。两处均已改，并新增 `_onetime_scripts/_verify_paper_e13_numbers_20261013.py`
> （30 项论文数字 ↔ 落盘 JSON 逐位对账，全绿）与 §4 的回归锚点防再犯。
> §3.2 的 `e13_bound_tightness.py` 行数 657 → **706**（按 UTF-8 实测刷新）。
> 论文三节（§5/§6/§7）同轮改写；`paper/main.tex` 等 14 个文件的先前会话改动
> **仍未提交**（不在本批次内）。
>


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
| `requirements.txt` | 29 | 依赖清单。⚠️ torch 写 `>=2.6.0`，而 README / CONTRIBUTING 写 `2.3.0+cpu`，**三处不一致**（待决 C5）；`transformers` 已于 2026-10-14 由 4.41.0 改锁 5.5.4（数字产出版本，见 commit_log 第 46 条）。头部「对齐 experiment_matrix.yaml」一句已加注：该规划件不存在（用户 2026-09-15 确认），对齐无从核对 |
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
| `attention_kernel.py` | 563 | **G1**：把 `CompactKV` 送进注意力的算子核（device-agnostic）。这是补齐 GPU 侧缺口的**第一块** —— 此前"紧凑 KV 算 attention"只以逐 query 的 Python 循环存在于 `src/distributed/dcc_kv_sync_cpu.py`（CPU、单 head、只做数值等价性验证），既不能批量化也跑不了 GPU。接口：`causal_visibility`（只认 `selected_indices`，**不**按块/锚点数组的行序切片）、`identity_compact`（B = L_s 的参照块、β≡0）、`compact_kv_attention`、`dense_attention`、`merge_partial_attention`、`dcc_kv_attention`（G2 与 H0 钩子共用的前向入口）。三条"形状对、量级对、只算错"的约束：SDPA 加性掩码的 ndim ≤ query 的 ndim；跨块归并必须等于"拼接后一次算完"（这条不成立则 A3/A4/E6 的 `dcc_kv` 行全错、而且错得很安静）；`query_chunk` 只到 ULP 级、非逐位。**2026-09-21 补（`b44a7b3`）**：`dcc_kv_attention` 的因果策略拆成**两个独立开关**（`local_causal` / `remote_causal`），使「源块全可见 + 本地因果」这一组合可表达（H0 的钩子要的正是它）；不给开关时按 `query_positions` 推导，旧调用点行为**逐位不变** | **2026-10-14（第 48 条）**：`dense_attention` 新增 `query_chunk`（514→563 行）—— 源端 Lq == Lk == L_s，整段一次前向每层都物化 `[g, Lq, Lk]` 显式 logits 及 3–4 个同尺寸中间张量（Qwen g=7、32K 时单张量 7.9 GB），是 32K 档 OOM 的主因。分块**不改变"刻意不走 SDPA"的性质**（每个子块仍是写开的三步，交叉验证价值保留），`query_chunk=None` 时逐位不变；`dcc_kv_attention` 同时把 `query_chunk` 透传给本地块（此前只覆盖远端紧凑块） |

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
| `attention_hook.py` | 894 | **H0 的方法侧**（2026-09-21）：把 G1 的算子核挂进真实模型 attention 的钩子。要点：注册**专用键**而非覆盖 `eager`/`sdpa`（后者一旦用 `pop` 还原就是把 transformers 自带实现删掉，不是还原）；source / destination 两阶段由调用方**显式声明**（不许靠「past 是不是 None」去猜 —— 猜错时产物看着完全正常）；目的端用 state 里的源端 K/V，**不是**传进来的 key（`cache.update` 在钩子之前，传进来的是「完整 cache + 本次新 token」）；返回值必须转置回 `[B, Lq, H_q*D_v]`；GQA 分组按 `repeat_interleave` 连续切；**不吃 `attention_mask`**（整列被遮蔽即抛错，否则 padding 会被当真 key）；`dcc_world` 单卡模拟**不体现代价化**（summary 里如实记 False 并附原因）。本机用本地构造的 tiny Llama 端到端验证：dense 参照臂与原生前向差 9.7e-08 | **2026-09-21 同日重写（`b44a7b3`，597→844 行）**：修掉两个被接线撞出来的契约漏洞 —— ① 本地块契约改为「传入的 cache **只含目的端本地 KV**」（旧实现取尾部 Lq，会丢掉评测路径的目的端本地上下文），并加**指纹守卫** `_assert_local_cache_excludes_source`（传入 cache 的前缀与 state 里源端 K 的 4 行采样逐位相同即抛错）；② `destination_phase(compacts=...)` 的 `compacts` 改为**必填**（`build` / `reuse`），杜绝「拿被打分的 token 去条件化」。另加 `budget_ratio`（评测源长逐样本变化）、`query_chunk`、`resolve_budget` / `edge_budget` / `describe`，并把因果策略交给 `K.dcc_kv_attention(local_causal=True, remote_causal=False)` | **2026-10-14（第 48 条，844→894 行）**：新增 `DEFAULT_LOCAL_QUERY_CHUNK = 512` 与 `resolve_query_chunk(cfg)`，源端 / 本地块 / 远端块三条路径**默认就分块**（`HookConfig.query_chunk` 为 None 时不再是不分块）。原文档写「本地块走 `dense_attention`，不受它影响」—— 那不成立：本地块与源端的 `[Lq, Lk]` 显式 logits 正是 32K OOM 的主因。`_source_attention` 双层保险（调用点传 `resolve_query_chunk(cfg)`，函数内 None 再兜底）；单测 Lq 均 < 512 ⇒ 分块不触发 ⇒ **逐位不变** |

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
| `run_gpu.sh` | 231 | GPU 实验入口（封装 `torchrun`）。⚠️ 数组展开必须用 `"${ARR[@]}"`，**不能**写 `"${ARR[@]:-}"`（后者会退化成空字符串参数）。2026-10-14 一天内修四批缺陷：①②（第 47 条）退出码透传（汇总取最大非零码，gate 的 3 不再被吞成 1）＋环境自举（解释器缺 torch→exit 3 带指引，不再 import 阶段崩掉）；③④（第 49 条）`usage()` 带退出码参数（无参/未知实验→2，首参 `-h/--help` 认帮助→0，旧实现恒 0 会让用法错误在 CI 里假装成功）＋torchrun 探测回落（PATH 找不到时查 `$PY` 同目录，覆盖 Windows venv 未激活时不在 PATH 的布局）。守卫 `tests/test_run_gpu_entry_2026_10_14.py`（13 项，真跑 bash 断言退出码） |

### 3.1 `experiments/common/` —— 公共模块（纯 CPU，只依赖 torch + numpy）

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `__init__.py` | 9 | 声明 `common` 包 |
| `synthetic.py` | 854 | 合成场景生成、完整注意力参考、误差与分布度量（含 KL / JS 等）。`mass_error` 已标废弃，公开偏移版本为 `absolute_mass_error`（列名 `eps_mass_abscommon_*`） |
| `report.py` | 390 | 结果汇总：`summarize`（median / p5 / p95 / bootstrap CI）、配对检验、`save_json` / `save_csv` |
| `hypotheses.py` | 875 | **H1–H5 阈值的单一事实源**（blueprint §3 的机器可读版）：阈值本体 + `h1_pass`…`h5_pass` + `HYPOTHESES` 注册表 + `UNJUDGED`（尚未接判据的假设，现为 `{H2,H3,H5}`）。2026-09-16 新增 H2 前提「质量相近」的判定程序 `quality_comparable_non_inferior`（入参显式含 `delta_pp` / `noise_floor_pp`，越出可行区间即 `ValueError` 而**拒绝执行**）与两个常量（`QUALITY_COMPARABLE_REFERENCE = "dense"`、非劣边界上界 = 1.5 pp）。2026-09-16 补 H2 的**跨长度聚合入口** `h2_pass_across_lengths`（逐长度分别判定、全通过才成立；前提未定记 `unresolved` 而非 `failed`），已接入 `e6_main_table.py`。**2026-10-17**：新增 `H2_UNRESOLVED_REASON_KINDS = ('resolution','instrument','structural')` 与 `H2PerLength.unresolved_reason`（默认回落 `resolution`）—— 「补参数就能判」与「补参数也判不了（单卡模拟）」从此分开记账 |
| `discrimination.py` | 418 | **「本条对照可否用于比较」判据的单一事实源**（2026-09-30 用户裁定新增）：判据 = **不一致对 > 0 且（方向不对称 或 参照臂两端都有余量）**。含 `mcnemar_exact_p`（精确双尾，`b+c=0` 时**拒绝执行**）、`discrimination_verdict`（返回原始统计量 / 三条判据分量 / `grade` / `licence` 四层结果）、`discrimination_verdict_from_arms`（直接吃 `niah_eval*.json` 的 `arms`）。三个登记常量：`DISCRIMINATION_ALPHA = 0.05`、`DISCRIMINATION_CI_LEVEL = 0.95`、`REF_HEADROOM_MIN_ACC = 0.10`。⚠️ 两处刻意设计：①`required_resolution_pp` **必填无默认值**；②`resolution_halfwidth_pp` 是**分辨率读数、不是可推断的置信区间**（推断以 `p_value` 为准；`b=0,c=4,n=20` 是实测反例）。与 `hypotheses.py` 分工：那个管**假设判据**、这个管**量具判据** |
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
| `e9_knob_localization.py` | 438 | **M–B 旋钮定位**：525 格扫描（`L_s=2048`、`B/L_s<=0.05`），分离"可达上界（B）"与"能否触及上界（M）" |
| `e10_beta_stability.py` | 656 | **β 稀疏塌缩**的定位与修复；把 β 分解为"块级常数分量（收益）"与"per-key 离散分量（代价）" |
| `e11_lambda_tuning.py` | 774 | λ_β 调参（尺度无关的相对正则强度） |
| `e12_representative_query.py` | 421 | **代表 Query 投影维度 $d_p$** 的扫描（`$d_p\in\{4,8,16,32,64\}$` + 无投影参照），报 JL 畸变 / 覆盖率 / 下游误差；5 种子 + 留出 |
| `e13_bound_tightness.py` | 706 | **误差界式 (37) 的紧致度检验**：逐 Query 比对 RHS/LHS（`rel_tol=1e-9`）与前置条件违反率，含块数轴（性质 3）；5760 个 Query、零反例。**2026-10-13 修正轮**：RHS 改为修正界 `[ε_out + ε_mass·(max_s‖Ô_s‖+max_s‖O_s‖)]/Ŝ`（旧界漏计分子扰动中的 `δM_s·Ô_s` 项、可被构造反例击穿，见模块 docstring）；旧界产物留档于同目录 `*_oldbound_20260922.*`；`ratio_stats` 的容差判据显式区分「取等号的浮点噪声」与「真反例」 |
| `c10_baseline_diagnosis.py` | 379 | 基线诊断：为何 `test_fast_kv_*` / `test_apb_*` 从未被 pytest 收集（包导出缺失 + `budget == L_s` 根本没压缩） |

### 3.3 `experiments/gpu/` —— 任务级与系统级实验（需 A100/H100 + NCCL + 真实权重）

> ⚠️ **本机无 CUDA ⇒ E5–E8 全部未执行**。环境闸门不达标时脚本**不写结果文件**、以退出码 3 结束。
> 因此"代码写完"不等于"实验做了"；仓库内**没有任何 GPU 数值**。

| 文件 | 行数 | 作用与内容 |
|---|---|---|
| `README.md` | 97 | GPU 实验说明、入口脚本、环境闸门与 device 修复记录 |
| `__init__.py` | 17 | **声明 CPU/GPU 两套证据的量纲关系**（E3↔A2、A3 机制↔A3 任务、E0↔E8），并明确不可互推 |
| `_env.py` | 656 | 环境闸门（`probe` / `enforce`，不达标退出码 3）、计时（`benchmark_ms`；窗口内只做 `device_sync()`，集合 barrier 留窗口外）、设备绑定（`local_rank_of` / `local_device`，取 `LOCAL_RANK`）、`CUDA_CONSTRUCTION_DEFECTS` 历史清单（第 50 条起含探针自身缺陷一条）、元数据构造。**2026-10-16（第 50 条）修设备比较误判**：新增 `_same_device`（type + index 规范化）——旧判定 `t.device == torch.device('cuda')` 中无索引 device 与张量 `cuda:0` 恒不相等，致 `probe_gpu_construction` 在**所有** GPU 机器上误报 `ok=false`；checks 补记 keys/values 设备消记账盲区。守卫 `tests/test_gpu_construction_probe_2026_10_16.py`（7 项） |
| `_comm.py` | 469 | 变长 All-to-Allv（阻塞 / 异步）、同步与异步流水线（`run_sync_pipeline` / `run_async_pipeline`）、分块 `_split_chunks`（逐 dst 取片，尺寸自洽）、体积口径（`make_uniform_plan`） |
| `_hf.py` | 918 | HF 模型加载、KV 预算裁剪（`apply_kv_budget`：identity / topk_rms / topk_norm / stride / random）、多项选择打分（`score_choices`，含独立 cache 副本 + 显式 position_ids）、评测（`evaluate`）、prefill 计时（**2026-09-21 起三段化**：源端构造 → 压缩 → 目的端前向，由 `PREFILL_TIMING_CONSUMES_COMPACT_KV=True` 声明且压缩臂缺 `dest_len` 即抛错）、KV 字节数 | **2026-09-21 补（`b44a7b3`）**：`attn_hook` 参数打通三个入口（`measure_prefill` / `score_choices` / `evaluate`）；新增 `prompt_split`，成为源段 / 目的端本地段切分的**唯一出处**（此前计时侧与评测侧各算一遍，实测导致同一行的预算读数对不上质量侧）；钩子路径下 `measure_prefill` 开**两个**计时窗口（含构造的 `prefill_ms_*` + 只目的端的 `dest_ms_*`，后者才是 kernel-matched 的量）；`_token_bound` 从**真实词表**取随机 token 上界（写死 1000 会让词表 < 1000 的模型死在 `F.embedding`，症状指向 embedding、真因在输入生成） | **2026-10-14（第 48 条，897→918 行）**：全部 12 处 `lm.model(...)` 显式传 `logits_to_keep`（源段/目的段 prefill = 1；continuation = T）。transformers 5.5.4 默认 0 = 全长过 lm_head ⇒ 32K 源段白物化 5.9–7.0 GB，且 `score_choices` 的 `base_logit` 是**视图**会把整份 logits 拖到打分结束。该口径写进 `score_choices` 文档第 5 条与 `measure_prefill` 的 Args，并有 AST 守卫（见 `test_gpu_memory_guards_2026_10_14.py`） |
| `_forward.py` | 426 | **G2**：异步 All-to-Allv 接**真实前向**的桥接层。`make_uniform_layout`（统一布局，供等价性验证）/ `pack_edges`（变长 edge 打包）/ `chunk_source_sizes`（源端分段约定，与 `attention_hook.source_partition` 同款 —— 两处必须一致，否则"每段长度"在两套代码里不同值）/ `decode_edges` / `pipelined_attention`（同步、异步两条流水）/ `assert_same_answer`。计时纪律：`T_build / T_comm / T_comp / T_total` 必须**拆开报**；异步的 `comp_ms` 是**上界**（`wait(handle)` 之后不允许任何 device-wide 同步，否则把正在飞的集合通信也等掉 ⇒ overlap 恒为 0）；同步与异步之间**只有 `total_ms` 可比**。⚠️ 单卡下真 `dist.all_to_all_single` 退化成一次本地拷贝、`chunks_effective=1`、**没有可重叠窗口** |
| `build_eval_set.py` | 557 | **G5**：评测集转换器 —— 把 LongBench 转成 `_hf.EvalSample` 的 JSONL。为什么需要它：E5-A2/A3、E6、E7 的**准确率列全部依赖它**，而 `_hf.score_choices` 只认 `{prompt, choices, answer, task, length_tag}` 这种多项选择形态。含 `convert_records` / `validate_samples`（转换后自检）/ `write_jsonl` / `write_report` 与 `run_selftest`（无数据也能跑）。⚠️ **评测集本体仍需生成** —— 本脚本是纯转换器，需要外部的 LongBench 输入数据 |
| `e5_gpu_ablation.py` | 1211 | **A1** 通信集大小、**A2** 压缩预算扫描、**A3** 组件拆分（被阻断，拒绝产假数字）、**A5** 异步 vs 同步（H4 判据走 `hypotheses.h4_pass`）。**A4**（压缩 × 异步交互，2026-09-20 已实现）。**A3 仍被阻断** —— 但**理由已换**（2026-09-21）：G1 算子核与 H0 的钩子都已就绪（`src/distributed/attention_hook.py`），真正剩下的是**逐边条件化的多设备语义**：单进程 harness 只有单一目的端，无法区分 DCC-KV 与 FastKV，故 A3 的对照必须走多设备路径。在此之前拒绝产假数字 |
| `e6_main_table.py` | 1386 | 主表与可扩展性。`gpu_count` 记为**方法要求**而非自由轴；sync/async 轴对单卡方法**折叠**（不生成两行相同的数）。⚠️ **2026-09-21 更新**：H0 已接（量具灵敏了），但 `dcc_kv` 等 4 个方法**仍** `measurable: False` —— 真实阻塞换了：`dcc_kv` 与 `kv_budget_shared` 仍走同一条 `apply_kv_budget`（共享裁剪）⇒ 质量差恒为 0。此时翻 `measurable` 会得到**假阳性**（拿一条不是本文方法的通路去判质量臂）。prefill 加速比分子已改判为 **dense**。**同日补**：H0 的**方法侧**钩子已就绪（`src/distributed/attention_hook.py`，本机 tiny Llama 端到端验证），但它**尚未接进 `measure_point` 的方法行** —— E6 的 `--dcc-world` **尚未加入 CLI**（钩子侧已有 `HookConfig.dcc_world`），接线完成前 `dcc_kv` 行仍与 `kv_budget_shared` 同路 | ⚠️ **已被 `b44a7b3` 修正**（上句保留）：`dcc_kv` 的 **`measurable` 已翻为 `True`**，其方法行改走 `attention_hook`；E6 加 `--dcc-world`（**无默认值**）/ `--dcc-budget-mode` / `--ranks`（默认 1）/ `--lambda-beta`。同轮另三处口径修正：折叠判据推广为「**本次运行真的没有多卡**」（`--ranks 1` 时 `dcc_kv` 的 sync 轴也折叠）；prefill 加速比落**两列**（`prefill_speedup_kernel_matched` 给 H2 用，`prefill_speedup_native` 只作端到端参考；**缺同核列时记 unresolved 而非 failed**）；计时与评测共用同一份 `_hf.prompt_split`。**2026-10-17 三处修正**：① 产物 `paper_claim` 写死的「3×4×2×2×3=144」**废止**（该式漏掉 DCC-KV 自身、单卡时 sync 轴折叠、代码无「GPU 数」轴）——点数一律读 `n_points_planned`；② 新增**逐点落盘** `e6_rows.partial.jsonl`（append + fsync，断连不丢；旧 partial 改名归档、不自动续跑）；③ `ranks=1` 时 H2 的 unresolved 原因记 `structural` 并在 diagnostics 里写明「补参数无用」 |
| `e7_negative_results.py` | 491 | 负结果四条件：短上下文 / 低预算 / 强 retrieval / batch=1。**2026-10-17**：`--model` 默认由 Llama-3.1-8B 改为 `Qwen/Qwen2.5-7B-Instruct`（Llama 为 gated，未授权时后台长跑到加载模型才 401） |
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
| `test_gpu_construction_probe_2026_10_16.py` | 224 | **G6 探针（`probe_gpu_construction`）守卫**（2026-10-16 新增，7 项，全 CPU 可跑）。修第 50 条的设备比较误判：`torch.device('cuda') != torch.device('cuda:0')`（index None vs 0，任何 torch 版本恒不相等）曾令探针在**所有** GPU 机器上误报 `ok=false`。锚点：① device 相等性语义钉死（坑的根）；② mock 假张量复现「四条全在 cuda:0 ⇒ 必须 PASS」（修复前恒 FAIL，CPU 上即可验服务器场景）；③ 反向「真滞留 CPU ⇒ 必须 FAIL 且 checks 指名肇事者」（防被改成永真）；④ checks 四设备字段齐全；⑤ AST 源码守卫（probe 内裸 `X.device == dev` 不许回归） |
| `test_forward_pipeline.py` | 532 | **G2 的锚点**（stub 顶掉集合通信后可纯 CPU 跑）。核心不变式只有一条：**流水只改变“什么时候算”，不改变“算什么”**。另有打包/解码互逆、块内源切分、退化为单块时如实上报。⚠️ 为躲开 conftest 的按名自动打标（含 `gpu` / `nccl` / `end_to_end` / `async_overlap` 会被默认排除），本文件**刻意避开这些词** |
| `test_a4_interaction.py` | 409 | **G4（A4 压缩 × 异步交互）的锚点**。A4 是**唯一**把测量与判定都放在设备无关路径上的消融，故本文件用 stub **真跑一遍** `a4_interaction_grid` 而非只查字符串（字符串检查挡不住“接线接反”）。三类锚点：判定语义、接线纪律（判据不得被重新实现、CI 水平不得硬编码）、测量正确性（A4 计算侧必须是真实算子核，T_comm/T_comp 取自同步臂） |
| `test_selfcheck_2026_09_18.py` | 362 | **2026-09-18 自查的回归锚点**：每条对应一个已修缺陷，且都是“形状/量级/文件名都正常、不报错但结论错或永远出不来”的那一类（如 `paired_bootstrap` 方向反了、`save_csv` 遇混合行整表 `ValueError`、`h4_pass=None` 被折进“未达标”） | **2026-09-21 补（`b44a7b3`）**：T6 新增一条 —— **单卡下表里写出的 `dcc_kv` 行 `sync_async` 也是 `"n/a"`**，H2 配对必须仍能找到它（旧实现只给基线做了轴无关查找，dcc 那一侧仍按 `sync_modes[0]` 查 ⇒ 折叠后配对恒为空；这是 `bee3388` 给基线修过的同一个坑在 `dcc_kv` 身上的第二次） |
| `test_adversarial_2026_09_20.py` | 1008 | **2026-09-20 对抗性审查 + 2026-09-21 H0 接线的回归锚点**：12 处已修缺陷（D1–D12）各一条，每条先证明「它本可以不被发现」。**A 组已于 2026-09-21 改写** —— 原锁「量具失敏 ⇒ 加速比结构上恒不超 1」，现锁「目的端看到的 KV 必须等于 B、两臂唯一差别就是那个长度、漏给 `dest_len` 必须抛错、分子必须是 dense」；B 组异步臂 `comm_ms` 是上界（落 `nan` 而非 0）；C 组重复次数与 `lambda_beta` 默认值必须进产物且与项目默认**同源**；D 组用 AST 扫设备索引（`set_device(rank)` / `f"cuda:{rank}"`）；E 组双边同步次数断言；F/G 组 E8 元数据与 `json_safe`；**H 组元守卫**保证本文件自身的用例默认全部运行；**I 组（2026-09-21 新增、同日扩展）**断言源码与 `docs/*.md` 里引用的**仓库内任意 `.py` 路径**真实存在（原先只守 `tests/test_*.py`，于是 `src/distributed/attention_hook.py` 被两处源码引用却不存在这件事它看不见） | **2026-09-21 第三次更新（`b44a7b3`）**：A 组的分子由 dense 改为 **kernel-matched**（同核），并新增「缺列时不得回落 native、且不得把 nan 比成未达标」；另新增 **I4**：`docs/*.md` 里不得有**未闭合的表格行**（`FILE_MAP.md` 曾有 4 行以 ``| **G1`` 这种半截形态存在，而它的职责恰是逐项说明每个受控文件）。**2026-10-17**：A2 扩为三类原因（`resolution` / `instrument` / `structural`），新增 `ranks=1` 时记 `structural` 且 diagnostics 含「结构性不可判」的断言 |
| `test_audit_2026_09_23.py` | 229 | **2026-09-23 论文架构 ↔ 代码工程一致性审查的回归锚点**（2026-10-13 随 D8 批次入库，10 项）。守卫本轮 D8 修复的核心事实：论文 §4.4 点名的三项机制（循环错位 `[s+p]` 调度 / `cuda.Stream`+`cuda.Event` 独立双流 / §4.2 的 `N-1` 轮 Ring All-Gather）在代码中**不存在** —— 实际是单次全局 `all_to_all_single` + 块级双缓冲。故处理方式是**改纸面追上实现**而非改代码圆纸面：单次集合通信在带宽利用上本就优于 `N-1` 轮环形，且改代码需 GPU 验证而本机无 CUDA。每条断言先说明「它本可以不被发现」：`grep async` 与 `grep All-to-Allv` **都会**命中，唯独三个具体机制名（错位轮次 / CUDA Stream / all_gather）零命中 —— 这正是「公式忠实、区间正确、假设对齐，而机制从未被建造」那一类偏差 |
| `test_discrimination.py` | 434 | **量具判据（`experiments/common/discrimination.py`）的锚点**（2026-09-30 新增，29 项）。四类断言：①**判据字面**（用户裁定的三段式不得被改动）；②**真实黄金值** —— NIAH x20 / LongBench n=20 的逐样本对错**抄自落盘产物**（抄进来而非读文件，因为产物在 `_onetime_scripts/_out/`、**仓库测试不得依赖仓外文件**）；③**边界语义**（`b=0,c=5` 的 `p=0.0625` **不算**方向可分辨 —— 这正是 n=8 那轮把「方向复现」误当结论的坑）；④**声明↔实现绑定**（阈值只有一处赋值，用 AST 按**结构**判定而非 grep 字符串；文档必须点到该模块）。另含一条反例锚点：`b=0,c=4,n=20` 时分辨率读数 19.6 pp < `|Δ|` 20 pp 却**不显著** ⇒ 钉死「读数不是置信区间、推断以 `p_value` 为准」 |
| `test_errbound_fix_2026_10_13.py` | 301 | **式(37) 误差界修正的回归锚点**（2026-10-13 新增，11 项）。四组守卫：①**反例回归** —— 初版界必须在构造反例（O₁=0、O₂=e₁、δM₁=0.1、δO₁=6e₁，前置条件满足而 LHS=3.119 > RHS=3.100）上违反（哨兵：证明反例有杀伤力），修正界两种形式必须成立，且 δM≡0 时必须退化为旧界的取等情形（「修正不丢旧界已正确的部分」）；②**对抗搜索哨兵** —— 固定种子 5000 次多块随机搜索，新界在 1e-9 相对容差外零真反例，同时旧界必须被击穿 > 0 次（证明搜索有击穿能力，不是「搜不到」）；③**实现 ↔ 论文学术对齐** —— E13 脚本的 RHS 必须含 `eps_mass * (o_hat_max + o_max) / S_hat_q`（三项缺一不可），论文方程必须含 `\widehat{S}` 与紧凑输出项，且论文三节必须携带本轮数字（2.23/7.37/9.56/190/14.6%）而不残留旧轮次排他数字（2.10/3.51/2.08/3.61/6.2%/4/60）；④**落盘产物复算** —— 论文引用的汇总数字从 `results/cpu/e13/e13_results.json` 逐位复算，并断言旧界留档副本存在（它承载「旧界为什么错」的对照证据，按项目删除边界不得删除） |
| `test_fstring_backslash_guard_2026_10_13.py` | 117 | **f-string 反斜杠全仓守卫**（2026-10-13 新增，2 项）。背景：e9 表头 `f"{'M\B':>6}"` 在 py3.11 是 SyntaxError（整个模块导入即失败）、py3.12+（PEP 701）合法 ⇒ 该类 bug 在 repo-tests（py3.13）全绿下能隐身多轮。守卫按**解释器精度分流**：py3.12+ 查 `FormattedValue` 源码段（位置精确）；py3.11 改查「表达式内字符串常量的解码值」（`get_source_segment` 在 py3.11 会溢出到字面部分，实测误报 7 个文件） |
| `test_gpu_memory_guards_2026_10_14.py` | 323 | **GPU 前向显存口径守卫**（2026-10-14 新增，10 项）。四道：① AST 守卫 `_hf.py` 每个 `lm.model(...)` 必须显式含 `logits_to_keep`（含 10 处单点变异自测 —— 删任一处的参数都必须变红）；② AST 守卫 `attention_hook.py` 中 `dense_attention` / `dcc_kv_attention` / `_source_attention` 调用必须传 `query_chunk`，且钩子内必须走 `resolve_query_chunk(cfg)`（不能退回裸 `cfg.query_chunk` = 默认不分块）；③ `dense_attention` 分块后与不分块 ULP 级等价（数值，含 `query_chunk=None` 逐位不变的锚点）；④ `_source_attention` 默认分块**真的发生**（spy 抓实收参数）且结果与显式不分块 ULP 级等价。设计原则：默认分块的阈值（512）远大于全部单测的 Lq ⇒ 单测逐位不变，只有真实长序列才分块 |
| `test_paper_hypothesis_wiring_2026_10_14.py` | 109 | **论文↔代码「假设接线状态」一致性守卫**（2026-10-14 新增，3 项）。修 2026-09-23 审查记账的自相矛盾：`06-experiment.tex` 曾写「五条假设中只有 H1 与 H4 在代码侧接了判据」，而同文件后文与代码事实（`hypotheses.py` 登记 + `e6_main_table.py` 调用 `h2_pass_across_lengths`）都表明 H2 入口已接进 E6 —— 错的是前一句。守卫钉两侧不再各自漂移：① stale 二元表述不许回来；② 论文**四态**表述（已判定 `judged` / 入口已接·参数缺 / 结构性不可判（单卡模拟）/ 尚未接线）与登记的 judged 集合、H2 的 judge 名与 `code_status`、E6 调用点逐项对应。**2026-10-17**：H2 那一态拆为「参数缺」与「结构性不可判」，并加回退守卫 —— 论文再写「仅剩参数待测」即红 |
| `test_run_gpu_entry_2026_10_14.py` | 388 | **`run_gpu.sh` 入口守卫**（2026-10-14 新增，13 项；⚠️ 第 47 条入库时**漏建本行**，第 49 条补上）。全部真跑 bash、以退出码断言，不 mock：gate 3 透传 / 参数拒绝 2 透传 / `--plan` 仍 0 / 缺 torch→3 带指引 / 解释器不存在→127 / 无参→2 / 未知实验→2 / `-h`·`--help`（首参与实验名后两种位置）→0 / `--plan` 位置陷阱→2 / torchrun 同目录回落（收窄 PATH + 假 torchrun 实弹，顺带验参数拼装与退出码透传）。⚠️ 文件自带一个**测试基建坑**的修复：本机 PATH 里第一个 `bash` 可能是 WSL 启动器（`system32\bash.EXE`），它收下 Windows 路径也找不到脚本 ⇒ BASH 解析加了「能跑 Windows 路径脚本」的实弹验证，候选全不可用才 skip（不是 fail：换台没 bash 的机器不该把守卫染红） |
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
| `sections/06-experiment.tex` | 1159 | **第 6 章 实验与证据**。设置、评测协议与元数据规范、**已验证的 CPU 实验**（E0–E11 / A3 机制级）、**GPU 实验的执行状态与协议**（E5–E8，含 2026-09-16 新设的 **A4** 压缩 × 异步交互）、结果报告规范。§6.2 的 H1–H5 阈值已显式标注为**待验假设**。**2026-10-14 修一处自相矛盾**（第 49 条）：原文「截至撰写时五条假设中只有 H1 与 H4 在代码侧接了判据」与同文件「假设判定的接线状态」段及代码登记矛盾（H2 入口 `h2_pass_across_lengths` 早已接进 E6）—— 已改为三态表述（H1/H4 已判定 / H2 入口已接·参数待测 / H3/H5 尚未接线），守卫见 `tests/test_paper_hypothesis_wiring_2026_10_14.py`。**2026-10-17 口径修订**：① 章节名去「方案」、去「待执行」；② 模型清单 4→3（删 70B/72B，Llama 标注 gated）；③ E6 点数 144 废止，改「3×4×6 单卡=72 计划 / 36 可测 / 36 阻断」现算；④ H2 改「结构性不可判（单卡模拟）」并区分 `structural`/`resolution`；⑤ §6.1.3 写死「质量只测一次、长度轴只作用于计时侧」；⑥ §6.5 报告规范由两态改三态（已执行 / 未执行 / 结构性不可判） |
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
| `git_strategy.md` | 419 | Git 管理策略：分支、commit 体例、tag、实验可追溯。§8.1 记录 **`results/` 入库**的决定与体积策略。头部「配套文档」已加注：blueprint v1.1 与 experiment_matrix.yaml v1.2.0 均不在库内（用户 2026-09-15 确认手边无） |
| `release_checklist.md` | 177 | 投稿前 / Camera Ready 清单。§4 是 failure_thresholds 校核（H2/H3/H4/H5）；2026-09-16 起该节附「质量相近」的定义摘要 |
| `reproducibility.md` | 314 | 可复现性说明。§6 的 H1–H5 及其阈值：2026-09-21 起口径来源改为**论文原件（`AuthorKit27` 第 7 章）+ 仓库内机器可读实现 `hypotheses.py`**（原先声明「以 blueprint §3 为准」，该原件不在库内、已裁决不再作为权威锚点，原 C11 关闭）；2026-09-16 起附「质量相近」的完整定义摘要；H2 的 prefill 加速比分子于 2026-09-21 改判为 **dense**。文末「配套文档」清单里 `../experiment_matrix.yaml` 已加不存在标注 |
| `writing_scope_and_metrics.md` | 210 | **撰写范围与指标口径**。以第 1–4 章原始建模（桌面 `AuthorKit27 (1).pdf`）为标尺的「建模条款 → 应报指标 → 口径 → 现状 → 缺口」对照表：六个误差术语 ↔ 四个代码度量函数 ↔ §5 理论量的三方映射；缺口 M1–M9 及各自判定标准；**8 条**口径硬约束。§6 记录历次裁决的落实（D1–D6）与仍开放项。**2026-09-16 第八轮：M1/M2/M3/M5/M9 全部关闭**（见 §4 后附注）。第九轮补上 E0 的「置换次数」轴，论文 §6 里最后一条非 GPU 待补项关闭。**2026-09-30**：§5 第 4 条**显式修订**为「单一事实源有**两个**模块」（原写「一律走 `hypotheses.py`」，现按职能拆为「假设判据 `hypotheses.py` / 量具判据 `discrimination.py`」）；新增 §5 第 8 条 = 量具判据本体，并声明其**非 AuthorKit27 原件内容** |
| `gpu_execution_plan.md` | 460 | **GPU 实验执行计划与预算控制**（2026-09-16 新增）。结论先行：现在不能开跑 —— 缺的不是卡是代码，依据 `e6_main_table.py --plan` 实测 6 方法中 4 个 `blocked`（含本文方法 `dcc_kv`）。含 G1–G6 前置缺口、S0–S3 四级梯队、数据量与时长估算、预算纪律十条、租卡前 checklist、论文完整性核对。**§0/§5/§6/§7 已于 2026-09-21 第二次同步**：G1–G5 已**提交**（`c9ca6b0`），**H0 已接**（`measure_prefill` 三段化 + `PREFILL_TIMING_CONSUMES_COMPACT_KV` 翻正 + E6 加 `--dest-fraction`）。**未变的是结论，变了的是理由**：现在阻的是「`dcc_kv` 的构造路径还没独立」（仍与 `kv_budget_shared` 共用共享裁剪 ⇒ 质量差恒为 0）与 **G6 未做**；翻 `measurable` 仍不可以（否则得到假阳性） | ⚠️ **已被 `b44a7b3` 修正**（上句保留）：**接线已完成、`measurable` 已翻**；§0/§5/§7 已第三次同步，口径变化三条（sync 轴折叠判据推广、prefill 加速比两列、计时与评测同一份切分） |
| `ssh_setup.md` | 141 | SSH / 远程机器配置 |
| `FILE_MAP.md` | 本文 | 文件说明（你正在读的这份） |
| `reports/2026-09-30-实验结果复核操作手册.md` | 950 | **对外复核操作手册**（2026-09-30，第 46 条首次入库）。「准备篇」讲路径替换与自建环境；§11 是给发件人的打包清单。⚠️ 手册全部数字在 transformers **5.5.4** 上跑出，而 `requirements.txt` 当时锁 4.41.0 —— 第 46 条已把锁改成 5.5.4 并同步手册三处版本口径 |
| `reports/2026-10-14-GPU显存需求预估与机型建议.md` | 370 | **GPU 显存需求预估与机型建议**（2026-10-14，第 48 条入库）。委托：实验室配 GPU（48G vs 96G、独占式分配）前的申请依据。结论：两处代码级显存风险修复后**全部场景 ≤ 27.7 GB，48 GB 卡全覆盖**。含量化口径（三条建模口径不说明就会被读错）、修复前/后双口径汇总表、分量拆解、机型判定、申请梯队与费用对照（校内 A800 3.5 元/卡时 ≈ 144 元全轮）。⚠️ 复算脚本 `gpu_memory_estimate_20261014.py` 在**仓库外** `_onetime_scripts/`（纯算术、约 250 行），未随仓库分发；报告 §1.1/§2.2 列全了输入参数与分量定义，可依其重写复算 |
| `reports/2026-09-29-H2可判性说明.md` | 162 | **H2 可判性说明**（2026-09-29）。「质量相近」前提的两个参数（非劣边界 δ 与噪声底线）为何按设计**无默认值**、未声明时记 `unresolved` 而非 `failed` 的结构性论证；H2 维持 `no-judge` 的原因沿革（定义缺失→聚合未定→参数待测→质量侧通路未独立→数据未落盘） |
| `reports/2026-09-29-链路可执行性证据说明.md` | 218 | **链路可执行性证据**（2026-09-29）。无 GPU 本机上「真权重 → 长文本切分 → 压缩 → 评测」链路完整跑通的证据链；核心质量主张必须真多卡、本机结构上测不下的论证 |
| `reports/2026-09-30-扩样复验-LongBenchx20.md` | 318 | **LongBench 扩样复验 n=20**（2026-09-30，第 49 条入库）。四臂配对 McNemar 全部 p=1.0000 ⇒ **推翻**「跨探针次序翻转（topk>rand）」这条曾被称为"本节最重要结论"的表述（原表述在源报告中已留原文加修正标记）；分辨率读数 `resolution_halfwidth_pp` 不是可推断置信区间的反例（b=0,c=4,n=20 读数 19.6pp 而 p=0.125）。⚠️ 入库时两处处理：复现命令的本机绝对路径换成占位符（替换表见复核手册准备篇）；文末「尚未提交」状态加 dated 补注（该轮产物早已入库推送，`cdc312d` 已证为远端祖先） |
| `reports/2026-09-30-扩样复验-NIAHx20.md` | 152 | **NIAH 扩样复验 n=20**（2026-09-30，第 49 条入库）。五臂配对统计：topk 相对 rand/stride **b=0、c=14/15、p=0.0001**（真效应）；与 LongBench 侧 b≈c、p=1.0（测不出）的对照 = 「无显著差异两种成因不可混读」的实证。⚠️ 入库时复现命令的本机绝对路径换成占位符 |
| `reports/2026-09-30-评测集与压缩敏感性验证.md` | 399 | **评测集区分度验证**（2026-09-30，第 49 条入库）。答案在文本中重复 ⇒ 对压缩零敏感（旧 FILLER×400 即此病）的实测；量具判据的成型过程（不一致对>0 且方向不对称）。注意其中 §9.2「次序翻转」表述**已被 LongBenchx20 报告修正**，引用以修正后为准 |
| `reports/2026-09-30-长文本长度档决策记录.md` | 106 | **长文本长度档决策记录**（2026-09-30，第 49 条入库）。4k/8k/16k/32k 档位取舍的依据与代价 |
| `reports/2026-10-04-PPT汇报素材包.md` | 475 | **PPT 制作素材包**（2026-10-04，第 49 条入库）。给第三方的汇报素材，不是 PPT 本身：三条引用红线（不写性能数字 / 不写压缩无损 / 不提基线）、逐页素材与「阴性结果是亮点」的呈现要求。数字均从落盘产物现读核对 |
| `reports/2026-10-04-组会汇报素材-4页.md` | 204 | **组会汇报素材（4 页）**（2026-10-04，第 49 条入库）。同上素材的四页精简版，47 项断言核对通过 |

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
| `2026-09-22-lambda-zero-and-interval-verdicts.md` | 302 | **Q1–Q3 裁决的落地与证据**（2026-09-22）。Q1：`λ=0` 非绝对限制但实测更差（超定域也差 4.3%）⇒ 维持 `3e-2`；Q2b：E9 搬进声明区间（现为 `results/cpu/e9/`），M 主导性 1.83× → **2.89×**（⚠️ 报告正文的 0.4407/0.1364/3.23× 取自已删除的 `e9_L2048/`，**已被 `ad2f091` 修正**）；Q3：FPS 起点 / 复杂度按论文改代码、箱约束实测 0.038% 保留。文末列**三项待裁** |

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
