# 提交变更报告日志

> 本文件按提交顺序倒序记录 `dcc-kv` 仓库的每一次提交：动机、改动清单、验证证据、遗留项。
> 与 `docs/git_strategy.md`（规范）互补 —— 那份说「应该怎么提交」，这份说「实际提交了什么、验没验证过」。
>
> 生成时间：2026-09-13 20:35 (GMT+8)
> 当前 HEAD：`8536bbb`（分支 `paper/sections-5-8`，**有 5 次提交未推送**）

---

## 0. 总览

| # | 短哈希 | 日期 | 作者 | 类型 | 文件数 | +行 | −行 |
|---|---|---|---|---|---|---|---|
| 10 | `8536bbb` | 2026-09-13 20:34 | Saluneo | docs | 1 | 535 | 0 |
| 9 | `e95927f` | 2026-09-13 20:24 | Saluneo | docs(paper) | 2 | 324 | 92 |
| 8 | `511ca1e` | 2026-09-13 20:24 | Saluneo | exp(cpu) | 4 | 1538 | 0 |
| 7 | `5b5ce98` | 2026-09-13 20:24 | Saluneo | fix(dcc_kv_ref) | 10 | 213 | 76 |
| 6 | `72af7db` | 2026-09-13 18:25 | Saluneo | exp | 24 | 6255 | 2 |
| 5 | `907d356` | 2026-09-13 17:44 | Saluneo | fix(paper) | 2 | 24 | 20 |
| 4 | `f0677f4` | 2026-09-13 17:20 | Saluneo | docs(paper) | 14 | 1823 | 0 |
| 3 | `8a1d275` | 2026-09-08 09:15 | Mavis | fix(requirements) | 1 | 6 | 6 |
| 2 | `fcd9718` | 2026-09-08 07:40 | Mavis | docs | 5 | 463 | 0 |
| 1 | `7ebff65` | 2026-09-08 06:36 | Mavis | feat | 43 | 5382 | 0 |
| | | | | **合计** | **106 次文件变更** | **16563** | **196** |

仓库当前规模：89 个受版本控制文件，其中 54 个 `.py`、9 个 `.tex`。

分支与推送状态：

```
* paper/sections-5-8   →  origin/paper/sections-5-8
                           (落后 5 个提交：5b5ce98, 511ca1e, e95927f, 72af7db, 8536bbb)
  main                 →  origin/main
```

---

## 10. `8536bbb` — 补记前三次提交的变更报告（本文件首次纳入版本控制）

类型：`docs` ｜ 1 文件 ｜ +535 / −0 ｜ 2026-09-13 20:34

### 动机

用户要求「给出每次提交的修改完善报告日志」。本文件即该日志，
在 `72af7db` 时以未跟踪状态存在（记录了 1–6 号提交），本次补入 7–9 号提交并将其纳入版本控制。

### 改动清单

- 新增 §7 / §8 / §9 三节，体例与前六节一致（动机 / 改动清单 / 验证证据 / 遗留项）。
- §0 总览加入 7–10 号提交，合计由 `89 次文件变更 / +13953 / −28` 更新为
  `106 次 / +16563 / −196`。
- §P 由单一表格拆为三段：**P.1** 已解决的 C1–C4（含解决方式），
  **P.2** 仍开放的 C5–C11，**P.3** 三项待用户决策。
- §Q 明确区分 H2 的机制级证据与任务级结论。
- §4（`f0677f4`）的写作边界中，给"E3 标注为仅量级验证"一条加修正批注。

### 一条体例决定

**提交日志记录的是每次提交当时的真实状态，不是事后修订稿。**
因此 §4 中已被推翻的表述保留原文，另加 `⚠️ 已被 e95927f 修正` 批注说明其被推翻的两点，
而不是把原文改写掉。理由是：日志的价值一部分正在于留下"当时为何这样判断"的痕迹 ——
若事后统一抹平，就看不出哪些判断是随证据推进而改变的。

### 遗留项

- 本日志需在每次提交后同步更新（当前已同步至 `8536bbb`）。
- 若后续将 §P.3 的三项决策落定，应把决策结果回填到对应条目，而非仅更新状态。

---

## 9. `e95927f` — 重写 §6 实验设计，补 E2b/E9，修正两处公式溢出

类型：`docs(paper)` ｜ 2 文件 ｜ +324 / −92 ｜ 2026-09-13 20:24

### 动机

`511ca1e` 与 `5b5ce98` 产出了新的判定实验与实现修复，§6 必须与之一致；
此外上一轮写作留下的两处问题必须处理：一是状态表与本节小结仍是旧口径
（E3 标为"仅量级验证"），二是 E9 结论里有一处我自己复制粘贴产生的数字错误。

### 改动清单

**内容**

| 位置 | 改动 |
|---|---|
| §6.3.5（新增） | **E2b**：β 链路口径的判定。判定依据（源论文 Eq.(2) 与本文 Eq.(15)(16) 系数均为 1）+ 45 配对样本逐项对照表（单块 / 归并 / 绝对质量误差）+ 三条要点 |
| §6.3.6（新增） | **E9**：M×B 网格表（fit / oracle / gap），B 决定可达上界、M 决定能否触及上界 |
| §6.3.3 重写 | E2 增加"关于既有测试的一处更正"与"一处度量更正" |
| §6.3.4 重写 | E3 明确 H2 证据等级限定在机制级（20 组中 17 显著 / 3 不显著 / 0 反向） |
| §6.3.9（新增） | 「由实验反查并修复的实现问题」：八处问题逐条记录，七修一待办 |
| §6.5 更新 | 状态表新增 E2b/E9 行，E3 由"仅量级验证"改为"机制级证据"；本节小结改为三条特点 |

**关键判断（值得单独记录）**

1. **E2 引用既有测试是错的，已停止引用。** 早期版本引用
   `test_fast_kv_vs_dense_within_tolerance`（声称 $B=64$ 时偏差 $<0.1$）与
   `test_apb_vs_dense_within_tolerance`（$<0.3$）作为压缩误差量级参照。复核结论：
   两项断言**当前均不成立**（实测 1.09 与 1.99，超阈值 10.9× / 6.6×）；更关键的是该测试
   `chunk_size = budget = 64`，即 $B=L_s$，**根本未发生压缩** —— 所以即使断言成立，
   也不能作为压缩保真度证据。这是一个"数值对不上"之外的**逻辑不成立**问题。
2. **E9 结论中的数字错误。** 原写"oracle 沿 B 由 8→128 下降 86%（M=4）到 86%（M=48）"，
   两个 86% 是复制粘贴所致。重跑 E9 取准数后改为 67.5%（M=4）与 83.8%（M=48）。
   按既定规则，拿不准的数字不得留在正文里。
3. **E2b 置于 E3 之后是刻意的。** 该口径问题并非设计时预见，而是在 E3 配对检验出现
   "方向随关注强度翻转"的异常时才暴露 —— 叙述顺序与发现顺序一致，已在正文写明理由。
4. **§6.3.9 中必须记录的后果**：在补全包导出之前，`tests/test_dist_equivalence.py`
   无法被 pytest 收集，**该文件的历史测试从未真正执行过**。因此早期版本中作为 E2/E3
   证据引用的测试名，其证据效力为零。

**版式**

- 新增段落中的长标识符/路径加 `\allowbreak` 断点（`test_*_vs_dense_within_tolerance`、
  `experiments/cpu/*.py`、`rope_extension_*` 等）。
- 实验状态表改 `p{}` 列并收紧 `tabcolsep`；E9 表降为 `footnotesize`。
- `main.tex` 第 4 章两处行间公式超栏宽 25–32pt：eq(19) 岭回归闭式解、eq(22)
  Online Softmax 状态定义，改 `split` 折行。这两处是**用户原稿第 1–4 章**的内容，
  属版式修正，不涉语义。

### 验证证据

```
xelatex → bibtex → xelatex ×2     通过
未定义引用                          0
输出                                main.pdf, 12 页
Overfull hbox                       24 处 → 8 处（剩余均 ≤15pt，中文标点悬挂）
PyMuPDF 逐块越界检测                 无文本越出右边距（修复前 p3 有两处 25–32pt）
```

### 遗留项

- 剩余 8 处 overfull 均为中文标点悬挂（`”` 紧跟于 CJK 之后），
  属 xeCJK 的标点压缩设置问题，非内容问题。若投稿前要求零 overfull，
  需在 `main.tex` 调整 `\xeCJKsetup{PunctStyle=...}` 或加 `\sloppy` 全局。
- 参考文献表最后一条（`[17]` Attention Matching）在页边界附近，需在最终版复核。

---

## 8. `511ca1e` — 新增 β 口径判定（E2b）与 M–B 旋钮定位（E9）

类型：`exp(cpu)` ｜ 4 文件 ｜ +1538 / −0 ｜ 2026-09-13 20:24

### 动机

用户对上一轮提出的两个问题给出指令，本提交是这两个指令的兑现：

1. 关于「β 的训练/推理口径相差 M 倍」——用户判定"大概率是笔误，按照最合理的情况判断即可"。
   但"最合理"需要有依据，因此先建一个判定实验（E2b），而不是直接改代码了事。
2. 关于「信息瓶颈是 M 还是 B，论文主线是否要重写」——用户要求"先补一组 M 扫描数再定"。
   本提交即补上这组扫描（E9），**不**改动论文主线（改与不改留给数据决定）。

两个脚本都不产出可写入论文的**结果**，只产出结论成立所需的**前提**。

### 改动清单

| 文件 | 行数 | 内容 |
|---|---|---|
| `experiments/common/beta_variants.py` | 549 | 七种 β 口径变体的独立实现（不改 `src/`），PRESETS 映射每个变体的 mass/fit/apply/solver/λ/scale |
| `experiments/cpu/e2b_beta_convention.py` | 420 | E2b：45 配对样本 × 7 变体 + 基线，单块/归并误差、JS、绝对质量误差、配对 bootstrap、结构诊断 |
| `experiments/cpu/e9_knob_localization.py` | 349 | E9：375 格 M×B 网格，fit/oracle/gap 三曲线，秩诊断与选键质量占比 |
| `experiments/common/synthetic.py` | 220 | 新增 `HeldoutSplit`、`heldout_split`、`absolute_mass_error`、`mixture_output`、`mixture_relative_error`、`make_fixed_context` |

### 关键技术判断

**（一）度量必须选对，否则 β 的效果量不出来。**
这是本次最值得记录的发现：β 加在 logit 上，而 `softmax` 对 logit 的整体常数位移免疫。
因此**只要独立地看单个块，β 的常数分量完全不可观测** —— 单块度量只能看到 β 的跨 Key
离散度。而 β 的职责恰恰是恢复块的**未归一化质量**，质量只在与其他块共处一个
softmax 分母时才有意义（最终输出是各块贡献的加权混合，权重正比于各块的未归一化质量）。

→ 首次运行 E2b 时 `am ≈ am_nobeta`（看不出 β 有效果），原因就是度量选错。
新增 `mixture_output` / `mixture_relative_error`（源块 + 固定上下文块拼接后再与 Dense 比）
之后，β 的作用立刻显形。

**（二）`absolute_mass_error` 的早期实现同样是"盲"的。**
旧实现"两侧各自减去自身最大值"，这依然是常数位移，仍然看不见 β 的常数项。
改为使用**公共偏移** $c=\max(\ell_{\text{full}})$ 后 β 才可观测。

**（三）E9 的留出集不可省。**
评估 Query 绝不能参与选键 / β 拟合 / V 回归，否则"增大 M"会因为见过更多评估点
而自动获胜，结论退化为同义反复。E9 的 fit / oracle / gap 三曲线全部在留出 Query 上测量。

**（四）oracle 是作弊上界。** 它固定同一套 Key 与同一个 β，但用**评估 Query 自己**拟合 V。
它的作用是界定 $B$ 轴的**表示能力**，不代表可实现精度 —— 正文已显式标注。

### 验证证据

**E2b（45 配对样本，$\lambda_\beta=10^{-3}$，留出 Query 中位数）**

| 口径 | 单块误差 | 归并误差 | 绝对质量误差 |
|---|---|---|---|
| 仓库原状 | 0.748 | 1.000 | 1.826 |
| 仅换收敛求解器 | 1.091 | 1.381 | 0.999 |
| 拟合侧迁就为 β/M | 0.532 | 0.768 | 0.776 |
| **修正口径 per-key** | 0.550 | **0.446** | 0.183 |
| 修正口径 log 尺度 | 0.593 | 0.498 | 0.192 |
| 修正口径标量 β | 0.525 | **0.484** | **0.171** |
| β 全链路关闭（基线） | 0.525 | 0.558 | 0.577 |

- 原状口径归并侧 40/45 配对落败（$p<10^{-4}$）→ **原实现对结果有实质损害**。
- 修正后 β 在归并侧显著优于关闭 β（$p<10^{-4}$），绝对质量误差 0.577 → 0.171。
- β 退化为单一标量时配对优势最一致（44/1）→ 可采纳的设计简化。

**E9（375 格）**

| | B=8 | B=128 | 结论 |
|---|---|---|---|
| fit（M=4 → M=48） | 0.758 → 0.460 | 0.677 → 0.214 | 两轴都有效 |
| oracle（M=4 → M=48） | 0.477 → 0.408 | 0.155 → 0.066 | B 定上界 |
| gap（M=4 → M=48） | 0.228 → 0.055 | 0.516 → 0.150 | M 定能否触及上界 |

- oracle 沿 B 下降 67.5%（M=4）与 83.8%（M=48）。
- 可实现链路沿 M 轴平均改善 0.553，沿 B 轴 0.265 → M 约为 B 的 2.1 倍。
- **但 B 不是次要旋钮**：其收益是 M 的函数 —— M=4 时 B 由 8→128 仅带来 10.7%，
  M=48 时带来 53.5%。"只扫 B 而固定 M"会系统性低估 B 的收益。
- 秩诊断：`rank_upper_bound = min(M,B)`，多格数值秩低于上界（$X^\top X$ 在正则下仍退化）。
- 选键保留质量：$B/L_s = 0.031$ → 实测 0.291（**9.32×**）；$0.5$ → 0.919。
  名义压缩比系统性低估信息保留率。

### 结论与对论文主线的影响

用户担心的是"扫描预算 B 画帕累托前沿"可能在优化次要旋钮。数据给出的答案是：
**不必推翻主线，但必须把 M 与 B 一起扫描**，并在报告任何"预算–精度"曲线时
同时报告所使用的 M —— 否则读者无法区分"预算不够"与"估计不足"。

### 原始产出

`results/cpu/e9/summary.json`、`results/cpu/e9/rows.csv`（未纳入版本控制）。

---

## 7. `5b5ce98` — 修正 β 链路口径等 7 处实现缺陷并补全包导出

类型：`fix(dcc_kv_ref)` ｜ 10 文件 ｜ +213 / −76 ｜ 2026-09-13 20:24

### 动机

这是上一轮"CPU 实验跑起来之后反查"的兑现。上一轮只**记录**了 5 处缺陷（`72af7db` 的
「由运行代码反查出的 5 处实现缺陷（**未改 `src/`**）」一节），本轮改用最合理口径
统一修复，共 8 处（新增 3 处）。

### 改动清单

| # | 问题 | 文件 | 修复 |
|---|---|---|---|
| 1 | 块质量目标退化为常数 | `compact_kv.py` | `softmax(...).sum(-1)` ≡ 1 → 未归一化 `exp(logits - shift).sum(-1)`，`shift = max(logits)` 仅防溢出 |
| 2 | β 训练/推理口径相差 M 倍 | `value_regression.py` | 移除设计矩阵对 β 的 `1/M`；`bias_per_query = logit_bias.unsqueeze(0).expand(M, B)` |
| 3 | 非负岭回归求解器不收敛 | `calibration.py` | 固定步长 `1e-2` → 步长 `1/L`（`L = λmax(GᵀG+λI)`）；正则改尺度无关 `λ_eff = λ·mean(diag(GᵀG))`；新增 `lambda_mode` 与 `return_diag` |
| 4 | FP64 被阻塞 | `representative_query.py` | `rademacher_projection` 写死 `.float()` → 按输入 dtype 生成 |
| 5 | CUDA device 不匹配 | `representative_query.py`、`value_regression.py` | `torch.Generator`/`torch.randint` 随输入 device；`torch.eye(B, dtype=, device=)` |
| 6 | 包导出缺失 | `__init__.py` | 补 `merge_softmax_states_list`、`attention_output_from_state` |
| 7 | 双导入根不兼容 | `baselines/*_cpu.py`、`distributed/*_cpu.py` | `from ..dcc_kv_ref` → `try/except ImportError` 回退 `from dcc_kv_ref` |
| 8 | β 稀疏塌缩 | — | **未修，待办**（见下） |

### 关键判断

**缺陷 2 是二选一决策，不是单纯修 bug。** 用户判定"大概率是笔误"。判定过程：
核验源论文 Eq.(2) 的 $\sum_j \exp(\ell_j+\beta_j)$ 与本文自身 Eq.(15)(16) 中
$w=\exp(\beta)$ 直接乘在 $\exp(\ell)$ 上 —— 两侧系数均为 **1**。
故 `1/M` 属推理侧写法残留，**拟合侧去掉 1/M**，而不是让推理侧除以 M。

**缺陷 6 的后果比它看起来严重得多。** `tests/test_dist_equivalence.py` 因
`cannot import name 'merge_softmax_states_list'` 在**收集阶段**即中止，导致该文件
零个测试被执行。也就是说，早期版本中作为 E2/E3 证据引用的测试名从未真正运行过。
修复后该文件 3 失败 / 10 通过；全仓库 44 通过 / 11 失败，
其中 8 项失败属于本机 Windows 多进程 spawn 的环境限制（`OSError: [WinError 6] 句柄无效`），
非代码缺陷。

**缺陷 3 与缺陷 1 是连带的。** 缺陷 1 修正后目标量级由 $\mathcal{O}(1)$ 跳到
$\mathcal{O}(L_s)$，梯度 Lipschitz 常数随之增大，原本勉强能跑的固定步长随即越界，
$w$ 被反复 clamp 到 0，β 全部顶到 $\log(10^{-6})=-13.8155$ 下界。
所以缺陷 3 是缺陷 1 暴露出来的，修好 1 就必须同时修 3。

### 未修项（缺陷 8）：β 的稀疏塌缩

非负最小二乘允许 $w_j\to0$，实测相当比例的 β 落到 clamp 下界，效果接近"软剔除部分 Key"。
这与本文把 β 描述为"质量重加权"而非"二次选键"的意图不符。
需引入下界约束或改换参数化。**已写入 §6.3.9 与 §P 的 C9。**

### 验证证据

```
xelatex 全文编译                     通过（本次修复不涉论文版式，但保证未破坏编译）
E2b / E9 在统一口径下可复现           通过（见 511ca1e）
tests/test_dist_equivalence.py      修复前：收集失败（0 测试执行）
                                    修复后：3 失败 / 10 通过
全仓库 pytest                        44 通过 / 11 失败（其中 8 项为 Windows spawn 限制）
```

### 与缺陷 2 相关的一处交叉验证

`test_apb_vs_dense_within_tolerance` 的偏差值在本次修复前后**逐位一致**
（`1.9943916551584877`），说明该测试的失败与本文的 β 改动无关，
是既有的独立问题 —— 这一点排除了"修复引入新失败"的可能。

---

## 6. `72af7db` — 实验代码按 CPU/GPU 拆分，并修正 §6.4 的算术错误

**时间** 2026-09-13 18:25:08 ｜ **作者** Saluneo \<2991493617@qq.com\>
**规模** 24 files changed, 6255 insertions(+), 2 deletions(−)

### 动机

论文 §6 此前只有实验方案的文字描述，没有可执行代码；且方案内部把「CPU 可验证」与「需 GPU」的
实验混在一起，无法判断哪些结论已经有证据支撑。本次把 §6 落地为两套**不可互替**的代码，并顺带
修正一处算术错误。

### 改动清单

**A. 新增 `experiments/`（CPU 侧，任意多核机器可复现）**

| 文件 | 行数 | 对应实验 | 作用 |
|---|---|---|---|
| `common/synthetic.py` | 602 | 基础设施 | 合成场景 `make_scenario`、稠密/紧凑注意力、KL/JS 散度、有符号质量误差、β 三模式、dtype 支持探测 |
| `common/report.py` | 203 | 基础设施 | `summarize`（median/p5/p95/bootstrap CI）、`paired_bootstrap`（配对 CI + 单侧置换检验） |
| `cpu/e0_order_invariance.py` | 277 | E0 | 顺序 vs 平衡树 vs 随机置换/树 × FP32/FP64 |
| `cpu/e1_interface_shapes.py` | 181 | E1 | 形状、索引合法性、同异种子可复现等 11 项不变量 |
| `cpu/e2_fidelity_curve.py` | 211 | E2 | 扫描预算 B 测 ε_mass / ε_out 曲线 |
| `cpu/e3_edge_conditioning.py` | 584 | E3 | H1（边级条件化确实不同）+ H2（配对比较，非「各自有界」） |
| `cpu/e4_dist_equivalence.py` | 283 | E4 | 同步 DCC-KV vs 稠密注意力，三档对照定位误差来源 |
| `cpu/e5a_mechanism_ablation.py` | 229 | A3（机制级） | β 与 Value 回归各自的重构误差贡献 |

**B. 新增 `experiments/`（GPU 侧，需真实 GPU + NCCL）**

| 文件 | 行数 | 作用 |
|---|---|---|
| `_env.py` | 516 | 环境闸门（不达标则**不产出任何结果**，退出码 3）+ 元数据 + 逐次计时 |
| `_comm.py` | 374 | 变长 All-to-Allv、同步/异步流水、消息体积口径 |
| `_hf.py` | 419 | 模型加载、KV 预算约束、多项选择打分、prefill 计时 |
| `e5_gpu_ablation.py` | 604 | A1 / A2 / A3 / A5 |
| `e6_main_table.py` | 431 | 主表（模型 × 上下文长度 × 同步模式 × 方法） |
| `e7_negative_results.py` | 485 | 负结果的四个条件 |
| `e8_low_precision.py` | 335 | 低精度下顺序无关性的失效边界 |

**C. 入口与文档**

- `run_cpu.sh` (65 行)：一键跑全部 CPU 实验，`--quick` 为冒烟模式
- `run_gpu.sh` (118 行)：`torchrun` 封装；`--print-env` / `--plan` 无需 GPU 即可体检
- `README.md`、`gpu/README.md`：两侧的边界与使用说明

**D. `paper/sections/06-experiment.tex`（+90 / −2）**

- **修正算术错误**：原写「3 模型 × 5 上下文长度 × 2 GPU 数 × 2 同步模式 × 3 基线 = 144 个数据点」，
  但 `3×5×2×2×3 = 180 ≠ 144`。反推 `144/(3×2×2×3) = 4`，上下文长度应为 **4 档**。已改为 4 档并写全等式。
- §6.3 补入四组 CPU 实测结果与三处实现问题
- §6.4 补入执行框架、四条计时口径纪律、四类需显式登记的前置缺口
- 明确 E6 当前**不能**支撑 H5（多卡方法均被阻断，`gpu_count` 对其余方法不是自由轴）

### 验证证据

CPU 侧 6 个脚本全部以 `--quick` 冒烟通过，退出码 0：

| 脚本 | 实测结果 |
|---|---|
| E0 | 置换最大误差 FP32 ≈ 5e-7、FP64 ≈ 7e-16 |
| E1 | 11/11 PASS；`logit_bias` 范围触及 clamp 下界 −13.8155 |
| E2 | ε_out(B) 非单调；保留质量显著高于名义 B/L |
| E3 | H1 成立；H2 方向在 `strength=8` 处**翻转** |
| E4 | 隔离对照 max\|Δ\| = 2.5e-7，压缩误差全部来自构造链路 |
| A3 | β 贡献 −0.605，Value 回归贡献 +0.485 |

GPU 侧在本机（无 CUDA）只做静态与逻辑验证：`--print-env` 正确以退出码 3 拦截。

### 由运行代码反查出的 5 处实现缺陷（**未改 `src/`**）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | `src/dcc_kv_ref/compact_kv.py:101` | `block_mass = softmax(x,dim=-1).sum(dim=-1)` 数学上恒为 1 | β 的拟合目标不携带信息 → 解释 β 贡献≈0 |
| 2 | `value_regression.py:72` vs `dcc_kv_sync_cpu.py:73/151` | 拟合侧 `/M`，推理侧用全量，相差 M=64 倍 | E3 中 H2 方向翻转的直接原因 |
| 3 | `value_regression.py` 设计矩阵 | X 仅 M 行，`rank(X) ≤ M`；B > M 时严重欠定 | M=16/B=128 时回归 Value 与真值差 94%；**真正瓶颈是 M 而非 B** |
| 4 | `representative_query.py:27,30` | `torch.Generator()` / `randint` 默认建在 CPU | CUDA 张量上直接 device mismatch |
| 5 | `value_regression.py:32` | `torch.eye(B, dtype=X.dtype)` 未指定 device | 同上 |

第 4、5 条各一行可修，是 GPU 侧 A1/A2/A3 的**硬前置**。

### 遗留项

- 缺陷 1–5 只记录、未修（等待口径决策，见下文 §P）
- E6 主表在 `dcc_kv` 多卡实现就绪前无法产出 H5 证据
- 本次提交**未推送**

---

## 5. `907d356` — 修复参考文献排版错误并补齐两条未引用文献

**时间** 2026-09-13 17:44:50 ｜ **作者** Saluneo
**规模** 2 files changed, 24 insertions(+), 20 deletions(-) ｜ `paper/main.tex`、`paper/refs.bib`

### 问题一：`note` 字段泄漏进参考文献表

`ACM-Reference-Format.bst` 定义了 `output.note`，因此 `refs.bib` 中的 `note` 字段（存放本地 PDF
文件名等作者备注）会被排进正式参考文献表。note 文本中的下划线在文本模式下开启了数学模式，引发
8 处 `Missing $ inserted` 与 16 处 `Missing character`（中文被送入数学字体），bibtex 后的两遍
xelatex 因此中断。另有一个 ⚠️（U+26A0 + U+FE0F）在本机任何字体中都不存在。

**处理**：字段 `note` 改名 `annote`。该 `.bst` 完全不引用 `annote`，因此注释仍保留在 `refs.bib`
中供作者核对，但不再进入正式参考文献表，中文注释也不会出现在投稿稿里。

### 问题二：两条文献存在但从未被引用

`dao2023flashattention2` 与 `milakov2018online` 在 `refs.bib` 中，但正文从未 `\cite`。BibTeX 只收录
被引条目，故 17 条中只有 15 条进入参考文献表。

**补齐引用**：
- `milakov2018online` → §4.4 归并算子 ⊕ 的定义处，并写明「沿用在线 Softmax 的归并算子」，与本文的原创边界声明一致
- `dao2023flashattention2` → §2.1 引言部分

### 验证证据

`xelatex + bibtex + 2×xelatex` 三遍编译：

- LaTeX 错误 **0**，缺失字符 **0**，未解析引用 **0**
- `refs.bib` 17 条 = `main.bbl` 17 条，**无未引用条目**
- 10 页，8 章齐备，Figure 1–4 全部嵌入

---

## 4. `f0677f4` — 续写第 5–8 章、重建参考文献库并补矢量图

**时间** 2026-09-13 17:20:37 ｜ **作者** Saluneo
**规模** 14 files changed, 1823 insertions(+) ｜ 全部为新增

### 动机

第 1–4 章（摘要 / 引言 / 相关工作 / 问题定义 / 方法 / Alg.1）已由既有材料完成，本次按仓库
**实际实现**续写后续章节，使论文可由 `main.tex` 完整编译。

### 改动清单

| 文件 | 行数 | 内容 |
|---|---|---|
| `sections/05-analysis.tex` | 260 | 复杂度与误差分析：通信量、计算量、误差分解与误差界、异步流水的理论加速上界 |
| `sections/06-experiment.tex` | 252 | 实验方案 E0–E8，严格区分「CPU 已验证」与「GPU 待执行」 |
| `sections/07-discussion.tex` | 132 | 讨论、结论、可复现性说明 |
| `refs.bib` | 238 | 17 条参考文献，标题/作者/编号均经 arXiv 原文首页核对 |
| `figures/*.tex` | 264 | 4 张图的 TikZ 源码 + 共享前言（PDF 由构建生成，只提交源码） |
| `main.tex` | 443 | 全文组装版，第 1–8 章可完整编译 |
| `build.sh` / `fetch_refs.sh` | 144 | 构建脚本与文献拉取脚本 |
| `README.md` / `.gitignore` | 90 | 构建说明与忽略规则 |

### 写作边界（本提交确立的硬约束）

- 严格遵循 README「不主张的内容」：**不主张**任务质量结果、通信性能结果、多 GPU 可扩展性、与基线的对比优势
- §4.3 紧凑 KV 构造机制明确声明沿用 Attention Matching（MIT CSAIL），**非本文原创**
- §6 中 E3 标注为「仅量级验证」：`test_dcc_kv_lower_error_than_shared` 只断言两者误差**各自有界**，
  并未断言 DCC-KV 更优，不构成 H2 证据
  > ⚠️ **已被 `e95927f` 修正。** 该条中的两个前提后来都被推翻：
  > （i）`test_dcc_kv_lower_error_than_shared` 因包导出缺失，**从未真正执行过**（见 `5b5ce98` 缺陷 6）；
  > （ii）β 口径修正后 E3 重跑，配对检验方向稳定（20 组中 17 显著 / 3 不显著 / 0 反向），
  > 因此 E3 的状态由「仅量级验证」改为「**机制级证据**」。但 H2 的**任务级**结论仍未取得，
  > 该条"不构成任务级 H2 证据"的精神继续有效。
- 图 4 为解析结果，图注已显式标注「非实测数据」

### 验证证据

- 全文 10 页，0 处未解析引用
- 4 张图均为 **vector PDF**（满足 `docs/release_checklist.md` §5 的硬性要求）
- 8 章齐备（Intro / Related / Preliminary / Method / Experiment / Analysis / Discussion / Conclusion）

### 关联产出（不在本提交内）

`paper/literature/` 已存放 18 份文献 PDF 与 `_download_log.txt`（含 APB `2502.12085`、
Attention Matching `2602.16284` 等）。

---

## 3. `8a1d275` — 依赖修正：torch 版本下限

**时间** 2026-09-08 09:15:59 ｜ **作者** Mavis \<Mavis@mavis.local\>
**规模** 1 file changed, 6 insertions(+), 6 deletions(-) ｜ `requirements.txt`

PyPI 索引中已不存在 `2.3.0+cpu`，安装直接失败。改为 `torch>=2.6.0`。

> **遗留冲突（至今未解）**：`CONTRIBUTING.md` 与 `README.md` L79 仍写 `torch==2.3.0+cpu`，
> 与 `requirements.txt` 不一致，三处口径需统一。

---

## 2. `fcd9718` — 协作与流程文档

**时间** 2026-09-08 07:40:25 ｜ **作者** Mavis
**规模** 5 files changed, 463 insertions(+)

| 文件 | 行数 | 内容 |
|---|---|---|
| `docs/ssh_setup.md` | 141 | Deploy Key 配置流程（沙箱-合作者 onboarding） |
| `CONTRIBUTING.md` | 178 | 贡献者完整指南（分支 / commit / 测试 / review） |
| `.github/PULL_REQUEST_TEMPLATE.md` | 55 | PR 标准模板 |
| `.github/ISSUE_TEMPLATE/bug_report.md` | 49 | Bug 报告模板 |
| `.github/ISSUE_TEMPLATE/feature_request.md` | 40 | 功能请求模板 |

---

## 1. `7ebff65` — 初始参考实现与 Phase A 骨架

**时间** 2026-09-08 06:36:42 ｜ **作者** Mavis
**规模** 43 files changed, 5382 insertions(+)

### `src/` 与 `tests/`

| 目录 | 内容 |
|---|---|
| `src/dcc_kv_ref/` | M0–M1 参考实现：`compact_kv` / `online_softmax` / `representative_query` / `key_selection` / `calibration` / `value_regression` |
| `src/distributed/` | Phase A 多进程与通信：`comm` / `var_len_msg` / `dcc_kv_sync_cpu` / `full_attention_cpu` / `launch_dist` |
| `src/baselines/` | Ring / FastKV / APB 的 CPU mock |
| `src/experiment_metadata.py` | `RunResult` 元数据契约 |
| `tests/` | 23 个 CPU 测试 + 6 个 GPU 测试（deferred） |

### `docs/` 与工程配置

`git_strategy.md` (410) / `release_checklist.md` (155) / `reproducibility.md` (118) /
`.github/workflows/test.yml` (90) / `.gitignore` (137) / `README.md` (145) / `conftest.py` / `pytest.ini` /
`scripts/run_m2_real.sh` / `scripts/run_m3_async.sh`

### 状态声明（提交信息原文）

> M0-M1 status: done (CPU PyTorch FP64)
> M2 status: pending GPU

---

## P. 各提交遗留项汇总（待决策）

### P.1 已在 `5b5ce98` / `511ca1e` 中解决

| 项 | 原性质 | 解决方式 |
|---|---|---|
| **C1** β 拟合目标无信息量 | 实现缺陷 | 改为未归一化质量目标（缺陷 1） |
| **C2** β 训练/推理口径相差 M 倍 | **二选一决策** | 用户判定为笔误；经源论文 Eq.(2) 与本文 Eq.(15)(16) 核验，系数均应为 1 → **拟合侧去掉 1/M**（E2b 定量支撑） |
| **C3** Value 回归在 B > M 时欠定 | 动摇了「预算 B 为主旋钮」的叙事 | 用户要求"先补一组 M 扫描数再定" → E9（375 格）给出：两轴都有效但职责不同，**主线不必推翻，但必须同时扫 M 与 B** |
| **C4** CUDA 不可用的两处 device 缺陷 | 一行可修 | 缺陷 4/5 已修，GPU 侧前置条件成立 |

用户对 H2 的指令是"拿不准的结论和结果不能写入论文"，已落实为：§6 只声称
H2 具备**机制级**证据（合成数据 + 混合输出误差），任务级结论明确留待 E6。

### P.2 仍未解决

| 项 | 来源 | 性质 | 阻塞什么 |
|---|---|---|---|
| **C5** torch 版本三处不一致（`requirements.txt` ≥2.6.0 / `CONTRIBUTING.md` 与 `README.md` L79 `==2.3.0+cpu`） | `8a1d275` | 文档漂移 | 环境可复现性 |
| **C6** README 中 `YOUR_USERNAME` / `author@example.com` 仍为占位符 | `7ebff65` | 文档 | 仓库对外可用性 |
| **C7** `launch_dist.py` 的 `import datetime` 置于文件末尾（264 行）但 100 行已使用 | `7ebff65` | 潜在运行时错误 | 多进程启动路径 |
| **C8** APB 编号 `2502.12085` 待二次确认 | `f0677f4` | 事实核对 | 引用准确性 |
| **C9** β 的稀疏塌缩（`w_j → 0` 落到 clamp 下界，接近"软剔除部分 Key"） | `5b5ce98`（缺陷 8） | 设计层面开放问题 | β 的"质量重加权"定位是否成立；GPU 侧 A2 消融的结论是否可解释 |
| **C10** `test_fast_kv_vs_dense_within_tolerance`（实测 1.09 vs 阈值 0.1）与 `test_apb_vs_dense_within_tolerance`（1.99 vs 0.3）失败 | `5b5ce98` | 既有独立问题（经交叉验证与本轮 β 改动无关） | 这两项不能作为压缩保真度证据；需单独定位 |
| **C11** 仓库外配套文档缺失：`../dcc_kv_plan/research_execution_blueprint_v1.md`、`experiment_matrix.yaml` v1.2.0、`references.bib`（README 称 26 条）、`contribution_boundary_section.md` | `7ebff65` | 素材缺失 | 与既有规划的一致性核对 |

### P.3 待用户决策

| 项 | 选项 |
|---|---|
| **D1** 是否把这些提交推送到 `origin/paper/sections-5-8`（当前 **4 次提交未推送**） | 推送 / 继续本地累积 |
| **D2** 是否把 §6 的 E2b/E9 拆成独立的补充材料，而非留在正文 | 留正文（当前）/ 移补充材料 |
| **D3** 剩余 8 处中文标点悬挂导致的 overfull 是否要在投稿前处理 | 处理（需改 `\xeCJKsetup`）/ 接受 |

## Q. 尚未产生的验证

以下内容截至本次统计**仍无实测数据**，任何文档中都不得声称已有结论：

- E5–E8 的 GPU 实测结果（含 A1–A5 的 GPU 版本）—— 本机无 CUDA，仅有执行框架
- E6 主表 → H5（设备数翻倍加速 ≥1.5×）的证据；主表中四个需多卡的方法**当前均被阻断**，
  因此 **E6 当前不能用于支撑 H5**
- 与基线（Ring / FastKV / APB）的任何对比优势
- **H2 的任务级结论**（准确率 / 生成质量）—— 机制级证据已有（见 `511ca1e`），
  两者不可混同
- β 机制在 GPU 侧消融（E5/A2）中的净贡献 —— 机制级有效不等于任务级显著
- 多进程通信路径在本机 Windows 上的通过记录（`WinError 6` 限制，8 项测试失败）

