# 可复现性说明（Reproducibility Statement）

> 投稿时附在论文 supplementary 或 GitHub README
> 配套 blueprint v1.1 / release_checklist.md

## 1. 代码可获取性

- **仓库**：https://github.com/lfei7199-star/dcc-kv
- **commit hash**：仓库当前**未打 release tag**（`v1.0.0` 是发布时的计划占位，
  尚不存在）。引用时请填具体 commit hash，取自 `git rev-parse HEAD`；
  本文件不硬编码哈希，以免与代码演进而失同步。
- **许可证**：Apache 2.0
- **匿名版**（如果双盲 review）：https://github.com/anonymous-dcc-kv

## 2. 模型可获取性

| 模型 | 许可 | 获取方式 |
|---|---|---|
| Llama-3.1-8B-Instruct | Meta License | HuggingFace `meta-llama/Llama-3.1-8B-Instruct` |
| Llama-3.1-70B-Instruct | Meta License | 同上（需申请） |
| Qwen2.5-72B-Instruct | Apache 2.0 | HuggingFace `Qwen/Qwen2.5-72B-Instruct` |
| Mistral-7B-Instruct | Apache 2.0 | HuggingFace `mistralai/Mistral-7B-Instruct-v0.3` |

## 3. 数据集可获取性

| 数据集 | 许可 | 获取方式 |
|---|---|---|
| LongBench | Apache 2.0 | https://github.com/THUDM/LongBench |
| LongBench-v2 | Apache 2.0 | HuggingFace `THUDM/LongBench-v2` |
| ∞Bench | MIT | https://github.com/OpenBMB/InfiniteBench |
| RULER | Apache 2.0 | https://github.com/NVIDIA/RULER |
| Needle-in-a-Haystack | MIT | 公开 benchmark |
| PG-19 | Public Domain | https://huggingface.co/datasets/emozilla/pg19 |
| wikitext-103 | Creative Commons | https://huggingface.co/datasets/Salesforce/wikitext |

## 4. 硬件需求

> **编号说明（2026-09-16）**：下表的 `M0`–`M5` 是 blueprint v1.1 时期的实验编号，
> 仓库现有的脚本用的是另一套编号（CPU 侧 E0–E4、GPU 侧 E5–E8），
> 而 `writing_scope_and_metrics.md` §4 的 `M1`–`M9` 是**缺口编号**，与实验编号无关。
> 三套编号并存是历史遗留，引用时务必写清是哪一套。可确证的对应关系已在表中标出，
> 未确证的不臆测。**建议后续统一为 E 编号。**

| 实验（历史编号） | 对应当前脚本 | 最低配置 | 推荐配置 |
|---|---|---|---|
| 单元测试（CPU） | `pytest tests/`（默认跳过 GPU） | 任意 | 4 核 |
| M0-M1 | 未确证（疑为环境冒烟） | 1 GPU（仅 sanity）| 1x A100 |
| M2（多卡同步） | E5 的 `--parts a1 a2`（`scripts/run_m2_real.sh` 是它的单元测试入口） | 2x A100 80G | 4x A100 80G |
| M3（异步） | E5 的 `--parts a5`（`scripts/run_m3_async.sh`） | 4x A100 80G NVLink | 8x A100 80G |
| M4（端到端 8B） | E6（`e6_main_table.py`） | 4x A100 80G | 8x A100 80G |
| M4（70B/72B） | **无对应脚本**（E6 只实现了 8B 档） | 8x A100 80G | 8x H100 80G |
| M5（跨节点） | **无对应脚本**（跨节点未实现） | 2 节点 × 8x A100 | Lambda Labs 多节点 |

按脚本实测的最低门槛（与上表不同源，取更保守者）：

| 脚本 | 最小 GPU | 最小显存 | 备注 |
|---|---|---|---|
| `e8_low_precision.py` | 1 卡 | 4 GB | 只做归并算子，无通信、无模型 |
| `e6_main_table.py` | 1 卡 | 40 GB | 当前只有 `dense` / `kv_budget_shared` 可测 |
| `e7_negative_results.py` | 1 卡 | 40 GB | 32K 上下文 + 8B 模型 |
| `e5_gpu_ablation.py` | **2 卡** | 16 GB | A1/A2/A5 需要 NCCL；A3 被实现缺口阻断 |

GPU 单元测试（`pytest -m gpu`）默认被 `pytest.ini` 的 `-m "not gpu"` 排除，
**上机后第一件事就应该是跑它** —— 它只验证 NCCL/通信通路，成本极低，
却是唯一能在真机上暴露 device/通信缺陷的手段。

## 5. 完整复现命令

> **更正（2026-09-16）**：本节此前给的 `scripts/run_main_table.sh`、
> `scripts/run_ablations.sh`、`scripts/run_robustness.sh` **三个文件在仓库中不存在**
> （`scripts/` 下只有 `run_m2_real.sh` 与 `run_m3_async.sh`）。照原文执行会直接失败。
> 现改为实际可用的入口 `experiments/run_gpu.sh`。

### 5.1 一次性安装

```bash
git clone https://github.com/lfei7199-star/dcc-kv
cd dcc-kv
# 版本：仓库当前**未打 release tag**。引用时请用具体 commit hash
# （`git rev-parse HEAD`），不要写 v1.0.0 —— 该 tag 尚不存在。
pip install -r requirements.txt
# torch 版本以 requirements.txt 为准；各文档间的版本声明不一致，
# 这是已知问题（见 commit_log 的 C5），安装前请自行核对 CUDA 版本。
```

### 5.2 跑主表（E6）

```bash
# 单卡；需要真实模型权重与评测集 JSONL
bash experiments/run_gpu.sh e6 --model meta-llama/Llama-3.1-8B-Instruct \
    --eval-file <评测集.jsonl>

# 先看网格与前置条件（任何机器可跑，不产出结果）
bash experiments/run_gpu.sh e6 --plan
```

⚠️ **当前 E6 的 6 个方法里有 4 个被实现缺口阻断**（含本文方法 `dcc_kv` 自己），
详见 `experiments/gpu/README.md`。在缺口补齐前，主表只会产出
`dense` 与 `kv_budget_shared` 两行。70B/72B 档**无对应脚本**。

> ⚠️ **2026-09-21 更新（`b44a7b3`）**：上段**已过期** —— 现为 **3 个**被阻断
> （`fastkv_official` / `ring` / `apb`），`dcc_kv` 的方法行已接线。
> 跑它必须显式给出源端段数：
>
> ```bash
> bash experiments/run_gpu.sh e6 --model <模型> --eval-file <评测集.jsonl> \
>     --methods dense kv_budget_shared dcc_kv --dcc-world 4
> ```
>
> `--dcc-world` **没有默认值**（它决定每边预算 `B_total/world`，给个默认值会把
> "本次模拟了几个源端设备"变成没人声明过的假设；不给而选了 `dcc_kv` 时脚本以
> 退出码 2 拒绝执行）。另外两条必须与数字一起读的口径：`--ranks` 默认 1，
> 此时 `dcc_kv` 的 sync/async 轴**被折叠**（行里 `sync_async` 记 `n/a`）；
> prefill 加速比落两列，H2 消费 `prefill_speedup_kernel_matched`（同核），
> **缺这一列时该长度记 unresolved 而不是未达标**。

### 5.3 跑消融（E5）

```bash
# 多卡 + NCCL
bash experiments/run_gpu.sh e5 --nproc 4 --parts a1 a2 a5

# 多卡通路的单元测试（上机后建议最先跑，成本最低）
bash scripts/run_m2_real.sh          # 或等价地：pytest -m gpu
```

A3 与 A4 当前**不可执行**：A3 缺 `CompactKV → GPU attention kernel`，
A4 只有设计（论文 §6.3）、尚未写代码。

### 5.4 跑鲁棒性与负结果（E7）

```bash
bash experiments/run_gpu.sh e7 --model <path> --eval-file <评测集.jsonl>
bash experiments/run_gpu.sh e7 --plan      # 只打印四条条件清单
```

条件 3（检索类样本）需要评测集里带 `task` 字段；缺数据时记为 `no_data`，
**不得记为 pass**。

### 5.5 跑 CPU 机制级实验（E0–E13，无需权重 / NCCL）

```bash
bash experiments/run_cpu.sh              # 全部跑一遍（秒 ~ 分钟级）
bash experiments/run_cpu.sh --quick      # 快速冒烟
python experiments/cpu/e0_order_invariance.py       --out results/cpu/e0
#   E0 的置换次数轴可单独调：--perm-ladder 1 10 100 1000
python experiments/cpu/e12_representative_query.py --out results/cpu/e12
python experiments/cpu/e13_bound_tightness.py       --out results/cpu/e13
# E3 默认按留出协议；--protocol in-sample 仅用于复现历史数字（不得作 H2 证据）
python experiments/cpu/e3_edge_conditioning.py      --out results/cpu/e3
```

CPU 侧结果全部落在 `results/cpu/**`，与论文 §6 的机制级数字一一对应。
其中 **E13** 是式 (37) 误差界的首次数值检验（5760 个 Query、零反例），
**E12** 是代表 Query 投影维度 d_p 的扫描。**E0** 给出在线归并的数值边界：
置换次数 $n$ 的累计最大相对误差在 $n\le25$ 内即达 $n=1000$ 时的 $88\%$ 以上，
且 $200\to1000$（$5$ 倍）的最大增长仅 $4.54\%$（$\mathrm{FP32}$）；
平衡树相对顺序归并**没有系统性方向**（$12$ 格中 $7$ 格不劣、$5$ 格更差），
故该规模下误差量级由浮点精度而非归并结构支配。**E3 的旧样本内落盘保留在
`results/cpu/e3_in_sample/`，仅供口径对照，不得作为 H2 的证据。**

## 6. 预期结果

按**论文原件**（`AuthorKit27`，第 7 章）与仓库内机器可读实现
（`experiments/common/hypotheses.py`）的可检验假设：

| 假设 | 预期 |
|---|---|
| H1 | 同一源块对不同目的端的紧凑 KV 显著不同（KL 散度 > 0.5） |
| H2 | **质量提升 ≥ 1.5 个百分点**（vs 共享压缩）**且** 在质量相近前提下 **prefill 加速 ≥ 1.10×**（vs 精确注意力） |
| H3 | 移除 β 后质量退化 ≥ 0.5 点；移除 V 回归后 ≥ 1.0 点 |
| H4 | 异步 vs 同步 p50 加速 ≥ 1.05× |
| H5 | 设备数翻倍，加速 ≥ 1.5× |

> **本表是 blueprint §3 的下游快照，不是权威原件** —— 若与原件冲突，以原件为准。
>
> ⚠️ **2026-09-21 更正（上句原文保留）**：该 blueprint **不在本仓库、也不在相邻
> 目录**，已于 2026-09-21 裁决**不再作为权威锚点**（原 C11 关闭）。本表的口径
> 现在以**论文原件（`AuthorKit27`，第 7 章）+ 仓库内机器可读实现
> `experiments/common/hypotheses.py`** 为准；两者不一致时以机器可读实现为准
> （它是 `H1–H5` 阈值的单一事实源）。
>
> **H2 的口径已定稿**（2026-09-15）。质量侧与性能侧是同一假设的两个侧面，
> 合并方式为**逻辑与**（两条同时满足才算达标）：
>
> > 质量提升 ≥ 1.5 个百分点 **且** 在质量相近前提下 prefill 加速 ≥ 1.10×
>
> 阈值的机器可读版本见 `experiments/common/hypotheses.py`（**唯一事实源**）；
> 实验脚本判定 H2 一律调用 `h2_pass`，不得再写魔法数字。
>
> **「质量相近」的定义已给出**（2026-09-16）—— 完整论证见论文 §7「「质量相近」的
> 判定」，机器可读实现见 `experiments/common/hypotheses.py`。要点：
>
> - **参照物 = 精确注意力**，不是共享压缩基线。若与基线同源，则「相近」与「提升
>   >= 1.5 pp」只有在边界 > 1.5 pp 时才可能同时成立，而那时前半句已无主张 ⇒ 自我矛盾。
> - **判定主体 = 预先固定的留出评测协议 + 事先声明的检验程序**，不由作者事后目测。
> - **判据 = 单侧非劣检验**：`CI_low(Q_DCC - Q_dense) > -delta`。取单侧的依据是该前提
>   的功能是排除「以质量换速度」；若同时显著更优则前提更强成立，但必须披露方向。
> - **delta 的可行区间**：`噪声底线 <= delta < min(1.5, delta_bad)`，其中
>   `delta_bad = Q_dense - Q_share`。下界保证判定不是噪声伪影；上界 1.5 pp 保证边界
>   不大于所要检测的效应量；上界 `delta_bad` 保证共享压缩本身不会落入「相近」。
> - **配对性**：质量侧与性能侧须取自同一组 run。
> - **聚合**：四个上下文长度分别判定，全部通过才算前提成立。
>
> `h2_pass` 仍把 `quality_comparable` 作为**必填**关键字参数 —— 定义解决的是
> 「按什么程序判」，而 `delta` 与噪声底线两个**参数**依赖尚未取得的测量，
> 所以 H2 的 `code_status` 仍是 `no-judge`。缺口已由「定义缺失」降为「**参数待测**」。
>
> **prefill 加速比的分子已改判为 dense**（2026-09-21）。E6 原先实现为
> `prefill_shared / prefill_dcc`，与论文 §7.5 的三段划分不符 —— 那三段是
> 「与**精确注意力**不劣 / 相对**共享压缩**更高 / prefill **更快**」，第三段讲的是
> 系统设计相对**精确注意力**的收益。`shared` 同样压过 KV、同样交付 B 长的 KV，
> 拿它做分子得到的比值结构上恒 ≈ 1，与 1.10× 的阈值不自洽。现为
> `prefill_dense / prefill_dcc`，方向由
> `tests/test_adversarial_2026_09_20.py::test_a4_*` 锁住。
>
> **2026-09-21 再改判为 kernel-matched**（同日，`b44a7b3`）。上段解决了"比错对象"，没解决
> "用错核"：dense 行走 SDPA 融合核、dcc 行走 `attention_kernel` 的显式核，两者的
> 实现差距会整个人进比值。现取 **kernel-matched** —— 分子是**钩子 dense 臂**的目的端
> 耗时、分母是**钩子 dcc 臂**的目的端耗时，两臂走同一个算子核，唯一差别只剩目的端消费
> 的远端 KV 长度。H2 消费这一列；`prefill_speedup_native`（dense 行端到端 / dcc 行
> 端到端）仍会落盘，但**只作端到端参考，不得当作机制收益**。
>
> **仍未闭合的部分**：量具虽已灵敏（H0，2026-09-21），但 `dcc_kv` 在
> `measure_point` 里与 `kv_budget_shared` 走**同一条共享裁剪路径** ⇒ 两者质量差
> 恒为 0，**质量臂为空**。故 H2 维持 `no-judge`：现在不是「参数待测」，
> 而是**质量侧的通路尚未独立**。详见 `docs/commit_log.md` 第 31 条。
>
> ⚠️ **2026-09-21 第三次更新（`b44a7b3`）**：上面这段**已被修** —— `dcc_kv` 现在走
> `attention_hook`（源端由 state 携带、紧凑块目的端条件化构造），质量臂不再为空。
> H2 仍维持 `no-judge`，但**原因又换了一次**：现在是
> `quality_comparable` 恒为 `None` —— 本表每格只落一个聚合准确率、不落逐样本判对错，
> 算不出"非劣"所需的**配对 95% CI 下界**。缺口已由「通路未独立」降为
> **「判定所需的逐样本数据未落盘」**，是一个能在 `measure_point` 里补的落盘项。
>
> ⚠️ **2026-09-21 第四次更新（`12ece9d`）**：上段说的那个落盘项**已补**。
> `measure_point` 现在把逐样本判对错（`accuracy_per_sample`）与配对身份键
> （`eval_sample_keys`）与该行的聚合准确率**同行落盘**；配对 CI 由
> `report.paired_bootstrap` 计算，且**先逐位比对两行的键再算数** —— 键不同源 /
> 样本数不等 / 缺数据三种情况分别记原因并**拒绝配对**，而不是产出一个"数值正常、
> 实际无意义"的 CI。
>
> 于是 H2 的判定链路**完整闭合**，`code_status` 仍为 `no-judge`，但**唯一剩下的
> 原因是参数取值**：`--h2-delta-pp` 与 `--h2-noise-floor-pp` 按设计**无默认值**
> （有默认值等于假称前提永远成立），未声明时该长度记 unresolved，并在
> `h2_comparable_diagnostics` 里写明缺的是哪一个。**这不是「结论未定」** ——
> 缺的是一次带参数的运行，以及一次真实的重复 run 来把噪声底线测出来（O1）。
>
> ⚠️ **该前提的 CI 方向极易写反**：`report.paired_bootstrap` 在
> `higher_is_better=True` 时把配对差翻了符号，故 `Q_DCC − Q_dense` 的下界是
> **`−ci_95_upper`** 而不是 `ci_95_lower`。实测：A 比 B 好 20pp ⇒
> `ci=[−0.28, −0.12]`、下界 = `+12pp`。写反后符号与量级都正常，只能靠定向断言
> （`tests/test_adversarial_2026_09_20.py::test_j2_*`）守住。
>
> 另注：**「质量相近」与「质量提升 ≥1.5pp」的参照物不同** —— 前者比 **dense**
> （前提的参照物），后者比 **kv_budget_shared**（H2 前半句）。两个数都在同一行
> `h2` 字段里，**不可互相代入**，`h2_comparable_note` 已写明这一点。

## 7. 随机性控制

- **多种子**：`e2b` / `e9` / `e10` / `e11` / **`e12` / `e13`** 的 `--seeds` 默认 **5**
  （该参数是**种子个数**，由基种子派生为 `42, 43, 44, 45, 46`）。
- **单种子**：`e0` / `e1` / `e2` / `e3` / `e4` / `e5a` 与 GPU 侧各脚本仅有
  单个 `--seed`（默认 42）。**这些实验尚无种子敏感性证据** —— 这是已知局限，
  见 `docs/commit_log.md` 第 22 条与 §S。
- 统计口径：中位数 + p5/p95 + bootstrap 95% CI，由
  `experiments/common/report.py` 统一实现（不是「每个 run 重复若干次」）。
- 代码 release 时附 `requirements.lock`

> **更正（2026-09-15）**：本节此前写「默认 42, 123, 1024, 7 个备选」与「每次 run 10 次
> 取统计」，与代码实况**均不符** —— 全仓没有任何脚本以 123 / 1024 / 7 为种子，
> 也不存在「10 次重复」的机制。该错位由独立监督查出（见 `docs/commit_log.md` 第 22 条）。

## 8. 已知差异

按 blueprint §3 failure_thresholds：
- 若 H2 / H3 / H4 / H5 任一未达，主张已收敛
- 负结果在论文 Section 5.4 报告
- 适用边界已明确

## 9. 联系

- Issue: GitHub Issues
- Email: 见论文作者信息
- Slack: 见项目 README

---

> 配套文档：
> - `release_checklist.md`
> - `git_strategy.md`
> - `../experiment_matrix.yaml` v1.2.0
> - blueprint v1.1（见父目录）
