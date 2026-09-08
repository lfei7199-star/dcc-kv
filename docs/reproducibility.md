# 可复现性说明（Reproducibility Statement）

> 投稿时附在论文 supplementary 或 GitHub README
> 配套 blueprint v1.1 / release_checklist.md

## 1. 代码可获取性

- **仓库**：https://github.com/YOUR_USERNAME/dcc-kv
- **commit hash**：`v1.0.0` （或具体版本）
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

| 实验 | 最低配置 | 推荐配置 |
|---|---|---|
| 单元测试（CPU） | 任意 | 4 核 |
| M0-M1 | 1 GPU（仅 sanity）| 1x A100 |
| M2（多卡同步）| 2x A100 80G | 4x A100 80G |
| M3（异步）| 4x A100 80G NVLink | 8x A100 80G |
| M4（端到端 8B）| 4x A100 80G | 8x A100 80G |
| M4（70B/72B）| 8x A100 80G | 8x H100 80G |
| M5（跨节点）| 2 节点 × 8x A100 | Lambda Labs 多节点 |

## 5. 完整复现命令

### 5.1 一次性安装

```bash
git clone https://github.com/YOUR_USERNAME/dcc-kv
cd dcc-kv
git checkout v1.0.0
pip install -r requirements.txt
pip install torch==2.3.0  # 按你的 CUDA 版本调整
```

### 5.2 跑主表

```bash
# 8B 主表
bash scripts/run_main_table.sh --model 8b --gpus 4

# 70B 主表
bash scripts/run_main_table.sh --model 70b --gpus 8
```

### 5.3 跑消融

```bash
bash scripts/run_ablations.sh
```

### 5.4 跑鲁棒性

```bash
bash scripts/run_robustness.sh
```

## 6. 预期结果

按 blueprint v1.1 §3 的可检验假设：

| 假设 | 预期 |
|---|---|
| H1 | 同一源块对不同目的端的紧凑 KV 显著不同（KL 散度 > 0.5） |
| H2 | DCC-KV 相对共享压缩，在长上下文任务上提升 ≥ 1.5 个百分点 |
| H3 | 移除 β 后质量退化 ≥ 0.5 点；移除 V 回归后 ≥ 1.0 点 |
| H4 | 异步 vs 同步 p50 加速 ≥ 1.05× |
| H5 | 设备数翻倍，加速 ≥ 1.5× |

## 7. 随机性控制

- 所有实验固定 seed（默认 42, 123, 1024, 7 个备选）
- 每次 run 10 次取统计
- 报告 median + p5/p95 + bootstrap 95% CI
- 代码 release 时附 `requirements.lock`

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
