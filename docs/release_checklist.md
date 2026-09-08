# 发布清单（Release Checklist）

> 配套 blueprint v1.1 §3、§4、§5 / experiment_matrix.yaml v1.2.0
> 投稿前 / Camera Ready 前逐项打勾

## 0. 触发时机

- [ ] **Milestone 完成**（M0 / M1 / M2 / M3 / M4 / M5）
- [ ] **投稿前**（论文 + 代码 + 附录）
- [ ] **Camera Ready 前**（接收通知后）

---

## 1. 代码

- [ ] 所有 milestone 代码已合并到 `main`
- [ ] 无未提交修改（`git status` 干净）
- [ ] 当前 commit 是 release tag
- [ ] 至少 2 个 reviewer approve
- [ ] CI 全过
- [ ] `pytest tests/ -v` 全过（CPU 部分）
- [ ] GPU 测试（如果资源可用）全过
- [ ] 大文件已从 git 移除（`git rev-list --objects --all | sort -k 2 | uniq -f 1`）

---

## 2. 元数据

- [ ] `requirements.txt` 锁定版本（`pip freeze > requirements.lock`）
- [ ] `ExperimentMetadata` schema 完整
- [ ] 实验 commit hash 记录在 `results/`
- [ ] 模型权重 commit hash 记录（sha256）
- [ ] 硬件规格记录（GPU 型号 / 显存 / 互联）
- [ ] 软件版本记录（PyTorch / CUDA / NCCL / Transformers）

---

## 3. 实验结果

- [ ] **主表完成**（Table 1 / Table 2）：
  - 3 模型 × 5 长度 × 2 GPU × 2 sync × 3 基线 = 144 数据点
  - 每个数据点 ≥ 10 次 run
  - 报告 median + p5/p95 + bootstrap 95% CI
- [ ] **消融完成**：
  - A1 通信集大小
  - A2 压缩预算（5 档）
  - A3 组件拆分（4 变体）
  - A5 异步 vs 同步
- [ ] **鲁棒性**：
  - 模型族迁移
  - 硬件迁移
  - 任务多样性
- [ ] **负结果**：
  - 短上下文（L < 4K）
  - 低预算（B < 1%）
  - 强 retrieval 任务
  - batch=1

---

## 4. failure_thresholds 校核（blueprint §3）

- [ ] H2：质量相近时 prefill 加速 ≥ 1.10×？或收敛 claim
- [ ] H3：β / V 回归各组件不可缺？或收敛 claim
- [ ] H4：异步 vs 同步 p50 加速 ≥ 1.05×？或弱化"异步"
- [ ] H5：扩展性符合分析？或讨论限制

如果任一阈值未达，**主张已按 failure_thresholds 收敛**。

---

## 5. 论文

- [ ] 8 个章节齐：Intro / Related / Preliminary / Method / Experiment / Analysis / Discussion / Conclusion
- [ ] 符号统一（前后一致）
- [ ] 单位统一（ms / GB / %）
- [ ] 引用格式正确（按 venue）
- [ ] 中文/英文摘要完整
- [ ] 所有 [?] 已替换为正式引用
- [ ] 所有图是 vector PDF（不用 PNG）
- [ ] 表格用 booktabs 风格
- [ ] 补充材料完整（算法伪代码、复现指南）
- [ ] 复现声明（GitHub 链接 + commit hash）

---

## 6. 可复现性

- [ ] GitHub 仓库公开（或匿名版）
- [ ] 仓库链接在论文 abstract 末尾
- [ ] 一次性运行脚本：`bash scripts/reproduce_main.sh`
- [ ] README 包含完整步骤
- [ ] 模型权重说明（公开 / 申请 / 私有）
- [ ] 数据集说明（公开 / 申请）
- [ ] 硬件最低要求

---

## 7. 双盲 review 友好

- [ ] 仓库匿名（如果 review 阶段）
- [ ] 论文 PDF 匿名
- [ ] 致谢里无个人信息
- [ ] 链接里无个人主页 / 实验室主页
- [ ] 引用里没引用未发表的相关工作（自己/同事）

---

## 8. 投稿平台

- [ ] 选刊（TPAMI / NeurIPS / MLSys / OSDI / TMLR）
- [ ] 阅读 author guideline
- [ ] 准备 cover letter
- [ ] 准备 supplementary
- [ ] 提交（OpenReview / EasyChair / TPAMI submission）

---

## 9. 决策树

```
实验完成？
  ├─ 主表 + 消融 + 鲁棒性 + 负结果齐？
  │   ├─ failure_thresholds 全过？
  │   │   ├─ Yes → 投稿（MLSys / NeurIPS）
  │   │   └─ No  → 收敛 claim 后再投稿
  │   └─ 缺关键实验 → 补
  └─ 缺大块 → 延期 / 转投低一档会议
```

---

## 10. 实际时间预算

| 阶段 | 估时 | 累计 |
|---|---|---|
| 补实验 | 4-8 周 | 4-8 周 |
| 写论文 | 4-6 周 | 8-14 周 |
| 内部 review | 2-3 周 | 10-17 周 |
| 投稿 | 1 周 | 11-18 周 |

**最坏情况**：6 个月（如果需要大改）

---

## 11. 一句话总结

> **在 v1.0 release 之前，所有检查项都是"must"；在 v1.0 release 之后，新增功能可降级为"should"。**

---

> 配套文档：
> - `git_strategy.md` — 仓库管理
> - `reproducibility.md` — 可复现性
> - `../experiment_matrix.yaml` v1.2.0 — 实验矩阵
