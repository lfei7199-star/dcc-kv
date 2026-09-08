# Git 仓库管理策略

> 配套文档：blueprint v1.1 / experiment_matrix.yaml v1.2.0

## 0. 设计目标

1. **实验可追溯**：每次跑实验的代码版本可追溯
2. **结果可复现**：别人拿到 commit hash 就能复现
3. **保护隐私**：模型权重、数据集、敏感信息不外传
4. **双盲 review 友好**：匿名化容易做

---

## 1. 仓库结构

```
dcc-kv/                                  ← GitHub 私人仓库
├── .git/                                 ← git 内部
├── .github/
│   ├── workflows/test.yml               ← CI
│   ├── ISSUE_TEMPLATE/                  ← issue 模板
│   ├── PULL_REQUEST_TEMPLATE.md         ← PR 模板
│   └── CODEOWNERS                       ← 代码负责人
│
├── src/                                  ← 源代码（公开）
│   ├── dcc_kv_ref/                      ← M0-M1 reference
│   ├── distributed/                     ← 分布式通信
│   ├── baselines/                       ← 基线实现
│   └── experiment_metadata.py
│
├── tests/                                ← 测试
│   ├── cpu/                              ← CPU-only 测试
│   └── gpu/                              ← GPU 测试（deferred）
│
├── scripts/                              ← 启动脚本
├── docs/                                 ← 文档
│   ├── blueprint.md                     ← 实验蓝图
│   ├── experiment_matrix.md             ← 实验矩阵
│   ├── git_strategy.md                  ← 本文件
│   ├── release_checklist.md             ← 发布清单
│   └── reproducibility.md               ← 可复现性说明
│
├── configs/                              ← 实验配置
│   ├── main_experiments.yaml
│   ├── ablations.yaml
│   └── gpu_resources.yaml
│
├── paper/                                ← 论文（投稿前）
│   ├── main.tex
│   ├── refs.bib
│   ├── figures/
│   └── supplementary/
│
├── results/                              ← 实验结果（不传）
├── logs/                                 ← 日志（不传）
├── traces/                               ← profiling trace（不传）
│
├── .gitignore
├── README.md
├── requirements.txt
├── conftest.py
├── pytest.ini
└── LICENSE
```

---

## 2. 分支策略

### 2.1 主分支

- **`main`**：稳定版本，每发布一个 milestone 打 tag
- **`develop`**：日常开发分支

### 2.2 工作分支

| 分支类型 | 命名 | 用途 | 合并到 |
|---|---|---|---|
| `feature/*` | `feature/...` | 新功能 | `develop` |
| `experiment/*` | `experiment/M2_nccl` | 单个实验 | `develop` |
| `paper/*` | `paper/mlsys2025` | 投稿专用 | `main` |
| `hotfix/*` | `hotfix/...` | 紧急修复 | `main` + `develop` |

### 2.3 实验分支的命名规范

```
experiment/<M_阶段>_<简短描述>

# 例子：
experiment/M2_nccl_2proc
experiment/M2_4proc_scaling
experiment/M3_async_overlap
experiment/M4_8b_main_table
experiment/M4_70b_scaling
```

每个实验分支对应一个 experiment_matrix.yaml 里的数据点。

---

## 3. Commit 规范

### 3.1 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 3.2 type 类型

| type | 用途 |
|---|---|
| `feat` | 新功能 |
| `fix` | bug 修复 |
| `docs` | 文档（仅文档） |
| `test` | 测试（仅测试） |
| `refactor` | 重构（无功能变化） |
| `perf` | 性能优化 |
| `exp` | 实验数据 / 结果 |
| `chore` | 杂项（CI、依赖） |

### 3.3 例子

```
feat(distributed): add var-len all-to-allv primitive

实现 M2 阶段核心通信原语。变长消息通过
"先交换 sizes，再交换数据"两阶段完成。

Reference: blueprint v1.1 §6 M2 acceptance

Tests: tests/test_var_len_msg.py (5 tests)
```

```
exp(M2): run 2proc gloo all_to_allv

Command:
    pytest tests/gpu/test_nccl_basic.py -v

Results:
    max_diff = 1.2e-6
    mean_diff = 3.4e-7
    Status: PASS

Hardware: 2x A100 80G
Date: 2026-09-08
```

---

## 4. Tag 策略

### 4.1 格式

```
v<major>.<minor>.<patch>-<stage>

# 例子：
v0.1.0-m0        # M0 完成
v0.2.0-m1        # M1 完成
v0.3.0-m2        # M2 完成
v0.4.0-m3        # M3 完成
v1.0.0-m4        # M4 主表完成（投稿基础）
v1.1.0-m5        # M5 扩展完成
v2.0.0-camera    # Camera Ready
```

### 4.2 触发条件

打 tag 的时机：
- 完成一个 milestone（M0/M1/M2/M3/M4/M5）
- 论文投稿前
- Camera Ready

### 4.3 Tag 信息

```bash
git tag -a v0.2.0-m1 -m "M1 complete: single-process compact KV construction"
git tag -a v1.0.0-m4 -m "M4 complete: end-to-end 8B/70B main table"
```

---

## 5. 实验可追溯

### 5.1 每次实验必须记录

1. **Git commit hash**（`git rev-parse HEAD`）
2. **分支名**（`git branch --show-current`）
3. **是否有未提交修改**（`git status`）
4. **模型权重版本**（`pytorch_model.bin` 的 sha256）
5. **代码版本**（commit hash）
6. **依赖版本**（`pip freeze > requirements.lock`）
7. **硬件规格**
8. **完整配置**（configs/*.yaml）
9. **原始数据**（可重现）

### 5.2 ExperimentMetadata 自动收集

```python
from src.experiment_metadata import ExperimentMetadata

meta = ExperimentMetadata(
    run_id="m2_2proc_nccl_001",
    git_commit=get_git_commit(),  # 自动获取
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    # ...
)
meta.save(f"results/{run_id}/metadata.json")
```

### 5.3 实验 commit 模式

**禁止**：实验结果散落在 commit 里

**推荐**：实验结果用专门的 branch + tag 关联
```bash
# 1. 切到实验分支
git checkout -b experiment/M4_8b_baseline

# 2. 跑实验
python scripts/run_m4.py

# 3. 提交实验配置 + 结果
git add configs/M4_8b_baseline.yaml results/M4_8b_baseline/
git commit -m "exp(M4): 8B baseline (no compression)"

# 4. 打 tag
git tag -a m4-8b-baseline-20260908 -m "8B baseline at 2026-09-08"

# 5. 合并到 develop
git checkout develop
git merge --no-ff experiment/M4_8b_baseline
```

---

## 6. 仓库设置

### 6.1 私人仓库（推荐）

```bash
# 在 GitHub 创建私人仓库（不要勾选 README / .gitignore / License）
# 仓库名建议：dcc-kv

# 远程添加
git remote add origin git@github.com:YOUR_USERNAME/dcc-kv.git

# 推送到 main（首次）
git push -u origin main
```

### 6.2 保护分支（建议）

GitHub Settings → Branches → Branch protection rules：
- `main`：必须 PR + 1 个 review + CI 通过
- `develop`：必须 PR + CI 通过（review 可选）

### 6.3 密钥管理

**绝不能 commit 到 git**：
- HuggingFace token
- AWS / 云服务密钥
- 模型权重 URL（如果需要 auth）

**做法**：
- 用 GitHub Secrets 存 CI 用的密钥
- 用 `.env` 文件（加进 `.gitignore`）
- 模型权重 URL 只放 configs/*.yaml，运行时注入

### 6.4 双盲 review 友好

投稿前：
- 仓库转为匿名（`anonymous-dcc-kv`）
- 历史 commit 里如有作者信息，**不能改**（git 历史不可改）
- 解决：投稿时单独 fork 一份新仓库做匿名版

---

## 7. 日常操作速查

```bash
# 1. 看当前状态
git status
git log --oneline -10
git branch --show-current

# 2. 写代码
git checkout develop
git pull
git checkout -b feature/M2_var_len_msg

# 编辑代码
git add .
git commit -m "feat(distributed): var-len all-to-allv"

# 3. 跑测试
pytest tests/ -v
# 全过 → push
git push -u origin feature/M2_var_len_msg
# 提 PR 到 develop

# 4. 合并
# GitHub 上 review + merge

# 5. 同步
git checkout develop
git pull

# 6. 跑实验（重要 commit）
git checkout -b experiment/M2_2proc
python scripts/run_m2.py
git add configs/ results/
git commit -m "exp(M2): 2-proc gloo all_to_allv"
git tag m2-2proc-20260908
git push -u origin experiment/M2_2proc
# 提 PR

# 7. 完结
git checkout develop
git merge --no-ff experiment/M2_2proc
```

---

## 8. 数据管理（不传仓库）

### 8.1 哪些数据不入仓

| 数据 | 位置 | 大小 |
|---|---|---|
| 模型权重 | 本地 `/workspace/models/` | 16GB+ |
| 数据集 | 本地 `/workspace/data/` | 10GB+ |
| 实验结果 | `results/` | 1-100MB/run |
| 训练日志 | `logs/` | 10MB/run |
| Profile trace | `traces/` | 1-10GB/run |

### 8.2 怎么分享给合作者

- 压缩：`tar czf results_M4.tar.gz results/M4_*/`
- 上传：Dropbox / 内部 FTP / 百度网盘
- 标注：哪个 commit hash + 哪台机器 + 完整配置

---

## 9. 提交前自检清单

每次 push 前：

- [ ] `git status` 干净（无未提交修改）
- [ ] 所有新代码有测试
- [ ] `pytest tests/ -v` 全过
- [ ] 文档更新（README / docs/）
- [ ] 没有大文件（检查 `git status` 输出）
- [ ] 没有敏感信息（`git log -p` 检查）
- [ ] commit message 符合规范
- [ ] branch 命名符合规范

---

## 10. 紧急情况处理

### 10.1 误传敏感信息

```bash
# 1. 立刻从 git 移除（但历史还在）
git rm --cached path/to/secret
git commit -m "chore: remove accidentally committed secret"

# 2. **关键**：换密钥！git 历史里还在
# 3. 通知合作者
# 4. 考虑用 git filter-branch 清理历史（复杂）
```

### 10.2 push 错分支

```bash
# 1. 撤销最后一次 push（不删远程 commit）
git push --force-with-lease origin LAST_GOOD_COMMIT:main

# 2. 撤销 commit（本地）
git reset --hard HEAD~1

# 3. 重新 commit 到正确分支
```

### 10.3 实验结果不可复现

按 blueprint §3 failure_thresholds 收敛主张：
- H2 失败：只报通信节省
- H3 失败：去掉"必要"表述
- H4 失败：弱化"异步"，保留"变长调度"
- H5 失败：讨论扩展性限制

---

## 11. 一句话总结

> **实验代码不是 commit 完就完——commit hash + 配置 + 元数据 + 结果打包，才是"可复现"的最小单位。**

---

> 配套文档：
> - `release_checklist.md` — 投稿前 / Camera Ready 检查清单
> - `reproducibility.md` — 可复现性说明（投稿时用）
