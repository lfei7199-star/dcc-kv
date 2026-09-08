# 贡献指南（Contributing Guide）

> 配合 `docs/git_strategy.md` 一起读。
> 第一次贡献请从头到尾读一遍。

## 工作流程

```
1. 选 issue / 新 issue
2. 在 develop 分支切 feature/* 子分支
3. 写代码 + 写测试
4. 跑 pytest
5. 提 PR 到 develop
6. 至少 1 个 reviewer approve
7. 合并 → 删除子分支
```

## 分支命名

| 类型 | 命名 | 合并到 |
|---|---|---|
| `feature/*` | `feature/M2_var_len_msg` | `develop` |
| `experiment/*` | `experiment/M2_2proc` | `develop` |
| `paper/*` | `paper/mlsys2025` | `main` |
| `hotfix/*` | `hotfix/...` | `main` + `develop` |

**禁止**：直接 commit 到 `main` / `develop`

## Commit 规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

**type**：

| type | 用途 |
|---|---|
| `feat` | 新功能 |
| `fix` | bug 修复 |
| `docs` | 仅文档 |
| `test` | 仅测试 |
| `refactor` | 重构（无功能变化） |
| `perf` | 性能优化 |
| `exp` | 实验数据 / 结果 |
| `chore` | 杂项（CI、依赖） |

**例子**：

```
feat(distributed): add var-len all-to-allv primitive

实现 M2 阶段核心通信原语。变长消息通过
"先交换 sizes，再交换数据"两阶段完成。

Reference: blueprint v1.1 §6 M2 acceptance

Tests: tests/test_var_len_msg.py (5 tests)
```

## 写代码

### 编码风格

- Python 3.10+
- 行长 ≤ 120
- 用 `black` 格式化（CI 不强制但建议）
- 用 `flake8` 检查（`max-line-length=120 --ignore=E501,W503`）
- 类型注解：尽量写（CI 不强制但建议）

### 写测试

**新功能必须带测试**。位置：
- `tests/test_<module>.py` — 单元测试
- `tests/gpu/test_<module>.py` — GPU 测试（加 `@pytest.mark.gpu`）

**测试规范**：
- 1 测试函数 = 1 行为
- 用 `@pytest.mark.distributed` / `@pytest.mark.gpu` / `@pytest.mark.slow` 分类
- 跑测试前 `pytest tests/ -v` 全过

### 文档

- 新模块 → 更新 README / docs/
- 改方法 → 更新相关 blueprint / 实验矩阵
- 实验结果 → 用 `ExperimentMetadata` schema 记录

## 跑测试

```bash
# 装依赖
pip install -r requirements.txt
pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 仅 mock 模式（< 5 秒）
pytest tests/ -v

# 包含 distributed 测试
pytest tests/ -v -m distributed

# 完整
pytest tests/ -v -a
```

## 提 PR

1. 推到 origin：`git push -u origin <branch>`
2. GitHub 上点 **Compare & pull request**
3. 填写 PR template（见 `.github/PULL_REQUEST_TEMPLATE.md`）
4. 关联 issue：`Closes #123` 或 `Refs #123`
5. 至少 1 个 reviewer approve
6. CI 全过（cpu-tests job）
7. Squash merge（保持 main 干净）

## 报告 issue

用 `.github/ISSUE_TEMPLATE/bug_report.md` 或 `feature_request.md`：
- 复现步骤
- 预期 / 实际行为
- 环境（OS、Python、PyTorch、CUDA 版本）
- 日志 / traceback

## 实验记录

**M2+ 阶段每次实验都做**：

1. **独立分支**：`git checkout -b experiment/M2_2proc`
2. **元数据**：用 `ExperimentMetadata` 自动收集
3. **结果目录**：`results/<run_id>/`
   - `metadata.json`
   - `raw_values.json`
   - `metrics.json`（含 median/p5/p95/bootstrap CI）
4. **配置**：`configs/<experiment>.yaml`
5. **commit**：`exp(M2): <description>`
6. **tag**：`m2-2proc-20260908`（日期命名）
7. **PR** → develop

## Code Review

reviewer 关注：

- [ ] 代码符合现有风格
- [ ] 新代码有测试
- [ ] 测试有断言（不是 `assert True`）
- [ ] 文档 / 注释更新
- [ ] commit message 符合规范
- [ ] 没有调试代码 / print
- [ ] 没有大文件（检查 `git diff`）
- [ ] 没有敏感信息

## 不做的事

- ❌ 直接 commit 到 main / develop
- ❌ 强制 push（用 `git push --force-with-lease` 替代 `--force`）
- ❌ 把模型权重 / 数据集 commit 进 git
- ❌ 把 token / API key commit 进 git
- ❌ 在 PR 里讨论实验结果对错（用 issue / discussion）

## 风格参考

参考现有代码：
- `src/dcc_kv_ref/online_softmax.py` — 类型注解 + 完整 docstring
- `src/distributed/comm.py` — 接口统一 + mock 模式
- `tests/test_dist_equivalence.py` — 测试组织

## 紧急联系

- Issue: https://github.com/lfei7199-star/dcc-kv/issues
- 论文相关：参考 paper/ 目录下的 README
- 内部讨论：开 discussion / slack

## 致谢

贡献者列表见 `CONTRIBUTORS.md`（首次合并后生成）。
