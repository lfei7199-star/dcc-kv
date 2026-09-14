# DCC-KV

**D**estination-**C**onditioned **C**ompact **KV** Communication for Long-Context Collaborative Inference.

> **GitHub**: https://github.com/lfei7199-star/dcc-kv （私人仓库）
> **Status**: M0-M1 done, M2 pending GPU
> **License**: Apache 2.0

## 不主张的内容（重要）

- ❌ 任何任务质量结果（LongBench / RULER / Needle / ∞Bench）
- ❌ 任何通信性能结果
- ❌ 任何多 GPU 可扩展性
- ❌ 任何与基线的对比

详见 `docs/` 下的边界声明（blueprint v1.1 附录 D）。

## 项目结构

```
dcc-kv/
├── src/
│   ├── dcc_kv_ref/           # M0-M1 参考实现（CPU 可跑）
│   │   ├── online_softmax.py        # 归并算子 ⊕
│   │   ├── representative_query.py  # 目的端代表 Query 选取
│   │   ├── key_selection.py         # 选键
│   │   ├── calibration.py           # β（质量偏置）拟合
│   │   ├── value_regression.py      # V 岭回归
│   │   └── compact_kv.py            # 构造链路总入口
│   ├── distributed/          # 多进程 + 通信原语
│   ├── baselines/            # 基线 mock（Ring / FastKV / APB）
│   └── experiment_metadata.py
├── experiments/
│   ├── common/               # 合成场景与结果汇总
│   ├── cpu/                  # E0–E11 / A3 机制级（CPU 可复现）
│   ├── gpu/                  # E5–E8 任务级与系统级（需 GPU + NCCL）
│   ├── run_cpu.sh
│   └── run_gpu.sh
├── tests/
│   ├── test_*.py             # CPU 测试
│   └── gpu/                  # GPU 测试（pytest 默认跳过）
├── paper/                    # 论文（xelatex；sections/ + figures/）
├── docs/
│   ├── commit_log.md         # 逐次提交报告
│   ├── FILE_MAP.md           # 文件说明（每个文件做什么）
│   ├── git_strategy.md
│   ├── release_checklist.md
│   ├── reproducibility.md
│   └── ssh_setup.md
├── scripts/                  # M2 / M3 运行脚本
├── conftest.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

**逐文件说明见 [docs/FILE_MAP.md](docs/FILE_MAP.md)。**

## 当前状态

✅ **M0 完成**（Online Softmax 数值验证，CPU PyTorch FP64）  
✅ **M1 完成**（单进程核心构造，CPU PyTorch FP64）  
⏳ **M2 准备中**（Phase A 启动器就绪；待 GPU 资源）

## 快速开始

### 安装

```bash
pip install -r requirements.txt
# CPU-only：
pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu  # 论文实验环境：2.11.0+cpu
```

### 跑测试

```bash
# 仅 mock 模式（< 5 秒，不需要 distributed）
pytest tests/ -v

# 包括 2 进程 gloo 真实 distributed 测试
pytest tests/ -v -m distributed

# Phase A 验收
pytest tests/test_phase_a.py -v
```

### 使用启动器

```python
from src.distributed import launch_dist, DistributedComm

def my_worker(rank, args):
    comm = DistributedComm(backend="gloo")
    comm.init(rank=rank, world_size=2)
    
    tensor = torch.tensor([float(rank)])
    result = comm.all_reduce(tensor)  # 1.0
    
    comm.cleanup()
    return f"rank_{rank}_done"

launch_dist(my_worker, nproc_per_node=2, args=None, backend="gloo")
```

### 构建 CompactKV

```python
from src.dcc_kv_ref import build_compact_kv

compact = build_compact_kv(
    source_keys=K,              # [L_s, d_h]
    source_values=V,            # [L_s, d_v]
    destination_queries=Q,      # [L_r, d_h]
    budget=64,
)
# compact.keys, compact.logit_bias, compact.values, compact.selected_indices
```

## 文档导航

- **[docs/FILE_MAP.md](docs/FILE_MAP.md)** — **文件说明：每个文件的作用与内容**
- **[docs/commit_log.md](docs/commit_log.md)** — 逐次提交报告（做了什么 / 验证了什么 / 还有什么没解决）
- **[docs/git_strategy.md](docs/git_strategy.md)** — Git 仓库管理策略（分支、commit、tag、实验可追溯）
- **[docs/release_checklist.md](docs/release_checklist.md)** — 投稿前 / Camera Ready 清单
- **[docs/reproducibility.md](docs/reproducibility.md)** — 可复现性说明

## 上级文档

研究执行蓝图书（v1.1，2026-09-08 冻结）：

- `../dcc_kv_plan/research_execution_blueprint_v1.md`
- `../dcc_kv_plan/experiment_matrix.yaml` v1.2.0
- `../dcc_kv_plan/M2_pre_launch_checklist.md`
- `../dcc_kv_plan/references.bib` 26 条
- `../dcc_kv_plan/contribution_boundary_section.md`

> ⚠️ **以上 5 份文件当前均不在仓库内，也不在相邻目录**（全盘搜索未找到，记账为 `docs/commit_log.md` C11）。
> 其中 blueprint v1.1 §3 / §4 的内容已大体抄入本仓库（见 `docs/reproducibility.md` §6、
> `docs/release_checklist.md` §4、`src/experiment_metadata.py`），因此**不阻塞开发**，
> 但**无法做一致性核对**。另注：本行原写 `references.bib` 26 条，而仓库内
> `paper/refs.bib` 实际为 **17 条**（且全部被引用）—— 该差额待取回原件后确认。

## License

Apache 2.0
