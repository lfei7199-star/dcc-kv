# DCC-KV

**D**estination-**C**onditioned **C**ompact **KV** Communication for Long-Context Collaborative Inference.

> **GitHub**: https://github.com/YOUR_USERNAME/dcc-kv （私人仓库）
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
dcc_kv/
├── src/
│   ├── dcc_kv_ref/           # M0-M1 reference（CPU 可跑）
│   │   ├── online_softmax.py
│   │   ├── representative_query.py
│   │   ├── key_selection.py
│   │   ├── calibration.py
│   │   ├── value_regression.py
│   │   └── compact_kv.py
│   ├── distributed/          # Phase A 多进程 + 通信原语
│   │   ├── launch_dist.py
│   │   ├── comm.py
│   │   ├── full_attention_cpu.py
│   │   ├── dcc_kv_sync_cpu.py
│   │   └── var_len_msg.py
│   ├── baselines/            # 基线 mock
│   │   ├── ring_attention_cpu.py
│   │   ├── fast_kv_cpu.py
│   │   └── apb_cpu.py
│   └── experiment_metadata.py
├── tests/
│   ├── test_dist_equivalence.py
│   ├── test_var_len_msg.py
│   ├── test_experiment_metadata.py
│   ├── test_smoke.py
│   ├── test_phase_a.py
│   └── gpu/                  # GPU 测试（deferred）
│       ├── test_nccl_basic.py
│       ├── test_async_overlap.py
│       ├── test_end_to_end_8b.py
│       └── profiling/
├── scripts/                  # 启动脚本
│   ├── run_m2_real.sh
│   └── run_m3_async.sh
├── docs/                     # 文档
│   ├── git_strategy.md
│   ├── release_checklist.md
│   └── reproducibility.md
├── conftest.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

## 当前状态

✅ **M0 完成**（Online Softmax 数值验证，CPU PyTorch FP64）  
✅ **M1 完成**（单进程核心构造，CPU PyTorch FP64）  
⏳ **M2 准备中**（Phase A 启动器就绪；待 GPU 资源）

## 快速开始

### 安装

```bash
pip install -r requirements.txt
# CPU-only：
pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
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

## License

Apache 2.0
