# tests/gpu/ — GPU 需求测试（待 GPU 资源就绪）

⚠️ **警告：这些测试必须有真实 GPU + NCCL backend 才能跑。**

在 CPU 环境下运行会被 pytest 自动跳过（pytest.ini 里 `-m "not gpu"`）。

## 跑 GPU 测试

```bash
# 跑全部 GPU 测试
pytest tests/gpu/ -v

# 跑单个文件
pytest tests/gpu/test_nccl_basic.py -v

# 跑特定 marker
pytest tests/ -m "gpu and not slow" -v
```

## GPU 资源需求

按 M2_pre_launch_checklist：

| 测试文件 | 最小 GPU | 最小显存 | 推荐配置 |
|---|---|---|---|
| `test_nccl_basic.py` | 1 卡 | 8GB | 2 卡 A100 80G |
| `test_async_overlap.py` | 2 卡 | 16GB | 4 卡 A100 80G |
| `test_end_to_end_8b.py` | 2 卡 | 32GB | 8 卡 A100 80G |
| `test_70b_scaling.py` | 4 卡 | 80GB | 8 卡 A100 80G |

## 文件清单

- `test_nccl_basic.py` — NCCL backend 基本验证（2 进程 / 4 进程）
- `test_async_overlap.py` — 异步通信-计算 overlap 量化
- `test_end_to_end_8b.py` — 端到端 8B 模型 + 真实 attention kernel
- `test_70b_scaling.py` — 70B/72B 模型 + 8 卡 + scaling
- `profiling/nsys_runner.py` — Nsight Systems trace 收集
- `profiling/torch_profiler_runner.py` — PyTorch Profiler trace 收集

## 资源就绪后的启动顺序

1. **先跑 `test_nccl_basic.py`**（最简单，验证 NCCL 能用）
2. **再跑 `test_async_overlap.py`**（核心创新点）
3. **再跑 `test_end_to_end_8b.py`**（接真实模型）
4. **最后 `test_70b_scaling.py`**（需要最大资源）

## CI 状态

GPU 测试在 `.github/workflows/test.yml` 里只对 `self-hosted` runner 启用。
push 到 main 时自动跑；PR 阶段不跑（避免消耗 GPU 资源）。
