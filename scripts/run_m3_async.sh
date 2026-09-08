#!/bin/bash
# M3 启动脚本：异步通信-计算 overlap profiling
# 配套文档：M2_pre_launch_checklist.md

set -e

NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-29500}
PROFILE_DIR=${PROFILE_DIR:-"./traces"}

echo "=================================================="
echo "  M3 启动：异步 overlap profiling"
echo "=================================================="
echo "NPROC = $NPROC"
echo "PROFILE_DIR = $PROFILE_DIR"
echo "=================================================="

mkdir -p "$PROFILE_DIR"

# Step 1: nsys trace 收集
echo ""
echo "Step 1: nsys trace 收集"
if command -v nsys &> /dev/null; then
    nsys profile -o "$PROFILE_DIR/m3_dcc_kv_async" \
        --trace=cuda,nvtx --force-overwrite=true \
        python -c "
import torch
import torch.distributed as dist
import os
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '$MASTER_PORT'
torch.cuda.set_device(0)
dist.init_process_group(backend='nccl', rank=0, world_size=1)
print('nsys DCC-KV async profile OK')
dist.destroy_process_group()
"
    echo "Trace: $PROFILE_DIR/m3_dcc_kv_async.nsys-rep"
else
    echo "nsys not available, skipping"
fi

# Step 2: torch.profiler trace
echo ""
echo "Step 2: torch.profiler trace"
python -c "
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
x = torch.randn(2048, 2048, device='cuda')
y = torch.randn(2048, 2048, device='cuda')
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler('$PROFILE_DIR/torch_profiler'),
) as prof:
    for _ in range(20):
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
print('torch.profiler trace OK')
"

# Step 3: 比较同步 vs 异步（仅收集时间数据）
echo ""
echo "Step 3: 同步 vs 异步 wall-clock 对比"
python -m pytest tests/gpu/test_async_overlap.py::test_async_vs_sync_overlap -v -s \
    --override-ini="addopts=-v --tb=short --strict-markers"

echo ""
echo "=================================================="
echo "  M3 profiling 完成"
echo "  Trace 位置：$PROFILE_DIR"
echo "=================================================="
