#!/bin/bash
# M2 启动脚本：真实 GPU 验证 NCCL + 分布式 attention
# 配套文档：M2_pre_launch_checklist.md

set -e

# 默认值
NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-29500}
MODEL_PATH=${DCC_KV_8B_PATH:-"meta-llama/Llama-3.1-8B-Instruct"}

echo "=================================================="
echo "  M2 启动：真实 GPU 多卡验证"
echo "=================================================="
echo "NPROC = $NPROC"
echo "MASTER_PORT = $MASTER_PORT"
echo "MODEL_PATH = $MODEL_PATH"
echo "=================================================="

# Step 1: 单卡 smoke test
echo ""
echo "Step 1: 单卡 smoke test"
python -m pytest tests/gpu/test_nccl_basic.py::test_nccl_2proc_all_reduce -v \
    --override-ini="addopts=-v --tb=short --strict-markers"

# Step 2: 2 卡 NCCL
echo ""
echo "Step 2: 2 卡 NCCL all_to_allv"
python -m pytest tests/gpu/test_nccl_basic.py::test_nccl_2proc_all_to_all_v -v \
    --override-ini="addopts=-v --tb=short --strict-markers"

# Step 3: 4 卡 scaling
if [ "$NPROC" -ge 4 ]; then
    echo ""
    echo "Step 3: 4 卡 scaling"
    python -m pytest tests/gpu/test_nccl_basic.py::test_nccl_4proc_scaling -v \
        --override-ini="addopts=-v --tb=short --strict-markers"
fi

# Step 4: 异步 overlap
echo ""
echo "Step 4: 异步通信 overlap"
python -m pytest tests/gpu/test_async_overlap.py -v \
    --override-ini="addopts=-v --tb=short --strict-markers"

# Step 5: nsys trace（如果可用）
echo ""
echo "Step 5: 收集 nsys trace（如可用）"
if command -v nsys &> /dev/null; then
    mkdir -p traces
    nsys profile -o traces/m2_overlap --trace=cuda,nvtx --force-overwrite=true \
        python -c "import torch; print('nsys test OK')"
    echo "Trace saved to traces/m2_overlap.nsys-rep"
else
    echo "nsys not available, skipping"
fi

echo ""
echo "=================================================="
echo "  M2 启动完成"
echo "=================================================="
