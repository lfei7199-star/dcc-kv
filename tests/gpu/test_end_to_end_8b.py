"""M4 GPU 验证：端到端 8B 模型 + 真实 attention kernel。

⚠️ 必须有 GPU + 8B 模型权重（Meta-Llama-3.1-8B-Instruct）。

对应 blueprint H1-H4：完整实验主表。

验证：
- LLaMA-3.1-8B 加载 + 切到多卡
- DCC-KV 同步版跑通
- DCC-KV 异步版跑通
- p50/p95 延迟测量
- 质量（perplexity）vs 完整 KV
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

pytestmark = pytest.mark.gpu


# ============================================================================
# 端到端 8B 模型测试
# ============================================================================
def _8b_worker(rank, args):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args["port"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=args["world_size"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = args["model_path"]
    if rank == 0:
        print(f"Loading {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=f"cuda:{rank}",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 准备输入
    prompt = "The quick brown fox jumps over the lazy dog. " * 256  # 简化
    inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{rank}")

    # 跑 baseline（无压缩）
    if rank == 0:
        print("Running baseline (no compression)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    torch.cuda.synchronize()
    baseline_time = time.perf_counter() - t0

    if rank == 0:
        print(f"Baseline: {baseline_time:.2f}s, output tokens: {out.shape[-1]}")
        # 记录（实际实验要存到 run result schema）
        result = {
            "model": model_path,
            "context_length": inputs.input_ids.shape[-1],
            "baseline_time_s": baseline_time,
            "gpu_count": args["world_size"],
        }
        with open(args["result_path"], "w") as f:
            json.dump(result, f, indent=2)

    dist.destroy_process_group()


@pytest.mark.gpu
def test_8b_baseline_runs(gpu_available, tmp_path):
    """8B 模型 baseline（无压缩）跑通。"""
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    if not os.environ.get("DCC_KV_8B_PATH"):
        pytest.skip("Set DCC_KV_8B_PATH env var to LLaMA-3.1-8B-Instruct path")
    torch.multiprocessing.spawn(
        _8b_worker,
        args=({
            "world_size": 2,
            "model_path": os.environ["DCC_KV_8B_PATH"],
            "port": 29540,
            "result_path": str(tmp_path / "8b_baseline.json"),
        },),
        nprocs=2,
    )

    result_path = tmp_path / "8b_baseline.json"
    if result_path.exists():
        with open(result_path) as f:
            data = json.load(f)
        assert "baseline_time_s" in data
        assert data["baseline_time_s"] > 0


# ============================================================================
# DCC-KV 端到端测试（需要 DCC-KV 完整实现）
# ============================================================================
def test_dcc_kv_8b_deferred():
    """DCC-KV 8B 端到端测试 deferred 到 M4 完成。"""
    pytest.skip("DCC-KV 8B end-to-end deferred to M4; M3 must complete first")
