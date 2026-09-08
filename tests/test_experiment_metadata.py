"""元数据 + 结果 schema 测试。"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiment_metadata import (
    ExperimentMetadata,
    RunResult,
    get_git_commit,
)


class TestMetadata:
    """元数据 schema 测试。"""

    def test_metadata_required_fields(self):
        """必填字段都在。"""
        m = ExperimentMetadata(
            run_id="test_001",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            context_length=8192,
            gpu_count=4,
        )
        d = m.to_dict()
        # blueprint §4.1 末尾要求
        for key in [
            "git_commit", "python_version", "pytorch_version",
            "cuda_version", "nccl_version", "transformers_version",
            "model_name", "model_commit_or_path", "tokenizer_commit_or_path",
            "context_length", "gpu_count", "gpu_model", "gpu_memory_gb",
            "interconnect", "precision",
            "rope_extension_used", "rope_extension_disclosed",
        ]:
            assert key in d, f"Missing required field: {key}"

    def test_rope_extension_disclosure(self):
        """blueprint §4.1 禁止 RoPE 扩展但不披露。"""
        m_disclosed = ExperimentMetadata(
            run_id="t",
            rope_extension_used="yarn-128k",
            rope_extension_disclosed=True,
        )
        m_undisclosed = ExperimentMetadata(
            run_id="t",
            rope_extension_used="yarn-128k",
            rope_extension_disclosed=False,
        )
        # 这两个是有意为之的区分；提交时 reviewer 知道 disclosed
        assert m_disclosed.to_dict()["rope_extension_disclosed"] is True
        assert m_undisclosed.to_dict()["rope_extension_disclosed"] is False

    def test_run_result_statistics(self):
        """RunResult 正确算统计量。"""
        # 已知数据：median ≈ 5, p5 ≈ 1, p95 ≈ 9
        values = list(range(1, 11))
        result = RunResult.from_values(values, metric_name="latency_ms", unit="ms")
        assert result.median == 5.5
        assert result.p5 == pytest.approx(1.45, abs=0.1)
        assert result.p95 == pytest.approx(9.55, abs=0.1)
        assert result.n_runs == 10
        assert 0 < result.ci_95_lower < result.median < result.ci_95_upper

    def test_metadata_save_load(self, tmp_path):
        """元数据能 save / load 回来。"""
        m = ExperimentMetadata(
            run_id="test_001",
            model_name="test-model",
            context_length=4096,
        )
        path = tmp_path / "metadata.json"
        m.save(str(path))
        assert path.exists()

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["run_id"] == "test_001"
        assert loaded["model_name"] == "test-model"
        assert loaded["context_length"] == 4096

    def test_git_commit_safe(self):
        """get_git_commit 不报错（即使不是 git 仓库）。"""
        commit = get_git_commit()
        # 应该是字符串，要么是 hash，要么是 "unknown"
        assert isinstance(commit, str)
        assert len(commit) > 0

    def test_run_result_str(self):
        """RunResult 字符串化。"""
        result = RunResult.from_values([1.0, 2.0, 3.0], metric_name="acc", unit="%")
        s = str(result)
        assert "acc" in s
        assert "%" in s
        assert "median=" in s
