"""2026-09-18 自查发现回归锚点。

每一条都对应一个**已修复**的缺陷。它们是"看起来对"的那一类：形状、量级、
文件名都正常，不报错，但结论是错的或永远出不来。逐条：
  T1 paired_bootstrap(higher_is_better=True) 的结论方向曾经是反的（H2 正是准确率）
  T2 save_csv 表头取 rows[0].keys() ⇒ 混合三类行的表在落盘时整表 ValueError
  T3 A4/A5 的加速比上界写成 1 + comm/comp，与论文 1 + min/max 不一致
  T4 A5 把 h4_pass=None（未判定）折进"未达标"
  T5 dcc_kv_attention 多块缺 remote_offsets 时静默按偏移 0 算因果掩码
  T6 E6 的 H2 配对永远查不到被折叠 sync 轴的基线行 ⇒ 判定恒为「无配对点」
  T7 E7 的两个通配条件从未被任何调用方判定
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys
import tempfile

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.common import report as R  # noqa: E402
from src.dcc_kv_ref import dcc_kv_attention, identity_compact  # noqa: E402


def _read_src(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# =============================================================================
# T1 方向
# =============================================================================

def test_t1_verdict_direction_when_higher_is_better():
    """A 的准确率更高 ⇒ 必须报「A 显著更优」。修复前报的是「B 显著更优」。"""
    r = R.paired_bootstrap([0.9] * 20, [0.5] * 20,
                           metric_name="acc", higher_is_better=True)
    assert r.mean_diff > 0            # 原始方向：A 更大
    assert r.diff_unified < 0         # 判定方向：已统一成"越小越好"
    assert r.higher_is_better is True
    assert r.verdict().startswith("A 显著更优")


def test_t1_one_sided_test_never_declares_b_better():
    """单侧检验的备择假设固定为 A 更优 ⇒ B 更优时**不能**由这个 p 值下结论。

    写这条测试时才发现：原先的「B 显著更优」分支是**不可达**的 ——
    零分布是围绕 0 的符号翻转分布，观测到 diff_unified > 0 时 p 恒接近 1。
    因此这里断言的是"不会误报"，而不是"会报 B 更优"。
    """
    r = R.paired_bootstrap([0.5] * 20, [0.9] * 20,
                           metric_name="acc", higher_is_better=True)
    v = r.verdict()
    assert not v.startswith("A 显著更优")
    assert not v.startswith("B 显著更优"), "单侧检验不该宣称 B 更优"
    assert "单侧" in v and "反向检验" in v
    # 观测值确实在"B 更优"那一侧，且 p 值因此失去判别力
    assert r.diff_unified > 0
    assert r.p_value_one_sided > 0.5


def test_t1_lower_is_better_path_unchanged():
    """误差类指标（默认 higher_is_better=False）的结论方向不受本次修复影响。"""
    r = R.paired_bootstrap([0.1] * 20, [0.5] * 20, metric_name="err")
    assert r.mean_diff < 0
    assert r.diff_unified < 0
    assert r.verdict().startswith("A 显著更优")


# =============================================================================
# T2 落盘
# =============================================================================

def test_t2_save_csv_uses_union_of_keys():
    """E6 主表混有 measured / blocked / error 三类行，必须能一次写完且不丢列。"""
    rows = [
        {"model": "m", "status": "ok", "gpu_count_note": "x", "accuracy": 0.5},
        {"model": "m", "status": "blocked", "blockers": ["b1"]},
        {"model": "m", "status": "error", "error_type": "ValueError"},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.csv")
        R.save_csv(path, rows)          # 修复前这里抛 ValueError
        with open(path, newline="", encoding="utf-8") as f:
            got = list(csv.DictReader(f))
    assert len(got) == 3
    # 三类行的专有列都必须在表头里（静默丢列比报错更危险）
    for col in ("gpu_count_note", "blockers", "error_type", "accuracy"):
        assert col in got[0], f"表头缺列 {col}"
    assert got[1]["blockers"] == "['b1']"
    assert got[2]["error_type"] == "ValueError"


# =============================================================================
# T3 上界公式
# =============================================================================

def test_t3_speedup_bound_matches_paper_formula():
    from experiments.gpu.e5_gpu_ablation import speedup_bound
    # 论文 eq:speedup-bound = (comm+comp)/max(comm,comp) = 1 + min/max
    for comm, comp in [(1.0, 1.0), (10.0, 1.0), (1.0, 10.0), (3.0, 7.0), (0.25, 4.0)]:
        expect = (comm + comp) / max(comm, comp)
        assert speedup_bound(comm, comp) == pytest.approx(expect, rel=1e-12)
    assert speedup_bound(1.0, 1.0) == pytest.approx(2.0)      # 平衡 ⇒ 2x
    assert speedup_bound(10.0, 1.0) == pytest.approx(1.1)     # 通信主导 ⇒ 趋 1
    assert speedup_bound(1.0, 10.0) == pytest.approx(1.1)     # 计算主导 ⇒ 趋 1


def test_t3_old_formula_is_gone_from_source():
    """旧的 `1 + comm/comp` 只在 comp>=comm 时与论文一致，必须已删净。"""
    src = _read_src("experiments/gpu/e5_gpu_ablation.py")
    assert "1.0 + (comm_ms / max(comp_ms" not in src
    assert "1.0 + t_comm_ms / max(t_comp_ms" not in src
    assert "def speedup_bound(" in src


# =============================================================================
# T4 未判定 ≠ 未达标
# =============================================================================

def test_t4_a5_does_not_fold_unjudged_into_failed():
    src = _read_src("experiments/gpu/e5_gpu_ablation.py")
    assert 'bool(r["h4_pass"]) for r in rows' not in src, \
        "又把 h4_pass=None 折进未达标了（行级注释明写 None=未判定）"
    assert "n_rows_unjudged" in src
    assert "n_rows_judged" in src


# =============================================================================
# T5 因果偏移不能静默按 0
# =============================================================================

def _blocks(L=12, d_h=8, d_v=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(L, d_h, generator=g, dtype=torch.float64)
    V = torch.randn(L, d_v, generator=g, dtype=torch.float64)
    Q = torch.randn(L, d_h, generator=g, dtype=torch.float64)
    return K, V, Q


def test_t5_multi_block_without_offsets_raises():
    K, V, Q = _blocks()
    qp = torch.arange(12)
    with pytest.raises(ValueError, match="remote_offsets"):
        dcc_kv_attention(Q, [identity_compact(K[:4], V[:4]),
                             identity_compact(K[4:8], V[4:8])],
                         query_positions=qp)


def test_t5_offsets_length_must_match_block_count():
    K, V, Q = _blocks()
    qp = torch.arange(12)
    b = [identity_compact(K[:4], V[:4]), identity_compact(K[4:8], V[4:8])]
    with pytest.raises(ValueError, match="长度"):
        dcc_kv_attention(Q, b, query_positions=qp, remote_offsets=[0])
    with pytest.raises(ValueError, match="长度"):
        dcc_kv_attention(Q, b[:1], query_positions=qp, remote_offsets=[0, 4, 99])


def test_t5_single_block_without_offsets_is_offset_zero():
    """单块缺省等价于"该块起始于位置 0" —— 这条保留，且必须与显式 [0] 相同。"""
    K, V, Q = _blocks()
    qp = torch.arange(12)
    b = [identity_compact(K[:8], V[:8])]
    implicit = dcc_kv_attention(Q, b, query_positions=qp)
    explicit = dcc_kv_attention(Q, b, query_positions=qp, remote_offsets=[0])
    assert torch.equal(implicit, explicit)


def test_t5_correct_offsets_still_equal_dense_causal():
    from src.dcc_kv_ref import dense_attention
    K, V, Q = _blocks()
    qp = torch.arange(12)
    out = dcc_kv_attention(Q, [identity_compact(K[:4], V[:4]),
                               identity_compact(K[4:8], V[4:8])],
                           query_positions=qp, remote_offsets=[0, 4])
    ref = dense_attention(Q, K[:8], V[:8], query_positions=qp, causal=True).out
    assert torch.allclose(out, ref, atol=1e-12, rtol=0)


# =============================================================================
# T6 E6 的 H2 配对
# =============================================================================

def _e6_ns(**kw):
    import argparse
    base = dict(models=["m"], context_lengths=[4096],
                sync_modes=["sync", "async"], h2_delta_pp=None,
                h2_noise_floor_pp=None, h2_delta_bad_pp=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_t6_h2_pairs_with_collapsed_baseline_row():
    """基线是单卡方法 ⇒ 它的行 sync_async == "n/a"；配对必须能找到它。

    修复前 h2_points_from_rows 用 sync_modes[0] 去查，配对点恒为 0。
    """
    from experiments.gpu import e6_main_table as E
    rows = [
        {"method": "dcc_kv", "model": "m", "context_length": 4096,
         "sync_async": "sync", "accuracy": 0.70, "prefill_ms_median": 100.0},
        {"method": "kv_budget_shared", "model": "m", "context_length": 4096,
         "sync_async": E.SYNC_MODE_NA, "accuracy": 0.60, "prefill_ms_median": 120.0},
    ]
    pts = E.h2_points_from_rows(_e6_ns(), rows)
    assert len(pts) == 1, "H2 配对又断了"
    assert pts[0].quality_gain_pp == pytest.approx(10.0)
    assert pts[0].prefill_speedup == pytest.approx(1.2)
    assert pts[0].quality_comparable is None   # 未测出容差 ⇒ 记 unresolved，不是 False

    out = E.compute_h2(_e6_ns(), rows)
    assert out["h2_n_lengths"] == 1
    assert out["h2_unresolved_lengths"] == [4096]   # 分辨率不足，不是"未达标"
    assert out["h2_n_failed"] == 0


def test_t6_h2_absent_dcc_row_still_no_points():
    from experiments.gpu import e6_main_table as E
    rows = [{"method": "kv_budget_shared", "model": "m", "context_length": 4096,
             "sync_async": E.SYNC_MODE_NA, "accuracy": 0.6, "prefill_ms_median": 1.0}]
    assert E.h2_points_from_rows(_e6_ns(), rows) == []


def test_t6_ambiguous_baseline_rows_are_refused_not_guessed():
    """同一 (model, ctx) 上有多行且没有 "n/a" 标记 ⇒ 拒绝配对（宁可不判）。"""
    from experiments.gpu import e6_main_table as E
    rows = [
        {"method": "dcc_kv", "model": "m", "context_length": 4096,
         "sync_async": "sync", "accuracy": 0.7, "prefill_ms_median": 1.0},
        {"method": "kv_budget_shared", "model": "m", "context_length": 4096,
         "sync_async": "sync", "accuracy": 0.6, "prefill_ms_median": 2.0},
        {"method": "kv_budget_shared", "model": "m", "context_length": 4096,
         "sync_async": "async", "accuracy": 0.6, "prefill_ms_median": 2.0},
    ]
    assert E.h2_points_from_rows(_e6_ns(), rows) == []


# =============================================================================
# T7 E7 通配条件
# =============================================================================

def test_t7_short_context_reported_missing_without_lt4k():
    from experiments.gpu import build_eval_set as B
    recs = [{"dataset": "trec", "input": "q", "context": "c", "answers": ["entity"],
             "all_classes": ["description", "entity"], "length": 20000}]
    _, rep = B.convert_records(recs, source="t")
    assert "short_context" in rep.missing_e7_conditions()


def test_t7_short_context_covered_when_lt4k_present():
    from experiments.gpu import build_eval_set as B
    recs = [{"dataset": "trec", "input": "q", "context": "c", "answers": ["entity"],
             "all_classes": ["description", "entity"], "length": 100}]
    _, rep = B.convert_records(recs, source="t")
    assert "short_context" not in rep.missing_e7_conditions()
    assert rep.length_tag_counts == {"lt4k": 1}


def test_t7_low_budget_declared_not_decidable():
    """预算轴不是数据属性 ⇒ 必须显式声明"本工具判不了"，不能算成已覆盖。"""
    from experiments.gpu import build_eval_set as B
    _, rep = B.convert_records(B.SELFTEST_RECORDS, source="t")
    assert "low_budget" in rep.not_decidable_e7_conditions()
    assert "low_budget" not in rep.missing_e7_conditions()
    d = rep.to_dict()
    assert "e7_conditions_not_decidable" in d and "length_tag_counts" in d


def test_t7_unconfigured_wildcard_defaults_to_missing():
    """新加通配条件却忘了配规则时，默认报缺失（吵，但不会静默少报一个条件）。"""
    from experiments.gpu import build_eval_set as B
    old = dict(B.E7_REQUIRED_TASKS)
    try:
        B.E7_REQUIRED_TASKS["brand_new"] = ["*"]
        _, rep = B.convert_records(B.SELFTEST_RECORDS, source="t")
        assert "brand_new" in rep.missing_e7_conditions()
    finally:
        B.E7_REQUIRED_TASKS.clear()
        B.E7_REQUIRED_TASKS.update(old)


# =============================================================================
# T8 A4 计算侧的接收入参必须能均分（整除去切会静默丢掉余数行）
# =============================================================================

def test_t8_a4_comp_work_refuses_non_divisible_rows():
    from experiments.gpu import e5_gpu_ablation as E5
    work = E5._a4_comp_work(torch.randn(6, 3), world=2, d_h=3, d_v=1)
    work(torch.randn(6, 3 + 1 + 1))                    # 6 = 3×2：可以
    with pytest.raises(ValueError, match="整除"):
        work(torch.randn(5, 3 + 1 + 1))                # 5 // 2 = 2 ⇒ 会丢 1 行
    with pytest.raises(ValueError, match="少于"):
        work(torch.randn(1, 3 + 1 + 1))                # 少于卡数


def test_t8_a4_comp_work_uses_all_rows_when_divisible():
    """可整除时不得丢行：partial 的 key 数之和应等于块行数。"""
    from experiments.gpu import e5_gpu_ablation as E5
    from src.dcc_kv_ref import attention_kernel as K
    seen = {}
    orig = K.compact_kv_attention

    def spy(query, compact, **kw):
        seen[compact.keys.shape[0]] = seen.get(compact.keys.shape[0], 0) + 1
        return orig(query, compact, **kw)

    K.compact_kv_attention = spy
    try:
        work = E5._a4_comp_work(torch.randn(6, 3), world=3, d_h=3, d_v=1)
        work(torch.randn(9, 3 + 1 + 1))                # 9 = 3×3
    finally:
        K.compact_kv_attention = orig
    assert seen == {3: 3}, f"各源边应各拿到 3 行，实得 {seen}"
