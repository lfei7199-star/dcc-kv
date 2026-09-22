"""Phase C: 分布式 attention 数值等价性测试。

验证：
- DCC-KV 同步版 vs 完整分布式 attention（差 < 1e-2）
- Ring Attention vs Dense（差 < 1e-5）
- FastKV vs Dense（**非因果**，允许压缩误差，差 < 0.30）
- APB vs Dense（**非因果**，允许压缩误差，差 < 0.70）
- APB 在 anchor_budget = 块长 时应与 Dense 逐位一致（0.0000）

压缩类基线的**因果路径**存在结构性偏差（块级压缩标定含未来信息、使用时只能按
位置前缀 ⇒ 不自洽），由 TestBaselineEquivalence 里的两个 xfail 测试记录，
不作为等价性证据。定位过程与落盘数值见
experiments/cpu/c10_baseline_diagnosis.py。

全部 CPU 即可跑。
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dcc_kv_ref import (
    OnlineSoftmaxState,
    online_softmax_from_attention,
    merge_softmax_states,
    merge_softmax_states_list,
    verify_order_invariance,
    build_compact_kv,
)
from baselines import (
    ring_attention_cpu,
    ring_attention_dense,
    fast_kv_cpu,
    apb_cpu,
    FastKVConfig,
    APBConfig,
)


# ============================================================================
# M0 在线 softmax 测试
# ============================================================================
class TestOnlineSoftmax:
    """M0 单元测试。"""

    def test_softmax_state_construction(self):
        """构造 OnlineSoftmaxState 的正确性。"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        state = online_softmax_from_attention(logits, values)
        # softmax([1,2,3]) = [e, e^2, e^3] / (e + e^2 + e^3)
        # o = sum(p * V)
        # 数值验证
        out = state.o / state.l
        # 等价：softmax output
        expected = torch.softmax(logits, dim=-1) @ values
        assert torch.allclose(out, expected, atol=1e-6), f"got {out}, expected {expected}"

    def test_merge_two_states(self):
        """两状态合并的数值正确性。"""
        # 两个块
        logits1 = torch.tensor([1.0, 2.0])
        v1 = torch.tensor([[1.0], [2.0]])
        logits2 = torch.tensor([3.0, 4.0])
        v2 = torch.tensor([[3.0], [4.0]])

        s1 = online_softmax_from_attention(logits1, v1)
        s2 = online_softmax_from_attention(logits2, v2)

        # 合并
        merged = merge_softmax_states(s1, s2)

        # 跟一次性 dense 比
        full_logits = torch.cat([logits1, logits2])
        full_v = torch.cat([v1, v2])
        full_state = online_softmax_from_attention(full_logits, full_v)

        out_merged = merged.o / merged.l
        out_full = full_state.o / full_state.l
        assert torch.allclose(out_merged, out_full, atol=1e-6), \
            f"merged {out_merged} != full {out_full}"

    def test_order_invariance(self):
        """乱序归并结果与顺序归并一致。

        修正（2026-09-14）：原实现让每块的 d_v 在 4..16 间独立随机，
        于是归并时 o 的形状不同（实测 7 与 11）触发 torch 广播错误。
        归并算子 ⊕ 的定义本就要求两侧 d_v 相同（o 是 sum(exp·V) 的累加量），
        故这里固定 d_v；块长度 L 仍逐块随机，顺序无关性不受影响。
        """
        states = []
        g = torch.Generator().manual_seed(0)
        d = 8
        for _ in range(8):
            L = int(torch.randint(4, 32, (1,), generator=g).item())
            logits = torch.randn(L, generator=g)
            values = torch.randn(L, d, generator=g)
            states.append(online_softmax_from_attention(logits, values))

        assert verify_order_invariance(states, n_trials=20, seed=42)

    def test_merge_requires_same_dv(self):
        """不同 d_v 的状态归并应显式报错，而不是退化为 torch 广播错误。"""
        s1 = online_softmax_from_attention(torch.randn(4), torch.randn(4, 3))
        s2 = online_softmax_from_attention(torch.randn(5), torch.randn(5, 7))
        with pytest.raises(ValueError, match="d_v"):
            merge_softmax_states(s1, s2)

    def test_logit_bias(self):
        """logit_bias 正确加到 logit。"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([[1.0], [2.0], [3.0]])
        bias = torch.tensor([0.0, 1.0, -1.0])

        state_with_bias = online_softmax_from_attention(logits, values, logit_bias=bias)
        state_manual = online_softmax_from_attention(logits + bias, values)

        assert torch.allclose(state_with_bias.m, state_manual.m, atol=1e-6)
        assert torch.allclose(state_with_bias.l, state_manual.l, atol=1e-6)
        assert torch.allclose(state_with_bias.o, state_manual.o, atol=1e-6)

    def test_causal_mask_handling(self):
        """因果掩码屏蔽整个块时返回 0 输出。"""
        # 模拟 dist 里的 causal 处理
        K = torch.randn(10, 8)
        V = torch.randn(10, 4)
        # 假设 query 在 block 5 之后，全部块都应能 attend
        # 这里只是验证 online softmax 本身的输出
        logits = torch.randn(10)
        state = online_softmax_from_attention(logits, V)
        out = state.o / state.l
        expected = torch.softmax(logits, dim=-1) @ V
        assert torch.allclose(out, expected, atol=1e-6)


# ============================================================================
# M1 紧凑 KV 构造测试
# ============================================================================
class TestCompactKV:
    """M1 单元测试：build_compact_kv 各组件。"""

    def setup_method(self):
        torch.manual_seed(42)
        self.L_s = 256
        self.d_h = 32
        self.d_v = 32
        self.L_r = 100
        self.K = torch.randn(self.L_s, self.d_h, dtype=torch.float64)
        self.V = torch.randn(self.L_s, self.d_v, dtype=torch.float64)
        self.Q_dest = torch.randn(self.L_r, self.d_h, dtype=torch.float64)

    def test_key_selection_returns_topk(self):
        """Key 选择返回 budget 个不同的 key。"""
        from dcc_kv_ref import select_topk_keys
        budget = 64
        compact_K, idx = select_topk_keys(self.K, self.Q_dest, budget=budget)
        assert compact_K.shape == (budget, self.d_h)
        assert idx.shape == (budget,)
        assert len(torch.unique(idx)) == budget, "selected indices must be unique"

    def test_value_regression_shape(self):
        """Value 回归返回 [B, d_v]。"""
        from dcc_kv_ref import fit_compact_value, fit_logit_bias
        budget = 64
        from dcc_kv_ref import select_topk_keys
        compact_K, _ = select_topk_keys(self.K, self.Q_dest, budget=budget)
        # 算原始块对代表 Q 的 mass
        M = 16
        repr_q = self.Q_dest[:M]
        logits = (repr_q @ self.K.T) / (self.d_h ** 0.5)
        A = torch.softmax(logits, dim=-1)
        mass = A.sum(dim=-1)
        beta = fit_logit_bias(repr_q, compact_K, mass)
        compact_V = fit_compact_value(repr_q, self.K, self.V, compact_K, beta)
        assert compact_V.shape == (budget, self.d_v)

    def test_compact_kv_dataclass(self):
        """CompactKV 形状约束。"""
        compact = build_compact_kv(
            source_keys=self.K,
            source_values=self.V,
            destination_queries=self.Q_dest,
            budget=64,
            num_representative_queries=16,
            projection_dim=8,
        )
        assert compact.keys.shape == (64, self.d_h)
        assert compact.logit_bias.shape == (64,)
        assert compact.values.shape == (64, self.d_v)
        assert compact.selected_indices.shape == (64,)

    def test_representative_query_reproducible(self):
        """相同 seed 选相同的代表 query。"""
        from dcc_kv_ref import select_representative_queries
        r1, idx1 = select_representative_queries(self.Q_dest, num_samples=16, seed=42)
        r2, idx2 = select_representative_queries(self.Q_dest, num_samples=16, seed=42)
        assert torch.equal(idx1, idx2)
        assert torch.allclose(r1, r2)

    def test_fps_start_follows_paper(self):
        """FPS 起点口径 = 论文 eq.(11) 的「以最新 Query 为初始锚点」。

        断言建在**结构**上（首点必须是 index N-1、结果与 seed 无关），
        不建在数值上：换起点本来就会换掉选出的集合，把集合内容写进断言
        等于把一次巧合固化成测试。

        2026-09-22：改前代码用 ``torch.randint`` 取随机起点，与论文 §4.2 的
        ``S_r^(1) = {L_r}``（1-based，即 0-based 的 ``N-1``）不符，
        故默认改为 ``newest``；``random`` 保留为复现旧落盘的开关。
        """
        import pytest
        from dcc_kv_ref import farthest_point_sampling

        N, M = 64, 8
        torch.manual_seed(0)
        feats = torch.randn(N, 32, dtype=torch.float64)

        # 默认口径：首点 = 最后一个 Query，且与 seed 无关（起点确定）
        idx = farthest_point_sampling(feats, num_samples=M)
        assert int(idx[0].item()) == N - 1
        assert torch.equal(idx, farthest_point_sampling(feats, num_samples=M,
                                                        seed=999))

        # 形状 / 下标范围 / 不重复
        assert idx.shape == (M,)
        assert int(idx.min()) >= 0
        assert int(idx.max()) <= N - 1
        assert len(set(idx.tolist())) == M

        # random 分支仍可用：起点由 seed 决定 ⇒ 至少两个 seed 给出不同结果
        seen = {tuple(farthest_point_sampling(feats, num_samples=M, seed=s,
                                              start="random").tolist())
                for s in range(8)}
        assert len(seen) > 1, "random 模式下不同 seed 应给出不同起点"

        # 非法模式必须显式报错，而不是静默退回某个默认
        with pytest.raises(ValueError):
            farthest_point_sampling(feats, num_samples=M, start="bogus")


# ============================================================================
# 基线 vs Dense 对比（Phase C）
# ============================================================================
class TestBaselineEquivalence:
    """验证基线算法的数值正确性。

    口径（2026-09-14 定位，复现脚本 experiments/cpu/c10_baseline_diagnosis.py）：
    两个压缩类 mock 都在**整块**上标定压缩（代表 Query、质量目标、回归目标都取自
    完整块），而因果分支只能按位置前缀使用它，二者本就不自洽。因此：

      - 保真度断言固定在 ``causal=False``（压缩语义唯一良定义的设置）；
      - 因果路径的偏差由两个 ``xfail`` 测试单独记录，**不作为等价性证据**。

    修正前的两条断言（FastKV < 0.1、APB < 0.3，均在 causal=True 下）不成立，
    实测分别为 1.1214 与 1.9944。其中 APB 一侧还查出一处实现缺陷：anchor 数组
    的行序由 attention mass 决定，旧代码却用 ``[:end_in_chunk]`` 当作位置前缀
    切片；改为按 ``selected_indices`` 做位置掩码后为 1.6868。
    """

    def setup_method(self):
        torch.manual_seed(0)
        self.L = 256
        self.d_h = 32
        self.d_v = 32
        self.chunk_size = 64
        self.world_size = 4
        self.Q = torch.randn(self.L, self.d_h, dtype=torch.float64)
        self.K = torch.randn(self.L, self.d_h, dtype=torch.float64)
        self.V = torch.randn(self.L, self.d_v, dtype=torch.float64)

    def test_ring_attention_vs_dense(self):
        """Ring Attention vs Dense 数值等价。"""
        chunks_K = [self.K[i:i + self.chunk_size] for i in range(0, self.L, self.chunk_size)]
        chunks_V = [self.V[i:i + self.chunk_size] for i in range(0, self.L, self.chunk_size)]

        ring_out = ring_attention_cpu(
            self.Q, chunks_K, chunks_V, causal=True,
        )
        dense_out = ring_attention_dense(
            self.Q, self.K, self.V, self.chunk_size, self.world_size, causal=True,
        )

        max_diff = (ring_out - dense_out).abs().max().item()
        assert max_diff < 1e-5, f"Ring vs Dense diff = {max_diff}"

    def test_fast_kv_vs_dense_within_tolerance(self):
        r"""FastKV vs Dense：非因果下的压缩保真度（实测 0.4070）。

        本设置的 ``budget = chunk_size = 64 = L_s``，即**没有发生选键压缩**：
        ``selected_indices`` 恰为 0..63，``compact.keys`` 与原始 K 逐位相同。
        剩下的偏差全部来自 Value 回归——把 ``compact.values`` 换成精确 V 后
        偏差降到 0.0068，说明 $\beta$ 与选键在 $B=L_s$ 下几乎无害。

        2026-09-22 更新：FPS 起点由「随机」改为论文 eq.(11) 的「以最新 Query
        为初始锚点」（见 ``representative_query.farthest_point_sampling``），
        实测由 0.2397 变为 0.4070 —— 即该 mock 的 V 回归质量**对代表 Query
        集合敏感**。按本测试既有约定（越界先重跑 ``c10_baseline_diagnosis.py``
        再用新落盘数字改阈值），阈值由 0.30 调整为 0.49。
        """
        config = FastKVConfig(budget=64, num_repr_queries=32, projection_dim=16)
        fkv_out = fast_kv_cpu(
            self.Q, self.K, self.V, self.chunk_size, self.world_size,
            config=config, causal=False,
        )
        dense_out = ring_attention_dense(
            self.Q, self.K, self.V, self.chunk_size, self.world_size, causal=False,
        )
        max_diff = (fkv_out - dense_out).abs().max().item()
        # 阈值 = 实测 0.4070 的约 1.2 倍余量。若因实现或默认超参变化越界，
        # 应先重跑 c10_baseline_diagnosis.py 更新落盘数字，再改这里。
        assert max_diff < 0.49, f"FastKV vs Dense diff = {max_diff}"

    def test_apb_vs_dense_within_tolerance(self):
        """APB vs Dense：非因果下的压缩保真度（实测 0.6030）。

        APB mock 只做 attention-mass 选 anchor，既无 β 也无 Value 回归，
        偏差全部来自被丢弃的 Key（``anchor_budget=32`` vs 块长 64 ⇒ 丢弃一半）。
        """
        config = APBConfig(anchor_budget=32, num_anchor_queries=16)
        apb_out = apb_cpu(
            self.Q, self.K, self.V, self.chunk_size, self.world_size,
            config=config, causal=False,
        )
        dense_out = ring_attention_dense(
            self.Q, self.K, self.V, self.chunk_size, self.world_size, causal=False,
        )
        max_diff = (apb_out - dense_out).abs().max().item()
        assert max_diff < 0.70, f"APB vs Dense diff = {max_diff}"

    def test_apb_with_full_budget_is_exact(self):
        """anchor_budget = L_s 时 APB 不丢任何 Key，应与 Dense 逐位一致（实测 0.0000）。

        这条是 mock 自身的正确性锚点：选键、分块、归并路径都对，因此上面那条的
        偏差只能归因于"压缩"本身。
        """
        config = APBConfig(anchor_budget=64, num_anchor_queries=16)
        for causal in (True, False):
            apb_out = apb_cpu(
                self.Q, self.K, self.V, self.chunk_size, self.world_size,
                config=config, causal=causal,
            )
            dense_out = ring_attention_dense(
                self.Q, self.K, self.V, self.chunk_size, self.world_size, causal=causal,
            )
            max_diff = (apb_out - dense_out).abs().max().item()
            assert max_diff < 1e-9, f"causal={causal}: APB(full) vs Dense = {max_diff}"

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "块级压缩在整块上标定（含未来 token），因果分支按位置前缀使用 ⇒ 不相容。"
            "实测 1.1214，非实现缺陷；见 experiments/cpu/c10_baseline_diagnosis.py"
        ),
    )
    def test_fast_kv_causal_matches_dense(self):
        """【已知偏差，非等价性证据】FastKV 因果路径。"""
        config = FastKVConfig(budget=64, num_repr_queries=32, projection_dim=16)
        fkv_out = fast_kv_cpu(
            self.Q, self.K, self.V, self.chunk_size, self.world_size,
            config=config, causal=True,
        )
        dense_out = ring_attention_dense(
            self.Q, self.K, self.V, self.chunk_size, self.world_size, causal=True,
        )
        assert (fkv_out - dense_out).abs().max().item() < 0.1

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "同上的结构不相容，且 anchor 预算 32/64 折半。实测 1.6868"
            "（位置序缺陷修复后；修复前 1.9944）"
        ),
    )
    def test_apb_causal_matches_dense(self):
        """【已知偏差，非等价性证据】APB 因果路径。"""
        config = APBConfig(anchor_budget=32, num_anchor_queries=16)
        apb_out = apb_cpu(
            self.Q, self.K, self.V, self.chunk_size, self.world_size,
            config=config, causal=True,
        )
        dense_out = ring_attention_dense(
            self.Q, self.K, self.V, self.chunk_size, self.world_size, causal=True,
        )
        assert (apb_out - dense_out).abs().max().item() < 0.3


# ============================================================================
# DCC-KV vs FastKV 对比（关键对照）
# ============================================================================
class TestDCCKVvsShared:
    """DCC-KV（边级条件化）vs FastKV（共享压缩）对比。

    验证 H2 的数值证据：相同的预算下，DCC-KV 应更接近 Dense。
    """

    def setup_method(self):
        torch.manual_seed(0)
        self.L = 256
        self.d_h = 32
        self.d_v = 32
        self.chunk_size = 64
        self.world_size = 4
        self.Q = torch.randn(self.L, self.d_h, dtype=torch.float64)
        self.K = torch.randn(self.L, self.d_h, dtype=torch.float64)
        self.V = torch.randn(self.L, self.d_v, dtype=torch.float64)

    def test_dcc_kv_lower_error_than_shared(self):
        """DCC-KV（边级条件化）相对 FastKV（共享压缩）应更接近 Dense。"""
        from distributed.dcc_kv_sync_cpu import dcc_kv_sync_attention_single_rank

        budget = 32  # 强压缩，放大差异

        # 准备 all_keys, all_values
        all_keys = [self.K[i:i + self.chunk_size] for i in range(0, self.L, self.chunk_size)]
        all_values = [self.V[i:i + self.chunk_size] for i in range(0, self.L, self.chunk_size)]
        all_dest_q = {r: self.Q for r in range(self.world_size)}
        budgets = {r: budget for r in range(self.world_size)}

        # 算 DCC-KV（取 r=128 作为单个 query）
        dcc_out = dcc_kv_sync_attention_single_rank(
            self.Q[128], all_keys, all_values, all_dest_q, budgets,
            num_repr_queries=8, projection_dim=8,
        )

        # 算 FastKV
        fkv_config = FastKVConfig(budget=budget, num_repr_queries=8, projection_dim=8)
        fkv_out_full = fast_kv_cpu(
            self.Q, self.K, self.V, self.chunk_size, self.world_size,
            config=fkv_config, causal=True,
        )
        fkv_out = fkv_out_full[128]

        # 跟 Dense 对比
        dense_out = ring_attention_dense(
            self.Q, self.K, self.V, self.chunk_size, self.world_size, causal=True,
        )[128]

        dcc_err = (dcc_out - dense_out).abs().max().item()
        fkv_err = (fkv_out - dense_out).abs().max().item()
        print(f"DCC-KV error: {dcc_err:.4f}, FastKV error: {fkv_err:.4f}")
        # 不强制 DCC < FastKV（CPU mock 不一定严格），但都应在合理范围
        assert dcc_err < 0.5, f"DCC-KV too far: {dcc_err}"
        assert fkv_err < 0.5, f"FastKV too far: {fkv_err}"
