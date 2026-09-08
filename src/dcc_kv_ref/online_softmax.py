"""M0: Online Softmax 状态与合并算子。

Reference: Milakov & Gimelshein 2018 (arXiv:1805.02867)
Adapted for attention with optional logit bias (DCC-KV extension).

提供：
- OnlineSoftmaxState: 三元组 (max, sum_exp, output) 表示一个块
- online_softmax_from_attention: 从 attention 权重和 value 构造状态
- merge_softmax_states: 归并算子 ⊕（满足交换律+结合律）
- verify_order_invariance: 单元测试：乱序归并结果与顺序归并一致
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class OnlineSoftmaxState:
    """Online Softmax 状态：单 head 单 query 视角。

    表示对一个块算出的 (m, l, o) 三元组：
        m = max(logits)             用于数值稳定
        l = sum(exp(logits - m))   softmax 分母
        o = sum(exp(logits - m) * V)  softmax 分子

    最终 attention output = o / l

    维度约定：
        logits: [L_block]
        V:      [L_block, head_dim]
        m:      scalar
        l:      scalar
        o:      [head_dim]
    """
    m: torch.Tensor    # scalar
    l: torch.Tensor    # scalar
    o: torch.Tensor    # [head_dim]

    def __post_init__(self):
        # 确保类型
        if not isinstance(self.m, torch.Tensor):
            self.m = torch.tensor(self.m, dtype=torch.float64)
        if not isinstance(self.l, torch.Tensor):
            self.l = torch.tensor(self.l, dtype=torch.float64)
        if not isinstance(self.o, torch.Tensor):
            self.o = torch.tensor(self.o, dtype=torch.float64)


def online_softmax_from_attention(
    logits: torch.Tensor,
    values: torch.Tensor,
    logit_bias: Optional[torch.Tensor] = None,
) -> OnlineSoftmaxState:
    """从 attention logits 和 values 构造 Online Softmax 状态。

    Args:
        logits: [L] 单个 query 对当前块的 attention logits
        values: [L, head_dim] 块的 value
        logit_bias: [L] 可选的 token-level bias（DCC-KV 的 β）

    Returns:
        OnlineSoftmaxState (m, l, o)
    """
    assert logits.dim() == 1, f"logits must be 1D, got {logits.dim()}D"
    assert values.dim() == 2, f"values must be 2D, got {values.dim()}D"
    assert logits.shape[0] == values.shape[0], "logits/values length mismatch"

    if logit_bias is not None:
        assert logit_bias.shape == logits.shape, "logit_bias shape mismatch"
        effective_logits = logits + logit_bias
    else:
        effective_logits = logits

    # m = max(effective_logits)
    m = effective_logits.max()

    # exp(logits - m) — 数值稳定
    exp_shifted = torch.exp(effective_logits - m)

    # l = sum(exp)
    l = exp_shifted.sum()

    # o = sum(exp * V)
    o = (exp_shifted.unsqueeze(-1) * values).sum(dim=0)

    return OnlineSoftmaxState(m=m, l=l, o=o)


def merge_softmax_states(
    state_a: OnlineSoftmaxState,
    state_b: OnlineSoftmaxState,
) -> OnlineSoftmaxState:
    """归并算子 ⊕（满足交换律+结合律）。

    Reference: Milakov & Gimelshein 2018, eq.(25) in DCC-KV paper.

    Args:
        state_a, state_b: 两个块的 Online Softmax 状态

    Returns:
        合并后的状态
    """
    m_ab = torch.maximum(state_a.m, state_b.m)
    exp_a = torch.exp(state_a.m - m_ab)
    exp_b = torch.exp(state_b.m - m_ab)
    l_ab = exp_a * state_a.l + exp_b * state_b.l
    o_ab = exp_a * state_a.o + exp_b * state_b.o
    return OnlineSoftmaxState(m=m_ab, l=l_ab, o=o_ab)


def merge_softmax_states_list(states: List[OnlineSoftmaxState]) -> OnlineSoftmaxState:
    """归并多个状态（按列表顺序）。"""
    if len(states) == 0:
        raise ValueError("Empty state list")
    if len(states) == 1:
        return states[0]
    result = states[0]
    for s in states[1:]:
        result = merge_softmax_states(result, s)
    return result


def verify_order_invariance(
    states: List[OnlineSoftmaxState],
    n_trials: int = 10,
    seed: int = 42,
) -> bool:
    """验证归并的顺序无关性（commutativity + associativity）。

    单元测试用：随机打乱 states 多次，比较归并结果是否一致。

    Args:
        states: 状态列表
        n_trials: 随机打乱次数
        seed: 随机种子

    Returns:
        True 如果所有打乱顺序归并结果一致
    """
    if len(states) < 2:
        return True

    g = torch.Generator().manual_seed(seed)
    reference = merge_softmax_states_list(states)
    ref_o = reference.o
    ref_l = reference.l

    for _ in range(n_trials):
        perm = torch.randperm(len(states), generator=g).tolist()
        permuted = [states[i] for i in perm]
        result = merge_softmax_states_list(permuted)
        # 比较 o/l 比例（attention output）
        ref_output = ref_o / ref_l
        result_output = result.o / result.l
        if not torch.allclose(ref_output, result_output, atol=1e-5, rtol=1e-5):
            return False

    return True


def attention_output_from_state(state: OnlineSoftmaxState) -> torch.Tensor:
    """从状态提取 attention output (o / l)。"""
    return state.o / state.l
