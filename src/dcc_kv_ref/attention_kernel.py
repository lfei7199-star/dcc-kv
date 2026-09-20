"""G1：把 CompactKV 送进注意力的算子核（device-agnostic）。

这是补齐 GPU 侧缺口的**第一块**：一条能把紧凑 K/β/V 喂进 SDPA 的通路。
在此之前的仓库里，"紧凑 KV 算 attention"只以逐 query 的 Python 循环形式存在于
`src/distributed/dcc_kv_sync_cpu.py`（CPU、单 head、面向数值等价性验证），
既不能批量化、也无法在 GPU 上跑。

设计约束
--------

**1. 本模块 device-agnostic。** 不出现任何 `cuda` 字面量，全部由输入张量的
`device` / `dtype` 决定。理由不是"优雅"，而是**可验证性**：本机没有 CUDA，
只有 device-agnostic 的实现才能在本地对拍，才谈得上"写完即验证"。
GPU 上它是同一段代码（SDPA 自动选 flash/efficient/mem-efficient 核）。

**2. β 走加性注意力掩码，不折进 head_dim。**
一个看似更省的技巧是把 β 折进内积：令 k' = [k, 1]、q' = [q, β]，则
q'·k' = q·k + β，省掉整张 mask。**这里刻意不采用**：它把 head_dim 从 d_h
变成 d_h+1（如 128→129），而 flash-attention 类核对 head_dim 有对齐要求，
129 会掉出向量化路径 —— 为了省一张 mask 而毁掉计时实验要测的那个算子，
是拿测量口径换显存。故一律用加性 mask。

**3. 加性 mask 的形状有硬约束（实测）。**
`torch.nn.functional.scaled_dot_product_attention` 的 `attn_mask`
**维数既不能少于 2，也不能多于 query 的维数**。两条边界都是实测踩出来的
（torch 2.11，CPU）：

    q.ndim = 2, mask [S]       →  IndexError: Dimension out of range
                                  (expected to be in range of [-1, 0], but got -2)
    q.ndim = 4, mask [1,1,1,S] →  ok
    q.ndim = 2, mask [1,1,1,S] →  RuntimeError: output with shape [Lq,d_v]
                                  doesn't match the broadcast shape [1,1,Lq,d_v]

第二条尤其阴险：mask 的**前导 1 维会让输出张量也多出这些维**，于是报错信息
指向"输出形状不匹配"，而真正的原因是 mask 比 query 多了一维 —— 排查时容易
往 query 上找。**唯一在四种维数下都成立的做法是让 mask 与 query 同维**：
形状取 `(*[1]*(q.ndim-2), Lq_or_1, B)`。本模块照此生成。

**4. 因果掩码按"原始位置"而不是"压缩块的行序"。**
紧凑块的行序由 RMS 选键决定，与位置序无关（仓库里这个坑在
`fast_kv_cpu` / `apb_cpu` 各犯过一次，定位见
`experiments/cpu/c10_baseline_diagnosis.py`）。故本模块的因果可见性一律
由 `CompactKV.selected_indices` 计算，见 `causal_visibility`。

**5. 归并需要 lse，而 lse 与融合核不可兼得。**
跨块归并的权重是 `exp(lse_i - max_j lse_j)`，因此归并路径必须拿到每块的
log-sum-exp。而 `torch` 的 SDPA **不返回 lse**。于是：

    return_lse=False  →  走 SDPA 融合核，只出输出，拿不到 lse
    return_lse=True   →  走显式 logits 路径（多一次 QK^T 的 exp/sum 归约），出 (out, lse)

**这两条路径的 T_comp 不可混比**：前者是融合核口径，后者是未融合口径。
计时实验若只关心单块的 T_comp，用前者；只要涉及多块归并（真实语义），
就必须用后者，并在报告里声明口径。这不是实现瑕疵，而是"要测的量"决定的选择。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .compact_kv import CompactKV


def default_scale(head_dim: int) -> float:
    """1/sqrt(d_h)。与 `compact_kv.build_compact_kv` 内部用的缩放一致。"""
    return 1.0 / (head_dim ** 0.5)


@dataclass
class PartialAttention:
    """一个源块贡献的部分注意力结果。

    out: [..., Lq, d_v]        该块贡献的**已归一化**输出
                                 （即 softmax 只在该块内部做过）
    lse: [..., Lq]             该块的 log-sum-exp；归并权重由它导出
    lse_available: bool        False 表示走了融合核，lse 无意义（不可归并）
    n_keys: int                该块的 key 数（= budget）
    """
    out: torch.Tensor
    lse: Optional[torch.Tensor]
    lse_available: bool
    n_keys: int

    def is_finite(self) -> torch.Tensor:
        """每个 query 位置是否至少看见一个 key（[..., Lq] bool）。

        lse 为 -inf 表示该 query 在**这个块里**没有任何可见 key
        （因果掩码把整块挡掉了，或块为空）。归并时必须把它们剔除，
        否则 `-inf - (-inf)` 会产出 nan 并把整行污染。
        """
        if self.lse is None:
            raise ValueError("该 PartialAttention 走的是融合核路径，没有 lse，无法判断可见性")
        return torch.isfinite(self.lse)


def causal_visibility(
    selected_indices: torch.Tensor,   # [B] 该块被选中 key 在**原块内**的索引（CompactKV.selected_indices）
    query_positions: torch.Tensor,    # [Lq] 目的端 query 的**全局**位置
    block_offset: int,                # 该源块首 key 的全局位置
) -> torch.Tensor:
    """因果可见性矩阵 [Lq, B]，True = 可见。

    判据：key 的原块内索引 + 块偏移 < query 全局位置 + 1。
    注意这里用的是 `selected_indices`（位置序），**不是**紧凑块的行序 ——
    行序是按 RMS 分数排的，把它当位置前缀用会静默改变数值。
    """
    si = selected_indices.reshape(1, -1)
    qp = query_positions.reshape(-1, 1)
    return si < (qp + 1 - int(block_offset))


def identity_compact(keys: torch.Tensor, values: torch.Tensor) -> CompactKV:
    """构造"无压缩"的 CompactKV：β ≡ 0、索引取恒等、V 原样。

    用途有两个，都不是玩具：

    1. **B = L_s 的精确退化的可检验形式。** 论文性质 1 说预算等于块长时
       应无残差 —— 但那是"存在零残差解"，而 `build_compact_kv` 带
       ridge 正则（λ_β、λ_v > 0），其解**不是**恒等，故不能用它检验性质 1。
       本函数把"解"直接给定为恒等，从而把**算子核**与**构造链**的误差分离开：
       用本函数喂核，输出必须等于 dense。
    2. 钩子里的 `dense` 参照臂（见 `src/distributed/attention_hook.py`）。
    """
    B, d_h = keys.shape
    return CompactKV(
        keys=keys,
        logit_bias=torch.zeros(B, dtype=keys.dtype, device=keys.device),
        values=values,
        selected_indices=torch.arange(B, device=keys.device, dtype=torch.long),
    )


def _check_shapes(query: torch.Tensor, compact: CompactKV) -> int:
    if query.dim() < 2:
        raise ValueError(f"query 至少 2 维 [..., Lq, d_h]，得到 {tuple(query.shape)}")
    d_h = int(query.shape[-1])
    if int(compact.keys.shape[-1]) != d_h:
        raise ValueError(
            f"query 的 d_h={d_h} 与 compact.keys 的 {int(compact.keys.shape[-1])} 不一致；"
            "紧凑块与本层 head 维度必须匹配"
        )
    if compact.keys.shape[0] == 0:
        raise ValueError("compact 的 budget 为 0：空块不应进入核，调用方应先过滤")
    return d_h


def _build_additive_mask(
    logit_bias: torch.Tensor,          # [B]
    visible: Optional[torch.Tensor],   # [Lq, B] bool 或 None
    dtype: torch.dtype,
    q_ndim: int,
    lq: int,
) -> torch.Tensor:
    """加性 mask，形状 [1,1,1,B] 或 [1,1,Lq,B]（见模块文档 3）。

    不可见位置用 -inf。**不能用大负数**（如 -1e30）：在 float16 下
    -1e30 会溢出成 -inf，看起来一样，但在 float32 的某些核里它只是
    一个很负的数，softmax 后是极小的非零权重，会以 1e-30 级别的量污染
    输出，且这种污染随块数累积。用真正的 -inf。
    """
    B = int(logit_bias.shape[0])
    # 掩码必须与 query **同维**（见模块文档第 3 条）
    lead = (1,) * max(0, int(q_ndim) - 2)
    base = logit_bias.to(dtype=dtype).reshape(*lead, 1, B)
    if visible is None:
        return base
    if int(visible.shape[0]) != int(lq):
        raise ValueError(
            f"visible 的行数 {int(visible.shape[0])} 与 Lq={int(lq)} 不符；"
            "可见性掩码必须逐 query 给出"
        )
    vis = visible.reshape(*lead, int(lq), B)
    return base.expand(*lead, int(lq), B).clone().masked_fill(~vis, float("-inf"))


def compact_kv_attention(
    query: torch.Tensor,                 # [..., Lq, d_h]
    compact: CompactKV,
    *,
    visible: Optional[torch.Tensor] = None,   # [Lq, B] bool；None = 全可见
    scale: Optional[float] = None,
    return_lse: bool = False,
    query_chunk: Optional[int] = None,
) -> PartialAttention:
    """用紧凑 KV 算注意力。

    Args:
        query:       目的端 query，[..., Lq, d_h]（支持任意前导维，如 [H, Lq, d_h]）
        compact:     单个源块的紧凑表示（K/β/V/索引）
        visible:     因果或其它可见性掩码 [Lq, B]，按 `selected_indices` 的位置序构造
        scale:       缩放；默认 1/sqrt(d_h)
        return_lse:  True 走显式 logits 路径并返回 lse（可归并）；False 走 SDPA 融合核
        query_chunk: 沿 query 维分块，限制中间张量 [Lq, B] 的峰值显存

    分块改变的是"怎么算"，不是"算的是什么"（但两者不重合，见下）
    ----------------------------------------------------------
    `query_chunk` 沿 **query 维**切，而归约（softmax 的归一化）沿 **key 维**做。
    对任意输出元素 `out[q, :]`，它参与的 key 集合与求和顺序**只由 q 决定**，
    与 q 落在哪个分块无关 ⇒ 数学上分组不改变每个元素的值。反之若沿 key 维切，
    就真的会改变数值（求和集合被切开）。

    **但这不等于逐位相同 —— 这条最初写错了，是实测纠正的。**
    本模块原先把 `_hf._pooled_key_energy` 的结论（沿 S 维分块、沿 (B,H,d) 归约
    ⇒ 逐位相同）直接搬到这里，理由是"两维正交"。该推理在本函数上**不成立**：
    在 `_pooled_key_energy` 里分块维 S 只是逐元素张量的一个轴，每个输出元素
    由同一个归约算完；而在注意力里 query 维是 GEMM 的 **M 维**，BLAS 会按 M
    选择分块核与累加策略 ⇒ 同一元素的浮点累加顺序随 M 改变。

    实测（float64，Lq=13，chunk ∈ {1,2,3,4,7}，`return_lse` 两种路径）：

        与不分块的最大偏差 ≤ 6.7e-16（约 1–3 ULP），**无一处逐位相同**

    因此本模块只断言「ULP 量级内等价」，并把分块定位为**显存控制手段**而非
    数值等价手段。需要逐位可复现的场合不要用 `query_chunk`。

    锚点见 `tests/test_attention_kernel.py::test_query_chunk_ulP_equivalent`。
    """
    d_h = _check_shapes(query, compact)
    sc = default_scale(d_h) if scale is None else float(scale)

    if query_chunk is not None and int(query_chunk) > 0 and int(query.shape[-2]) > int(query_chunk):
        outs: List[torch.Tensor] = []
        lses: List[torch.Tensor] = []
        step = int(query_chunk)
        for lo in range(0, int(query.shape[-2]), step):
            hi = min(int(query.shape[-2]), lo + step)
            vis = None if visible is None else visible[lo:hi]
            part = compact_kv_attention(
                query[..., lo:hi, :], compact, visible=vis, scale=sc,
                return_lse=return_lse, query_chunk=None,
            )
            outs.append(part.out)
            if part.lse is not None:
                lses.append(part.lse)
        out = torch.cat(outs, dim=-2)
        lse = torch.cat(lses, dim=-1) if lses else None
        return PartialAttention(out=out, lse=lse, lse_available=return_lse,
                                n_keys=int(compact.keys.shape[0]))

    keys = compact.keys
    values = compact.values
    bias = compact.logit_bias

    if return_lse:
        # 显式 logits 路径：QK^T → +β → 掩码 → logsumexp → 加权求和。
        # 成本与融合核相当（同样的两趟矩阵乘），但多一次 [.., Lq, B] 的 exp/sum。
        logits = torch.matmul(query, keys.transpose(-1, -2)) * sc      # [..., Lq, B]
        logits = logits + bias.to(dtype=logits.dtype).reshape(
            *([1] * (logits.dim() - 1)), int(keys.shape[0])
        )
        if visible is not None:
            # 形状必须是 [1,...,1,Lq,B]，总维数 = logits.dim()。写成
            # (1, *([1]*(D-2)), Lq, B) 会多出一维（D+1 维），而 PyTorch 对
            # 多余的**前导** 1 维是容忍的 —— 于是掩码广播到错误的轴上也照样
            # 跑完，静默给出错的结果。这里显式断言，堵住这条静默路径。
            vis = visible.reshape(*([1] * (logits.dim() - 2)),
                                  visible.shape[0], int(keys.shape[0]))
            assert vis.dim() == logits.dim(), (
                f"可见性掩码维数 {vis.dim()} 与 logits 维数 {logits.dim()} 不符"
            )
            logits = logits.masked_fill(~vis, float("-inf"))
        lse = torch.logsumexp(logits, dim=-1)                          # [..., Lq]
        # 全 -inf 行：lse = -inf，exp(-inf - (-inf)) = nan ⇒ 显式置零
        safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
        p = torch.exp(logits - safe.unsqueeze(-1))
        p = torch.where(torch.isfinite(logits), p, torch.zeros_like(p))
        out = torch.matmul(p.to(values.dtype), values)
        return PartialAttention(out=out, lse=lse, lse_available=True,
                                n_keys=int(keys.shape[0]))

    mask = _build_additive_mask(bias, visible, query.dtype,
                                q_ndim=query.dim(), lq=int(query.shape[-2]))
    out = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask, scale=sc)
    return PartialAttention(out=out, lse=None, lse_available=False,
                            n_keys=int(keys.shape[0]))


def dense_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    query_positions: Optional[torch.Tensor] = None,
    block_offset: int = 0,
    causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = True,
) -> PartialAttention:
    """精确注意力（不压缩）—— 独立参考实现。

    刻意**不走 SDPA**，而是把 matmul + softmax + matmul 三步写开：
    这样它与 `compact_kv_attention` 的融合核是两条完全不同的计算路径，
    二者的吻合才是交叉验证，而不是"同一个核调用两次"。

    因果掩码按全局位置：key 的全局位置 = block_offset + i。
    """
    d_h = int(query.shape[-1])
    sc = default_scale(d_h) if scale is None else float(scale)
    logits = torch.matmul(query, keys.transpose(-1, -2)) * sc
    if causal:
        if query_positions is None:
            raise ValueError("causal=True 时必须给出 query_positions")
        pos_k = torch.arange(int(keys.shape[0]), device=keys.device) + int(block_offset)
        logits = logits.masked_fill(
            pos_k.reshape(1, -1) > query_positions.reshape(-1, 1), float("-inf")
        )
    lse = torch.logsumexp(logits, dim=-1)
    safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    p = torch.exp(logits - safe.unsqueeze(-1))
    p = torch.where(torch.isfinite(logits), p, torch.zeros_like(p))
    out = torch.matmul(p.to(values.dtype), values)
    return PartialAttention(out=out, lse=lse, lse_available=True, n_keys=int(keys.shape[0]))


def merge_partial_attention(
    partials: Sequence[PartialAttention],
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    """把多个源块的部分结果归并成最终输出。

    这是 flash-attention 的 combine 步，也是论文式(25)归并算子的向量化形式：
    每个块只在自己内部做过归一化，跨块的权重由各块的 lse 决定：

        LSE = logsumexp_i lse_i
        w_i = exp(lse_i - LSE)
        out = Σ_i w_i · out_i            （Σ_i w_i = 1）

    为什么不用 `exp(lse_i)` 直接加权：那会在 lse 量级偏大时溢出。这里先取
    最大值再减，是标准做法。

    剔除规则：lse 非有限的块（在该块内无可见 key）权重为 0，直接跳过。
    若**全部**块都无可见 key，返回全零（而不是 nan）—— 这与"没有任何注意力
    可用"的语义一致，且能让上层用 `n_visible == 0` 判定而不是在 nan 上打补丁。
    """
    if len(partials) == 0:
        raise ValueError("merge_partial_attention 需要至少一个 PartialAttention")
    for p in partials:
        if not p.lse_available or p.lse is None:
            raise ValueError(
                "归并需要每块的 lse，但存在走融合核（return_lse=False）的块。"
                "跨块归并必须用 return_lse=True 重新计算 —— 见模块文档第 5 条。"
            )

    shapes = {tuple(p.out.shape) for p in partials}
    if len(shapes) != 1:
        raise ValueError(
            f"归并要求各块的 out 形状一致，得到 {sorted(shapes)}；"
            "d_v 不同的块不能直接归并（与 merge_softmax_states 的约束同源）。"
        )
    lses = torch.stack([p.lse for p in partials], dim=0)          # [N, ..., Lq]
    outs = torch.stack([p.out for p in partials], dim=0)          # [N, ..., Lq, d_v]
    finite = torch.isfinite(lses)
    if not bool(finite.any()):
        return torch.zeros_like(outs[0])

    neg_inf = torch.full_like(lses, float("-inf"))
    lses_safe = torch.where(finite, lses, neg_inf)
    lse_total = torch.logsumexp(lses_safe, dim=0)                 # [..., Lq]
    # 全 -inf 的 query 位置：logsumexp 给 -inf，减它会得到 nan，显式兜底
    all_invalid = ~torch.isfinite(lse_total)
    denom = torch.where(all_invalid, torch.ones_like(lse_total), lse_total)

    w = torch.exp(lses_safe - denom.unsqueeze(0))                 # [N, ..., Lq]
    w = torch.where(finite, w, torch.zeros_like(w))
    if eps > 0:
        w = torch.where(w < eps, torch.zeros_like(w), w)
    w = w / w.sum(dim=0, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)

    merged = (w.unsqueeze(-1) * outs).sum(dim=0)                   # [..., Lq, d_v]
    return torch.where(all_invalid.unsqueeze(-1), torch.zeros_like(merged), merged)


def dcc_kv_attention(
    query: torch.Tensor,                              # [..., Lq, d_h]
    compact_blocks: Sequence[CompactKV],              # 远端块的紧凑表示
    *,
    local_keys: Optional[torch.Tensor] = None,        # 本地块原始 K
    local_values: Optional[torch.Tensor] = None,      # 本地块原始 V
    query_positions: Optional[torch.Tensor] = None,   # [Lq] 全局位置（因果用）
    remote_offsets: Optional[Sequence[int]] = None,   # 各远端块的全局起始位置
    local_offset: int = 0,
    scale: Optional[float] = None,
    query_chunk: Optional[int] = None,
) -> torch.Tensor:
    """DCC-KV 的完整前向：本地精确块 + 若干远端紧凑块，归并成一个输出。

    这是 A3 / E6 的 `dcc_kv` 行、以及注意力替换钩子共用的入口。

    因果语义（**两条分支，别混说**）
    -------------------------------
    - `query_positions is None` ⇒ **全可见**：不施加任何因果掩码，
      适用于 prefill 中源块整体位于目的端 query 之前的情形。
    - `query_positions` 给定 ⇒ 按"key 全局位置 ≤ query 全局位置"掩码，
      此时**各远端块的全局起始位置必须由 `remote_offsets` 给出**。

    ⚠️ 自查修正（2026-09-18）：本函数的文档原先写「`remote_offsets` 缺省视为
    全可见」，而实现是「缺省把偏移当 0，**并施加因果掩码**」—— 两者不是一回事，
    且后者在块数 > 1 时是**静默错**：实测（12 个位置、两个远端块）与正确解的
    最大绝对差 1.03，而形状与量级都正常，从输出上看不出问题。
    现在改为显式拒绝：块数 > 1 且未给 `remote_offsets` ⇒ `ValueError`；
    块数 == 1 时缺省按偏移 0 处理（等价于"该块起始于位置 0"）。
    """
    if not compact_blocks and local_keys is None:
        raise ValueError("既无远端块也无本地块，无法计算注意力")
    if remote_offsets is not None and len(remote_offsets) != len(compact_blocks):
        raise ValueError(
            f"remote_offsets 长度 {len(remote_offsets)} != 远端块数 "
            f"{len(compact_blocks)}；长度不足会 IndexError，长度多余会被静默忽略 ——"
            "两种都是在没有共同约定时错配因果掩码（块与偏移必须一一对应）"
        )
    if (query_positions is not None and remote_offsets is None
            and len(compact_blocks) > 1):
        raise ValueError(
            f"给了 query_positions 但未给 remote_offsets，且有 "
            f"{len(compact_blocks)} 个远端块：此时无法确定各块的全局起始位置。"
            "缺省按 0 处理会把所有块都当成起始于位置 0，得到形状与量级都正常、"
            "但数值错误的结果（实测与正确解差 1.03）。请显式传入各块的全局偏移；"
            "若确实想要全可见，请改为不传 query_positions。"
        )

    partials: List[PartialAttention] = []

    if local_keys is not None:
        if local_values is None:
            raise ValueError("给出 local_keys 就必须同时给出 local_values")
        if query_positions is not None:
            part = dense_attention(
                query, local_keys, local_values,
                query_positions=query_positions, block_offset=local_offset,
                causal=True, scale=scale, return_lse=True,
            )
        else:
            part = dense_attention(query, local_keys, local_values,
                                   causal=False, scale=scale, return_lse=True)
        partials.append(part)

    Lq = int(query.shape[-2])
    for i, ck in enumerate(compact_blocks):
        vis = None
        if query_positions is not None:
            off = int(remote_offsets[i]) if remote_offsets is not None else 0
            vis = causal_visibility(ck.selected_indices, query_positions, off)
            if not bool(vis.any()):
                continue
        partials.append(compact_kv_attention(
            query, ck, visible=vis, scale=scale, return_lse=True,
            query_chunk=query_chunk,
        ))

    return merge_partial_attention(partials)


__all__ = [
    "PartialAttention",
    "default_scale",
    "causal_visibility",
    "identity_compact",
    "compact_kv_attention",
    "dense_attention",
    "merge_partial_attention",
    "dcc_kv_attention",
]
