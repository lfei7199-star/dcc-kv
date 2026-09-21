"""G1 紧凑 KV 注意力算子核的锚点（纯 CPU，不需要 GPU）。

这个文件把三件"看起来对但实测不对"的事情冻成断言，它们是本机（无 CUDA）能在
缺口补齐后**真正验证**的部分：

1. **SDPA 加性掩码的形状规则**（`q.ndim` 与 `mask.ndim` 的配对）
   两条边界都是实测踩出来的：mask 少于 2 维抛 `IndexError`，多于 query 维数
   抛的是**形状不匹配**的 `RuntimeError`（症状指向输出形状，真因在掩码维数，
   排查时容易往 query 上找）。本模块统一生成与 query 同维的掩码。

2. **归并 = 拼接后一次算完**
   跨块归并（flash combine）必须与"把所有 key 拼起来一次算"数值一致。
   这是 DCC-KV 多块语义的正确性根基：**若这条不成立，A3/A4/E6 的 dcc_kv 行
   全是错的，而且错得很安静**（形状、量级都正常）。

3. **分块不是逐位等价，只是 ULP 级等价**
   本模块一度把 `_pooled_key_energy` 的"两维正交 ⇒ 逐位相同"直接搬过来，
   实测推翻：注意力里 query 维是 GEMM 的 M 维，BLAS 按 M 选分块核。
   此文件按实测值立锚，防止后人再次把"数学等价"读成"逐位相同"。
"""
from __future__ import annotations

import pytest
import torch

from src.dcc_kv_ref import (
    compact_kv_attention,
    causal_visibility,
    dcc_kv_attention,
    dense_attention,
    identity_compact,
    merge_partial_attention,
)
from src.dcc_kv_ref.attention_kernel import _build_additive_mask, default_scale

# 实测容差：float64 下各路径互差 ≤ 7e-16（1–3 ULP），float32 下 ≈ 1.8e-07。
# 不用 rtol：这里比较的是同一量纲的输出，绝对容差更严格也更好读。
ATOL_F64 = 1e-12
ATOL_F32 = 1e-5


def _blocks(dtype=torch.float64, seed=0, n_blocks=4, block=8, d_h=16, d_v=8, lq=13):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(n_blocks * block, d_h, generator=g, dtype=dtype)
    V = torch.randn(n_blocks * block, d_v, generator=g, dtype=dtype)
    Q = torch.randn(lq, d_h, generator=g, dtype=dtype)
    return K, V, Q, n_blocks, block, d_h


# =============================================================================
# 1. 掩码形状规则
# =============================================================================

def test_mask_is_generated_with_query_ndim():
    """掩码维数必须等于 query 维数 —— 少了抛 IndexError，多了抛形状错误。"""
    bias = torch.zeros(5, dtype=torch.float64)
    for q_ndim, lead in ((2, ()), (3, (1,)), (4, (1, 1))):
        m = _build_additive_mask(bias, None, torch.float64, q_ndim=q_ndim, lq=7)
        assert m.dim() == q_ndim, f"q_ndim={q_ndim} 但掩码维数 {m.dim()}"
        assert tuple(m.shape) == (*lead, 1, 5)


def test_kernel_accepts_2d_3d_4d_queries():
    """四种 query 维数下都要能跑，且结果在各维数间一致（此处测 2/3/4 维）。"""
    K, V, Q, _, _, _ = _blocks()
    ck = identity_compact(K, V)
    ref = compact_kv_attention(Q, ck, return_lse=False).out

    q3 = Q.unsqueeze(0).expand(2, -1, -1).contiguous()
    q4 = Q.reshape(1, 1, *Q.shape).expand(1, 3, -1, -1).contiguous()
    for q, tag in ((q3, "3-D"), (q4, "4-D")):
        out = compact_kv_attention(q, ck, return_lse=False).out
        assert out.shape[-2:] == ref.shape
        # 按元素数之比算重复次数：out 与 ref 的 d_v 相同，故只能用 numel；
        # 用 out.shape[0] 会把批量维当行数（3-D 时得到 2 // 13 = 0）。
        reps = int(out.numel()) // int(ref.numel())
        assert reps >= 1
        assert torch.allclose(out.reshape(-1, ref.shape[-1]),
                              ref.repeat(reps, 1),
                              atol=ATOL_F64, rtol=0), f"{tag} 与 2-D 结果不一致"


def test_bias_actually_enters_the_logits():
    """β 必须真的进 logits：加一个大负偏置必须改变输出，且方向可验证。

    构造方式：单块 = 两个 key，让某一个 key 的权重从 0.5 推到接近 0/1。
    """
    K = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    V = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    Q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    bias0 = torch.zeros(2, dtype=torch.float64)
    bias1 = torch.tensor([10.0, -10.0], dtype=torch.float64)
    ck0 = identity_compact(K, V)
    from src.dcc_kv_ref import CompactKV
    ck1 = CompactKV(keys=K, logit_bias=bias1, values=V,
                    selected_indices=torch.arange(2))

    o0 = compact_kv_attention(Q, ck0, return_lse=False).out
    o1 = compact_kv_attention(Q, ck1, return_lse=False).out
    assert abs(float(o0[0, 0]) - 0.5) < 1e-12, "无偏置时两 key 权重应各半"
    assert float(o1[0, 0]) > 0.99, "正偏置应把权重推向对应 V"
    assert ck0.logit_bias.abs().sum().item() == 0.0
    assert default_scale(4) == pytest.approx(0.5)


# =============================================================================
# 2. 无压缩退化（论文性质 1 的算子侧形式）
# =============================================================================

@pytest.mark.parametrize("dtype,atol", [(torch.float64, ATOL_F64), (torch.float32, ATOL_F32)])
def test_identity_compact_equals_dense(dtype, atol):
    """B = L_s 且解为恒等时，紧凑路径必须等于精确注意力。

    刻意用 `identity_compact` 而不是 `build_compact_kv(budget=L_s)`：
    后者带 ridge 正则，解**不是**恒等（β≠0、V 被收缩），把它与 dense 比较
    测的是"构造链的正则强度"，不是算子核。此处的分离是有意的。
    """
    K, V, Q, _, _, _ = _blocks(dtype=dtype)
    ck = identity_compact(K, V)
    fused = compact_kv_attention(Q, ck, return_lse=False).out
    explicit = compact_kv_attention(Q, ck, return_lse=True).out
    ref = dense_attention(Q, K, V).out
    assert torch.allclose(fused, ref, atol=atol, rtol=0)
    assert torch.allclose(explicit, ref, atol=atol, rtol=0)


def test_two_paths_disagree_at_most_one_ulp_in_float64():
    """融合核路径与显式路径是两条独立实现，其吻合才算交叉验证。"""
    K, V, Q, _, _, _ = _blocks()
    ck = identity_compact(K, V)
    a = compact_kv_attention(Q, ck, return_lse=False).out
    b = compact_kv_attention(Q, ck, return_lse=True).out
    assert torch.allclose(a, b, atol=ATOL_F64, rtol=0)


# =============================================================================
# 3. 因果可见性按位置序，不按行序
# =============================================================================

def test_causal_visibility_uses_position_not_row_order():
    """把 `fast_kv_cpu` / `apb_cpu` 各犯过一次的历史 bug 冻成断言。

    compact 的行序由选键分数决定，与位置序无关。若误把"行序前缀"当"位置前缀"
    使用，掩码会算错，而形状与量级都正常 —— 静默错误。
    """
    si = torch.tensor([7, 0, 5, 2])          # 非单调：典型的 topk 输出形态
    qp = torch.tensor([10, 11, 12])
    vis = causal_visibility(si, qp, block_offset=10)

    # 期望：selected_indices[j] < qp - offset + 1
    # 阈值 = qp + 1 - offset：qp=10 → 1，qp=11 → 2，qp=12 → 3
    # si < 阈值 ⇒ qp=11 时 si=2 仍不可见（2 < 2 为假）。
    # 此处第一版手算把"0 与 2"错记成阈值 2 下的可见集，实现是对的。
    expect = torch.tensor([
        [False, True, False, False],   # 阈值 1：si < 1 ⇒ 只有 0
        [False, True, False, False],   # 阈值 2：si < 2 ⇒ 只有 0
        [False, True, False, True],    # 阈值 3：si < 3 ⇒ 0 与 2
    ])
    assert torch.equal(vis, expect)

    # 行序前缀（错误做法）在第 1 行给出 [True, False, False, False]，与正确解不同
    wrong = torch.zeros_like(vis)
    wrong[:, 0] = True
    assert not torch.equal(vis, wrong), "若两者相等，本测试就失去了意义"


def test_dcc_kv_attention_local_plus_remote_equals_dense_causal():
    """本地精确块 + 远端紧凑块（恒等解）在因果下必须等于整体 dense。"""
    K, V, Q, _, block, _ = _blocks(dtype=torch.float64)
    K1, K2 = K[:block], K[block:]
    V1, V2 = V[:block], V[block:]
    qpos = torch.arange(Q.shape[0], dtype=torch.long)

    out = dcc_kv_attention(
        Q, [identity_compact(K2, V2)],
        local_keys=K1, local_values=V1,
        query_positions=qpos, remote_offsets=[block], local_offset=0,
    )
    ref = dense_attention(Q, K, V, query_positions=qpos, causal=True).out
    assert torch.allclose(out, ref, atol=ATOL_F64, rtol=0)


# =============================================================================
# 4. 归并
# =============================================================================

def test_merge_equals_dense_over_concatenated_keys():
    """归并的核心锚点：分块算再归并 == 拼接后一次算。"""
    K, V, Q, n_blocks, block, _ = _blocks()
    parts = [
        compact_kv_attention(Q, identity_compact(K[i * block:(i + 1) * block],
                                                 V[i * block:(i + 1) * block]),
                             return_lse=True)
        for i in range(n_blocks)
    ]
    merged = merge_partial_attention(parts)
    ref = dense_attention(Q, K, V).out
    assert torch.allclose(merged, ref, atol=ATOL_F64, rtol=0)


def test_merge_drops_blocks_with_no_visible_key():
    """因果把整块挡掉时，该块的 lse = -inf，必须被剔除而不是产出 nan。"""
    K, V, Q, n_blocks, block, _ = _blocks()
    parts = [
        compact_kv_attention(Q, identity_compact(K[i * block:(i + 1) * block],
                                                 V[i * block:(i + 1) * block]),
                             return_lse=True)
        for i in range(n_blocks)
    ]
    # 人为把第 2 块整块标为"不可见"
    dead = parts[2]
    dead = type(dead)(out=dead.out, lse=torch.full_like(dead.lse, float("-inf")),
                      lse_available=True, n_keys=dead.n_keys)
    keep = [parts[0], dead, parts[1], dead, parts[3]]
    merged = merge_partial_attention(keep)
    ref = merge_partial_attention([parts[0], parts[1], parts[3]])
    assert torch.isfinite(merged).all()
    assert torch.allclose(merged, ref, atol=ATOL_F64, rtol=0)


def test_merge_all_invalid_returns_zeros_not_nan():
    """全部无可见 key 时返回全零 —— 让上层用计数判断，而不是在 nan 上打补丁。"""
    K, V, Q, _, _, _ = _blocks()
    p = compact_kv_attention(Q, identity_compact(K, V), return_lse=True)
    dead = type(p)(out=p.out, lse=torch.full_like(p.lse, float("-inf")),
                   lse_available=True, n_keys=p.n_keys)
    merged = merge_partial_attention([dead, dead])
    assert torch.equal(merged, torch.zeros_like(merged))


def test_merge_requires_lse():
    """走融合核的块没有 lse，参与归并必须显式报错（不能静默算错）。"""
    K, V, Q, _, _, _ = _blocks()
    fused = compact_kv_attention(Q, identity_compact(K, V), return_lse=False)
    with pytest.raises(ValueError, match="lse"):
        merge_partial_attention([fused])


def test_merge_rejects_shape_mismatch():
    K, V, Q, _, _, _ = _blocks()
    a = compact_kv_attention(Q, identity_compact(K, V), return_lse=True)
    b = compact_kv_attention(Q, identity_compact(K, V[:, :4]), return_lse=True)
    with pytest.raises(ValueError, match="out 形状"):
        merge_partial_attention([a, b])


def test_merge_is_order_invariant_at_ulp_level():
    """归并对块顺序不敏感 —— 这是异步接收顺序任意的前提。

    断言的是 ULP 级一致（实测 ≤ 2.3e-16），不是逐位：不同求和顺序必然
    改变浮点累加序，把"顺序无关"读成"逐位相同"是同一类错误的另一面。
    """
    K, V, Q, n_blocks, block, _ = _blocks()
    parts = [
        compact_kv_attention(Q, identity_compact(K[i * block:(i + 1) * block],
                                                 V[i * block:(i + 1) * block]),
                             return_lse=True)
        for i in range(n_blocks)
    ]
    ref = merge_partial_attention(parts)
    g = torch.Generator().manual_seed(7)
    for _ in range(20):
        perm = torch.randperm(n_blocks, generator=g).tolist()
        got = merge_partial_attention([parts[i] for i in perm])
        assert torch.allclose(got, ref, atol=ATOL_F64, rtol=0)


# =============================================================================
# 5. 分块：显存控制手段，不是数值等价手段
# =============================================================================

def test_query_chunk_ulp_equivalent():
    """分块必须落在 ULP 量级内，但**不承诺**逐位相同（实测）。

    教训：本模块曾照搬 `_pooled_key_energy` 的"两维正交 ⇒ 逐位相同"。
    在注意力里，query 维是 GEMM 的 M 维，BLAS 按 M 选分块核与累加策略，
    因此同一输出元素的浮点累加顺序会随 M 改变。数学等价 ≠ 逐位等价。
    """
    K, V, Q, _, _, _ = _blocks()
    ck = identity_compact(K, V)
    for return_lse in (False, True):
        full = compact_kv_attention(Q, ck, return_lse=return_lse).out
        for chunk in (1, 2, 3, 4, 7):
            got = compact_kv_attention(Q, ck, return_lse=return_lse,
                                       query_chunk=chunk).out
            assert torch.allclose(got, full, atol=1e-14, rtol=0), (
                f"query_chunk={chunk} 超出 ULP 量级"
            )
            assert got.shape == full.shape


# =============================================================================
# 6. 入参自洽性：错误必须显式失败
# =============================================================================

def test_head_dim_mismatch_raises():
    K, V, Q, _, _, _ = _blocks()
    K_bad = K[:, :8]
    with pytest.raises(ValueError, match="d_h"):
        compact_kv_attention(Q, identity_compact(K_bad, V))


def test_zero_budget_raises():
    K, V, Q, _, _, _ = _blocks()
    from src.dcc_kv_ref import CompactKV
    empty = CompactKV(keys=K[:0], logit_bias=torch.zeros(0, dtype=K.dtype),
                      values=V[:0], selected_indices=torch.zeros(0, dtype=torch.long))
    with pytest.raises(ValueError, match="budget 为 0"):
        compact_kv_attention(Q, empty)


def test_empty_query_stack_raises():
    K, V, Q, _, _, _ = _blocks()
    with pytest.raises(ValueError, match="至少 2 维"):
        compact_kv_attention(Q[0], identity_compact(K, V))


def test_visible_row_count_mismatch_raises():
    K, V, Q, _, _, _ = _blocks()
    bad = torch.ones(Q.shape[0] + 1, K.shape[0], dtype=torch.bool)
    with pytest.raises(ValueError, match="visible"):
        compact_kv_attention(Q, identity_compact(K, V), visible=bad, return_lse=False)


def test_returns_partial_with_expected_fields():
    K, V, Q, _, _, _ = _blocks()
    ck = identity_compact(K, V)
    p = compact_kv_attention(Q, ck, return_lse=True)
    assert p.lse is not None and p.lse_available
    assert p.lse.shape == (Q.shape[0],)
    assert p.out.shape == (Q.shape[0], V.shape[-1])
    assert p.n_keys == K.shape[0]
    assert bool(p.is_finite().all())

def test_dense_causal_mask_supports_leading_dims():
    """`dense_attention` 的因果掩码必须支持带前导维的 query/keys。

    2026-09-21 修：旧实现写的是

        pos_k = torch.arange(int(keys.shape[0]), ...)
        logits.masked_fill(pos_k.reshape(1, -1) > query_positions.reshape(-1, 1), -inf)

    两处都把「最后一维是 key 维」的假设写成了「第 0 维」：

    * `keys.shape[0]` 在 keys 为 `[H, Lk, d_h]` 时取到 H —— 实测 H=2、Lk=24
      直接 `RuntimeError` 报尺寸 2 vs 24（症状指向 matmul/广播，真因在索引）；
    * 掩码按二维形状构造，前导维会被当作 query 维广播 —— 而 **H == Lk 时
      不报错、只算错**，比第一条更坏。

    它是在接 H0 的注意力钩子时被真实前向撞出来的：钩子按 kv head 分组后，
    交给 dense_attention 的 query 就是 `[g, Lq, d_h]`（g = H_q / H_kv > 1）。

    本锚点比的是 **ULP 级等价**，不是逐位 —— 分组改变了 GEMM 的 M 维。
    """
    torch.manual_seed(0)
    H, Lq, Lk, d = 3, 5, 7, 4
    assert H != Lk, "取 H 与取 Lk 若相等，这个缺陷会隐身，锚点就没判别力了"
    q = torch.randn(H, Lq, d, dtype=torch.float64)
    k = torch.randn(H, Lk, d, dtype=torch.float64)
    v = torch.randn(Lk, d, dtype=torch.float64)
    pos = torch.arange(Lq)

    got = dense_attention(q, k, v, causal=True, query_positions=pos).out
    exp = torch.stack([
        dense_attention(q[h], k[h], v, causal=True, query_positions=pos).out
        for h in range(H)
    ])
    assert got.shape == exp.shape == (H, Lq, d)
    # 只能 ULP 级一致，**不是**逐位：把 H 个 head 一次算与逐 head 算，GEMM 的
    # M 维不同（H*Lq 对 Lq），BLAS 据此选分块核 ⇒ 同一元素的浮点累加顺序会变。
    # 这与 query_chunk 是同一现象（那里沿 query 维分块），实测 float64 差
    # 1.11e-16 与 float32 差 1.19e-07，各约 1 ULP，**两档都不逐位相同**。
    dev = (got - exp).abs().max().item()
    assert dev < 1e-12, f"带前导维的因果掩码与逐 head 展开差 {dev:.3e}"

    # 逐 head 那一档必须真的施加了因果，否则上面是「两个都错」
    full = dense_attention(q[0], k[0], v, causal=False, query_positions=pos).out
    assert not torch.allclose(exp[0], full), "因果掩码没有起作用"

    with pytest.raises(ValueError, match="query_positions"):
        dense_attention(q, k, v, causal=True, query_positions=torch.arange(Lq - 1))
