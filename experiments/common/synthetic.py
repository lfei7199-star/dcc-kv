"""合成数据生成与误差度量 —— 所有 CPU 实验的共同基础。

为什么用合成数据
----------------
本仓库当前**没有** GPU 资源，"任务指标"（LongBench 准确率等）无法测量。
但第 5 章的误差分解（ε_mass / ε_out）与 E3 的边级条件化主张，
本质上是对**注意力分布重构精度**的陈述，不依赖真实模型。

因此这里构造一个可控的合成场景：
- 源块 K/V 完全随机；
- 目的端被组织成若干"关注带"（band）：目的端 r 的 Query 与源块中
  第 r 个位置带内的 Key 有高内积，从而**真实地**只关注该带。
- 于是"同一源块对不同目的端的紧凑 KV 应当不同"成为一个可检验命题，
  而不是一个同义反复：若压缩机制忽略目的端，KL 散度会接近 0。

诚实边界
--------
合成数据上的误差是**机制级**证据，不是**任务级**证据。
本模块产出的任何数字都不得用于主张任务质量（见 README「不主张的内容」）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from src.dcc_kv_ref import CompactKV, build_compact_kv


# =============================================================================
# 合成场景
# =============================================================================

@dataclass
class SyntheticScenario:
    """一个可控的"源块 + 多目的端"场景。

    Attributes:
        keys:           [L_s, d_h]  源块 Key（全网唯一，所有目的端共享）
        values:         [L_s, d_v]  源块 Value
        dest_queries:   {dest_id: [L_r, d_h]}  各目的端的 Query
        band_of_dest:   {dest_id: (start, end)} 该目的端关注的源位置区间
    """
    keys: torch.Tensor
    values: torch.Tensor
    dest_queries: Dict[int, torch.Tensor]
    band_of_dest: Dict[int, Tuple[int, int]]

    @property
    def L_s(self) -> int:
        return self.keys.shape[0]

    @property
    def d_h(self) -> int:
        return self.keys.shape[1]

    @property
    def d_v(self) -> int:
        return self.values.shape[1]

    @property
    def all_queries(self) -> torch.Tensor:
        """所有目的端 Query 的拼接，用作"共享压缩"基线（FastKV）的目的端。"""
        return torch.cat([self.dest_queries[k] for k in sorted(self.dest_queries)], dim=0)


def make_scenario(
    L_s: int = 256,
    d_h: int = 32,
    d_v: int = 32,
    num_dest: int = 4,
    queries_per_dest: int = 48,
    focus_strength: float = 8.0,
    focus_noise: float = 1.0,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> SyntheticScenario:
    """构造一个"多目的端各有关注带"的合成场景。

    构造方式：
        1. K/V 独立采样自 N(0, I)；
        2. 把源块等分为 num_dest 个位置带；
        3. 目的端 r 的 Query = focus_strength * unit(带 r 内 K 的均值) + focus_noise * N(0, I)。

    由于 attention logit 为 <q, k>/√d_h，目的端 r 的 Query 与带 r 内 Key 的内积
    系统性更大，因而其注意力质量集中在带 r。focus_strength 越大，各目的端的
    关注分布越分离 —— 这是控制 H1 可检验性的关键旋钮。

    Args:
        L_s: 源块长度
        d_h, d_v: head 维度
        num_dest: 目的端数量（也是关注带数量）
        queries_per_dest: 每个目的端的 Query 数 L_r
        focus_strength: 关注带的对齐强度
        focus_noise: 每个 Query 的随机扰动强度
        seed: 随机种子
        dtype: 张量精度。**默认 float32，因为 float64 目前在仓库代码中不可用** ——
            见 `compaction_dtype_support()`。论文 §6.3 的 E2 写的是 FP64，
            在本缺陷修复前无法按该规格运行。

    Returns:
        SyntheticScenario
    """
    g = torch.Generator().manual_seed(seed)

    keys = torch.randn(L_s, d_h, generator=g, dtype=dtype)
    values = torch.randn(L_s, d_v, generator=g, dtype=dtype)

    band_len = L_s // num_dest
    dest_queries: Dict[int, torch.Tensor] = {}
    band_of_dest: Dict[int, Tuple[int, int]] = {}

    for r in range(num_dest):
        start = r * band_len
        end = L_s if r == num_dest - 1 else (r + 1) * band_len
        band_of_dest[r] = (start, end)

        # 关注方向：带内 Key 的均值方向（归一化到单位长度）
        focus = keys[start:end].mean(dim=0)
        focus = focus / (focus.norm() + 1e-12)

        noise = torch.randn(queries_per_dest, d_h, generator=g, dtype=dtype)
        q = focus_strength * focus.unsqueeze(0) + focus_noise * noise
        dest_queries[r] = q

    return SyntheticScenario(
        keys=keys,
        values=values,
        dest_queries=dest_queries,
        band_of_dest=band_of_dest,
    )


# =============================================================================
# 环境能力探测
# =============================================================================

# 已知缺陷（2026-09-13 定位，尚未修复）：
#   src/dcc_kv_ref/representative_query.py::rademacher_projection 第 30 行写死
#   `proj = proj.float() / sqrt(projection_dim)`，不保留输入 dtype。
#   于是当 K/Q 为 float64 时，select_representative_queries 中的
#   `z = queries @ proj.T` 抛 RuntimeError: expected m1 and m2 to have the
#   same dtype, but got: double != float。
#
#   影响：build_compact_kv 整条链路**无法在 FP64 下运行**。
#   而论文 §6.3 的 E2 写的是"固定 L=256、d_h=d_v=32、FP64"，
#   §6.1 的 E0 也主张 FP64 下顺序无关性 —— 这两处规格在修复前无法复现。
#
#   修法（一行，需作者决定）：
#     proj = (torch.randint(...) * 2 - 1).to(queries.dtype) / (projection_dim ** 0.5)
DTYPE_BUG_FILE = "src/dcc_kv_ref/representative_query.py"
DTYPE_BUG_LINE = 30


def compaction_dtype_support(
    d_h: int = 32,
    d_v: int = 32,
    L_s: int = 32,
    budget: int = 8,
) -> Dict[str, bool]:
    """探测 `build_compact_kv` 在各精度下是否可用。

    在本缺陷修复前，预期返回 {"float32": True, "float64": False}。
    实验脚本应在启动时调用它，而不是等用户跑到一半收到 RuntimeError。

    Returns:
        {"float32": bool, "float64": bool}
    """
    result: Dict[str, bool] = {}
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        try:
            torch.manual_seed(0)
            K = torch.randn(L_s, d_h, dtype=dtype)
            V = torch.randn(L_s, d_v, dtype=dtype)
            Q = torch.randn(8, d_h, dtype=dtype)
            build_compact_kv(K, V, Q, budget=budget)
            result[name] = True
        except Exception:
            result[name] = False
    return result


def dtype_from_name(name: str) -> torch.dtype:
    """把 CLI 传入的精度名解析为 torch.dtype。"""
    table = {"float32": torch.float32, "float64": torch.float64,
             "fp32": torch.float32, "fp64": torch.float64}
    if name not in table:
        raise ValueError(f"不支持的精度: {name}（可选 {sorted(table)}）")
    return table[name]


# =============================================================================
# 参考实现：完整注意力
# =============================================================================

def dense_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """完整（无压缩）注意力输出：softmax(qK^T/√d) V。

    Args:
        query:  [d_h] 或 [N, d_h]
        keys:   [L_s, d_h]
        values: [L_s, d_v]

    Returns:
        [d_v] 或 [N, d_v]
    """
    scale = 1.0 / (keys.shape[-1] ** 0.5)
    logits = (query @ keys.T) * scale
    weights = torch.softmax(logits, dim=-1)
    return weights @ values


def dense_attention_weights(
    query: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """完整注意力的权重（用于与紧凑 KV 诱导的分布比较）。"""
    scale = 1.0 / (keys.shape[-1] ** 0.5)
    return torch.softmax((query @ keys.T) * scale, dim=-1)


# =============================================================================
# 紧凑 KV 的两个使用面：重建输出 与 诱导分布
# =============================================================================

# β 的三种用法 —— 用于定位训练/推理不一致
# -----------------------------------------------------------------------------
# 仓库现状（2026-09-13 核实）：
#   拟合侧 src/dcc_kv_ref/value_regression.py:72
#       bias_per_query = logit_bias.unsqueeze(0).expand(M, B) / M     ← 除以 M
#   推理侧 src/distributed/dcc_kv_sync_cpu.py:73,151
#          src/baselines/fast_kv_cpu.py:108
#       logits = ... + compact.logit_bias                              ← 全量 β
#
# 即 β 按 β/M 标定、却按 β 施加，相差 M 倍（默认 M=64）。
# 三种模式的定义：
#   "full"   ：β 原样加入 logit —— 与仓库所有推理路径一致（默认，保持可比性）
#   "over_m" ：β/M —— 与拟合时使用的标定一致
#   "none"   ：完全不加 β —— 对应论文 §6.4 的 A3 消融（移除质量偏置）
BETA_MODES = ("full", "over_m", "none")


def beta_applied(
    logit_bias: torch.Tensor,
    beta_mode: str,
    num_repr_queries: int = 1,
) -> torch.Tensor:
    """按指定模式把 β 变换为实际加到 logit 上的量。

    Args:
        logit_bias:      [B] 拟合得到的 β
        beta_mode:       "full" / "over_m" / "none"
        num_repr_queries: 拟合时使用的 M（"over_m" 模式需要）

    Returns:
        [B] 实际偏置（"none" 时返回全零）

    Raises:
        ValueError: beta_mode 非法
    """
    if beta_mode == "full":
        return logit_bias
    if beta_mode == "over_m":
        return logit_bias / max(int(num_repr_queries), 1)
    if beta_mode == "none":
        return torch.zeros_like(logit_bias)
    raise ValueError(f"未知 beta_mode: {beta_mode}（可选 {BETA_MODES}）")


def compact_attention(
    query: torch.Tensor,
    compact: CompactKV,
    beta_mode: str = "full",
    num_repr_queries: int = 1,
) -> torch.Tensor:
    """用紧凑 KV 计算注意力输出。

    β 的处理由 `beta_mode` 控制，默认 "full" 以与仓库现有推理路径
    （`dcc_kv_sync_cpu.py`、`fast_kv_cpu.py`）保持一致。这样做的目的是
    让实验首先反映"代码现在的实际行为"，而不是悄悄替它选一个更合理的约定。

    Args:
        query:   [d_h] 或 [N, d_h]
        compact: CompactKV（keys [B,d_h] / logit_bias [B] / values [B,d_v]）
        beta_mode: 见 `beta_applied`
        num_repr_queries: 拟合时的 M（"over_m" 模式需要）

    Returns:
        [d_v] 或 [N, d_v]
    """
    scale = 1.0 / (compact.keys.shape[-1] ** 0.5)
    bias = beta_applied(compact.logit_bias, beta_mode, num_repr_queries)
    logits = (query @ compact.keys.T) * scale + bias
    weights = torch.softmax(logits, dim=-1)
    return weights @ compact.values


def induced_distribution(
    compact: CompactKV,
    probe_queries: torch.Tensor,
    L_s: int,
    smooth: float = 1e-6,
    beta_mode: str = "full",
    num_repr_queries: int = 1,
) -> torch.Tensor:
    """把紧凑 KV 诱导的注意力分布"抬回"原始 token 空间 [L_s]。

    动机：两个目的端的紧凑 KV 选中的 Key 子集不同，其分布定义在**不同的支撑集**上，
    无法直接比较。做法是 scatter 回原始索引空间 —— 这样 P、Q 都落在同一个
    单纯形 Δ^{L_s} 上，KL 散度才有定义。

    Args:
        compact:      CompactKV
        probe_queries:[N, d_h] 探针 Query（应当对所有目的端用同一组，才可比）
        L_s:          源块原始长度
        smooth:       均匀混合系数 γ（P ← (1-γ)P + γ/L_s）。
                      必须 > 0：当两个分布支撑集不相交时 KL 会发散，
                      γ 给出一个有限上界（见 `kl_saturation_ceiling`）。
        beta_mode:    β 的用法，见 `beta_applied`
        num_repr_queries: 拟合时的 M

    Returns:
        [N, L_s] 每行是一个概率分布（和为 1）
    """
    scale = 1.0 / (compact.keys.shape[-1] ** 0.5)
    bias = beta_applied(compact.logit_bias, beta_mode, num_repr_queries)
    logits = (probe_queries @ compact.keys.T) * scale + bias
    p_compact = torch.softmax(logits, dim=-1)          # [N, B]

    N = probe_queries.shape[0]
    p_full = torch.zeros(N, L_s, dtype=p_compact.dtype, device=p_compact.device)
    idx = compact.selected_indices.to(p_full.device).long()
    p_full = p_full.index_add(1, idx, p_compact)        # [N, L_s]

    # 均匀平滑，保证被比较的两个分布处处有正密度
    p_full = (1.0 - smooth) * p_full + smooth / L_s
    return p_full / p_full.sum(dim=-1, keepdim=True)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """逐行 KL(P || Q)，返回 [N]。

    P、Q 均须是概率分布（行和为 1）。

    警告——**饱和**：若两个分布的支撑集几乎不相交（在 B ≪ L_s 的压缩场景下
    这是常态），KL 会被均匀平滑系数 γ 支配而饱和在 log(1/γ) 附近，
    失去区分"略有不同"与"完全不同"的能力。见 `kl_saturation_ceiling()`。
    因此本模块同时提供有界的 `jensen_shannon_divergence` 作为伴随度量。
    """
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def jensen_shannon_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """逐行 Jensen-Shannon 散度（以 nats 为单位，上界 ln2 ≈ 0.6931），返回 [N]。

    与 KL 相比的两个关键差别：
        1. **有界**：取值恒在 [0, ln2]，不会在支撑集不相交时发散或饱和；
        2. **对称**：JS(P||Q) = JS(Q||P)，无需再取两个方向的平均。

    因此在"两个紧凑 KV 差多少"这个问题上，JS 比 KL 更适合作为度量。
    KL 仍按论文 §6.3 的 H1 判据报告，JS 作为不会饱和的伴随证据。
    """
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def kl_saturation_ceiling(smooth: float = 1e-6, L_s: int = 256) -> float:
    """给定均匀平滑系数 γ，报告 KL 的饱和上界（量级）。

    当两个分布支撑集不相交时，P 在 Q 的支撑上只有 γ/L_s 的密度，
    故 KL ≈ log(P_mass / (γ/L_s)) ≈ log(1/γ) + log(L_s) - log(P_mass)。

    只要实测 KL 与这个量级相当，就说明度量已饱和，
    "KL > 阈值"这类判据在该设定下失去鉴别力。
    """
    import math

    return math.log(1.0 / smooth) + math.log(L_s)


# =============================================================================
# 误差度量（对应第 5 章 ε_mass 与 ε_out）
# =============================================================================

def mass_error(
    probe_queries: torch.Tensor,
    compact: CompactKV,
    keys: torch.Tensor,
    beta_mode: str = "full",
    num_repr_queries: int = 1,
) -> torch.Tensor:
    """ε_mass：紧凑块保留的 softmax 质量相对完整块的比例误差。

    定义（与 §5.3 一致）：
        ε_mass = |Σ_j exp(ℓ_compact_j) - Σ_i exp(ℓ_full_i)| / Σ_i exp(ℓ_full_i)

    实现上为避免 exp 溢出，两侧各自减去自己的 max 后比较 ——
    这与 softmax 的数值稳定性处理一致，衡量的是"质量被低估/高估"的相对程度。

    Args:
        probe_queries: [N, d_h]
        compact:       CompactKV
        keys:          [L_s, d_h] 完整源块 Key
        beta_mode:     β 的用法，见 `beta_applied`
        num_repr_queries: 拟合时的 M

    Returns:
        [N] 每个探针 Query 的质量相对误差
    """
    scale = 1.0 / (keys.shape[-1] ** 0.5)

    logits_full = (probe_queries @ keys.T) * scale
    full_shifted = torch.exp(logits_full - logits_full.max(dim=-1, keepdim=True).values)
    mass_full = full_shifted.sum(dim=-1)

    bias = beta_applied(compact.logit_bias, beta_mode, num_repr_queries)
    logits_c = (probe_queries @ compact.keys.T) * scale + bias
    c_shifted = torch.exp(logits_c - logits_c.max(dim=-1, keepdim=True).values)
    mass_c = c_shifted.sum(dim=-1)

    return (mass_c - mass_full).abs() / (mass_full + 1e-12)


def output_error(
    probe_queries: torch.Tensor,
    compact: CompactKV,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta_mode: str = "full",
    num_repr_queries: int = 1,
) -> torch.Tensor:
    """ε_out：紧凑块与完整块在输出侧的 L2 偏差（逐 query）。

    ε_out(q) = || Attn_compact(q) - Attn_dense(q) ||_2

    Returns:
        [N]
    """
    y_compact = compact_attention(
        probe_queries, compact, beta_mode=beta_mode,
        num_repr_queries=num_repr_queries,
    )
    y_dense = dense_attention(probe_queries, keys, values)
    return (y_compact - y_dense).norm(dim=-1)


def relative_output_error(
    probe_queries: torch.Tensor,
    compact: CompactKV,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta_mode: str = "full",
    num_repr_queries: int = 1,
) -> torch.Tensor:
    """归一化的 ε_out：除以完整输出的逐 query 范数。

    绝对 ε_out 会随 d_v 增大而增大，跨维度不可比；归一化版本可以。
    """
    y_dense = dense_attention(probe_queries, keys, values)
    return output_error(
        probe_queries, compact, keys, values,
        beta_mode=beta_mode, num_repr_queries=num_repr_queries,
    ) / (y_dense.norm(dim=-1) + 1e-12)


def signed_mass_error(
    probe_queries: torch.Tensor,
    compact: CompactKV,
    keys: torch.Tensor,
    beta_mode: str = "full",
    num_repr_queries: int = 1,
) -> torch.Tensor:
    """ε_mass 的**有符号**版本：(mass_compact - mass_full) / mass_full。

    `mass_error` 取绝对值，只能回答"差多少"；本函数保留符号，可以回答
    "偏高还是偏低"。这个区别是必要的 —— 论文 §5.3 的性质 2 是一个
    **方向性**预测："无偏置时质量被系统性低估"。
    只有有符号误差才能检验它；取绝对值的版本会把这个预测变成不可证伪的。

    Returns:
        [N] 正值表示紧凑块高估了质量，负值表示低估
    """
    scale = 1.0 / (keys.shape[-1] ** 0.5)

    logits_full = (probe_queries @ keys.T) * scale
    full_shifted = torch.exp(logits_full - logits_full.max(dim=-1, keepdim=True).values)
    mass_full = full_shifted.sum(dim=-1)

    bias = beta_applied(compact.logit_bias, beta_mode, num_repr_queries)
    logits_c = (probe_queries @ compact.keys.T) * scale + bias
    c_shifted = torch.exp(logits_c - logits_c.max(dim=-1, keepdim=True).values)
    mass_c = c_shifted.sum(dim=-1)

    return (mass_c - mass_full) / (mass_full + 1e-12)


def selected_index_overlap(a: CompactKV, b: CompactKV) -> float:
    """两个紧凑 KV 选中索引集的 Jaccard 重叠度。

    这是"不同目的端选到了不同的 Key"最直接的**无假设**证据：
    与 KL 散度不同，它不依赖任何平滑系数或分布假设。
    """
    sa = set(a.selected_indices.tolist())
    sb = set(b.selected_indices.tolist())
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


# =============================================================================
# 便捷封装：为某目的端构造紧凑 KV
# =============================================================================

def build_for_dest(
    scenario: SyntheticScenario,
    dest: int,
    budget: int,
    num_repr_queries: int = 32,
    projection_dim: int = 32,
    lambda_beta: float = 1e-3,
    lambda_value: float = 1e-3,
    seed: int = 42,
) -> CompactKV:
    """为单个目的端构造紧凑 KV（DCC-KV 的逐边语义）。"""
    return build_compact_kv(
        source_keys=scenario.keys,
        source_values=scenario.values,
        destination_queries=scenario.dest_queries[dest],
        budget=budget,
        num_representative_queries=num_repr_queries,
        projection_dim=projection_dim,
        lambda_beta=lambda_beta,
        lambda_value=lambda_value,
        seed=seed,
    )


def build_shared(
    scenario: SyntheticScenario,
    budget: int,
    num_repr_queries: int = 32,
    projection_dim: int = 32,
    lambda_beta: float = 1e-3,
    lambda_value: float = 1e-3,
    seed: int = 42,
) -> CompactKV:
    """构造"共享压缩"基线（FastKV 语义）：用**全部**目的端 Query 构造一份紧凑 KV。

    与 `build_for_dest` 的唯一差别就是 destination_queries 的来源 ——
    这正是 H2 要检验的自变量。M 与 B 两侧保持相同，因此误差差异不能归因于
    "可用 Query 数量"或"预算"不同。
    """
    return build_compact_kv(
        source_keys=scenario.keys,
        source_values=scenario.values,
        destination_queries=scenario.all_queries,
        budget=budget,
        num_representative_queries=num_repr_queries,
        projection_dim=projection_dim,
        lambda_beta=lambda_beta,
        lambda_value=lambda_value,
        seed=seed,
    )


__all__ = [
    "SyntheticScenario",
    "make_scenario",
    "compaction_dtype_support",
    "dtype_from_name",
    "DTYPE_BUG_FILE",
    "DTYPE_BUG_LINE",
    "dense_attention",
    "dense_attention_weights",
    "compact_attention",
    "beta_applied",
    "BETA_MODES",
    "induced_distribution",
    "kl_divergence",
    "jensen_shannon_divergence",
    "kl_saturation_ceiling",
    "mass_error",
    "signed_mass_error",
    "output_error",
    "relative_output_error",
    "selected_index_overlap",
    "build_for_dest",
    "build_shared",
]
