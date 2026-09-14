"""M1.3: 质量偏置（β）拟合——非负岭回归。

论文 DCC-KV eq.(15)(16)：
- ℓ_{s→r}_{a,k} = q̂_{r,a} C_{s→r}^K_k^T / √d_h
- m_{s→r}_a = sum_k exp(ℓ_{s→r}_{a,k})    ← **未归一化**质量（注意不是 softmax 之和）
- G_{s→r}_{a,j} = exp(q̂_{r,a} C_{s→r}^K_j^T / √d_h)
- w_j = exp(β_j) >= 0
- w* = argmin_{w>=0} ||G w - m||^2 + λ_β ||w - 1||^2
- β_j = log(max(w*_j, ε))

口径说明（2026-09-13 修正）
---------------------------
源论文（Attention Matching, arXiv:2602.16284）Eq.(2) 要求

    Σ_{k∈K} exp(ℓ(q,k))  ≈  Σ_{j∈Ck} exp(ℓ(q,Ck_j) + β_j)

即 β 以**系数 1** 加在 logit 上，作用是把每个保留 Key 对**未归一化质量**的
贡献乘性重加权 exp(β_j)。本文件与 `value_regression.py` 已统一到该口径。

箱约束（2026-09-14 新增，默认开启）
-----------------------------------
源论文 Appendix C.2 对**本文所用的选键分支**（RMS / Highest-Attention-Keys）
给出的实现是「有界 NNLS」，箱约束 β∈[−3,3]（其方法命名 `nnls2_-3_3`），
理由是：

    某些被选中的 Key 对质量目标的贡献很小，因此会被分到极小的权重
    （β 极负，乃至 β=−∞）；但这类 Key 对**降低输出误差**仍可能有用。
    一旦 β 极负，该 Key 无论 C_v 取什么都不能再参与输出。

本仓库实测（E10，`experiments/cpu/e10_beta_stability.py`，180 格）：不加箱约束时
约 5.4% 的 Key 被压到 log(1e-6)=−13.8155 的下界，即"软剔除"；加 [−3,3] 后
该比例降为 0，而误差变化在 ±0.002 以内。因此默认采用源论文的箱约束。

λ_β 的默认值（E11，`experiments/cpu/e11_lambda_tuning.py`）
-----------------------------------------------------------
箱约束解决了"顶到 −13.8155"，但没有解决"顶到 −3"：在 λ_β=1e-3 下仍有 5.7%
的 Key 停在箱的下界（E11 新增的"箱绑定率"诊断）。E11 在同样的 180 格上扫描
λ_β ∈ [1e-3, 1e3]，结论是：

* λ_β → ∞ 时 w → 1、β → 0，链路逐位退化到"关闭 β"：实测 max|Δ| ≤ 1.7·β_std，
  比值有界，故该极限成立。这意味着留出误差曲线**必然回弹**，存在内部最优；
* 输出与归并误差的最低点出现在 λ_β ≈ 1e-2 ~ 1e-1，比 λ_β=1e-3 低 0.004~0.019；
* 但最优 λ_β 随 (M,B) 与所选指标移动达 4 个数量级（单格 argmin 从 3e-3 到 1e3），
  因此**不存在全局最优的 λ_β**，默认值只能按公开判据取。

本文采用的判据（可从落盘产物复现）：取**最小的** λ_β，使两个指标相对旧默认
（1e-3）的改进在配对 bootstrap 上均**稳健**（95% CI 排除 0），且 B≤M 一侧
保持统计中性。满足该判据的最小值是 3e-2（输出 Δ=−0.0119、归并 Δ=−0.0118，
均 p<0.002；B≤M 侧 p>0.5），其最差单格退化 7.3%，远小于 1e-1 的 20.3%。
故 ``DEFAULT_LAMBDA_BETA = 3e-2``；显式传入 lambda_reg 即可复现任何其它配置。
"""
from __future__ import annotations

import math
import torch
from typing import Dict, Optional, Tuple

# β 的默认箱约束半宽（源论文 Appendix C.2 对 RMS 选键分支取 3）。
# 设为 None 可退回「w ≥ 0 无上界」的旧行为（复现旧结果时使用）。
DEFAULT_BETA_BOUND: Optional[float] = 3.0

# β 岭正则的默认强度 λ_β（E11 判定，见模块文档）。
# 旧值为 1e-3：它会让 5.7% 的 Key 停在箱下界 −3，且两个指标都被 3e-2 稳健地
# 超过（配对 bootstrap 95% CI 排除 0），故提升到判据允许的最小值 3e-2。
DEFAULT_LAMBDA_BETA: float = 3e-2


def nonneg_least_squares(
    G: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = DEFAULT_LAMBDA_BETA,
    n_iter: int = 2000,
    tol: float = 1e-12,
    lambda_mode: str = "relative",
    return_diag: bool = False,
    w_lower: float = 0.0,
    w_upper: Optional[float] = None,
):
    """箱约束岭回归：min_{w_lower <= w <= w_upper} ||G w - target||² + λ ||w - 1||²

    求解方式：投影梯度（步长 1/L，L 为 A = GᵀG + λ_eff·I 的最大特征值）。
    凸二次目标 + 箱约束下该步长保证收敛。

    为什么不能用固定步长（2026-09-13 修正）
    --------------------------------------
    旧实现用 lr = 1e-2 的固定步长。当 target 的量级是 L_s（未归一化质量，
    可达数百）时，A 的 Lipschitz 常数随之增大，固定步长越过稳定域，
    w 在 0 与正值之间反复跳变并被投影 clamp 到 0，最终 β 全部顶到
    log(1e-6) = −13.8155 的下界 —— 表现为「β 全部失效」。

    `lambda_mode` 决定 λ_eff：
        - ``"relative"``（默认）：λ_eff = λ · mean(diag(GᵀG))，**尺度无关**。
          未归一化质量跨 query 可相差数个数量级，写死的 λ 会在
          「正则完全压制数据项」与「正则形同虚设」之间反复失准。
        - ``"absolute"``：λ_eff = λ，与旧实现一致（仅供复现旧结果）。

    Args:
        G: [M, B] 设计矩阵（G_{a,j} = exp(ℓ_compact)）
        target: [M] 目标向量（未归一化质量 m_{s→r}）
        lambda_reg: 正则化强度 λ_β（默认 `DEFAULT_LAMBDA_BETA`=3e-2，见模块文档）
        n_iter: 最大迭代次数
        tol: 收敛容差（w 的最大变化量）
        lambda_mode: 见上
        return_diag: 为 True 时返回 (w, diag)
        w_lower: 下界（默认 0，即非负）
        w_upper: 上界；None 表示无上界。
            `fit_logit_bias` 由 β 的箱约束换算后传入。

    Returns:
        w: [B] 权重；return_diag 时返回 (w, diag)
    """
    M, B = G.shape
    if w_upper is not None and w_upper < w_lower:
        raise ValueError(f"w_upper({w_upper}) < w_lower({w_lower})")

    A_data = G.T @ G
    if lambda_mode == "relative":
        lam = lambda_reg * float(torch.diagonal(A_data).mean().item())
    elif lambda_mode == "absolute":
        lam = lambda_reg
    else:
        raise ValueError(f"未知 lambda_mode: {lambda_mode}（可选 absolute / relative）")

    A = A_data + lam * torch.eye(B, dtype=G.dtype, device=G.device)
    b = G.T @ target + lam * torch.ones(B, dtype=G.dtype, device=G.device)

    try:
        L = float(torch.linalg.eigvalsh(A).max().item())
    except Exception:
        L = float(A.abs().sum(dim=-1).max().item())
    L = max(L, 1e-30)

    hi = float("inf") if w_upper is None else float(w_upper)
    w = torch.full((B,), float(w_lower), dtype=G.dtype, device=G.device)
    n_used = 0
    for it in range(n_iter):
        grad = A @ w - b
        w_new = torch.clamp(w - grad / L, min=float(w_lower), max=hi)
        delta = float((w_new - w).abs().max().item())
        w = w_new
        n_used = it + 1
        if delta < tol:
            break

    n_at_lower = int((w <= float(w_lower) + 1e-12).sum().item())
    n_at_upper = (0 if w_upper is None
                  else int((w >= float(w_upper) - 1e-12).sum().item()))
    if not return_diag:
        return w

    diag = {
        "solver_iters": n_used,
        "solver_lipschitz": L,
        "solver_lambda_eff": lam,
        "solver_curvature": float(torch.diagonal(A_data).mean().item()),
        "solver_residual": float((G @ w - target).pow(2).sum().item()),
        "solver_w_lower": float(w_lower),
        "solver_w_upper": (float("inf") if w_upper is None else float(w_upper)),
        "solver_n_at_lower": n_at_lower,
        "solver_n_at_upper": n_at_upper,
    }
    return w, diag


def fit_logit_bias(
    representative_queries: torch.Tensor,
    compact_keys: torch.Tensor,
    original_block_mass: torch.Tensor,
    lambda_reg: float = DEFAULT_LAMBDA_BETA,
    mass_shift: float = 0.0,
    beta_bound: Optional[float] = DEFAULT_BETA_BOUND,
) -> torch.Tensor:
    """拟合 β 偏置（logit calibration）。

    Args:
        representative_queries: [M, d_h] 代表 Q
        compact_keys: [B, d_h] 选中的 K（紧凑块）
        original_block_mass: [M] 原始块的**未归一化**质量
            m_{s→r} = Σ_k exp(ℓ_k) —— 不是 softmax 沿 key 轴之和（那恒等于 1）
        lambda_reg: 正则化强度 λ_β（默认 `DEFAULT_LAMBDA_BETA`=3e-2，见模块文档）
        mass_shift: 全局常数偏移 c。设计矩阵取 exp(ℓ_compact − c)，
            调用方必须用同一 c 计算 original_block_mass。
            **偏移不改变拟合出的 β**：把 ℓ → ℓ − c 与 m → m·exp(−c) 同时施加后，
            方程 Σ_j exp(ℓ_j − c + β'_j) = m·exp(−c) 两边乘 exp(c) 与原方程完全相同，
            故 β' = β。唯一用途是防止 exp(ℓ) 溢出。
        beta_bound: β 的箱约束半宽，即 β ∈ [−beta_bound, +beta_bound]。
            默认 `DEFAULT_BETA_BOUND` = 3（源论文 Appendix C.2 对 RMS 选键分支的配置）。
            设为 None 时退回「w ≥ 0 无上界」，β 下限为 log(1e-6) = −13.8155
            —— 该下界会把部分 Key 的权重压到近似 0，相当于"软剔除"，
            与 β"质量重加权"的定位不符（见模块文档与 E10）。

    Returns:
        beta: [B] 偏置向量 β = log(w)
    """
    M, d_h = representative_queries.shape
    B = compact_keys.shape[0]
    scale = 1.0 / (d_h ** 0.5)

    if original_block_mass.shape[0] != M:
        raise ValueError(
            f"original_block_mass 的长度应为 M={M}，"
            f"得到 {original_block_mass.shape[0]}"
        )

    # G_{a,j} = exp(q̂_a C_k_j^T / √d_h − c)   [M, B]
    logits_compact = (representative_queries @ compact_keys.T) * scale
    G = torch.exp(logits_compact - mass_shift)

    # 把 β 的箱约束换算成 w 的箱约束：w = exp(β) 单调，故等价
    if beta_bound is None:
        w_lower, w_upper = 0.0, None
    else:
        if beta_bound <= 0:
            raise ValueError(f"beta_bound 应为正数或 None，得到 {beta_bound}")
        w_lower = math.exp(-float(beta_bound))
        w_upper = math.exp(float(beta_bound))

    # 求解 w ∈ [w_lower, w_upper]  最小化 ||G w - m||² + λ ||w - 1||²
    w = nonneg_least_squares(
        G, original_block_mass,
        lambda_reg=lambda_reg, w_lower=w_lower, w_upper=w_upper,
    )

    if beta_bound is None:
        # 无箱约束时下界为 0，需 clamp 防 log(0)
        beta = torch.log(torch.clamp(w, min=1e-6))
    else:
        # 有箱约束时 w ≥ exp(−bound) > 0，log 天然有定义；再 clamp 一次
        # 只为消除投影梯度末步的浮点误差
        b = float(beta_bound)
        beta = torch.clamp(torch.log(w), min=-b, max=b)
    return beta
