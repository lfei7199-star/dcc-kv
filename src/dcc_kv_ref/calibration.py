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
"""
from __future__ import annotations

import torch
from typing import Dict, Optional, Tuple


def nonneg_least_squares(
    G: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = 1e-3,
    n_iter: int = 2000,
    tol: float = 1e-12,
    lambda_mode: str = "relative",
    return_diag: bool = False,
):
    """非负岭回归：min_{w >= 0} ||G w - target||^2 + λ ||w - 1||^2

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
        lambda_reg: 正则化强度 λ_β
        n_iter: 最大迭代次数
        tol: 收敛容差（w 的最大变化量）
        lambda_mode: 见上
        return_diag: 为 True 时返回 (w, diag)

    Returns:
        w: [B] 非负权重；return_diag 时返回 (w, diag)
    """
    M, B = G.shape

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

    w = torch.zeros(B, dtype=G.dtype, device=G.device)
    n_used = 0
    for it in range(n_iter):
        grad = A @ w - b
        w_new = torch.clamp(w - grad / L, min=0.0)
        delta = float((w_new - w).abs().max().item())
        w = w_new
        n_used = it + 1
        if delta < tol:
            break

    if not return_diag:
        return w

    diag = {
        "solver_iters": n_used,
        "solver_lipschitz": L,
        "solver_lambda_eff": lam,
        "solver_curvature": float(torch.diagonal(A_data).mean().item()),
        "solver_residual": float((G @ w - target).pow(2).sum().item()),
    }
    return w, diag


def fit_logit_bias(
    representative_queries: torch.Tensor,
    compact_keys: torch.Tensor,
    original_block_mass: torch.Tensor,
    lambda_reg: float = 1e-3,
    mass_shift: float = 0.0,
) -> torch.Tensor:
    """拟合 β 偏置（logit calibration）。

    Args:
        representative_queries: [M, d_h] 代表 Q
        compact_keys: [B, d_h] 选中的 K（紧凑块）
        original_block_mass: [M] 原始块的**未归一化**质量
            m_{s→r} = Σ_k exp(ℓ_k) —— 不是 softmax 沿 key 轴之和（那恒等于 1）
        lambda_reg: 正则化强度 λ_β
        mass_shift: 全局常数偏移 c。设计矩阵取 exp(ℓ_compact − c)，
            调用方必须用同一 c 计算 original_block_mass。
            **偏移不改变拟合出的 β**：把 ℓ → ℓ − c 与 m → m·exp(−c) 同时施加后，
            方程 Σ_j exp(ℓ_j − c + β'_j) = m·exp(−c) 两边乘 exp(c) 与原方程完全相同，
            故 β' = β。唯一用途是防止 exp(ℓ) 溢出。

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

    # 求解 w >= 0  最小化 ||G w - m||^2 + λ ||w - 1||^2
    w = nonneg_least_squares(G, original_block_mass, lambda_reg=lambda_reg)

    # β = log(w)，clamp 防 -inf
    eps = 1e-6
    beta = torch.log(torch.clamp(w, min=eps))
    return beta
