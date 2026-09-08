"""M1.3: 质量偏置（β）拟合——非负岭回归。

论文 DCC-KV eq.(15)(16)：
- ℓ_{s→r}_{a,k} = q̂_{r,a} C_{s→r}^K_k^T / √d_h
- m_{s→r}_a = sum_k exp(ℓ_{s→r}_{a,k})    紧凑块的 softmax 质量
- G_{s→r}_{a,j} = exp(q̂_{r,a} C_{s→r}^K_j^T / √d_h)
- w_j = exp(β_j) >= 0
- w* = argmin_{w>=0} ||G w - m||^2 + λ_β ||w - 1||^2
- β_j = log(max(w*_j, ε))
"""
from __future__ import annotations

import torch
from typing import Optional


def nonneg_least_squares(
    G: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = 1e-3,
    n_iter: int = 100,
) -> torch.Tensor:
    """非负岭回归 (NNLS) 闭式/迭代解。

    解：min_{w >= 0} ||G w - target||^2 + λ ||w - 1||^2

    简化实现：用 projected gradient descent
    - 实际论文用了 NNLS 的 closed-form 求解（基于 Lawson 算法）
    - 这里是 reference 实现，迭代即可

    Args:
        G: [M, B] 设计矩阵
        target: [M] 目标向量（m_{s→r} 块级质量）
        lambda_reg: 正则化强度
        n_iter: 迭代次数

    Returns:
        w: [B] 非负权重
    """
    M, B = G.shape
    w = torch.ones(B, dtype=G.dtype) * 0.5  # 初始化

    # 闭式 ridge 起点（不带非负约束）
    GTG = G.T @ G + lambda_reg * torch.eye(B, dtype=G.dtype)
    GTt = G.T @ target
    w_ridge = torch.linalg.solve(GTG, GTt)
    w_ridge = torch.clamp(w_ridge, min=0.0)  # 投影到非负

    if M < B:
        # 小样本：用 ridge 解
        return w_ridge

    # 否则：projected gradient descent 精化
    w = w_ridge.clone()
    lr = 1e-2
    for _ in range(n_iter):
        grad = G.T @ (G @ w - target) + lambda_reg * (w - 1.0)
        w = w - lr * grad
        w = torch.clamp(w, min=0.0)
    return w


def fit_logit_bias(
    representative_queries: torch.Tensor,
    compact_keys: torch.Tensor,
    original_block_mass: torch.Tensor,
    lambda_reg: float = 1e-3,
) -> torch.Tensor:
    """拟合 β 偏置（logit calibration）。

    Args:
        representative_queries: [M, d_h] 代表 Q
        compact_keys: [B, d_h] 选中的 K（紧凑块）
        original_block_mass: [M] 原始块的 softmax 质量 m_{s→r}
        lambda_reg: 正则化强度 λ_β

    Returns:
        beta: [B] 偏置向量 β = log(w)
    """
    M, d_h = representative_queries.shape
    B = compact_keys.shape[0]
    scale = 1.0 / (d_h ** 0.5)

    # G_{a,j} = exp(q̂_a C_k_j^T / √d_h)   [M, B]
    logits_compact = (representative_queries @ compact_keys.T) * scale
    G = torch.exp(logits_compact)

    # 求解 w >= 0  最小化 ||G w - m||^2 + λ ||w - 1||^2
    w = nonneg_least_squares(G, original_block_mass, lambda_reg=lambda_reg)

    # β = log(w)，clamp 防 -inf
    eps = 1e-6
    beta = torch.log(torch.clamp(w, min=eps))
    return beta
