"""M1.4: Value 回归（ridge regression）。

论文 DCC-KV eq.(17)(18)(19)：
- Y_{s→r} = softmax(Q̂_r K_s^T / √d_h) V_s   [M, d_v] 原始条件输出
- X_{s→r} = softmax(Q̂_r C_{s→r}^K^T / √d_h + (1/M) β_{s→r}^T)   [M, B]
- C_v* = argmin_{C_v} ||X C_v - Y||_F^2 + λ_v ||C_v||_F^2
- = (X^T X + λ_v I)^{-1} X^T Y   闭式解
"""
from __future__ import annotations

import torch
from typing import Optional


def ridge_regression_value(
    X: torch.Tensor,
    Y: torch.Tensor,
    lambda_reg: float = 1e-3,
) -> torch.Tensor:
    """Ridge 回归闭式解。

    Args:
        X: [M, B] 设计矩阵
        Y: [M, d_v] 目标
        lambda_reg: λ_v

    Returns:
        C_v: [B, d_v] 紧凑 Value
    """
    XTX = X.T @ X
    B = X.shape[1]
    ridge = XTX + lambda_reg * torch.eye(B, dtype=X.dtype)
    XTY = X.T @ Y
    C_v = torch.linalg.solve(ridge, XTY)
    return C_v


def fit_compact_value(
    representative_queries: torch.Tensor,
    original_keys: torch.Tensor,
    original_values: torch.Tensor,
    compact_keys: torch.Tensor,
    logit_bias: torch.Tensor,
    lambda_reg: float = 1e-3,
) -> torch.Tensor:
    """拟合紧凑 Value。

    Args:
        representative_queries: [M, d_h] 代表 Q
        original_keys: [L_s, d_h] 原始块 K
        original_values: [L_s, d_v] 原始块 V
        compact_keys: [B, d_h] 紧凑 K
        logit_bias: [B] β 偏置
        lambda_reg: λ_v

    Returns:
        compact_values: [B, d_v] 紧凑 V
    """
    M, d_h = representative_queries.shape
    L_s = original_keys.shape[0]
    B = compact_keys.shape[0]
    scale = 1.0 / (d_h ** 0.5)

    # Y = softmax(Q̂ K^T / √d) V   [M, d_v]
    logits_orig = (representative_queries @ original_keys.T) * scale
    A_orig = torch.softmax(logits_orig, dim=-1)
    Y = A_orig @ original_values

    # X = softmax(Q̂ C^T / √d + (1/M) β^T)   [M, B]
    logits_compact = (representative_queries @ compact_keys.T) * scale
    # β broadcast: 每个 query 都有 β 加在 logit 上
    bias_per_query = logit_bias.unsqueeze(0).expand(M, B) / M
    X_logits = logits_compact + bias_per_query
    X = torch.softmax(X_logits, dim=-1)

    # 闭式解 C_v = (X^T X + λ I)^{-1} X^T Y
    C_v = ridge_regression_value(X, Y, lambda_reg=lambda_reg)
    return C_v
