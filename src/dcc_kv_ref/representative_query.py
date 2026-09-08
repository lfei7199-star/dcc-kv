"""M1.1: 代表 Query 选择（Rademacher 投影 + 最远点采样）。

论文 reference: DCC-KV eq.(10)(11)(12)
"""
from __future__ import annotations

import torch


def rademacher_projection(
    queries: torch.Tensor,
    projection_dim: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Rademacher 随机投影：queries [N, head_dim] → [N, projection_dim]

    满足 Johnson-Lindenstrauss 引理：投影后保留两两距离关系。

    Args:
        queries: [N, head_dim] 输入 queries
        projection_dim: 投影维度 d_p（论文默认 32）
        seed: 随机种子

    Returns:
        projection_matrix: [projection_dim, head_dim]（用于后续投影）
    """
    g = torch.Generator().manual_seed(seed)
    # P_{a,b} ~ 1/sqrt(d_p) * {+1, -1}
    proj = (torch.randint(0, 2, (projection_dim, queries.shape[-1]), generator=g) * 2 - 1)
    proj = proj.float() / (projection_dim ** 0.5)
    return proj


def farthest_point_sampling(
    features: torch.Tensor,
    num_samples: int,
    seed: int = 42,
) -> torch.Tensor:
    """最远点采样（FPS）：从 N 个点中选 M 个代表点。

    论文 DCC-KV eq.(11)：
    - S^(1) = {l_r}（任意起点）
    - Δ^(m)_u = min_{v ∈ S^(m)} D(z_u, z_v)
    - u_{m+1} = argmax_u Δ^(m)_u
    - S^(m+1) = S^(m) ∪ {u_{m+1}}

    Args:
        features: [N, d] 投影后的 queries（已归一化）
        num_samples: 要选的代表数 M
        seed: 起始点随机种子

    Returns:
        indices: [num_samples] 代表 query 的索引
    """
    N = features.shape[0]
    if num_samples >= N:
        return torch.arange(N)

    g = torch.Generator().manual_seed(seed)

    # 距离矩阵（用余弦距离：D = 1 - z^T z'）
    # features 已归一化，所以 z^T z' = cos sim
    normed = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    sim = normed @ normed.T  # [N, N]
    dist = 1.0 - sim

    # 起点：随机
    start = int(torch.randint(0, N, (1,), generator=g).item())
    selected = [start]
    min_dist = dist[start].clone()  # [N]

    for _ in range(num_samples - 1):
        # 选 min_dist 最大的点
        next_idx = int(min_dist.argmax().item())
        selected.append(next_idx)
        # 更新 min_dist
        min_dist = torch.minimum(min_dist, dist[next_idx])

    return torch.tensor(selected, dtype=torch.long)


def select_representative_queries(
    queries: torch.Tensor,
    num_samples: int = 64,
    projection_dim: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """DCC-KV 论文 4.2 节：选择代表 Query。

    Pipeline:
    1. Rademacher 投影：[N, d] → [N, d_p]
    2. 归一化：cos sim 距离
    3. 最远点采样：选 M 个代表

    Args:
        queries: [N, d] 原始 queries
        num_samples: M
        projection_dim: d_p
        seed: 随机种子

    Returns:
        representative_queries: [M, d] 原始维度的代表 query
        indices: [M] 选中的索引
    """
    g = torch.Generator().manual_seed(seed)
    proj = rademacher_projection(queries, projection_dim=projection_dim, seed=seed)
    z = queries @ proj.T  # [N, d_p]
    z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    indices = farthest_point_sampling(z_norm, num_samples=num_samples, seed=seed)
    return queries[indices], indices
