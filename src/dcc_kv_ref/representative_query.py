"""M1.1: 代表 Query 选择（Rademacher 投影 + 最远点采样）。

论文 reference: DCC-KV eq.(10)(11)(12)

设备与精度说明（2026-09-13 修正）
--------------------------------
旧实现在 `rademacher_projection` 与 `farthest_point_sampling` 中直接用
`torch.Generator()` + `torch.randint(...)`，两者都默认建在 **CPU**：

- 当 `queries` 在 CUDA 上时，`queries @ proj.T` 会因设备不一致直接报错，
  整条紧凑 KV 构造链在 GPU 上不可用；
- `proj.float()` 写死 float32，使 float64 输入报 dtype 不匹配，
  第 5 章的 FP64 误差分析因此在实现上无法复现。

现改为按输入张量的 device/dtype 生成。
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
    device = queries.device
    g = torch.Generator(device=device).manual_seed(seed)
    # P_{a,b} ~ 1/sqrt(d_p) * {+1, -1}；dtype 与 device 均随输入
    proj = (torch.randint(0, 2, (projection_dim, queries.shape[-1]),
                          generator=g, device=device) * 2 - 1)
    proj = proj.to(queries.dtype) / (projection_dim ** 0.5)
    return proj


def farthest_point_sampling(
    features: torch.Tensor,
    num_samples: int,
    seed: int = 42,
    start: str = "newest",
) -> torch.Tensor:
    """最远点采样（FPS）：从 N 个点中选 M 个代表点。

    论文 DCC-KV eq.(11)：
    - S^(1) = {l_r}（**最新 Query** 为初始锚点，见 §4.2）
    - Δ^(m)_u = min_{v ∈ S^(m)} D(z_u, z_v)
    - u_{m+1} = argmax_u Δ^(m)_u
    - S^(m+1) = S^(m) ∪ {u_{m+1}}

    复杂度：O(M · N · d)。**不预计算 N×N 相似度矩阵** —— 后者要 O(N^2)
    内存，L_r = 10^5 时不可行。论文 §4.2 声明的复杂度正是
    O(M · L_r · d_p)，此实现按论文口径给出。

    Args:
        features: [N, d] 投影后的 queries（已归一化）
        num_samples: 要选的代表数 M
        seed: 起点随机种子（**仅** ``start="random"`` 时使用）
        start: 起点模式。默认 ``"newest"``，即服从论文 eq.(11) 的
            「以最新 Query 为初始锚点」（index N-1）。``"random"`` 复现
            2026-09-22 之前的实现（由 seed 决定起点），仅供对照与复现旧落盘。

    Returns:
        indices: [num_samples] 代表 query 的索引（与 features 同 device）

    Raises:
        ValueError: ``start`` 不是 ``"newest"`` / ``"random"``。
    """
    N = features.shape[0]
    if num_samples >= N:
        return torch.arange(N, device=features.device)

    # features 已归一化，所以 z^T z' = cos sim，D = 1 - z^T z'
    normed = features / (features.norm(dim=-1, keepdim=True) + 1e-8)

    if start == "newest":
        first = N - 1
    elif start == "random":
        g = torch.Generator(device=features.device).manual_seed(seed)
        first = int(torch.randint(0, N, (1,), generator=g,
                                  device=features.device).item())
    else:
        raise ValueError(
            f"未知起点模式 start={start!r}（应为 'newest' 或 'random'）")

    selected = [first]
    # 到已选集合的最近距离：只维护 O(N) 向量，不建 N×N 距离矩阵
    min_dist = 1.0 - (normed @ normed[first])

    for _ in range(num_samples - 1):
        # 选 min_dist 最大的点
        next_idx = int(min_dist.argmax().item())
        selected.append(next_idx)
        # 只算新选点到其余点的距离（O(N·d)），再逐点取 min
        min_dist = torch.minimum(min_dist, 1.0 - (normed @ normed[next_idx]))

    return torch.tensor(selected, dtype=torch.long, device=features.device)


def select_representative_queries(
    queries: torch.Tensor,
    num_samples: int = 64,
    projection_dim: int = 32,
    seed: int = 42,
    start: str = "newest",
) -> torch.Tensor:
    """DCC-KV 论文 4.2 节：选择代表 Query。

    Pipeline:
    1. Rademacher 投影：[N, d] → [N, d_p]
    2. 归一化：cos sim 距离
    3. 最远点采样：选 M 个代表（起点口径见 ``farthest_point_sampling``）

    Args:
        queries: [N, d] 原始 queries
        num_samples: M
        projection_dim: d_p
        seed: 随机种子
        start: FPS 起点模式，默认 ``"newest"``（服从论文 eq.(11)）

    Returns:
        representative_queries: [M, d] 原始维度的代表 query
        indices: [M] 选中的索引
    """
    proj = rademacher_projection(queries, projection_dim=projection_dim, seed=seed)
    z = queries @ proj.T  # [N, d_p]
    z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    indices = farthest_point_sampling(z_norm, num_samples=num_samples,
                                      seed=seed, start=start)
    return queries[indices], indices

