"""β 链路的口径变体 —— 用于在**不改 `src/`** 的前提下判定哪一种约定最合理。

背景（2026-09-13 定位）
------------------------
`src/dcc_kv_ref/compact_kv.py` 的 β 拟合链路存在两处与源论文不符的地方：

1. **质量目标退化**（`compact_kv.py:101`）
   ```
   A_orig = torch.softmax(logits_orig, dim=-1)
   block_mass = A_orig.sum(dim=-1)          # 沿归一化维求和 ≡ 1.0
   ```
   而 `calibration.py` 的文档写明目标应为
   `m_{s→r}_a = Σ_k exp(ℓ_{a,k})` —— **未归一化**质量。
   两者相差一个 softmax 归一化，且前者恒为常数 1，不携带任何信息。

2. **β 系数不一致**（`value_regression.py:72` vs 推理侧）
   拟合侧 `bias_per_query = logit_bias / M`，推理侧 `logits + logit_bias`
   （`dcc_kv_sync_cpu.py:73,151`、`fast_kv_cpu.py:108`），相差 M 倍。

源论文口径（Attention Matching, arXiv:2602.16284）
--------------------------------------------------
Eq.(2)：`Σ_{k∈K} exp(ℓ(q,k)) ≈ Σ_{j∈Ck} exp(ℓ(q,Ck_j) + β_j)`
- 质量是 **未归一化** 的 `Σ exp(·)`；
- β **以系数 1** 加在 logit 上，作用是"按 exp(β_j) 乘性重加权每个保留 Key 对质量的贡献"。

因此最合理的约定是 `mass_target="unnorm"` + `fit_coeff="one"`，
即本模块的 `P1` 预设。本模块把四种候选口径都实现出来，用**留出 Query** 上的
实测误差做对照，而不是仅凭论证下结论。

诚实边界
--------
本模块产出的数字是**机制级**证据（注意力分布与线性输出的重构精度），
不构成任何任务质量主张。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from src.dcc_kv_ref import CompactKV
from src.dcc_kv_ref.calibration import nonneg_least_squares
from src.dcc_kv_ref.key_selection import select_topk_keys
from src.dcc_kv_ref.representative_query import select_representative_queries
from src.dcc_kv_ref.value_regression import ridge_regression_value


# =============================================================================
# 口径预设
# =============================================================================

#: 每种预设 = (质量目标, 拟合侧 β 系数, 推理侧 β 系数, 求解器, 正则尺度, 拟合尺度)
#:
#: - ``legacy``            ：仓库现状（含仓库求解器与绝对 λ）。softmax 质量（≡1）
#:                            + 拟合 β/M + 推理 β。三处互相矛盾。
#: - ``legacy_fixed_solver``：只把求解器换成收敛版，其余保持旧口径。
#:                            用于把"口径错"与"求解器错"两个变量分开。
#: - ``shift``             ：只把推理侧迁就到拟合侧的 β/M，质量目标仍是退化的常数。
#: - ``am``                ：源论文口径。未归一化质量 + 拟合 β + 推理 β，线性最小二乘。
#: - ``am_over_m``         ：未归一化质量 + 两侧都用 β/M。β 整体被压小 M 倍。
#: - ``am_logfit``         ：同 ``am``，但拟合改为 **log 尺度**（最小化相对误差）。
#:                           动机见 `fit_logit_bias_logspace`：未归一化质量跨 query
#:                           相差数个数量级，线性最小二乘会被大质量 query 独占。
#: - ``am_scalar``         ：β 退化为**单一标量** b（所有 Key 同一抬升），闭式解。
#:                           这是"只做质量补偿、不做 Key 间重分配"的最小实现。
#:                           它同时是一个诊断：若它优于 per-key β，
#:                           说明 per-key 自由度是在拟合噪声而非信号。
#: - ``am_nobeta``         ：**β 全链路关闭**（拟合与施加都不含 β）。
#:                           这是判断"β 到底有没有用"的**唯一正确基线** ——
#:                           旧脚本里"拟合时带 β、评估时去掉 β"的做法把
#:                           V 回归与评估口径割裂，会系统性夸大 β 的损害。
PRESETS: Dict[str, Dict[str, str]] = {
    "legacy":              {"mass": "softmax", "fit": "over_m", "apply": "full",
                            "solver": "repo_legacy", "lam": "absolute",
                            "scale": "linear"},
    "legacy_fixed_solver": {"mass": "softmax", "fit": "over_m", "apply": "full",
                            "solver": "pgd", "lam": "absolute", "scale": "linear"},
    "shift":               {"mass": "softmax", "fit": "over_m", "apply": "over_m",
                            "solver": "pgd", "lam": "relative", "scale": "linear"},
    "am":                  {"mass": "unnorm",  "fit": "one",    "apply": "full",
                            "solver": "pgd", "lam": "relative", "scale": "linear"},
    "am_over_m":           {"mass": "unnorm",  "fit": "over_m", "apply": "over_m",
                            "solver": "pgd", "lam": "relative", "scale": "linear"},
    "am_logfit":           {"mass": "unnorm",  "fit": "one",    "apply": "full",
                            "solver": "logfit", "lam": "relative", "scale": "log"},
    "am_scalar":           {"mass": "unnorm",  "fit": "one",    "apply": "full",
                            "solver": "scalar", "lam": "relative", "scale": "log"},
    "am_nobeta":           {"mass": "unnorm",  "fit": "one",    "apply": "none",
                            "solver": "pgd", "lam": "relative", "scale": "linear"},
}

PRESET_ORDER = ("legacy", "legacy_fixed_solver", "shift", "am", "am_over_m",
                "am_logfit", "am_scalar", "am_nobeta")


def _legacy_nonneg_least_squares(
    G: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = 1e-3,
    n_iter: int = 100,
) -> torch.Tensor:
    """修复前 `src/dcc_kv_ref/calibration.py` 的原始实现，逐行复刻。

    保留它的唯一目的是让 `legacy` 预设能够**精确复现修复前的行为**，
    这样"改前 vs 改后"的对照才是可复现的，而不是靠记忆比较。

    原实现的三个问题：
    1. 固定步长 lr = 1e-2，未按 Lipschitz 常数缩放；
    2. 正则项用绝对 λ，与目标量级无关；
    3. 非负性的唯一保证是每步后 clamp，收敛性无依据。
    """
    M, B = G.shape
    GTG = G.T @ G + lambda_reg * torch.eye(B, dtype=G.dtype, device=G.device)
    GTt = G.T @ target
    w_ridge = torch.linalg.solve(GTG, GTt)
    w_ridge = torch.clamp(w_ridge, min=0.0)

    if M < B:
        return w_ridge

    w = w_ridge.clone()
    lr = 1e-2
    for _ in range(n_iter):
        grad = G.T @ (G @ w - target) + lambda_reg * (w - 1.0)
        w = w - lr * grad
        w = torch.clamp(w, min=0.0)
    return w


# =============================================================================
# 非负岭回归求解器
# =============================================================================
#
# 仓库的 `calibration.nonneg_least_squares` 使用**固定步长** lr=1e-2 的投影梯度。
# 这对"目标恒为 1"的退化问题勉强能用，但一旦换成未归一化质量
# （目标量级可达 L_s ≈ 256），梯度的 Lipschitz 常数随之增大，
# 固定步长直接越过稳定域，w 在 0 与正值之间来回跳并被 clamp 到 0 ——
# 表现为 β 全部顶到 log(1e-6) = −13.8155 的下界。
#
# 因此这里提供步长取 1/L（L 为 Lipschitz 常数，由最大特征值给出）的投影梯度，
# 对凸二次 + 箱约束保证收敛。为了把"口径错"与"求解器错"分开，
# 预设里显式记录了用哪个求解器。

def nonneg_ridge_pgd(
    G: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = 1e-3,
    n_iter: int = 2000,
    tol: float = 1e-12,
    lambda_mode: str = "relative",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """求解 min_{w>=0} ||G w − target||² + λ||w − 1||²，用步长 1/L 的投影梯度。

    规范化后等价于 `A w = b` 在非负象限上的投影，其中
    `A = GᵀG + λ_eff·I`、`b = Gᵀ·target + λ_eff·1`。

    `lambda_mode` 决定 λ_eff：
        - ``"absolute"``：λ_eff = λ。与仓库 `nonneg_least_squares` 一致。
        - ``"relative"``：λ_eff = λ · mean(diag(GᵀG))。**尺度无关**。

    为什么需要尺度无关的正则
    ------------------------
    未归一化质量的目标量级是 L_s（可达数百），而 λ 的合适取值取决于
    ||G||² 的量级。若 λ 写死为 1e-3 而 ||G||² ~ 1e-6（低质量 query），
    正则项会完全压过数据项，解被拉回 w = 1（即 β ≡ 0）；
    反之若 ||G||² ~ 1e6，正则项形同虚设，解在零空间里乱跑。
    取 λ 相对平均曲率可以同时避免这两种失效，并且让全局偏移 c
    （见 `fit_logit_bias_variant`）真正成为无关变量。

    Returns:
        (w [B], 诊断字典)
    """
    B = G.shape[1]
    A_data = G.T @ G
    if lambda_mode == "relative":
        lam = lambda_reg * float(torch.diagonal(A_data).mean().item())
    elif lambda_mode == "absolute":
        lam = lambda_reg
    else:
        raise ValueError(f"未知 lambda_mode: {lambda_mode}（可选 absolute / relative）")

    A = A_data + lam * torch.eye(B, dtype=G.dtype)
    b = G.T @ target + lam * torch.ones(B, dtype=G.dtype)

    # Lipschitz 常数 = A 的最大特征值
    try:
        L = float(torch.linalg.eigvalsh(A).max().item())
    except Exception:
        L = float(A.abs().sum(dim=-1).max().item())
    L = max(L, 1e-12)

    w = torch.zeros(B, dtype=G.dtype)
    n_used = 0
    for it in range(n_iter):
        grad = A @ w - b
        w_new = torch.clamp(w - grad / L, min=0.0)
        delta = float((w_new - w).abs().max().item())
        w = w_new
        n_used = it + 1
        if delta < tol:
            break

    resid = float((G @ w - target).pow(2).sum().item())
    diag = {
        "solver_iters": n_used,
        "solver_lipschitz": L,
        "solver_lambda_eff": lam,
        "solver_curvature": float(torch.diagonal(A_data).mean().item()),
        "solver_residual": resid,
    }
    return w, diag


def fit_logit_bias_scalar(
    G: torch.Tensor,
    target: torch.Tensor,
    mass_shift: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """把 β 退化为**单一标量** b（所有 Key 同一抬升），有闭式解。

    条件：`Σ_j exp(ℓc_{a,j} + b) = m_a`  →  取对数得
    `b = log m_a − log Σ_j exp(ℓc_{a,j})`。对 a 取平均即为
    `min_b Σ_a [log m_a − log Σ_j exp(ℓc_{a,j}) − b]²` 的闭式解。

    这是"只做质量补偿、不做 Key 间重分配"的最小实现，也是一个诊断：
    若标量 β 优于 per-key β，说明 per-key 的额外自由度在拟合噪声而非信号。
    注意 λ_β 在此变体下不参与（标量版本无需正则——它只有 1 个参数，
    而 M 个方程远多于参数）。
    """
    logG = torch.log(G.clamp(min=1e-30))
    logm = torch.log(target.clamp(min=1e-30))
    d_a = torch.logsumexp(logG, dim=-1)          # [M] 无 β 时的对数质量
    b = float((logm - d_a).mean().item())
    beta = torch.full((G.shape[1],), b, dtype=G.dtype, device=G.device)
    w = torch.exp(beta)

    with torch.no_grad():
        resid = float(((logm - d_a - b) ** 2).mean().item())
        spread = float((logm - d_a).std().item()) if logm.numel() > 1 else 0.0

    diag = {
        "solver_iters": 1,
        "solver_lipschitz": float("nan"),
        "solver_lambda_eff": 0.0,
        "solver_curvature": spread ** 2,
        "solver_residual_logspace": resid,
        "solver_residual": resid,
        "scalar_beta": b,
        # 残差的标准差 = 标量 β 无法吸收的 per-query 差异（越小越说明"只需抬升"）
        "scalar_residual_std": spread,
    }
    return w, diag


def fit_logit_bias_logspace(
    G: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = 1e-3,
    n_iter: int = 300,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """在 **log 尺度**上拟合 β：最小化相对误差而非绝对误差。

    目标：`min_β  mean_a [ log Σ_j exp(ℓc_{a,j} + β_j) − log m_a ]²  + λ·mean_j (e^{β_j} − 1)²`

    为什么必须换尺度
    ----------------
    未归一化质量 `m_a` 跨 query 可以相差数个数量级（尖峰 query 的 m 可达平缓 query 的
    数百倍）。对 `Σ_a (G w − m)²` 而言，**平方后**这个差距再放大一次，
    于是拟合完全被少数大质量 query 支配，其余 query 的 β 需求被忽略。
    换成 log 尺度后，每个 query 的误差变成"相对误差"，
    与 β 的最终作用（乘性重加权质量）也一致。

    可辨识性说明
    ------------
    在 log 尺度上 β 的**整体常数**是可辨识的（挪动常数会乘性改变总质量），
    这与 softmax 内部对 β 常数位移不敏感的性质不同 —— 正是这一点让 β 能够
    承担"补回丢失质量"的职责。

    Returns:
        (beta [B], 诊断字典)
    """
    B = G.shape[1]
    logG = torch.log(G.clamp(min=1e-30))
    logm = torch.log(target.clamp(min=1e-30))

    # 正则项在 log 尺度上取一个与数据项同量级的尺度
    data_scale = float(logm.var().item()) if logm.numel() > 1 else 1.0
    lam = lambda_reg * max(data_scale, 1e-12)

    beta = torch.zeros(B, dtype=G.dtype, requires_grad=True)
    opt = torch.optim.LBFGS(
        [beta], max_iter=n_iter, line_search_fn="strong_wolfe",
        tolerance_grad=1e-10, tolerance_change=1e-12,
    )
    n_evals = {"n": 0}

    def closure():
        opt.zero_grad()
        lse = torch.logsumexp(logG + beta.unsqueeze(0), dim=-1)      # [M]
        data = ((lse - logm) ** 2).mean()
        reg = ((torch.exp(beta) - 1.0) ** 2).mean()
        loss = data + lam * reg
        loss.backward()
        n_evals["n"] += 1
        return loss

    opt.step(closure)
    beta_d = beta.detach()

    with torch.no_grad():
        lse = torch.logsumexp(logG + beta_d.unsqueeze(0), dim=-1)
        rel_resid = float((lse - logm).pow(2).mean().item())
        w_equiv = torch.exp(beta_d)

    diag = {
        "solver_iters": n_evals["n"],
        "solver_lipschitz": float("nan"),
        "solver_lambda_eff": lam,
        "solver_curvature": data_scale,
        "solver_residual_logspace": rel_resid,
    }
    return w_equiv, diag


def beta_factor(apply_mode: str, num_repr_queries: int) -> float:
    """推理侧施加 β 时的缩放因子。"""
    if apply_mode == "full":
        return 1.0
    if apply_mode == "over_m":
        return 1.0 / max(int(num_repr_queries), 1)
    if apply_mode == "none":
        return 0.0
    raise ValueError(f"未知 apply 模式: {apply_mode}")


# =============================================================================
# β 拟合（可切换质量目标与系数）
# =============================================================================

def fit_logit_bias_variant(
    representative_queries: torch.Tensor,
    compact_keys: torch.Tensor,
    original_logits: torch.Tensor,
    mass_target: str = "unnorm",
    lambda_reg: float = 1e-3,
    shift: float = 0.0,
    solver: str = "pgd",
    lambda_mode: str = "relative",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """按指定口径拟合 β。

    Args:
        representative_queries: [M, d_h]
        compact_keys:           [B, d_h]
        original_logits:        [M, L_s] 原始块的缩放 logit ℓ = qK^T/√d
        mass_target:            "unnorm"（Σ exp）或 "softmax"（Σ softmax ≡ 1，旧口径）
        lambda_reg:             λ_β
        shift:                  全局常数偏移 c，同时从两侧 logit 中减去。
                                **不改变拟合出的 β**（见下），仅用于防 exp 溢出。
        solver:                 "pgd"（步长 1/L，收敛）或 "repo"（仓库固定步长实现）
        lambda_mode:            正则项的尺度解释，见 `nonneg_ridge_pgd`

    Returns:
        (beta [B], 诊断字典)

    为什么 `shift` 是免费的
    ----------------------
    把 ℓ → ℓ − c 与目标 m → m·exp(−c) 同时施加，拟合方程
    `Σ_j exp(ℓ_j − c + β'_j) = m·exp(−c)` 两边乘 exp(c) 后与未偏移时**完全相同**，
    故 β' = β。因此可以放心地用全局 max 偏移换取数值安全。
    """
    M, d_h = representative_queries.shape
    B = compact_keys.shape[0]
    scale = 1.0 / (d_h ** 0.5)

    # ---- 质量目标 m [M] ----
    if mass_target == "softmax":
        # 旧口径：Σ_j softmax(ℓ)_j ≡ 1（数学恒等式，与输入无关）
        m = torch.softmax(original_logits, dim=-1).sum(dim=-1)
    elif mass_target == "unnorm":
        # 源论文口径：未归一化质量 Σ_j exp(ℓ_j)，做全局偏移防溢出
        m = torch.exp(original_logits - shift).sum(dim=-1)
    else:
        raise ValueError(f"未知 mass_target: {mass_target}")

    # ---- 设计矩阵 G = exp(ℓ_compact − shift)   [M, B] ----
    logits_compact = (representative_queries @ compact_keys.T) * scale
    G = torch.exp(logits_compact - shift)

    if solver == "pgd":
        w, sdiag = nonneg_ridge_pgd(G, m, lambda_reg=lambda_reg,
                                    lambda_mode=lambda_mode)
    elif solver == "repo_legacy":
        w = _legacy_nonneg_least_squares(G, m, lambda_reg=lambda_reg)
        sdiag = {"solver_iters": -1, "solver_lipschitz": float("nan"),
                 "solver_lambda_eff": lambda_reg,
                 "solver_curvature": float(torch.diagonal(G.T @ G).mean().item()),
                 "solver_residual": float((G @ w - m).pow(2).sum().item())}
    elif solver == "logfit":
        w, sdiag = fit_logit_bias_logspace(G, m, lambda_reg=lambda_reg)
    elif solver == "scalar":
        w, sdiag = fit_logit_bias_scalar(G, m, mass_shift=shift)
    else:
        raise ValueError(f"未知 solver: {solver}")

    # 各求解器返回的诊断字段名不完全一致，这里统一出一个 solver_residual
    sdiag.setdefault("solver_residual",
                     sdiag.get("solver_residual_logspace", float("nan")))

    eps = 1e-6
    beta = torch.log(torch.clamp(w, min=eps))

    # ---- 结构性诊断：单个静态 β 到底能不能救回质量 ----
    # 对第 a 个 query，若不加 β，紧凑块的质量是 (G·1)_a；要匹配目标 m_a，
    # 需要的乘性补偿是 c_a = m_a / (G·1)_a。β 是**逐 Key 的静态向量**，
    # 对同一个 query 内部所有 Key 施加相同的整体抬升最多只能做到常数倍的 c。
    # 因此只有当 c_a 跨 query 近似恒定，静态 β 才可能有效。
    # c_a 的离散程度因此是"β 机制是否可行"的直接判据 —— 与拟合算法无关。
    with torch.no_grad():
        base = (G @ torch.ones(G.shape[1], dtype=G.dtype)).clamp(min=1e-30)
        comp = m / base
        c_med = float(comp.median().item())
        c_p5 = float(torch.quantile(comp, 0.05).item()) if comp.numel() > 1 else c_med
        c_p95 = float(torch.quantile(comp, 0.95).item()) if comp.numel() > 1 else c_med
        need_spread = (c_p95 / c_p5) if c_p5 > 1e-30 else float("inf")

    diag = {
        "mass_target_mean": float(m.mean()),
        "mass_target_std": float(m.std()) if M > 1 else 0.0,
        "w_mean": float(w.mean()),
        "beta_mean": float(beta.mean()),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
        "clamp_hits": int((w <= eps).sum()),
        "needed_compensation_median": c_med,
        "needed_compensation_p5": c_p5,
        "needed_compensation_p95": c_p95,
        # 理想情况下静态 β 需要这个值接近 1；远大于 1 即"结构上救不回来"
        "needed_compensation_spread": need_spread,
    }
    diag.update(sdiag)
    return beta, diag


# =============================================================================
# 完整构造链（与 build_compact_kv 同序，仅口径可切）
# =============================================================================

def build_compact_kv_variant(
    source_keys: torch.Tensor,
    source_values: torch.Tensor,
    destination_queries: torch.Tensor,
    budget: int,
    preset: str = "am",
    num_representative_queries: int = 32,
    projection_dim: int = 32,
    lambda_beta: float = 1e-3,
    lambda_value: float = 1e-3,
    seed: int = 42,
) -> Tuple[CompactKV, Dict[str, float]]:
    """按指定口径构造紧凑 KV。

    与 `build_compact_kv` 的差别**仅在 β 与 V 的口径**；代表 Query 与 Key 选择
    调用的是同一个函数、同一个种子，因此四种预设选出的 Key 子集完全一致 ——
    误差差异不能归因于选键不同。
    """
    if preset not in PRESETS:
        raise ValueError(f"未知 preset: {preset}（可选 {PRESET_ORDER}）")
    cfg = PRESETS[preset]

    repr_queries, _ = select_representative_queries(
        destination_queries,
        num_samples=num_representative_queries,
        projection_dim=projection_dim,
        seed=seed,
    )
    compact_keys, selected_idx = select_topk_keys(
        source_keys, repr_queries, budget=budget,
    )

    M, d_h = repr_queries.shape
    scale = 1.0 / (d_h ** 0.5)
    logits_orig = (repr_queries @ source_keys.T) * scale
    # 全局偏移：同时作用于目标与设计矩阵，β 不变，只为防 exp 溢出
    shift = float(logits_orig.max().item()) if cfg["mass"] == "unnorm" else 0.0

    beta, diag = fit_logit_bias_variant(
        repr_queries, compact_keys, logits_orig,
        mass_target=cfg["mass"], lambda_reg=lambda_beta, shift=shift,
        solver=cfg["solver"], lambda_mode=cfg["lam"],
    )

    # ---- V 回归：β 的系数由 fit 决定 ----
    # apply == "none" 表示整条链路关闭 β：不仅推理时不加，V 回归的设计矩阵
    # 也必须用无 β 的 X。否则"拟合时带 β、评估时去掉 β"会把两个口径割裂，
    # 让 β 的损害被系统性夸大。
    if cfg["apply"] == "none":
        beta = torch.zeros_like(beta)

    logits_compact = (repr_queries @ compact_keys.T) * scale
    beta_in_fit = beta if cfg["fit"] == "one" else beta / max(M, 1)
    X = torch.softmax(logits_compact + beta_in_fit, dim=-1)

    A_orig = torch.softmax(logits_orig, dim=-1)
    Y = A_orig @ source_values
    C_v = ridge_regression_value(X, Y, lambda_reg=lambda_value)

    diag.update({
        "preset": preset,
        "M": int(M),
        "B": int(budget),
        "shift": float(shift),
        "beta_in_fit_coeff": 1.0 if cfg["fit"] == "one" else 1.0 / max(M, 1),
        "beta_apply_coeff": beta_factor(cfg["apply"], M),
        "solver": cfg["solver"],
        "lambda_mode": cfg["lam"],
        # X 的秩上限 —— 决定 B > M 时回归是否欠定
        "rank_upper_bound": int(min(M, budget)),
    })

    compact = CompactKV(
        keys=compact_keys,
        logit_bias=beta,
        values=C_v,
        selected_indices=selected_idx,
    )
    return compact, diag


def apply_beta_for_preset(
    logit_bias: torch.Tensor,
    preset: str,
    num_repr_queries: int,
) -> torch.Tensor:
    """按 preset 的推理侧约定把 β 变换为实际加到 logit 上的量。"""
    factor = beta_factor(PRESETS[preset]["apply"], num_repr_queries)
    return logit_bias * factor


__all__ = [
    "PRESETS",
    "PRESET_ORDER",
    "beta_factor",
    "nonneg_ridge_pgd",
    "fit_logit_bias_variant",
    "build_compact_kv_variant",
    "apply_beta_for_preset",
]
