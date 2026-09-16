#!/usr/bin/env python
"""E13：误差界式(37)的数值紧致度检验（CPU 可跑）

补的是哪个缺口
--------------
`docs/writing_scope_and_metrics.md` 的 M3 —— 论文 §5.4 给出了误差界

    ‖Âttn(q) − Attn(q)‖₂ ≤ ε_out(q)/S(q) + 2·ε_mass(q)/S(q)·max_s‖O_s(q)‖₂   (37)

但**这个界从未被数值检验过**。原因有二：其一，界里的 ε_mass、ε_out 是
**跨块绝对量**（对全部有效块求和），而论文 §6 报的所有误差数字都是
**单块版本**（见 §6 的度量定义表）；其二，界成立需要前置条件
Σ_s|δM_s| < S/2，而该条件的实际违反率也无人测过。

于是式(37) 目前的地位是"看起来严谨的装饰"：它支撑 §5 性质 3
（"误差不随块数累积发散"），而性质 3 又被用来论证 DCC-KV 在 N 大时仍可用。
**用一个从未检验过的界去支撑一个从未检验过的性质，是论文里最容易被
审稿人一击的地方。**

本脚本做三件事
--------------
1. **逐 Query 实测 LHS 与 RHS**，报比值 RHS/LHS 的分布（median/p5/min）与
   "RHS ≥ LHS"的成立比例。界必须**恒成立**，任何一个反例都是逻辑缺陷。
2. **测前置条件的违反率** |δM| ≥ S/2，并单独标注违反前置条件的格子 ——
   那些格子上的界本来就无保证，不能混进"界是否成立"的统计。
3. **测性质 3**：把源块切成 N 份、每份独立压缩，看 LHS 与 Σ|δM| 随 N 的变化。
   性质 3 主张"只要有界、就不随 N 线性放大"，这里直接量它的斜率。

口径（关键，勿混）
------------------
- 本脚本刻意构造**跨块**度量：源块（压缩）与一个**固定上下文块**（精确）
  共处同一个 softmax 分母，S(q) = M_源(q) + M_上下文(q)，ε_mass 与 ε_out
  按式(35)(36) 对块求和。**这与 E2/E5a 的单块口径不可互比。**
- 质量取公共偏移 c = max(ℓ_full)（全块拼接后的最大 logit），否则 β 的
  常数分量不可观测、且 S(q) 的尺度不可定。归一化常数在任何比值里都会
  约掉，故不影响界本身。
- O_s 取**归一化后的块输出** softmax(ℓ_s)·V_s，与式(37) 的 max_s‖O_s‖ 一致。
- 评估只用不相交的留出 Query（`heldout_split`）；构造只用 fit Query。
- 多种子：网格型实验按 `docs/reproducibility.md` §7 用 5 个种子（42–46）。

不产生的主张
------------
本脚本只在合成场景（K/V ~ N(0,I)）上检验一个解析命题的数值自洽性，
不涉及任何真实模型或任务指标。

用法
----
    python experiments/cpu/e13_bound_tightness.py --out results/cpu/e13
    python experiments/cpu/e13_bound_tightness.py --quick
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import synthetic as S  # noqa: E402
from experiments.common import report as R  # noqa: E402


# =============================================================================
# 单次归并的 LHS / RHS 分解
# =============================================================================

def _block_state(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor | None,
    offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (未归一化质量 M、块输出 O、logits)。

    M 用**公共偏移** offset 计算，故不同块之间可比、可直接相加得 S(q)。
    O 是归一化后的块输出（softmax 只对块内做，故与 offset 无关）。
    """
    scale = 1.0 / (keys.shape[-1] ** 0.5)
    logits = (q @ keys.T) * scale
    if bias is not None:
        logits = logits + bias
    mass = torch.exp(logits - offset).sum()
    output = torch.softmax(logits, dim=-1) @ values
    return mass, output, logits


def bound_terms(
    probe_queries: torch.Tensor,
    compact: Any,
    keys: torch.Tensor,
    values: torch.Tensor,
    fixed_keys: torch.Tensor,
    fixed_values: torch.Tensor,
    beta_mode: str,
    num_repr_queries: int,
) -> Dict[str, torch.Tensor]:
    """逐 Query 计算 LHS（真实偏差）与式(37) 的 RHS 及其分解。

    两个有效块：块 0 = 固定上下文（两侧都精确，δM₀ = δO₀ = 0），
    块 1 = 源块（精确 vs 紧凑）。因此

        ε_mass = |δM₁|,          ε_out = M₁·‖Oˆ₁ − O₁‖
        S      = M₀ + M₁,        max_s‖O_s‖ = max(‖O₀‖, ‖O₁‖)

    注意 ε_out 用 **M₁**（完整侧的质量）加权、ε_mass 取**绝对值** —— 与
    式(35)(36) 的定义一致，不能用紧凑侧质量或带符号误差替代。
    """
    from experiments.common.synthetic import beta_applied, mixture_output

    beta = beta_applied(compact.logit_bias, beta_mode, num_repr_queries)
    n = probe_queries.shape[0]

    lhs = torch.zeros(n, dtype=torch.float64)
    rhs = torch.zeros(n, dtype=torch.float64)
    term_out = torch.zeros(n, dtype=torch.float64)
    term_mass = torch.zeros(n, dtype=torch.float64)
    dM_signed = torch.zeros(n, dtype=torch.float64)
    s_total = torch.zeros(n, dtype=torch.float64)
    max_o = torch.zeros(n, dtype=torch.float64)

    for i in range(n):
        q = probe_queries[i]
        # 公共偏移：源块与固定块的全块拼接后的最大 logit
        scale = 1.0 / (keys.shape[-1] ** 0.5)
        off = torch.maximum(
            ((q @ keys.T) * scale).max(),
            ((q @ fixed_keys.T) / (fixed_keys.shape[-1] ** 0.5)).max(),
        )
        m_fx, o_fx, _ = _block_state(q, fixed_keys, fixed_values, None, off)
        m_src, o_src, _ = _block_state(q, keys, values, None, off)
        m_cmp, o_cmp, _ = _block_state(q, compact.keys, compact.values, beta, off)

        dM = m_cmp - m_src
        eps_mass = dM.abs()
        eps_out = m_src * (o_cmp - o_src).norm()
        S_q = m_fx + m_src
        o_max = torch.maximum(o_fx.norm(), o_src.norm())

        dM_signed[i] = dM
        s_total[i] = S_q
        max_o[i] = o_max
        term_out[i] = eps_out / S_q
        term_mass[i] = 2.0 * eps_mass / S_q * o_max
        rhs[i] = term_out[i] + term_mass[i]

        y_c = mixture_output(
            q.unsqueeze(0), fixed_keys, fixed_values, compact=compact,
            beta_mode=beta_mode, num_repr_queries=num_repr_queries,
        )
        y_d = mixture_output(
            q.unsqueeze(0), fixed_keys, fixed_values, keys=keys, values=values,
        )
        lhs[i] = (y_c - y_d).norm()

    return {
        "lhs": lhs, "rhs": rhs,
        "term_out": term_out, "term_mass": term_mass,
        "dM_signed": dM_signed, "S": s_total, "max_O": max_o,
    }


def ratio_stats(
    lhs: torch.Tensor, rhs: torch.Tensor, s_total: torch.Tensor, dM: torch.Tensor,
    rel_tol: float = 1e-9,
) -> Dict[str, float]:
    """把逐 Query 量聚成可报的标量。

    容差是必需的，不是宽容
    ----------------------
    当 δM ≡ 0 时（例如 B = L_s、源块不裁剪），式(37) 的右侧与左侧
    **在解析上恒等**：RHS = ε_out/S = M·‖Ô − O‖/S = LHS。
    但两侧走的是不同的数值路径 —— LHS 经 `mixture_output`（online softmax
    逐块归并），RHS 由质量/输出范数直接算出。于是取等号处会有约 40%
    的 Query 因舍入出现 rhs 略小于 lhs。**那是浮点误差，不是界失效。**
    若不设容差，就会把"界恰好取到等号"误报成"界被推翻"。
    """
    valid = lhs > 0
    ratio = rhs[valid] / (lhs[valid] + 1e-300)
    lhs_v, rhs_v = lhs[valid], rhs[valid]
    strict_violation = rhs_v < lhs_v * (1.0 - rel_tol)
    at_equality = (rhs_v - lhs_v).abs() <= rel_tol * lhs_v
    precondition = dM.abs() < 0.5 * s_total  # Σ|δM| < S/2
    return {
        "n_queries": int(lhs.numel()),
        "lhs_median": float(lhs.median()),
        "lhs_mean": float(lhs.mean()),
        "rhs_median": float(rhs.median()),
        "rhs_mean": float(rhs.mean()),
        "ratio_median": float(ratio.median()) if ratio.numel() else float("nan"),
        "ratio_p5": float(ratio.quantile(0.05)) if ratio.numel() else float("nan"),
        "ratio_min": float(ratio.min()) if ratio.numel() else float("nan"),
        "ratio_max": float(ratio.max()) if ratio.numel() else float("nan"),
        "frac_rhs_ge_lhs": float((rhs >= lhs).to(torch.float64).mean()),
        "frac_rhs_ge_lhs_tol": float((~strict_violation).to(torch.float64).mean()),
        "frac_at_equality": float(at_equality.to(torch.float64).mean()),
        "n_violating": int((rhs < lhs).sum()),
        "n_violating_strict": int(strict_violation.sum()),
        "rel_tol": rel_tol,
        "lhs_zero_frac": float((~valid).to(torch.float64).mean()),
        "precondition_violation_frac": float((~precondition).to(torch.float64).mean()),
        "precondition_margin_median": float(
            (0.5 * s_total - dM.abs()).median()
        ),
    }


def term_shares(bundle: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """两项 RHS 各自的占比（先逐 Query 取份额，再取中位数）。"""
    t_out, t_mass, rhs = bundle["term_out"], bundle["term_mass"], bundle["rhs"]
    ok = rhs > 0
    if not bool(ok.any()):
        return {"share_out_median": float("nan"), "share_mass_median": float("nan")}
    return {
        "share_out_median": float((t_out[ok] / rhs[ok]).median()),
        "share_mass_median": float((t_mass[ok] / rhs[ok]).median()),
    }


# =============================================================================
# 主流程
# =============================================================================

def run(args) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        for strength in args.focus_strengths:
            scenario = S.make_scenario(
                L_s=args.L_s, d_h=args.d_h, d_v=args.d_v,
                num_dest=1, queries_per_dest=args.queries_per_dest,
                focus_strength=strength, seed=seed, dtype=args._dtype,
            )
            split = S.heldout_split(scenario, eval_fraction=args.eval_fraction)
            fit_q = split.fit_queries[0]
            eval_q = split.eval_queries[0]
            if args.num_repr_queries > fit_q.shape[0]:
                raise ValueError(
                    f"M={args.num_repr_queries} 超过 fit 池 {fit_q.shape[0]}"
                )
            fixed_keys, fixed_values = S.make_fixed_context(
                d_h=args.d_h, d_v=args.d_v, length=args.fixed_length,
                seed=args.fixed_seed + seed, dtype=args._dtype,
            )

            for beta_mode in args.beta_modes:
                # ---------- 主网格：预算 × β ----------
                for budget in args.budgets:
                    budget = min(budget, scenario.L_s)
                    compact = S.build_compact_kv(
                        source_keys=scenario.keys,
                        source_values=scenario.values,
                        destination_queries=fit_q,
                        budget=budget,
                        num_representative_queries=args.num_repr_queries,
                        projection_dim=args.projection_dim,
                        seed=seed,
                    )
                    bundle = bound_terms(
                        eval_q, compact, scenario.keys, scenario.values,
                        fixed_keys, fixed_values, beta_mode, args.num_repr_queries,
                    )
                    stats = ratio_stats(
                        bundle["lhs"], bundle["rhs"], bundle["S"],
                        bundle["dM_signed"], rel_tol=args.rel_tol,
                    )
                    row: Dict[str, Any] = {
                        "seed": seed, "focus_strength": strength,
                        "beta_mode": beta_mode, "budget": budget,
                        "compression_ratio": budget / scenario.L_s,
                        "n_blocks": 2, "fixed_length": args.fixed_length,
                        "M": args.num_repr_queries,
                        "n_eval_queries": int(eval_q.shape[0]),
                    }
                    row.update(stats)
                    row.update(term_shares(bundle))
                    rows.append(row)

    # ---------- 性质 3：块数轴（独立循环，避免与预算轴混淆） ----------
    # 设计要点（2026-09-16 修正）：**每块长度固定、总长随 N 增长**。
    # 若反过来固定总长 L_s 而让 chunk = L_s/N，则 N 越大每块越短，
    # 当 chunk 缩到 ≤ 预算时压缩比变成 1（δM ≡ 0），N 轴会被"压缩消失"
    # 污染成一条假的下行曲线 —— 那是度量设计的问题，不是性质 3 的结论。
    # 固定 chunk 后，各 N 下**每块的压缩强度完全相同**，唯一变量是块数。
    if args.block_chunk_len % 1:
        raise ValueError("block_chunk_len 必须是整数")
    if args.block_budget >= args.block_chunk_len:
        raise ValueError(
            f"block_budget={args.block_budget} 必须严格小于 "
            f"block_chunk_len={args.block_chunk_len}，否则每块无压缩、δM ≡ 0"
        )
    for seed in args.seeds:
        for strength in args.focus_strengths:
            for beta_mode in args.beta_modes:
                for n_blk in args.block_counts:
                    chunk = args.block_chunk_len
                    L_total = chunk * n_blk
                    scenario = S.make_scenario(
                        L_s=L_total, d_h=args.d_h, d_v=args.d_v,
                        num_dest=1, queries_per_dest=args.queries_per_dest,
                        focus_strength=strength, seed=seed, dtype=args._dtype,
                    )
                    split = S.heldout_split(
                        scenario, eval_fraction=args.eval_fraction
                    )
                    fit_q = split.fit_queries[0]
                    eval_q = split.eval_queries[0]
                    fixed_keys, fixed_values = S.make_fixed_context(
                        d_h=args.d_h, d_v=args.d_v, length=args.fixed_length,
                        seed=args.fixed_seed + seed, dtype=args._dtype,
                    )
                    compacts = []
                    for b in range(n_blk):
                        lo, hi = b * chunk, (b + 1) * chunk
                        compacts.append(S.build_compact_kv(
                            source_keys=scenario.keys[lo:hi],
                            source_values=scenario.values[lo:hi],
                            destination_queries=fit_q,
                            budget=args.block_budget,
                            num_representative_queries=args.num_repr_queries,
                            projection_dim=args.projection_dim,
                            seed=seed + b,
                        ))
                    stats = multiblock_stats(
                        eval_q, compacts, scenario, fixed_keys, fixed_values,
                        beta_mode, args.num_repr_queries, n_blk, chunk,
                    )
                    block_rows.append({
                        "seed": seed, "focus_strength": strength,
                        "beta_mode": beta_mode, "n_blocks": n_blk,
                        "chunk_len": chunk, "per_chunk_budget": args.block_budget,
                        "L_total": L_total, "M": args.num_repr_queries,
                    } | stats)

    return {
        "experiment": "E13",
        "precision": str(args._dtype).replace("torch.", ""),
        "config": {
            "L_s": args.L_s, "d_h": args.d_h, "d_v": args.d_v,
            "budgets": list(args.budgets),
            "focus_strengths": list(args.focus_strengths),
            "beta_modes": list(args.beta_modes),
            "fixed_length": args.fixed_length,
            "fixed_seed": args.fixed_seed,
            "block_counts": list(args.block_counts),
            "block_budget": args.block_budget,
            "block_chunk_len": args.block_chunk_len,
            "num_repr_queries": args.num_repr_queries,
            "projection_dim": args.projection_dim,
            "queries_per_dest": args.queries_per_dest,
            "eval_fraction": args.eval_fraction,
            "rel_tol": args.rel_tol,
            "seeds": list(args.seeds),
        },
        "bound_note": (
            "RHS = ε_out/S + 2·ε_mass/S·max_s‖O_s‖，LHS = ‖Âttn − Attn‖，"
            "按式(37) 逐 Query 计算。ε_mass、ε_out 是**跨块绝对量**"
            "（本脚本 2 个有效块：固定上下文块 + 压缩源块），"
            "与 E2/E5a 的单块口径不可互比。"
        ),
        "verdict_rule": (
            "界必须在**每一个** Query 上成立（frac_rhs_ge_lhs_tol = 1.0）；"
            "任一超出相对容差的 rhs < lhs 都是该解析命题的反例，不得用中位数掩盖。"
            "相对容差（默认 1e-9）只用于吸收「界恰好取等号」时的浮点舍入 —— "
            "此时两侧解析上恒等（δM ≡ 0 ⇒ RHS = ε_out/S = LHS），"
            "但 LHS 走逐块归并、RHS 走解析式，两条数值路径不同。"
            "取等率 frac_at_equality 单列，正是为了把这个情形与真反例分开。"
        ),
        "heldout_note": "紧凑 KV 用 fit Query 构造，评估只用不相交的 eval Query。",
        "caveat": (
            "合成场景（K/V ~ N(0,I)）下的解析命题自洽性检验，"
            "不涉及真实模型或任务指标。RHS/LHS 的绝对水平由合成分布决定，"
            "不得外推为真实模型上的紧致度。"
        ),
        "aggregated": aggregate(rows, args),
        "block_aggregated": aggregate_blocks(block_rows, args),
        "rows": rows,
        "block_rows": block_rows,
    }


def multiblock_stats(
    eval_q, compacts, scenario, fixed_keys, fixed_values,
    beta_mode, m, n_blk, chunk,
) -> Dict[str, float]:
    """N 个等长压缩块 + 1 个固定上下文块：LHS 与 Σ|δM| 随 N 的变化。"""
    from experiments.common.synthetic import beta_applied, mixture_output

    n = eval_q.shape[0]
    lhs = torch.zeros(n, dtype=torch.float64)
    sum_abs_dM = torch.zeros(n, dtype=torch.float64)
    net_dM = torch.zeros(n, dtype=torch.float64)
    s_tot = torch.zeros(n, dtype=torch.float64)

    for i in range(n):
        q = eval_q[i]
        scale = 1.0 / (scenario.d_h ** 0.5)
        l_fx = (q @ fixed_keys.T) / (fixed_keys.shape[-1] ** 0.5)
        off = l_fx.max()
        for b in range(n_blk):
            lo, hi = b * chunk, (b + 1) * chunk
            off = torch.maximum(off, ((q @ scenario.keys[lo:hi].T) * scale).max())

        m_fx = torch.exp(l_fx - off).sum()
        s_q = m_fx
        net = torch.zeros((), dtype=torch.float64)   # ΣδM（带符号）
        sabs = torch.zeros((), dtype=torch.float64)  # Σ|δM|
        for b in range(n_blk):
            lo, hi = b * chunk, (b + 1) * chunk
            k_b = scenario.keys[lo:hi]
            cp = compacts[b]
            beta = beta_applied(cp.logit_bias, beta_mode, m)
            m_full = torch.exp((q @ k_b.T) * scale - off).sum()
            m_cmp = torch.exp((q @ cp.keys.T) * scale + beta - off).sum()
            s_q = s_q + m_full
            net = net + (m_cmp - m_full)
            sabs = sabs + (m_cmp - m_full).abs()
        sum_abs_dM[i] = sabs
        net_dM[i] = net
        s_tot[i] = s_q

        # LHS：逐块归并（固定块 + N 个压缩块）vs（固定块 + N 个完整块）
        def merged(use_compact: bool):
            st = None
            from src.dcc_kv_ref import merge_softmax_states, online_softmax_from_attention
            st = online_softmax_from_attention(l_fx, fixed_values)
            for b in range(n_blk):
                lo, hi = b * chunk, (b + 1) * chunk
                k_b, v_b = scenario.keys[lo:hi], scenario.values[lo:hi]
                if use_compact:
                    beta = beta_applied(compacts[b].logit_bias, beta_mode, m)
                    l = (q @ compacts[b].keys.T) * scale + beta
                    st = merge_softmax_states(
                        st, online_softmax_from_attention(l, compacts[b].values)
                    )
                else:
                    st = merge_softmax_states(
                        st, online_softmax_from_attention((q @ k_b.T) * scale, v_b)
                    )
            return st.o / st.l

        lhs[i] = (merged(True) - merged(False)).norm()

    return {
        "lhs_median": float(lhs.median()),
        "lhs_mean": float(lhs.mean()),
        "sum_abs_dM_over_S_median": float((sum_abs_dM / s_tot).median()),
        "net_abs_dM_over_S_median": float((net_dM.abs() / s_tot).median()),
        "cancellation_ratio_median": float(
            (net_dM.abs() / (sum_abs_dM + 1e-300)).median()
        ),
        "S_median": float(s_tot.median()),
    }


def aggregate(rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    """按 (strength, beta_mode, budget) 跨种子聚成中位数与极差。"""
    out: Dict[str, Any] = {}
    fields = [
        "ratio_median", "ratio_p5", "ratio_min", "lhs_median", "rhs_median",
        "frac_rhs_ge_lhs", "frac_rhs_ge_lhs_tol", "frac_at_equality",
        "precondition_violation_frac",
        "share_out_median", "share_mass_median",
    ]
    for strength in args.focus_strengths:
        for beta_mode in args.beta_modes:
            entries = []
            for budget in sorted({r["budget"] for r in rows}):
                g = [
                    r for r in rows
                    if r["focus_strength"] == strength
                    and r["beta_mode"] == beta_mode and r["budget"] == budget
                ]
                if not g:
                    continue
                e = {"budget": budget, "n_seeds": len(g)}
                for f in fields:
                    vals = [r[f] for r in g if r.get(f) is not None]
                    e[f + "_med"] = float(torch.tensor(vals).median()) if vals else None
                    e[f + "_range"] = (max(vals) - min(vals)) if vals else None
                e["n_violating_total"] = int(sum(r["n_violating"] for r in g))
                e["n_violating_strict_total"] = int(
                    sum(r["n_violating_strict"] for r in g)
                )
                entries.append(e)
            out[f"strength={strength},beta={beta_mode}"] = entries
    return out


def aggregate_blocks(block_rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for strength in args.focus_strengths:
        for beta_mode in args.beta_modes:
            entries = []
            for nb in sorted({r["n_blocks"] for r in block_rows}):
                g = [
                    r for r in block_rows
                    if r["focus_strength"] == strength and r["beta_mode"] == beta_mode
                    and r["n_blocks"] == nb
                ]
                if not g:
                    continue
                med = lambda f: float(torch.tensor([r[f] for r in g]).median())
                entries.append({
                    "n_blocks": nb, "n_seeds": len(g),
                    "lhs_median": med("lhs_median"),
                    "lhs_range": max(r["lhs_median"] for r in g)
                    - min(r["lhs_median"] for r in g),
                    "sum_abs_dM_over_S_median": med("sum_abs_dM_over_S_median"),
                    "net_abs_dM_over_S_median": med("net_abs_dM_over_S_median"),
                    "cancellation_ratio_median": med("cancellation_ratio_median"),
                })
            out[f"strength={strength},beta={beta_mode}"] = entries
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E13：误差界式(37)的数值紧致度检验（CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--focus-strengths", dest="focus_strengths", type=float, nargs="+",
                   default=[0.0, 8.0])
    p.add_argument("--beta-modes", dest="beta_modes", type=str, nargs="+",
                   default=["none", "full"])
    p.add_argument("--block-counts", dest="block_counts", type=int, nargs="+",
                   default=[1, 2, 4, 8],
                   help="性质 3 的块数轴（每块独立压缩、每块预算固定）")
    p.add_argument("--block-budget", dest="block_budget", type=int, default=16,
                   help="每块预算（必须严格小于 --block-chunk-len）")
    p.add_argument("--block-chunk-len", dest="block_chunk_len", type=int, default=64,
                   help="每块长度（固定；总长 = 本值 × 块数）")
    p.add_argument("--fixed-length", dest="fixed_length", type=int, default=128)
    p.add_argument("--fixed-seed", dest="fixed_seed", type=int, default=7)
    p.add_argument("--L-s", dest="L_s", type=int, default=256)
    p.add_argument("--d-h", dest="d_h", type=int, default=32)
    p.add_argument("--d-v", dest="d_v", type=int, default=32)
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=96)
    p.add_argument("--num-repr-queries", dest="num_repr_queries", type=int, default=32)
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=32)
    p.add_argument("--eval-fraction", dest="eval_fraction", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                   help="网格型实验按 reproducibility.md §7 用 5 个种子")
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--rel-tol", dest="rel_tol", type=float, default=1e-9,
                   help="判「真反例」的相对容差；取等号时的浮点舍入不计为反例")
    p.add_argument("--out", type=str, default="results/cpu/e13")
    p.add_argument("--quick", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    support = S.compaction_dtype_support()
    args._dtype = S.dtype_from_name(args.dtype)
    if not support.get(args.dtype, False):
        print(f"错误：{args.dtype} 不可用（见 synthetic.DTYPE_BUG_*）。")
        print(f"  可用精度：{[k for k, v in support.items() if v]}")
        return 2

    if args.quick:
        args.budgets = [32]
        args.focus_strengths = [8.0]
        args.block_counts = [1, 4]
        args.block_chunk_len = 64
        args.block_budget = 16
        args.seeds = [42]
        args.L_s = 128
        args.queries_per_dest = 64
        args.num_repr_queries = 16

    print("=" * 78)
    print("E13：误差界式(37)的数值紧致度检验")
    print("=" * 78)
    print(f"  dtype={args.dtype}  L_s={args.L_s}  d_h={args.d_h}  "
          f"M={args.num_repr_queries}  固定块长={args.fixed_length}")
    print(f"  预算={args.budgets}  β={args.beta_modes}  块数={args.block_counts}  "
          f"种子={args.seeds}")
    print(f"  性质 3 轴：每块 {args.block_chunk_len} token、每块预算 "
          f"{args.block_budget}（总长随块数增长）")
    print()

    t0 = time.time()
    payload = run(args)

    print("=" * 78)
    print("A. 界是否成立（RHS/LHS）与前置条件")
    print("=" * 78)
    for key, entries in payload["aggregated"].items():
        print(f"  {key}")
        print(f"    {'B':>5} {'LHS中位':>11} {'RHS中位':>11} {'RHS/LHS':>19} "
              f"{'界成立率':>10} {'取等率':>8} {'前置违反率':>10} {'ε_out占比':>9}")
        for e in entries:
            print(
                f"    {e['budget']:>5} {e['lhs_median_med']:>11.4e} "
                f"{e['rhs_median_med']:>11.4e} "
                f"{e['ratio_median_med']:>8.3f}[{e['ratio_median_range']:.3f}] "
                f"{e['frac_rhs_ge_lhs_tol_med']:>10.3f} "
                f"{e['frac_at_equality_med']:>8.3f} "
                f"{e['precondition_violation_frac_med']:>10.3f} "
                f"{e['share_out_median_med']:>9.3f}"
            )
        tot = sum(e["n_violating_total"] for e in entries)
        strict = sum(e["n_violating_strict_total"] for e in entries)
        print(f"    → 浮点意义上 rhs<lhs：{tot}；"
              f"**超过相对容差 {args.rel_tol:g} 的真反例：{strict}**")

    print()
    print("=" * 78)
    print("B. 性质 3：误差是否随块数线性放大")
    print("=" * 78)
    for key, entries in payload["block_aggregated"].items():
        print(f"  {key}")
        print(f"    {'N块':>5} {'LHS中位':>13} {'Σ|δM|/S':>11} {'|ΣδM|/S':>11} "
              f"{'抵消比':>10}")
        for e in entries:
            print(
                f"    {e['n_blocks']:>5} {e['lhs_median']:>13.4e} "
                f"{e['sum_abs_dM_over_S_median']:>11.4f} "
                f"{e['net_abs_dM_over_S_median']:>11.4f} "
                f"{e['cancellation_ratio_median']:>10.3f}"
            )
        if len(entries) >= 2:
            first, last = entries[0], entries[-1]
            dn = last["n_blocks"] / first["n_blocks"]
            dl = (last["lhs_median"] / (first["lhs_median"] + 1e-300))
            print(f"    → N 乘 {dn:.0f} 倍时 LHS 乘 {dl:.3f} 倍"
                  f"{'（次线性，与性质 3 一致）' if dl < dn else '（达或超线性，需复核）'}")

    print()
    print("读取方式：")
    print("  1. 界必须在**每个** Query 上成立；反例数 > 0 即为解析命题的反例。")
    print("  2. 前置条件 Σ|δM| < S/2 违反的格子，界本就无保证，应单独看。")
    print("  3. RHS/LHS 越小界越紧；它衡量的是「界是否可用」，不是误差本身。")
    print("  4. 抵消比 = |ΣδM| / Σ|δM|，接近 1 表示误差同向叠加，接近 0 表示相互抵消。")
    print(f"\n  总耗时 {time.time() - t0:.0f}s")

    out_dir = REPO_ROOT / args.out
    R.save_json(str(out_dir / "e13_results.json"), payload)
    R.save_csv(str(out_dir / "e13_rows.csv"), payload["rows"])
    R.save_csv(str(out_dir / "e13_block_rows.csv"), payload["block_rows"])
    print(f"结果已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
