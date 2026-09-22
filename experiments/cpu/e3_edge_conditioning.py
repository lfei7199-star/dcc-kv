#!/usr/bin/env python
"""E3：边级条件化 vs 共享压缩（CPU 可跑，无需 GPU）

为什么这个脚本最重要
--------------------
论文 §6.3 的 E3 是全文唯一直接检验核心主张的 CPU 实验，但仓库现有的
`tests/test_dist_equivalence.py::test_dcc_kv_lower_error_than_shared`
**只断言两者的误差各自 < 0.5，并未断言 DCC-KV 更优**。因此论文里
E3 被如实标注为"仅量级验证，不构成 H2 证据"。

本脚本把这个缺口补上，做两件事：

1. **H1**：同一源块对不同目的端构造出的紧凑 KV 是否显著不同。
   判据为 KL 散度 > 0.5。
   关键设计——**噪声底线对照**：把同一目的端换随机种子重建一次，
   测其 KL。若跨目的端的 KL 不显著高于这个种子噪声底线，
   那么"边级条件化"的差异就只是随机性，不是条件化。这一步是
   区分"真实信号"与"度量本身的噪声"所必需的。

2. **H2**：DCC-KV 与共享压缩的**配对**比较。
   对同一（目的端, 预算）组合，两者面对同一份源块与同一组探针 Query，
   属于同源配对量，可做配对 bootstrap 与置换检验。

诚实边界
--------
本脚本输出的是**机制级**证据（注意力分布重构误差），不是任务质量。
即使 H2 显著，也**不能**据此主张 LongBench 等任务指标上的优势 ——
那需要 GPU 实验（见 `experiments/gpu/`）。合成数据上目的端关注带是
人为构造的，真实文本中目的端差异可能更小。

用法
----
    # 默认扫描（约数十秒）
    python experiments/cpu/e3_edge_conditioning.py --out results/cpu/e3

    # 只跑 H1
    python experiments/cpu/e3_edge_conditioning.py --only h1 --out results/cpu/e3

    # 快速自检
    python experiments/cpu/e3_edge_conditioning.py --quick
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import synthetic as S  # noqa: E402
from experiments.common import report as R  # noqa: E402
from experiments.common import hypotheses as H  # noqa: E402


# =============================================================================
# H1：边级条件化是否使紧凑 KV 真的不同
# =============================================================================

def _dest_query_source(scenario, dest, args):
    """构造侧该用的 Query 集合。

    留出协议（heldout，默认）下只用 **fit 池** —— 它必须先于评估集被切开，
    且评估集从未参与选键、β 拟合或 V 回归。这是本仓库的硬约束
    （见 `synthetic.HeldoutSplit` 的 docstring）。in-sample 仅供复现历史数字。
    """
    if args._protocol == "in-sample":
        return scenario.dest_queries[dest]
    return args._fit[dest]


def _shared_query_source(scenario, args):
    """共享压缩的构造侧 Query：DCC 侧与共享侧**用同一个池**、同样的 M。

    留出协议下取"各目的端 fit 池的并集"，于是两侧都看不到评估 Query，
    差异唯一的来源就是「是否按目的端条件化」，而不是「谁见过评估点」。
    """
    if args._protocol == "in-sample":
        return scenario.all_queries
    return torch.cat([args._fit[d] for d in sorted(args._fit)], dim=0)


def _build_dest(scenario, dest, budget, args, seed):
    """统一入口：保证 H1 与 H2 用同一套 M / d_p / Query 源配置。"""
    return S.build_compact_kv(
        source_keys=scenario.keys,
        source_values=scenario.values,
        destination_queries=_dest_query_source(scenario, dest, args),
        budget=budget,
        num_representative_queries=args.num_repr_queries,
        projection_dim=args.projection_dim,
        seed=seed,
    )


def _build_shared(scenario, budget, args, seed):
    """共享压缩基线：与 `_build_dest` 只差 Query 来源这一个自变量。"""
    return S.build_compact_kv(
        source_keys=scenario.keys,
        source_values=scenario.values,
        destination_queries=_shared_query_source(scenario, args),
        budget=budget,
        num_representative_queries=args.num_repr_queries,
        projection_dim=args.projection_dim,
        seed=seed,
    )


def _stage_protocol(scenario, args):
    """按协议切分 Query 池，并把结果挂到 args 上供后续调用取用。"""
    if args._protocol == "in-sample":
        args._fit = dict(scenario.dest_queries)      # 历史行为：不切分
        args._eval = dict(scenario.dest_queries)
        args._n_fit = int(next(iter(scenario.dest_queries.values())).shape[0])
        args._n_eval = args._n_fit
    else:
        split = S.heldout_split(scenario, eval_fraction=args.eval_fraction)
        args._fit = split.fit_queries
        args._eval = split.eval_queries
        args._n_fit = int(next(iter(split.fit_queries.values())).shape[0])
        args._n_eval = int(next(iter(split.eval_queries.values())).shape[0])
    if args.num_repr_queries > args._n_fit:
        raise ValueError(
            f"M={args.num_repr_queries} 超过构造池 {args._n_fit}；"
            f"请调大 --queries-per-dest（留出协议下需要 ≥ 2M）。"
        )


def measure_kl_between(
    scenario: S.SyntheticScenario,
    dest_a: int,
    dest_b: int,
    budget: int,
    args,
    seed: int = 42,
    probe_queries: torch.Tensor = None,
) -> Dict[str, float]:
    """测量两个目的端的紧凑 KV 在同一组探针 Query 上诱导分布的 KL / JS / 索引重叠。"""
    if probe_queries is None:
        probe_queries = scenario.all_queries

    ca = _build_dest(scenario, dest_a, budget, args, seed)
    cb = _build_dest(scenario, dest_b, budget, args, seed)

    pa = S.induced_distribution(ca, probe_queries, L_s=scenario.L_s)
    pb = S.induced_distribution(cb, probe_queries, L_s=scenario.L_s)

    # KL 不对称，取两个方向的平均作为对称化度量
    kl_ab = S.kl_divergence(pa, pb).mean().item()
    kl_ba = S.kl_divergence(pb, pa).mean().item()
    # JS 本身对称且有界，是"差异多大"更可靠的度量
    js = S.jensen_shannon_divergence(pa, pb).mean().item()

    return {
        "kl_mean": 0.5 * (kl_ab + kl_ba),
        "kl_ab": kl_ab,
        "kl_ba": kl_ba,
        "js_mean": js,
        "jaccard_overlap": S.selected_index_overlap(ca, cb),
    }


def measure_noise_floor(
    scenario: S.SyntheticScenario,
    dest: int,
    budget: int,
    args,
    seed_a: int = 42,
    seed_b: int = 43,
) -> Dict[str, float]:
    """种子噪声底线：同一目的端、不同随机种子重建。

    这是 H1 的必要对照。但有一个**前提容易被破坏**：
    `farthest_point_sampling` 在 `num_samples >= N` 时直接 `return arange(N)`，
    完全绕过种子。因此当 M >= 该目的端的 Query 数时，
    换种子不会改变任何东西，噪声底线退化为恒 0 —— 这会**高估** H1 的显著性。

    本函数因此强制要求 M < N，否则抛错（由调用方保证）。
    """
    n_dest_queries = _dest_query_source(scenario, dest, args).shape[0]
    if args.num_repr_queries >= n_dest_queries:
        raise ValueError(
            f"噪声底线退化：M={args.num_repr_queries} >= 目的端 Query 数 "
            f"{n_dest_queries}。此时 farthest_point_sampling 走 "
            f"`return torch.arange(N)` 分支，种子不起作用，底线恒为 0。"
            f"请把 --num-repr-queries 设小，或把 --queries-per-dest 设大。"
        )

    probe = scenario.all_queries
    ca = _build_dest(scenario, dest, budget, args, seed_a)
    cb = _build_dest(scenario, dest, budget, args, seed_b)
    pa = S.induced_distribution(ca, probe, L_s=scenario.L_s)
    pb = S.induced_distribution(cb, probe, L_s=scenario.L_s)
    return {
        "kl": 0.5 * (
            S.kl_divergence(pa, pb).mean().item()
            + S.kl_divergence(pb, pa).mean().item()
        ),
        "js": S.jensen_shannon_divergence(pa, pb).mean().item(),
        "jaccard": S.selected_index_overlap(ca, cb),
    }


def run_h1(args) -> List[Dict[str, Any]]:
    """H1 全扫描：目的端分离度 × 压缩预算 → KL / JS / Jaccard / 噪声底线。"""
    rows: List[Dict[str, Any]] = []
    ceiling = S.kl_saturation_ceiling(L_s=args.L_s)

    for strength in args.focus_strengths:
        scenario = S.make_scenario(
            L_s=args.L_s,
            d_h=args.d_h,
            d_v=args.d_v,
            num_dest=args.num_dest,
            queries_per_dest=args.queries_per_dest,
            focus_strength=strength,
            seed=args.seed,
            dtype=args._dtype,
        )
        _stage_protocol(scenario, args)
        # 探针必须来自**留出集**：若探针本身参与过代表 Query 的选择，
        # 跨目的端的 KL 会被"两边都见过这些点"抬高，H1 随之虚高。
        probe = torch.cat(
            [args._eval[d] for d in sorted(args._eval)], dim=0
        )

        for budget in args.budgets:
            kls: List[float] = []
            jss: List[float] = []
            jaccards: List[float] = []

            dests = sorted(scenario.dest_queries.keys())
            for i in range(len(dests)):
                for j in range(i + 1, len(dests)):
                    m = measure_kl_between(
                        scenario, dests[i], dests[j], budget,
                        args=args, seed=args.seed, probe_queries=probe,
                    )
                    kls.append(m["kl_mean"])
                    jss.append(m["js_mean"])
                    jaccards.append(m["jaccard_overlap"])

            floor = measure_noise_floor(scenario, dests[0], budget, args=args,
                                        seed_a=args.seed)

            kl_sum = R.summarize(kls, "kl_between_destinations", "nats", seed=args.seed)
            js_sum = R.summarize(jss, "js_between_destinations", "nats", seed=args.seed)
            jc_sum = R.summarize(jaccards, "jaccard_overlap", "ratio", seed=args.seed)

            rows.append({
                "protocol": args._protocol,
                "n_fit_queries": args._n_fit,
                "n_eval_queries": args._n_eval,
                "focus_strength": strength,
                "budget": budget,
                "n_pairs": len(kls),
                "kl_median": kl_sum.median,
                "kl_ci_lower": kl_sum.ci_95_lower,
                "kl_ci_upper": kl_sum.ci_95_upper,
                "kl_ceiling": ceiling,
                "kl_saturated": bool(kl_sum.median > 0.8 * ceiling),
                "js_median": js_sum.median,
                "js_ci_lower": js_sum.ci_95_lower,
                "js_ci_upper": js_sum.ci_95_upper,
                "js_max_possible": 0.6931,
                "noise_floor_kl": floor["kl"],
                "noise_floor_js": floor["js"],
                # 信号/噪声比：跨目的端差异相对种子噪声的倍数
                "kl_over_floor": (kl_sum.median / floor["kl"]) if floor["kl"] > 1e-12 else float("inf"),
                "js_over_floor": (js_sum.median / floor["js"]) if floor["js"] > 1e-12 else float("inf"),
                "jaccard_median": jc_sum.median,
                # H1 判据必须走单一事实源（experiments/common/hypotheses.py）。
                # 这里曾经把阈值 0.5 直接写死在比较里，绕开了阈值表 —— 由独立
                # 监督（2026-09-15）查出，见 docs/commit_log.md 第 22 条。
                "h1_criterion_met": H.h1_pass(kl_sum.ci_95_lower),
                "beats_sampling_floor": bool(js_sum.ci_95_lower > floor["js"]),
                # 无假设证据：两个目的端选中的 Key 集合是否显著不同
                "key_sets_distinct": bool(jc_sum.ci_95_upper < 0.9),
            })

            print(
                f"  strength={strength:<5} B={budget:<4} "
                f"KL={kl_sum.median:7.3f}(ceil≈{ceiling:.1f})  "
                f"JS={js_sum.median:.4f}  "
                f"floor: KL={floor['kl']:.3f} JS={floor['js']:.4f}  "
                f"Jaccard={jc_sum.median:.3f}  "
                f"{'✓H1' if rows[-1]['h1_criterion_met'] else ' ✗ '}"
                f"{'饱和' if rows[-1]['kl_saturated'] else '  '} "
                f"{'✓Key集不同' if rows[-1]['key_sets_distinct'] else '✗Key集相同'}"
            )

    return rows


# =============================================================================
# H2：DCC-KV 与共享压缩的配对比较
# =============================================================================

def run_h2(args) -> Dict[str, Any]:
    """H2：在同样预算下，边级条件化是否显著降低重构误差。

    配对结构：每个 (focus_strength, budget, dest, probe_query) 是一个配对单位 ——
    同一份源块、同一组探针 Query，唯一差别是紧凑 KV 用谁的目的端 Query 构造。

    注意 M（代表 Query 数）在两侧相同，因此差异只来自"条件化"本身，
    不来自可用 Query 数量的多少。

    ⚠️ 该局限已于 2026-09-16 修复（缺口 M9）
    ---------------------------------------
    独立监督（2026-09-15）查出：M 在两侧确实相同，但两侧**取 M 的池子不同源** ——
    DCC 侧从 `scenario.dest_queries[dest]`（也正是评估用的那一批 Query）里抽
    代表 Query，shared 侧从 `scenario.all_queries` 里抽，而评估同样在
    `q_dest` 上进行 ⇒ **DCC 侧存在样本内优势**：它见过评估点，shared 侧没有。
    于是两侧的 Δ 不是纯粹的「条件化」效应。

    现在默认 `--protocol heldout`：每个目的端的 Query 按 `--eval-fraction`
    切成互不相交的 fit / eval 两份，**两侧的构造都只用 fit 池，评估只用
    eval 池**，且两侧取 M 的池子规模相同。唯一自变量回到「是否按目的端
    条件化」。`--protocol in-sample` 保留历史行为，仅供复现旧数字对照。

    与 `docs/commit_log.md` 第 22 条的登记一致：**旧（in-sample）Δ 不得作为
    H2 的机制级证据写入论文**；本协议下的新数字才是。
    """
    detail_rows: List[Dict[str, Any]] = []
    per_condition: List[Dict[str, Any]] = []
    per_beta_mode: List[Dict[str, Any]] = []

    # 场景只与 focus_strength 有关，与 β 模式无关 —— 构造一次复用
    scenarios: Dict[float, S.SyntheticScenario] = {}
    for strength in args.focus_strengths:
        scenarios[strength] = S.make_scenario(
            L_s=args.L_s,
            d_h=args.d_h,
            d_v=args.d_v,
            num_dest=args.num_dest,
            queries_per_dest=args.queries_per_dest,
            focus_strength=strength,
            seed=args.seed,
            dtype=args._dtype,
        )

    # 紧凑 KV 本身也与 β 模式无关（β 是在使用阶段施加的），故只构造一次
    compacts: Dict[Any, Any] = {}
    for strength, scenario in scenarios.items():
        _stage_protocol(scenario, args)   # 切池必须先于任何构造
        for budget in args.budgets:
            compacts[(strength, budget, "shared")] = _build_shared(
                scenario, budget=budget, args=args, seed=args.seed,
            )
            for dest in sorted(scenario.dest_queries.keys()):
                compacts[(strength, budget, dest)] = _build_dest(
                    scenario, dest, budget, args, args.seed
                )

    for beta_mode in args.beta_modes:
        mode_dcc: List[float] = []
        mode_shared: List[float] = []
        mode_conditions: List[Dict[str, Any]] = []

        print(f"  --- β 模式 = {beta_mode} ---")

        for strength in args.focus_strengths:
            scenario = scenarios[strength]
            # ⚠️ 必须在这里重新切池：args._eval 是在上面的"构造"循环里逐场景
            # 覆盖的，构造循环跑完后它只保留**最后一个场景**的 Query。
            # 若评估循环直接读 args._eval，除最后一个 strength 外全部读错，
            # 而错误方向是"安静地给出看似合理的数字"（实测过）。
            _stage_protocol(scenario, args)

            for budget in args.budgets:
                shared = compacts[(strength, budget, "shared")]
                cond_dcc: List[float] = []
                cond_shared: List[float] = []

                for dest in sorted(scenario.dest_queries.keys()):
                    # 评估只用留出集；构造用的是它的补集（见 _dest_query_source）
                    q_dest = args._eval[dest]
                    compact_dcc = compacts[(strength, budget, dest)]

                    # 逐 query 的相对输出误差（避免 d_v 量纲影响可比性）
                    e_dcc = S.relative_output_error(
                        q_dest, compact_dcc, scenario.keys, scenario.values,
                        beta_mode=beta_mode,
                        num_repr_queries=args.num_repr_queries,
                    )
                    e_shared = S.relative_output_error(
                        q_dest, shared, scenario.keys, scenario.values,
                        beta_mode=beta_mode,
                        num_repr_queries=args.num_repr_queries,
                    )

                    cond_dcc.extend(e_dcc.tolist())
                    cond_shared.extend(e_shared.tolist())

                    detail_rows.append({
                        "protocol": args._protocol,
                        "n_fit_queries": args._n_fit,
                        "n_eval_queries": args._n_eval,
                        "beta_mode": beta_mode,
                        "focus_strength": strength,
                        "budget": budget,
                        "dest": dest,
                        "dcc_mean_rel_err": e_dcc.mean().item(),
                        "shared_mean_rel_err": e_shared.mean().item(),
                        "delta_shared_minus_dcc": e_shared.mean().item() - e_dcc.mean().item(),
                    })

                # --- 关键：逐条件做配对检验，而不是一股脑汇总 ---
                # 不同 focus_strength / beta_mode 下效应方向可能相反。
                # 混在一起做汇总检验会产生辛普森式误读。
                cond_paired = R.paired_bootstrap(
                    cond_dcc, cond_shared,
                    metric_name="relative_output_error",
                    unit="ratio",
                    higher_is_better=False,
                    seed=args.seed,
                )
                entry = {
                    "protocol": args._protocol,
                    "beta_mode": beta_mode,
                    "focus_strength": strength,
                    "budget": budget,
                    "n_pairs": cond_paired.n_pairs,
                    "dcc_mean": cond_paired.mean_a,
                    "shared_mean": cond_paired.mean_b,
                    "mean_diff": cond_paired.mean_diff,
                    "ci_95_lower": cond_paired.ci_95_lower,
                    "ci_95_upper": cond_paired.ci_95_upper,
                    "p_value_one_sided": cond_paired.p_value_one_sided,
                    "dcc_wins": cond_paired.a_wins,
                    "shared_wins": cond_paired.b_wins,
                    "verdict": cond_paired.verdict(),
                }
                per_condition.append(entry)
                mode_conditions.append(entry)

                mode_dcc.extend(cond_dcc)
                mode_shared.extend(cond_shared)

                print(
                    f"    strength={strength:<5} B={budget:<4} "
                    f"DCC-KV={cond_paired.mean_a:.6f}  "
                    f"Shared={cond_paired.mean_b:.6f}  "
                    f"Δ={cond_paired.mean_diff:+.6f}  "
                    f"p={cond_paired.p_value_one_sided:.4f}  "
                    f"{'DCC' if cond_paired.mean_diff < 0 else 'Shr'} 优"
                )

        # 该 β 模式下的汇总
        pooled = R.paired_bootstrap(
            mode_dcc, mode_shared,
            metric_name="relative_output_error",
            unit="ratio",
            higher_is_better=False,
            seed=args.seed,
        )

        signs = [r["mean_diff"] for r in mode_conditions]
        n_favor_dcc = sum(s < 0 for s in signs)
        consistent = n_favor_dcc in (0, len(signs))

        per_beta_mode.append({
            "beta_mode": beta_mode,
            "paired_pooled": pooled.to_dict(),
            "pooled_sign_consistent": consistent,
            "n_conditions": len(mode_conditions),
            "n_conditions_favoring_dcc": n_favor_dcc,
            "mean_dcc": pooled.mean_a,
            "mean_shared": pooled.mean_b,
            "mean_diff": pooled.mean_diff,
            "p_value_one_sided": pooled.p_value_one_sided,
        })
        print(f"    [汇总] Δ={pooled.mean_diff:+.6f} "
              f"CI95=[{pooled.ci_95_lower:+.6f},{pooled.ci_95_upper:+.6f}] "
              f"p={pooled.p_value_one_sided:.6f}  "
              f"方向一致={'是' if consistent else '否'}")
        print()

    # --- 跨 β 模式的对比：方向是否随 β 约定翻转 ---
    diffs_by_mode = {r["beta_mode"]: r["mean_diff"] for r in per_beta_mode}
    signs_all = [d < 0 for d in diffs_by_mode.values()]
    beta_sensitive = len(set(signs_all)) > 1

    if beta_sensitive:
        beta_note = (
            f"⚠️ 结论对 β 的使用方式敏感：各模式下的汇总差异为 "
            f"{ {k: round(v, 6) for k, v in diffs_by_mode.items()} }，"
            f"方向不一致。这说明 E3 的结论目前主要由『β 在拟合侧除以 M、"
            f"在推理侧不除』这一训练/推理不一致所主导，"
            f"而不是由『边级条件化』本身决定。"
            f"在修复 β 约定之前，H2 无法被可靠检验。"
        )
    else:
        beta_note = (
            f"结论方向对 β 的使用方式不敏感（各模式汇总差异 "
            f"{ {k: round(v, 6) for k, v in diffs_by_mode.items()} }），"
            f"因此 E3 的结论不是 β 约定造成的伪影。"
        )

    return {
        "per_beta_mode": per_beta_mode,
        "beta_sensitivity_note": beta_note,
        "beta_sensitive": beta_sensitive,
        "per_condition": per_condition,
        "n_total_pairs": len(detail_rows),
        "detail": detail_rows,
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E3：边级条件化 vs 共享压缩（CPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--only", choices=["h1", "h2", "both"], default="both")
    p.add_argument("--focus-strengths", type=float, nargs="+",
                   default=[0.0, 2.0, 4.0, 8.0, 16.0],
                   help="目的端关注带的分离强度；0 表示各目的端无差异（负对照）")
    p.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 128],
                   help="压缩预算 B")
    p.add_argument("--L-s", dest="L_s", type=int, default=256, help="源块长度")
    p.add_argument("--d-h", dest="d_h", type=int, default=32, help="head 维度")
    p.add_argument("--d-v", dest="d_v", type=int, default=32, help="value 维度")
    p.add_argument("--num-dest", dest="num_dest", type=int, default=4,
                   help="目的端数量")
    p.add_argument("--protocol", type=str, default="heldout",
                   choices=["heldout", "in-sample"],
                   help="heldout：构造只用 fit 池、评估只用不相交的 eval 池（默认）；"
                        "in-sample：历史行为（评估 Query 参与构造），仅供对照复现")
    p.add_argument("--eval-fraction", dest="eval_fraction", type=float, default=0.5,
                   help="划给评估集的 Query 比例（仅 heldout 协议生效）")
    p.add_argument("--queries-per-dest", dest="queries_per_dest", type=int, default=96,
                   help="每个目的端的 Query 数。留出协议下需要 ≥ 2M："
                        "切分后 fit 池必须容得下 M 个代表 Query")
    p.add_argument("--num-repr-queries", dest="num_repr_queries", type=int, default=32,
                   help="代表 Query 数 M（两侧相同，保证差异只来自条件化）")
    p.add_argument("--projection-dim", dest="projection_dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta-modes", dest="beta_modes", type=str, nargs="+",
                   default=["full", "over_m", "none"],
                   help="β 的使用方式扫描（H2）。"
                        "full=推理路径现状（全量 β）；over_m=拟合时的标定（β/M）；"
                        "none=移除 β（A3 消融）。三者对比可定位训练/推理不一致的影响。")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float64"],
                   help="张量精度。float64 目前被仓库的一个 dtype 缺陷阻塞，"
                        "见 synthetic.compaction_dtype_support()")
    p.add_argument("--out", type=str, default="results/cpu/e3",
                   help="输出目录（相对仓库根）")
    p.add_argument("--quick", action="store_true",
                   help="快速自检：小规模、少量组合")
    return p


def main() -> int:
    args = build_parser().parse_args()
    # 内部别名：_stage_protocol / _dest_query_source 读 _protocol / _fit / _eval，
    # 一律带下划线以便被下方的 cfg 过滤掉（torch.dtype 与张量不可 JSON 序列化）。
    args._protocol = args.protocol
    args._fit = {}
    args._eval = {}
    args._n_fit = 0
    args._n_eval = 0

    # --- 精度能力探测：在跑之前暴露环境限制，而不是抛 RuntimeError ---
    support = S.compaction_dtype_support()
    args._dtype = S.dtype_from_name(args.dtype)
    if not support.get(args.dtype, False):
        print("!" * 78)
        print(f"错误：compact KV 构造链路不支持 {args.dtype}。")
        print()
        print("根本原因（已在 experiments/common/synthetic.py 中记录）：")
        print(f"  {S.DTYPE_BUG_FILE} 第 {S.DTYPE_BUG_LINE} 行")
        print("  rademacher_projection 写死 `.float()`，不保留输入 dtype，")
        print("  导致 float64 输入在 `queries @ proj.T` 处报 dtype 不匹配。")
        print()
        print("影响：论文 §6.3 的 E2 规格为 FP64，修复前无法按该规格运行。")
        print("修法（一行）：把 `.float()` 换成 `.to(queries.dtype)`。")
        print()
        print(f"当前可用精度：{ [k for k, v in support.items() if v] }")
        print("!" * 78)
        return 2

    if args.quick:
        args.focus_strengths = [0.0, 8.0]
        args.budgets = [16, 64]
        args.L_s = 128
        args.num_dest = 3
        args.queries_per_dest = 48
        args.num_repr_queries = 16

    print("=" * 78)
    print("E3：边级条件化 vs 共享压缩（CPU，合成数据）")
    print("=" * 78)
    print(f"  精度 dtype={args.dtype}  （可用: "
          f"{[k for k, v in support.items() if v]}）")
    print(f"  L_s={args.L_s}  d_h={args.d_h}  d_v={args.d_v}  "
          f"num_dest={args.num_dest}  M={args.num_repr_queries}")
    print(f"  预算扫描 B={args.budgets}")
    print(f"  目的端分离度扫描={args.focus_strengths}（0.0 为负对照）")
    print(f"  Query 协议={args.protocol}"
          + (f"（fit/eval 按 {args.eval_fraction:g}/{1 - args.eval_fraction:g} 切分）"
             if args.protocol == "heldout" else "（⚠️ 历史行为：评估 Query 参与构造）"))
    print()

    # 只保留公开参数（去掉内部解析出的 _dtype，它是 torch.dtype 不可 JSON 序列化）
    cfg = {k: v for k, v in vars(args).items() if not k.startswith("_")}

    payload: Dict[str, Any] = {
        "experiment": "E3",
        "environment": "cpu",
        "config": cfg,
        "caveats": [
            "机制级证据（注意力分布重构误差），不是任务质量指标。",
            "合成数据的关注带为人为构造，真实文本中目的端差异可能更小。",
            "H1 判据 KL > 0.5 来自 docs/reproducibility.md；本脚本同时报告噪声底线。",
            "H2 显著也不能主张任务指标优势，那需要 experiments/gpu/ 的实验。",
            "留出协议（--protocol heldout，默认）下构造只用 fit 池、评估只用 "
            "互不相交的 eval 池；旧 in-sample 数字含 DCC 侧的样本内优势，"
            "不得作为 H2 的机制级证据（commit_log 第 22 条、缺口 M9）。",
        ],
        "estimator_source": "Attention Matching（非本文原创）",
        "protocol": args.protocol,
        "eval_fraction": args.eval_fraction if args.protocol == "heldout" else None,
    }

    if args.only in ("h1", "both"):
        print("-" * 78)
        print("H1：同一源块对不同目的端的紧凑 KV 是否显著不同")
        print("    判据 KL > 0.5；同时要求 KL 显著高于同目的端的种子噪声底线")
        print("-" * 78)
        h1_rows = run_h1(args)
        payload["h1"] = h1_rows

        n_pass = sum(r["h1_criterion_met"] for r in h1_rows)
        n_above_floor = sum(r["beats_sampling_floor"] for r in h1_rows)
        n_distinct = sum(r["key_sets_distinct"] for r in h1_rows)
        n_sat = sum(r["kl_saturated"] for r in h1_rows)
        print()
        print(f"  KL > 0.5 满足:            {n_pass}/{len(h1_rows)} 组合")
        print(f"  JS 高于种子噪声底线:       {n_above_floor}/{len(h1_rows)} 组合")
        print(f"  选中 Key 集合显著不同:     {n_distinct}/{len(h1_rows)} 组合")
        print(f"  KL 已饱和(> 0.8×上限):     {n_sat}/{len(h1_rows)} 组合")
        print()
        if n_sat > 0:
            print("  ⚠️  KL 饱和警告：")
            print(f"     KL 的平滑上限约为 {h1_rows[0]['kl_ceiling']:.1f}。当两个紧凑 KV 选中的")
            print("     Key 集合几乎不相交时（B ≪ L_s 下的常态），KL 会被均匀平滑系数")
            print("     支配而顶到上限，从而无法区分'略有不同'与'完全不同'。")
            print("     因此『H1 满足』这个结论的价值有限 —— 它只能排除'机制完全")
            print("     忽略目的端'，不能证明条件化的程度。")
            print("     建议论文把 H1 改为以 JS 散度或有界的 Key 集合重叠度表述。")
            print()
        print("  说明：strength=0 不是「无差异」的零假设，而是「各目的端仅有采样噪声」")
        print("  差异」。真正的零假设是「无条件化」（共享压缩），其对比见下节 H2。")
        print()

    if args.only in ("h2", "both"):
        print("-" * 78)
        print("H2：DCC-KV(逐边条件化) vs 共享压缩 —— 配对比较")
        print("    同时扫描 β 的三种使用方式，以检验结论是否由 β 约定主导")
        print("-" * 78)
        h2 = run_h2(args)
        print("-" * 78)
        print("逐条件明细（以这一层为准，汇总仅作参考）：")
        for r in h2["per_condition"]:
            who = "DCC-KV" if r["mean_diff"] < 0 else "共享压缩"
            sig = "显著" if r["p_value_one_sided"] < 0.05 else "不显著"
            print(f"    β={r['beta_mode']:<7} strength={r['focus_strength']:<5} "
                  f"B={r['budget']:<4} Δ={r['mean_diff']:+.6f} "
                  f"p={r['p_value_one_sided']:.3f} {sig} → {who}优")
        print()
        print(f"  {h2['beta_sensitivity_note']}")
        print()
        payload["h2"] = h2

    out_dir = REPO_ROOT / args.out
    R.save_summary(str(out_dir / "e3_results.json"), payload)
    if "h1" in payload:
        R.save_csv(str(out_dir / "e3_h1.csv"), payload["h1"])
    if "h2" in payload:
        R.save_csv(str(out_dir / "e3_h2_detail.csv"), payload["h2"]["detail"])

    print("=" * 78)
    print(f"结果已写入 {out_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
