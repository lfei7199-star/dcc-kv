#!/usr/bin/env python
"""E6：主表与可扩展性 —— GPU 端。

论文 §6.4 现把 E6 描述为
    "3 模型 × 4 上下文长度 × 2 GPU 数 × 2 同步模式 × 3 基线 = 144 个数据点"

算术自洽（3 × 4 × 2 × 2 × 3 = 144）。

（历史）论文原稿写的是"5 上下文长度 = 144 个数据点"，与 180 对不上；
`72af7db` 已把长度档位改为 4 档修掉。本脚本曾把这条不一致硬编码成警告，
现在改为**按 `--context-lengths` 的实参现算**数据点数并打印，不再复述
一个已经过期、且与论文当前状态相反的结论。

本脚本另一个必须明确的点：**GPU 数不是所有方法都能调的自变量。**
    dense / kv_budget_shared —— 单卡测量。它们不涉及跨设备通信，
                               把它们的 "2 卡" 数字填进主表等于编造。
    dcc_kv / ring / apb / fastkv_official —— 才真正有 2 卡 / 4 卡 的维度，
                               而这四个当前都没有可用的 GPU 实现（见 --plan）。
因此本脚本把 gpu_count 记为**方法的要求**而非自由轴，并在结果里显式标注
`gpu_count_observed` 与 `gpu_count_required`。

同一个道理适用于**同步模式轴**。sync / async 描述的是跨设备通信的调度方式，
对单卡方法没有意义：若仍为 dense / kv_budget_shared 各生成 sync 与 async 两行，
这两行必然**逐位相同**，而"两行数字一模一样"在表里会被读成"异步没有收益" ——
那是把「这条轴不存在」错当成「这条轴上的测量结果为零」。
因此本脚本对这类方法**折叠**该轴：只生成一行，`sync_async` 记为 `n/a`；
`n_points_planned` 也按每个方法各自的轴累加，而不是用「方法数 × 同步模式数」一把乘。

用法
----
    # 任何机器上都能跑：打印完整网格与每个方法的前置条件
    python experiments/gpu/e6_main_table.py --plan

    # 真实测量（需要 GPU + 模型权重）
    python experiments/gpu/e6_main_table.py \
        --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct \
        --context-lengths 4096 8192 16384 32768 \
        --methods dense kv_budget_shared \
        --eval-file data/longbench_qa.jsonl --out results/gpu/e6
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import report as R      # noqa: E402
from experiments.gpu import _env, _hf            # noqa: E402

SCRIPT = "experiments/gpu/e6_main_table.py"

# H2 的阈值、前提判定与跨长度聚合规则**只写在 experiments/common/hypotheses.py**。
# 本文件不得再出现裸阈值（tests/test_hypotheses.py 有源码级断言守这条）。
from experiments.common import hypotheses as H  # noqa: E402

# 每个方法的前置条件。blocked 的项在这里给出**具体缺什么**，
# 而不是一句"待实现"。
METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "dense": {
        "measurable": True,
        "gpu_required": 1,
        "desc": "精确注意力（不压缩）—— 精度上界与性能参照",
        "impl": "HF 模型 + full KV cache",
    },
    "kv_budget_shared": {
        "measurable": True,
        "gpu_required": 1,
        "desc": "共享预算裁剪（按 Key 能量保留 top-B 位置，所有 Query 共用）",
        "impl": "_hf.apply_kv_budget(mode='topk_rms')",
        "caveat": (
            "**这不是 FastKV**。FastKV 是「用共享目的端构造一份紧凑 KV」"
            "（含 β 与 Value 回归）；本项是「直接裁剪缓存」，没有构造链路。"
            "它给出的是目的端无关族在同等预算下的参考曲线，"
            "不能用它声称「已与 FastKV 对比」。"
        ),
    },
    "fastkv_official": {
        "measurable": False,
        "gpu_required": 2,
        "desc": "FastKV 官方实现（共享压缩）",
        "blockers": [
            "（原 blocker「无 GPU kernel」已关闭：G3 的 src/baselines/operators.py "
            "给出 device-agnostic 的 fastkv_attention，复用 G1 的紧凑 KV 核。）",
            "（原 blocker「E6 的注意力钩子未接」已关闭：2026-09-21 的 H0 已接 —— "
            "measure_prefill 的压缩臂新增目的端前向（dest_len），裁剪后的 KV "
            "现在真的进入计时窗口。）",
            "仍需：把该算子接进 measure_point 的**方法行**。当前 dcc_kv / "
            "fastkv_official / apb 三行都还没走各自的真实链路。",
        ],
    },
    "dcc_kv": {
        "measurable": False,
        "gpu_required": 2,
        "desc": "DCC-KV（逐边条件化 + 异步 All-to-Allv）",
        "blockers": [
            "（原 blocker「CompactKV → GPU attention kernel 缺失」已关闭："
            "src/dcc_kv_ref/attention_kernel.py 提供 compact_kv_attention。"
            "原 blocker「异步流水无 GPU 入口」亦已关闭："
            "experiments/gpu/_forward.py 的 pipelined_attention。）",
            "（原 blocker「E6 注意力钩子未接」亦已关闭：2026-09-21 H0 —— "
            "measure_prefill 的压缩臂含目的端前向，裁剪后的 KV 真的进计时窗口。）",
            "**仍未接：方法实现本身。** measure_point 对 dcc_kv 与 "
            "kv_budget_shared 走的是**同一条** apply_kv_budget（共享裁剪）路径 —— "
            "没有目的端条件化构造，也没有 G2 的 pipelined_attention。二者质量差"
            "会恒为 0，H2 的 quality_gain 无从谈起。翻 measurable 之前必须让 "
            "dcc_kv 行改走 build_compact_kv + _forward.pipelined_attention 的真实链路。"
            "（2026-09-21 补：所需**工具已就绪** —— src/distributed/attention_hook.py "
            "的 dcc_attention 上下文管理器 + HookConfig(dcc_world=...)，本机 tiny Llama "
            "端到端验证到 ULP 级；尚未接进本脚本的方法行，E6 也还没有 --dcc-world 参数。）",
            "构造链路的 CUDA 可用性待 GPU 机实测（G6 未做；静态审计不能替代实测）。",
        ],
    },
    "ring": {
        "measurable": False,
        "gpu_required": 2,
        "desc": "Ring Attention（精确序列并行）",
        "blockers": [
            "（原 blocker 部分关闭：G3 的 operators.ring_attention 已是"
            "device-agnostic 实现。）但它复用 G1 的紧凑核、在**单进程内**模拟 ring，"
            "不是 NCCL P2P ring；跨设备 ring 通信仍无实现，这是它与 dcc_kv 的"
            "本质差别，不能拿单进程版当「Ring Attention 的 GPU 实现」。",
        ],
    },
    "apb": {
        "measurable": False,
        "gpu_required": 2,
        "desc": "APB（全网共享 anchor）",
        "blockers": [
            "（原 blocker「无 GPU kernel」已关闭：operators.apb_attention 已"
            "device-agnostic 化。）",
            "仍需：把 apb_attention 接进 measure_point 的方法行（同 dcc_kv）—— "
            "H0 只解决了量具，没有解决方法实现。",
            "（原 blocker「APB 编号 2502.12085 待二次确认」已关闭：编号有效。）",
        ],
    },
}

# sync/async 轴对不需要该轴的方法的占位取值。刻意不用 "sync" 顶上：
# 那会被下游读成一个真实的测量条件，而不是"该轴不适用"。
SYNC_MODE_NA = "n/a"


def sync_axis_applies(method: str) -> bool:
    """sync / async 是否是该方法的自变量。

    判据是**该方法是否存在跨设备通信**，即它是否要求多卡
    （`METHOD_SPECS[...]["gpu_required"] > 1`）。单卡方法没有可异步重叠的通信，
    该轴对它不构成自变量 —— 给它生成两行只会得到两行逐位相同的数。

    从 `gpu_required` 推导而不是写死 False：写死会在 dcc_kv 的 GPU 实现就绪后
    仍把它的异步行标成"该轴不适用"，把"还没测"固化成"没这条轴"。
    """
    return METHOD_SPECS[method]["gpu_required"] > 1


# =============================================================================
# 计划打印（无需 GPU）
# =============================================================================

def planned_points(a: argparse.Namespace) -> int:
    """按每个方法各自的轴累加计划点数。

    单卡方法折叠 sync/async 轴（贡献 1 而不是 len(sync_modes)）。若照旧用
    `n_models * n_ctx * n_sync * n_methods` 一把乘，就把"永远不会被生成的数据点"
    算进了计划数，计划数与实测数从此对不上，而差额会被误读成"漏跑"。
    """
    total = 0
    for m in a.methods:
        mult = len(a.sync_modes) if sync_axis_applies(m) else 1
        total += len(a.models) * len(a.context_lengths) * mult
    return total


def print_plan(a: argparse.Namespace) -> int:
    n_models = len(a.models)
    n_ctx = len(a.context_lengths)
    n_sync = len(a.sync_modes)
    n_methods = len(a.methods)
    n_points = planned_points(a)
    expanded = [m for m in a.methods if sync_axis_applies(m)]
    collapsed = [m for m in a.methods if not sync_axis_applies(m)]

    print("=" * 78)
    print("E6 网格计划")
    print("=" * 78)
    print(f"  模型      ({n_models})：{a.models}")
    print(f"  上下文长度 ({n_ctx})：{a.context_lengths}")
    print(f"  同步模式   ({n_sync})：{a.sync_modes}")
    print(f"  方法      ({n_methods})：{a.methods}")
    print(f"  => 数据点 = {n_points}（按各方法自身的轴累加，不是简单相乘）")
    print(f"     sync/async 轴展开的方法（{len(expanded)}）：{expanded}")
    if collapsed:
        overcount = n_models * n_ctx * (n_sync - 1) * len(collapsed)
        print(f"     sync/async 轴折叠的方法（{len(collapsed)}）：{collapsed}")
        print(f"       —— 单卡方法，无跨设备通信，该轴不是自变量；"
              f"若按简单相乘会虚增 {overcount} 个永不生成的数据点")
    if n_methods != 3:
        print(f"  ⚠ 注意：方法数是 {n_methods}，而论文的 144 是按「3 基线」算的。"
              f"折叠 sync/async 轴后本脚本算出 {n_points} 点，"
              f"与论文的 144 既不同值也不同义 —— 不要把本脚本的总数当成"
              f"论文声称的数据点数。（旧版不折叠轴时数字曾凑到 144，纯属巧合。）")
    print(f"  每点 ≥{a.iters} 次 run（报告规范要求）"
          f" => 总前向次数 ≥ {n_points * a.iters}")

    print()
    print("  论文 §6.4 现写「3 模型 × 4 上下文长度 × 2 GPU × 2 同步 × 3 基线 = 144」，")
    print("  算术自洽（3 × 4 × 2 × 2 × 3 = 144）。")
    print("  （历史）原稿写「5 上下文长度 = 144」，与 180 对不上；`72af7db` 已改为 4 档。")
    print("  下面按本次实参现算，不复述过期结论。")

    print()
    print("  方法前置条件：")
    print("  " + "-" * 74)
    for m in a.methods:
        spec = METHOD_SPECS[m]
        tag = "可测量" if spec["measurable"] else "被阻断"
        print(f"  [{tag}] {m}  (需 {spec['gpu_required']} 卡)")
        print(f"           {spec['desc']}")
        if spec.get("impl"):
            print(f"           实现：{spec['impl']}")
        for b in spec.get("blockers", []):
            print(f"           ✗ {b}")
        if spec.get("caveat"):
            print(f"           ! {spec['caveat']}")
    print("  " + "-" * 74)
    blocked = [m for m in a.methods if not METHOD_SPECS[m]["measurable"]]
    print(f"  本次计划中 {len(blocked)}/{n_methods} 个方法会被阻断并如实记账：{blocked}")
    print("=" * 78)
    return 0


# =============================================================================
# 测量
# =============================================================================

def measure_point(
    a: argparse.Namespace,
    model: str,
    ctx_len: int,
    sync_mode: str,
    method: str,
    lm: _hf.LoadedModel,
    samples: Sequence[_hf.EvalSample],
) -> Dict[str, Any]:
    """测量主表的一个数据点。"""
    spec = METHOD_SPECS[method]
    budget_ratio = None if method == "dense" else a.budget_ratio
    mode = "identity" if method == "dense" else a.compaction_mode

    # H0：目的端 query 长度。两臂必须取同一个值 —— 加速比的分子与分母要来自
    # 同一组 run（论文 §7.5 的「配对性」），否则比的不是同一件事。
    dest_len = max(1, int(round(a.dest_fraction * ctx_len)))

    pre = _hf.measure_prefill(
        lm, seq_len=ctx_len, batch_size=a.batch_size,
        warmup=a.warmup, iters=a.iters,
        budget_ratio=budget_ratio, compaction_mode=mode,
        dest_len=dest_len,
    )
    ps = R.summarize(pre["prefill_ms_samples"], "prefill", "ms", seed=a.seed)

    acc: Optional[float] = None
    acc_by_task: Optional[Dict[str, float]] = None
    n_eval = 0
    if samples:
        ev = _hf.evaluate(lm, samples, budget_ratio=budget_ratio,
                          compaction_mode=mode,
                          max_prompt_tokens=a.max_prompt_tokens)
        acc = ev["accuracy"]
        acc_by_task = ev["by_task"]
        n_eval = ev["n"]

    kv_ratio = 1.0 if method == "dense" else (
        budget_ratio if budget_ratio is not None else 1.0)
    keep = min(1.0, max(0.0, a.budget_ratio if budget_ratio is not None else 1.0))

    return {
        "model": model,
        "context_length": ctx_len,
        "gpu_count_observed": 1,
        "gpu_count_required": spec["gpu_required"],
        "gpu_count_note": ("单卡测量（该方法不涉及跨设备通信）"
                           if spec["gpu_required"] == 1 else
                           f"该方法需要 {spec['gpu_required']} 卡，本次未满足"),
        "sync_async": sync_mode,
        # sync/async 只对"有跨设备通信"的方法才是自变量。dense 与
        # kv_budget_shared 是单卡测量：本脚本对它们**折叠**该轴（见 main 的循环），
        # 只生成一行且 sync_async 记为 "n/a"，因此不会出现两行逐位相同的数。
        # 该标注由 gpu_required 推导而非写死 —— 写死会在 dcc_kv 的 GPU 实现
        # 就绪后仍把它的异步行标成"不适用"。
        "sync_mode_applicable": sync_axis_applies(method),
        "method": method,
        "method_measurable": spec["measurable"],
        "num_repr_queries": a.M,
        "projection_dim": a.d_p,
        # prefill 这一列的口径是否含压缩收益。False 时 prefill_speedup 不可用于
        # H2 判定（原因见 _hf.PREFILL_TIMING_CONSUMES_COMPACT_KV）。
        "prefill_timing_consumes_compact_kv": _hf.PREFILL_TIMING_CONSUMES_COMPACT_KV,
        "budget_ratio": budget_ratio,
        "compaction_mode": mode,
        "dest_len": dest_len,
        "dest_fraction": a.dest_fraction,
        "prefill_ms_median": ps.median,
        "prefill_ms_p5": ps.p5,
        "prefill_ms_p95": ps.p95,
        "prefill_ms_ci": [ps.ci_95_lower, ps.ci_95_upper],
        "tokens_per_s_median": pre["tokens_per_s_median"],
        "peak_memory_gb": pre["peak_memory_gb"],
        "kv_bytes_full": pre["kv_bytes_full"],
        "kv_bytes_kept": _hf.kv_cache_bytes(lm, ctx_len, keep_ratio=keep,
                                            batch_size=a.batch_size),
        "accuracy": acc,
        "accuracy_by_task": acc_by_task,
        "n_eval": n_eval,
        "status": "ok",
    }


def blocked_point(a: argparse.Namespace, model: str, ctx_len: int,
                  sync_mode: str, method: str) -> Dict[str, Any]:
    spec = METHOD_SPECS[method]
    return {
        "model": model,
        "context_length": ctx_len,
        "gpu_count_observed": 0,
        "gpu_count_required": spec["gpu_required"],
        "sync_async": sync_mode,
        "sync_mode_applicable": sync_axis_applies(method),
        "method": method,
        "method_measurable": False,
        "budget_ratio": a.budget_ratio,
        "status": "blocked",
        "blockers": spec.get("blockers", []),
        "accuracy": None,
        "prefill_ms_median": None,
    }


# =============================================================================
# main
# =============================================================================

def h2_points_from_rows(a: argparse.Namespace,
                        rows: List[Dict[str, Any]]) -> List[H.H2PerLength]:
    """从主表行数据里抽 H2 的三个原始量，构造逐长度的判据输入。

    配对结构：同一 (model, ctx_len, sync_mode) 下，`dcc_kv` 与
    `kv_budget_shared` 面对同一份提示、同一份评测集，因此两者的差是**配对量**；
    `prefill_speedup` 的分子取 `dense`（精确注意力），理由见下。

    为什么 prefill 的分子是 dense 而不是 shared（2026-09-21 修正）
    ---------------------------------------------------------------
    论文 §7.5 把 H2 拆成**互不重叠的三段**：「压缩后的 DCC-KV 与精确注意力
    **不劣**」「DCC-KV 相对共享压缩**更高**」「且 prefill **更快**」。第一段比
    的是 dense，第二段比的是 shared，第三段的「更快」讲的是**系统设计的收益**。
    在 prefill 这一格上，`kv_budget_shared` 同样压过 KV、同样交付 B 长的 KV，
    它不可能比 DCC-KV 慢 10%。原实现写的 `ms_shared / ms_dcc` 得到的是一个结构
    上恒 ≈1 的量：一个恒 ≈1 的比值配 1.10x 的阈值，只能说明它不是论文那句话
    在说的量。

    三个原始量的来源：
        quality_gain_pp  = (acc_dcc - acc_shared) * 100
        prefill_speedup  = prefill_dense / prefill_dcc
        quality_comparable = 由 quality_comparable_non_inferior 判定；当非劣边界
            `delta_pp` 或噪声底线尚未由命令行给出时取 **None**（= 分辨率不足），
            而不是 False —— 「没测出容差」与「确实不相近」是两个结论。

    任一必需量为 None（含被阻断的方法）时，该长度记为 None（unresolved）。
    """
    def find(method: str, model: str, ctx: int, mode: str):
        for r in rows:
            if (r.get("method") == method and r.get("model") == model
                    and r.get("context_length") == ctx
                    and r.get("sync_async") == mode):
                return r
        return None

    def find_axis_free(method: str, model: str, ctx: int):
        """查**不适用 sync/async 轴**的方法在该 (model, ctx) 上的行。

        为什么必须有这个函数（自查发现，2026-09-18）
        --------------------------------------------
        `kv_budget_shared` 是单卡方法，`sync_axis_applies()` 为 False，
        于是 main 的循环把它的 `sync_async` 写成 `SYNC_MODE_NA`（"n/a"）。
        而本函数原先用 `a.sync_modes[0]`（"sync"）去查它 ⇒ **永远查不到**
        ⇒ H2 的配对点恒为 0，`compute_h2` 每次都返回
        「无可配对的数据点（dcc_kv / kv_budget_shared 未同时产出）」。
        也就是说：折叠 sync 轴的修复（`bee3388` 的纪律）把 H2 的配对一起折叠掉了，
        而当时**没有任何测试覆盖 h2_points_from_rows**（现已由
        `tests/test_selfcheck_2026_09_18.py` 的 `test_t6_*` 三条 +
        `tests/test_adversarial_2026_09_20.py` 的 `test_a4_*` 覆盖）。
        ⚠️ 原文此处引用的 `test_e6_h2_pairing` **从未存在** —— 声称有覆盖
        而实际没有，比没写测试更坏；已改正，并由 I 组锚点防止再犯。

        语义：该轴对该方法不适用 ⇒ 它的行与模式无关，应按 (method, model, ctx) 查。
        并**拒绝歧义** —— 若同一 (model, ctx) 上有多个候选且没有一个是
        SYNC_MODE_NA，说明表的形状与预期不符，宁可返回 None（该长度记 unresolved）
        也不猜一个模式出来，否则配出来的是两个不同测量条件下的量。
        """
        hits = [r for r in rows
                if r.get("method") == method and r.get("model") == model
                and r.get("context_length") == ctx]
        if not hits:
            return None
        marked = [r for r in hits if r.get("sync_async") == SYNC_MODE_NA]
        if len(marked) == 1:
            return marked[0]
        if len(hits) == 1:
            return hits[0]
        return None

    delta = getattr(a, "h2_delta_pp", None)
    floor = getattr(a, "h2_noise_floor_pp", None)
    delta_bad = getattr(a, "h2_delta_bad_pp", None)

    points: List[H.H2PerLength] = []
    for model in a.models:
        for ctx in a.context_lengths:
            mode = a.sync_modes[0]
            r_dcc = find("dcc_kv", model, ctx, mode)
            # 基线是单卡方法，其 sync/async 轴在表里被折叠 ⇒ 不能按模式查
            # （否则配对恒为空；见 find_axis_free 的说明）。
            r_shr = find_axis_free("kv_budget_shared", model, ctx)
            # prefill 的分子是 dense（理由见函数 docstring）。
            r_den = find_axis_free("dense", model, ctx)
            if r_dcc is None or r_shr is None or r_den is None:
                continue
            acc_d, acc_s = r_dcc.get("accuracy"), r_shr.get("accuracy")
            ms_d = r_dcc.get("prefill_ms_median")
            ms_den = r_den.get("prefill_ms_median")

            comparable = None
            if delta is not None and floor is not None and acc_d is not None \
                    and acc_s is not None:
                # 非劣检验要的是 (Q_DCC − Q_dense) 的**配对 95% CI 下界**，
                # 而本表每格只落一个聚合准确率、不落逐样本判对错，故无法算 CI。
                # 这里**不猜**：保持 None ⇒ 记入 unresolved。
                # 要让 H2 真正落判据，需要在 measure_point 里落逐样本得分，
                # 那是 GPU 侧 E6 跑通后才能做的事（当前 dcc_kv 被阻断）。
                comparable = None

            # 量具是否体现压缩收益：由 _hf 的契约决定，不在本文件里写死。
            instrument_ok = bool(_hf.PREFILL_TIMING_CONSUMES_COMPACT_KV)
            if (acc_d is None or acc_s is None or not ms_d or not ms_den):
                points.append(H.H2PerLength(
                    length=ctx, quality_gain_pp=float("nan"),
                    prefill_speedup=float("nan"), quality_comparable=None,
                    prefill_instrument_valid=instrument_ok,
                ))
            else:
                points.append(H.H2PerLength(
                    length=ctx,
                    quality_gain_pp=(acc_d - acc_s) * 100.0,
                    prefill_speedup=ms_den / ms_d,
                    quality_comparable=comparable,
                    prefill_instrument_valid=instrument_ok,
                ))
    return points


def compute_h2(a: argparse.Namespace,
               rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """H2 的落盘字段。**规则与阈值全部来自 hypotheses 模块。**

    设计上刻意让它在 E6 被阻断时也能跑完并如实产出 `h2_passed=False`：
    判定链路先接线、后取数，这样 E6 一跑通就能自动判定，不必再改代码。
    """
    points = h2_points_from_rows(a, rows)
    if not points:
        return {
            "h2_passed": None,
            "h2_aggregation_rule": H.H2_LENGTH_AGGREGATION,
            "h2_note": "无可配对的数据点（dcc_kv / kv_budget_shared 未同时产出）",
        }
    agg = H.h2_pass_across_lengths(points)
    out = agg.to_dict()
    out["h2_note"] = (
        "quality_comparable 为 None 表示该长度上「质量相近」前提无法判定"
        "（delta 与噪声底线尚未测量），记入 h2_unresolved_lengths，"
        "**不计入** h2_n_failed —— 分辨率不足不等于未达标。"
    )
    out["h2_prefill_instrument_valid"] = bool(
        _hf.PREFILL_TIMING_CONSUMES_COMPACT_KV)
    if not _hf.PREFILL_TIMING_CONSUMES_COMPACT_KV:
        out["h2_note"] += (
            "；**prefill 量具无效**：当前 measure_point 的计时窗口里只有全长前向，"
            "裁剪后的 KV 未被使用（_hf.PREFILL_TIMING_CONSUMES_COMPACT_KV=False），"
            "prefill_speedup 结构上恒 <= 1，故本长度记 unresolved 而不是未达标。"
            "要让 H2 的第二个合取项可判，须先接 E6 的注意力钩子。")
    else:
        out["h2_note"] += (
            "；prefill 量具有效（H0 已接，2026-09-21）：压缩臂的计时窗口里含一次"
            "目的端前向，其 past cache 是裁剪后的 KV，故 prefill_speedup 能反映"
            "压缩带来的 KV 长度收益。分子口径为 dense（精确注意力），"
            "见 h2_points_from_rows 的说明。")
    out["h2_params_supplied"] = {
        "delta_pp": getattr(a, "h2_delta_pp", None),
        "noise_floor_pp": getattr(a, "h2_noise_floor_pp", None),
        "delta_bad_pp": getattr(a, "h2_delta_bad_pp", None),
    }
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E6 主表与可扩展性（GPU）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--plan", action="store_true",
                   help="只打印网格与前置条件并退出，无需 GPU")
    p.add_argument("--models", nargs="+", default=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ])
    p.add_argument("--context-lengths", dest="context_lengths", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768],
                   help="论文写 5 档但总数 144 对应 4 档；此处默认 4 档并与论文核对")
    p.add_argument("--sync-modes", dest="sync_modes", nargs="+",
                   default=["sync", "async"], choices=["sync", "async"])
    p.add_argument("--methods", nargs="+",
                   default=["dense", "kv_budget_shared", "fastkv_official",
                            "dcc_kv", "ring", "apb"],
                   choices=sorted(METHOD_SPECS))
    p.add_argument("--eval-file", dest="eval_file", type=str, default=None)
    p.add_argument("--eval-limit", dest="eval_limit", type=int, default=None)
    p.add_argument("--max-prompt-tokens", dest="max_prompt_tokens", type=int, default=None)
    p.add_argument("--budget-ratio", dest="budget_ratio", type=float, default=0.05)
    p.add_argument("--dest-fraction", dest="dest_fraction", type=float, default=0.25,
                   help="目的端 query 长度占上下文长度的比例（H0）。prefill 计时窗口里"
                        "目的端前向看到的 KV 长度才是压缩收益的载体；取 0 会让压缩臂"
                        "直接报错（_hf.measure_prefill 拒绝静默失效），而不是产出一个"
                        "结构上恒 <=1 的加速比")
    p.add_argument("--compaction-mode", dest="compaction_mode", type=str,
                   default="topk_rms",
                   choices=["identity", "topk_rms", "topk_norm", "stride", "random"])
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--d-p", dest="d_p", type=int, default=32)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=1)
    p.add_argument("--precision", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-impl", dest="attn_impl", type=str, default=None)
    p.add_argument("--interconnect", type=str, default="unknown")
    p.add_argument("--rope-extension-used", dest="rope_extension_used", type=str,
                   default=None)
    p.add_argument("--rope-extension-disclosed", dest="rope_extension_disclosed",
                   action="store_true")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--h2-delta-pp", dest="h2_delta_pp", type=float, default=None,
                   help="H2 前提「质量相近」的非劣边界（百分点）。**无默认值** —— "
                        "取值依赖重复 run 的噪声底线，未给出时该前提记入 "
                        "h2_unresolved_lengths，判定结果为「分辨率不足」")
    p.add_argument("--h2-noise-floor-pp", dest="h2_noise_floor_pp", type=float,
                   default=None,
                   help="同配置重复 run 的噪声底线（百分点）。**无默认值**")
    p.add_argument("--h2-delta-bad-pp", dest="h2_delta_bad_pp", type=float,
                   default=None,
                   help="共享压缩相对精确注意力的退化量（百分点），可选")
    p.add_argument("--out", type=str, default="results/gpu/e6")
    p.add_argument("--print-env", action="store_true")
    return p


def main() -> int:
    a = build_parser().parse_args()

    if a.plan:
        return print_plan(a)
    if a.print_env:
        return _env.print_env_only("E6 主表与可扩展性", min_gpus=1, need_nccl=False)

    measurable = [m for m in a.methods if METHOD_SPECS[m]["measurable"]]
    if not measurable:
        print_plan(a)
        print("\n阻断：所选方法全部被前置缺口阻断，无可测量项。")
        print("可测量方法：" + str(sorted(k for k, v in METHOD_SPECS.items()
                                          if v["measurable"])))
        return _env.GATE_EXIT_CODE

    gate = _env.probe(min_gpus=1, need_nccl=False, hf_backend=True,
                      model_name=a.models[0])
    code = _env.enforce(gate, "E6 主表与可扩展性", SCRIPT)
    if code is not None:
        return code

    if a.iters < 10:
        print(f"[警告] --iters={a.iters} < 10，不满足 §6.1 的重复次数规范。")

    samples: List[_hf.EvalSample] = []
    if a.eval_file:
        samples = _hf.load_eval_file(a.eval_file, limit=a.eval_limit)
        print(f"评测集：{len(samples)} 条（{a.eval_file}）")

    rows: List[Dict[str, Any]] = []
    for model in a.models:
        print()
        print("=" * 78)
        lm = _hf.load_model(model, a.precision, attn_implementation=a.attn_impl)
        print("=" * 78)
        for ctx_len in a.context_lengths:
            for method in a.methods:
                # 轴折叠：单卡方法（dense / kv_budget_shared）不生成 sync/async
                # 两行 —— 它们没有跨设备通信，两行必然逐位相同，而表里"两行
                # 一样"会被读成"异步没有收益"。详见 SYNC_MODE_NA 与 planned_points。
                modes = a.sync_modes if sync_axis_applies(method) else [SYNC_MODE_NA]
                for sync_mode in modes:
                    if not METHOD_SPECS[method]["measurable"]:
                        rows.append(blocked_point(a, model, ctx_len, sync_mode, method))
                        continue
                    try:
                        r = measure_point(a, model, ctx_len, sync_mode, method, lm, samples)
                    except Exception as e:
                        r = {
                            "model": model, "context_length": ctx_len,
                            "sync_async": sync_mode, "method": method,
                            "sync_mode_applicable": sync_axis_applies(method),
                            "status": "error", "error_type": type(e).__name__,
                            "error": str(e),
                            "traceback": traceback.format_exc().splitlines()[-8:],
                            "accuracy": None, "prefill_ms_median": None,
                        }
                    rows.append(r)
                    if r["status"] == "ok":
                        acc_str = ("n/a" if r["accuracy"] is None
                                   else f"{r['accuracy']:.4f}")
                        print(f"  {model.split('/')[-1]:<28} L={ctx_len:<6} "
                              f"{sync_mode:<5} {method:<17} "
                              f"prefill={r['prefill_ms_median']:9.2f}ms "
                              f"(p95={r['prefill_ms_p95']:9.2f})  "
                              f"{r['tokens_per_s_median']:10.1f} tok/s  "
                              f"peak={r['peak_memory_gb']:5.2f}GB  "
                              f"acc={acc_str}")
                    else:
                        print(f"  {model.split('/')[-1]:<28} L={ctx_len:<6} "
                              f"{sync_mode:<5} {method:<17} [{r['status']}] "
                              f"{r.get('error_type', '')}")
        del lm
        torch.cuda.empty_cache()

    # 达标性判定
    meta = _env.build_metadata(
        run_id="e6-main-table",
        model_name=",".join(a.models),
        context_length=max(a.context_lengths),
        seed=a.seed, budget_ratio=a.budget_ratio,
        sync_async="both", num_repr_queries=a.M, projection_dim=a.d_p,
        task="main-table", precision=a.precision,
        warmup=a.warmup, iters=a.iters,
        interconnect=a.interconnect,
        rope_extension_used=a.rope_extension_used,
        rope_extension_disclosed=a.rope_extension_disclosed,
        notes=f"methods={a.methods}",
    )
    admissible = _env.gate_guard_for_report(meta)

    payload: Dict[str, Any] = {
        "experiment": "E6",
        "rows": rows,
        "metadata": meta.to_dict(),
        "report_admissibility": admissible,
        "grid": {
            "models": a.models, "context_lengths": a.context_lengths,
            "sync_modes": a.sync_modes, "methods": a.methods,
            "n_points_planned": planned_points(a),
            "n_points_planned_naive_product": (len(a.models) * len(a.context_lengths)
                                               * len(a.sync_modes) * len(a.methods)),
            "sync_axis_expanded_methods": [m for m in a.methods if sync_axis_applies(m)],
            "sync_axis_collapsed_methods": [m for m in a.methods
                                            if not sync_axis_applies(m)],
            "n_points_measured": sum(1 for r in rows if r["status"] == "ok"),
            "n_points_blocked": sum(1 for r in rows if r["status"] == "blocked"),
            "paper_claim": "3 模型 × 4 上下文长度 × 2 GPU × 2 同步 × 3 基线 = 144",
            "paper_claim_check": "3×4×2×2×3 = 144，与论文 §6.4 自洽（原稿的 5 档已由 72af7db 改为 4 档）",
        },
        "repetitions": {
            "warmup": a.warmup,
            "iters": a.iters,
            "norm_required_iters": 10,
            "below_norm": bool(a.iters < 10),
            "norm_source": "论文 §6.4「每点 >= 10 次 run」",
            "note": ("below_norm 为真时该产物只作冒烟用，**不得**当作主表数据。"
                     "此前该信息只打印在 stdout，产物里无从判别 —— "
                     "于是 --iters 1 的冒烟结果与合规结果在 JSON 上无法区分。"),
        },
        "instrument": {
            "prefill_timing_consumes_compact_kv": _hf.PREFILL_TIMING_CONSUMES_COMPACT_KV,
            "dest_fraction": a.dest_fraction,
            "dest_len_by_context": {str(c): max(1, int(round(a.dest_fraction * c)))
                                    for c in a.context_lengths},
            "prefill_speedup_numerator": "dense（精确注意力）",
            "meaning": ("True ⇒ 压缩臂的计时窗口里含一次目的端前向，其 past cache "
                        "是裁剪后的 KV，prefill_speedup 可反映压缩收益；"
                        "False ⇒ prefill 一列对压缩方法只含开销、不含收益，"
                        "该长度记 unresolved。"),
        },
        "h2": compute_h2(a, rows),
        "caveat": (
            "gpu_count 在可测量方法（dense / kv_budget_shared）上不是自由轴："
            "它们是单卡测量，不涉及跨设备通信。主表中的「2 GPU」维度只对"
            "需要多卡的四个方法有意义，而那四个当前均被阻断，"
            "因此本表**不能**用于支撑 H5（设备数翻倍加速 ≥1.5×）。"
            "H5 需要 dcc_kv 的 GPU 多卡实现就绪后单独测量。"
        ),
    }

    out = REPO_ROOT / a.out
    R.save_json(str(out / "e6_results.json"), payload)
    R.save_csv(str(out / "e6_main_table.csv"), rows)
    print()
    print(f"数据点：计划 {payload['grid']['n_points_planned']} / "
          f"实测 {payload['grid']['n_points_measured']} / "
          f"阻断 {payload['grid']['n_points_blocked']}")
    h2 = payload["h2"]
    if h2.get("h2_passed") is None:
        print(f"H2 判定：跳过（{h2.get('h2_note', '')}）")
    else:
        print(f"H2 判定：passed={h2['h2_passed']}  规则={h2['h2_aggregation_rule']}  "
              f"{h2['h2_n_passed']}/{h2['h2_n_lengths']} 通过  "
              f"未定={h2['h2_unresolved_lengths']}  "
              f"最差质量提升={h2['h2_worst_quality_gain_pp']:.3f}pp  "
              f"最差 prefill 加速={h2['h2_worst_prefill_speedup']:.3f}x")
        if not h2["h2_passed"] and h2["h2_unresolved_lengths"]:
            print("  [分辨率不足] 存在无法判定的长度；这**不是**「H2 未达标」。")
    if not admissible["admissible"]:
        print(f"[不可入主表] {admissible['reason']}")
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
