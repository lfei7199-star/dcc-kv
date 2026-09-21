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

⚠️ **2026-09-21 补**：折叠的判据从「单卡方法」推广成「**本次运行没有真的用多卡**」。
`dcc_kv` 的 `gpu_required` 是 2，但 `--dcc-world` 是**单卡模拟**（把源序列切成 W 段
来模拟 W 个源端设备），跑起来 `gpu_count_observed` 仍是 1，**没有第二次设备参与**
⇒ sync/async 这条轴在本表里同样不存在。若照 `gpu_required>1` 生成两行，两行会
逐位相同 —— 那正是本节开头要避免的误读。故：

    sync_axis_applies(method, a) = (gpu_required > 1) and (--ranks > 1)

`--ranks` 默认 1，且**声称 >1 时本脚本会检查进程组**（没真的起多进程就直接
以闸门退出码拒绝），免得"声明多卡、实际单卡"把轴凭空造出来。

dcc_kv 行的 prefill 加速比有**两列**，各自命名、不得互相冒充
---------------------------------------------------------
H0 接上后，`dcc_kv` 行的目的端前向走钩子（源端由 state 携带、紧凑块带 β 与 V
回归）。于是这条比值可以有两种分子，**它们回答的不是同一个问题**：

    prefill_speedup_kernel_matched = 钩子dense臂的目的端耗时 / 钩子dcc臂的目的端耗时
        —— 两臂走**同一个算子核**（`attention_kernel` 的显式路径），唯一变量是
           目的端注意力消费的 KV 长度（dense = S，dcc = B）。
           **H2 的第二个合取项消费这一列**：它测的是机制。
    prefill_speedup_native = dense 行的端到端 prefill / dcc 行的端到端 prefill
        —— 分子分母**不是同一个核**（SDPA 融合核 vs 显式核），而且还差着构造
           代价与源端缓存是否留在 past 里。它是"我们的实现 vs 精确注意力"的
           端到端数，**含实现差距**，不得当作机制收益。

两列各自的产处（避免"哪一列缺了就静默降级"）
------------------------------------------
`prefill_speedup_kernel_matched` 的分子与分母**都在 dcc_kv 行内部测出**
（`measure_point` 对同一份输入分别跑一次钩子 dense 臂与钩子 dcc 臂），
因此它不依赖别的行是否在表里 —— H2 消费的正是这一列。
`prefill_speedup_native` 需要 dense 行的端到端耗时，是**跨行**的量，
由 `attach_prefill_speedups()` 在全部行收齐后补齐（必须在 `compute_h2`
之前调用）；两列都会落进每一行产物，并各自带口径说明。

**一列缺失时不得用另一列顶替**：kernel-matched 算不出时 H2 的那一格记
unresolved，绝不回落到 native 列 —— 那正是"两列互冒充"。

H2 的「质量相近」前提：数据来源与方向口径（2026-09-21）
------------------------------------------------------
该前提是**单侧非劣检验**，参照物是 **dense**（`hypotheses.QUALITY_COMPARABLE_
REFERENCE`），检验量是 `Q_DCC − Q_dense` 的**配对 95% CI 下界**。它此前恒为
`None`（记 unresolved），根因不是"通路没独立"，而是**逐样本判对错没落盘** ——
每格只有一个聚合准确率，配对 CI 在信息量上算不出来。

现在 `measure_point` 落两样（与聚合量**同行**）：

    accuracy_per_sample —— 逐样本是否答对（顺序同 `samples`）
    eval_sample_keys    —— 每样本的身份键，供**跨行**逐位比对

配对时**先验键再算数**：`dcc_kv` 与 `dense` 两行的键序列若有一位不同，就
拒绝配对并记下原因。少了这一步，样本错位只会产出一个"数值正常、实际无意义"
的 CI —— 它不会报错，只会悄悄地把结论带偏。

⚠️ CI 的**方向**容易写反，且写反后符号、量级都正常（`report.paired_bootstrap`
在 `higher_is_better=True` 时把配对差翻了符号）：`ci_low = −ci_95_upper`。
实测钉住的样例见 `_paired_quality_ci` 的 docstring，另有锚点测试守着。

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
import math
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
from src.distributed import attention_hook as _A  # noqa: E402

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
            "仍需：把该算子接进 measure_point 的**方法行**。"
            "（2026-09-21 补：dcc_kv 那行已接（attention_hook），"
            "本行与 apb 行仍未接 —— 三行不再同状态。）",
        ],
    },
    "dcc_kv": {
        "measurable": True,
        "gpu_required": 2,
        "desc": "DCC-KV（逐边条件化 + 紧凑 KV 注意力；源端切分为单卡模拟）",
        "impl": ("attn_hook: src/distributed/attention_hook.py（build_compact_kv + "
                 "attention_kernel.dcc_kv_attention）"),
        "blockers": [
            "（原 blocker「CompactKV → GPU attention kernel 缺失」已关闭："
            "src/dcc_kv_ref/attention_kernel.py 提供 compact_kv_attention。）",
            "（原 blocker「异步流水无 GPU 入口」已关闭："
            "experiments/gpu/_forward.py 的 pipelined_attention。）",
            "（原 blocker「E6 注意力钩子未接」已关闭：2026-09-21 H0。"
            "原 blocker「方法实现本身未接」亦已关闭：2026-09-21 稍后，"
            "measure_point 的 dcc_kv 行改走 attention_hook —— 源端由钩子 state 携带，"
            "紧凑块经 build_compact_kv 目的端条件化构造，注意力走 "
            "dcc_kv_attention（即 G2 `_comp` 闭包的同一段计算）。）",
            "**仍未满足预登记条件的字面部分**：G2 的 pipelined_attention 用**真** "
            "dist.all_to_all_single，单卡（本脚本的观测口径 gpu_count_observed=1）下"
            "它退化成一次本地拷贝、chunks_effective=1，**没有可重叠窗口**。"
            "故本表对 dcc_kv **折叠 sync/async 轴**，产物里 sync_async 记 n/a；"
            "async 的代价与收益只在 A5/E5 上测，不在这张表里。"
            "**2026-09-21 记录**：此条是对预登记表述（原文写「走 G2 的 "
            "pipelined_attention」）的**口径修正**，不是条件已满足 —— 修正理由与"
            "原文一并留在 docs/gpu_execution_plan.md 与 commit_log.md。",
            "构造链路的 CUDA 可用性待 GPU 机实测（G6 未做；静态审计不能替代实测）。",
            "单卡模拟**不体现逐边条件化的代价与收益**（所有源段看到同一批目的端 "
            "query ⇒ 与共享压缩在数值上不可区分），见钩子 summary 的 "
            "conditionalization_marginal_available=False。",
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


def ranks_exercised(a: argparse.Namespace) -> int:
    """本次运行**真的**用到了几个 rank（见模块文档的同步轴折叠一节）。

    取 `--ranks`（默认 1）而不是运行时探测：`--plan` 在任何机器上都要能跑，
    而规划数据点数时就得知道这条轴在不在。
    """
    return int(getattr(a, "ranks", 1))


def sync_axis_applies(method: str, a: argparse.Namespace) -> bool:
    """sync / async 是否是**本次运行**中该方法的自变量。

    两个条件都要成立：

    1. 该方法**存在**跨设备通信（`gpu_required > 1`）—— 单卡方法没有可异步
       重叠的通信，该轴对它不构成自变量；
    2. 本次运行**真的**起了多卡（`--ranks > 1`）—— 单卡跑多卡方法时该轴同样
       不存在（`--dcc-world` 是单卡模拟，没有第二次设备参与）。

    从这两个条件推导而不是写死 False：写死会在多卡跑通后仍把 dcc_kv 的异步行
    标成"该轴不适用"，把"还没测"固化成"没这条轴"。反过来，只看 `gpu_required`
    会在单卡模拟下生成两行**逐位相同**的数，而"两行一样"会被读成"异步没有收益"。

    `a` **必填**（2026-09-21 收紧）：此前允许省略，省略时退化成"该方法原则上
    有没有这条轴"。两个语义共用一个函数、且默认走的是**不安全**的那个（返回
    True = 这条轴存在），是个脚枪 —— 谁忘了传 `a`，谁就在单卡下凭空造出两行
    逐位相同的数。要问"原则上"，直接看 `METHOD_SPECS[m]["gpu_required"] > 1`。
    """
    if METHOD_SPECS[method]["gpu_required"] <= 1:
        return False
    return ranks_exercised(a) > 1


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
        mult = len(a.sync_modes) if sync_axis_applies(m, a) else 1
        total += len(a.models) * len(a.context_lengths) * mult
    return total


def print_plan(a: argparse.Namespace) -> int:
    n_models = len(a.models)
    n_ctx = len(a.context_lengths)
    n_sync = len(a.sync_modes)
    n_methods = len(a.methods)
    n_points = planned_points(a)
    expanded = [m for m in a.methods if sync_axis_applies(m, a)]
    collapsed = [m for m in a.methods if not sync_axis_applies(m, a)]

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
        print(f"       —— 本次运行 ranks={ranks_exercised(a)}："
              f"该轴对这些方法不是自变量（单卡方法本就没有跨设备通信；"
              f"多卡方法在单卡上跑时也没有第二次设备参与）。"
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
    """测量主表的一个数据点。

    dcc_kv 行走 H0 的方法侧钩子（2026-09-21 接线）
    ----------------------------------------------
    与 `dense` / `kv_budget_shared` 的差别不只在"压得更狠"：

        dense / kv_budget_shared —— `apply_kv_budget` 路径：源端 KV **留在**
            cache 里被裁剪，目的端前向直接消费这份 cache；
        dcc_kv                  —— 钩子路径：源端由钩子 state 携带，目的端
            前向传 `past=None`，紧凑块经 `build_compact_kv` 目的端条件化构造，
            注意力走 `dcc_kv_attention`（与 G2 的 `_comp` 闭包同一段计算）。

    两条路径下 `budget_ratio` / `compaction_mode` 的归属不同：钩子路径**必须**
    把它们交给 `HookConfig`，由 `_hf` 显式拒绝"两处各写一份预算"。

    ⚠️ 源端切分（`--dcc-world`）是**单卡模拟**：把源序列切成 W 段来模拟 W 个
    源端设备。它不产生第二次设备参与，故 `gpu_count_observed` 仍是 1、
    sync/async 轴被折叠、钩子 summary 里 `conditionalization_marginal_available`
    为 False（所有段看到同一批 query ⇒ 与共享压缩在数值上不可区分）。
    拿它当"多卡结果"是把模拟读成了测量。
    """
    spec = METHOD_SPECS[method]
    hook: Optional[Any] = None
    if method == "dcc_kv":
        if a.dcc_world is None:
            raise ValueError(
                "dcc_kv 行必须给出 --dcc-world（源端段数）。它没有默认值："
                "它决定每边的预算（per_edge 口径下 B_total/world），给个默认值"
                "会把「本次模拟了几个源端设备」变成没人声明过的假设。"
            )
        hook = _A.HookConfig(
            budget_ratio=a.budget_ratio,
            dcc_world=int(a.dcc_world),
            budget_mode=a.dcc_budget_mode,
            n_repr=int(a.M), projection_dim=int(a.d_p),
            lambda_beta=resolve_lambda_beta(a), seed=a.seed,
        )
    budget_ratio = (None if (method == "dense" or hook is not None)
                    else a.budget_ratio)
    mode = ("identity" if (method == "dense" or hook is not None)
            else a.compaction_mode)

    # H0：源段 / 目的端本地段的切分。**必须与评测路径同源**（同一个
    # `_hf.prompt_split`）：计时侧若把整条 ctx_len 当源端、再另加 dest_len，
    # 它压缩的源长就成了 ctx_len，而评测压缩的是 (1-f)·ctx_len —— 同一行里
    # 两个不同的压缩设置，产物里的 `budget_tokens_resolved` 也就对不上质量
    # 那一侧（实测：计时报 {"64": 32}，评测实际是 {"48": 24}）。
    # 切分只有一处定义，`--dest-fraction 0` 时 dest 为 0，由 `_hf` 对压缩臂
    # /钩子路径直接拒跑（原实现用 max(1, ...) 把 0 悄悄抬成 1）。
    n_src, n_dst = _hf.prompt_split(ctx_len, a.dest_fraction)
    dest_len = n_dst

    pre = _hf.measure_prefill(
        lm, seq_len=n_src, batch_size=a.batch_size,
        warmup=a.warmup, iters=a.iters,
        budget_ratio=budget_ratio, compaction_mode=mode,
        dest_len=dest_len, attn_hook=hook, seed=a.seed,
    )
    ps = R.summarize(pre["prefill_ms_samples"], "prefill", "ms", seed=a.seed)

    # kernel-matched 的分子：**同一个算子核**下的 dense 臂。它只能在这里测 ——
    # dense 行走的是 SDPA 融合核，与钩子的显式核不是同一个量，拿它的端到端数
    # 顶替就等于把"实现差距"算进"机制收益"（见模块文档「两列各自的产处」）。
    dense_kernel_dest_ms: Optional[float] = None
    if hook is not None:
        dense_hook = _A.HookConfig(
            budget_ratio=1.0, mode="dense", dcc_world=1,
            n_repr=int(a.M), projection_dim=int(a.d_p),
            lambda_beta=resolve_lambda_beta(a), seed=a.seed,
        )
        pre_dense = _hf.measure_prefill(
            lm, seq_len=n_src, batch_size=a.batch_size,
            warmup=a.warmup, iters=a.iters,
            dest_len=dest_len, attn_hook=dense_hook, seed=a.seed,
        )
        dense_kernel_dest_ms = pre_dense.get("dest_ms_median")

    acc: Optional[float] = None
    acc_by_task: Optional[Dict[str, float]] = None
    n_eval = 0
    # 逐样本判对错 + 身份键。H2 的「质量相近」前提要做**配对**非劣检验，
    # 光有聚合准确率算不出配对 CI（见模块文档）。两者与 acc 同源同批。
    acc_per_sample: Optional[List[int]] = None
    eval_sample_keys: Optional[List[str]] = None
    if samples:
        # `dest_fraction` 对**所有**方法都传：源段 / 目的端本地段的切分必须是
        # 同一份，否则两边的"保留了多少精确 token"不同，质量差里混进了预算不
        # 对齐（`_hf` 模块文档已把这条同口径写成契约，此前只有 prefill 那侧传了）。
        ev = _hf.evaluate(lm, samples, budget_ratio=budget_ratio,
                          compaction_mode=mode,
                          max_prompt_tokens=a.max_prompt_tokens,
                          attn_hook=hook,
                          dest_fraction=a.dest_fraction)
        acc = ev["accuracy"]
        acc_by_task = ev["by_task"]
        n_eval = ev["n"]
        acc_per_sample = list(ev["per_sample"])
        eval_sample_keys = list(ev["keys"])

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
        "sync_mode_applicable": sync_axis_applies(method, a),
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
        # 钩子行的配置与运行态（含 n_builds / resolved_budget_by_source_len /
        # conditionalization_marginal_available）。**必须与数值同行落盘**：
        # 读表的人要能看出这一行是"几个源端段、每段多少预算、构造了几次"。
        "attn_hook": pre.get("attn_hook"),
        "dcc_world": (None if hook is None else int(hook.dcc_world)),
        "dcc_budget_mode": (None if hook is None else hook.budget_mode),
        "lambda_beta": (None if hook is None else float(hook.lambda_beta)),
        "budget_tokens_resolved": pre.get("budget_tokens_resolved"),
        # kernel-matched 那两列的原生量：目的端窗口的耗时（两端都在本行内测）。
        "dest_ms_median": pre.get("dest_ms_median"),
        "dest_ms_median_dense_kernel": dense_kernel_dest_ms,
        "prefill_speedup_kernel_matched": _safe_ratio(
            dense_kernel_dest_ms, pre.get("dest_ms_median")),
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
        # 逐样本数据**与本行的聚合量同行落盘**（不另立一份）。理由和 attn_hook
        # 那边一样：读表的人要能就地看出这一行的 CI 是哪批样本配出来的。
        # CSV 里这两个 list 会被 str() 化（save_csv 不做特殊处理）—— 可读性差，
        # 但不丢信息；要看整齐的逐样本数据看 JSON。
        "accuracy_per_sample": acc_per_sample,
        "eval_sample_keys": eval_sample_keys,
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
        "sync_mode_applicable": sync_axis_applies(method, a),
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

def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """只在两端都是**有限正数**时给比值；否则 None。

    None 与 0.0 / 1.0 是三个不同的意思，混用会让"没测出来"看起来像"测出来是
    没有收益"。分母非正时同样返回 None：那说明窗口里根本没发生那次前向。
    """
    if num is None or den is None:
        return None
    try:
        n, d = float(num), float(den)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(n) and math.isfinite(d)) or d <= 0.0:
        return None
    return n / d


def resolve_lambda_beta(a: argparse.Namespace) -> float:
    """λ_β 的实际取值：命令行给了就用它，否则用仓库默认（与源码同源）。

    落盘的是**解析后的绝对值**而不是"用了默认"这个事实 —— 默认值会随
    `src.dcc_kv_ref.DEFAULT_LAMBDA_BETA` 变，而产物要能回答"当时是几"。
    """
    if getattr(a, "lambda_beta", None) is not None:
        return float(a.lambda_beta)
    return float(_A.DEFAULT_LAMBDA_BETA)


def _kernel_matched_speedup(row: Dict[str, Any]) -> Optional[float]:
    """dcc_kv 行的 kernel-matched 加速比（**H2 消费这一列**）。

        分子 = 钩子 dense 臂的目的端耗时（`dest_ms_median_dense_kernel`）
        分母 = 钩子 dcc 臂的目的端耗时   （`dest_ms_median`）

    两臂走**同一个算子核**（`attention_kernel.dcc_kv_attention`）：dense 臂的
    远端块是 `identity_compact` 出来的全长块（B = L_s），dcc 臂的远端块是条件化
    构造出来的紧凑块（B = budget）。唯一变量是远端 KV 长度 —— 块数随 `dcc_world`
    变，那是机制自身的属性（W 条边），不是量具的差异。

    两端都在 dcc_kv 行内部测得，故本函数**不依赖别的行是否存在**。
    返回 None 表示该行没测出这一对量（未接钩子 / 未跑 dcc 行），
    **不得**回落到 native 那一列。
    """
    return _safe_ratio(row.get("dest_ms_median_dense_kernel"),
                       row.get("dest_ms_median"))


def _axis_free_rows(rows: List[Dict[str, Any]], method: str, model: str,
                    ctx: int) -> List[Dict[str, Any]]:
    """取该方法在该 (model, ctx) 上的全部行（**不**筛 sync/async）。"""
    return [r for r in rows
            if r.get("method") == method and r.get("model") == model
            and r.get("context_length") == ctx]


def _pick_axis_free(hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """从候选行里挑出"该轴不适用"的那一行；有歧义时返回 None（不猜）。

    与 `h2_points_from_rows` 内 `find_axis_free` 同一套判据 —— 抽出来是为了让
    `attach_prefill_speedups` 找 dense 行时走同一条路，免得两处判据漂开。
    """
    if not hits:
        return None
    marked = [r for r in hits if r.get("sync_async") == SYNC_MODE_NA]
    if len(marked) == 1:
        return marked[0]
    if len(hits) == 1:
        return hits[0]
    return None


def attach_prefill_speedups(a: argparse.Namespace,
                            rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把两列 prefill 加速比补进每个 `dcc_kv` 行，并返回口径声明。

    为什么是"补"而不是在 `measure_point` 里算全
    -------------------------------------------
    kernel-matched 的两端都在 dcc_kv 行内部（`measure_point` 已就地算好，
    见 `_kernel_matched_speedup`）；native 的分子是 **dense 行**的端到端耗时，
    是跨行的量，只有全部行收齐后才存在。故这里只补 native，并把 kernel-matched
    再写一次到行上 —— 读者不必知道它是在哪一步算的。

    **两列不得互相冒充**：native 含"实现差距"（SDPA 融合核 vs 显式核 + 构造
    代价 + 源端缓存是否留在 past 里），只作端到端参考；H2 消费 kernel-matched。

    必须在 `compute_h2` 之前调用 —— 它写入的正是 H2 要读的那一列。
    """
    n_km = 0
    n_native = 0
    native_missing: List[str] = []
    for r in rows:
        if r.get("method") != "dcc_kv":
            continue
        km = _kernel_matched_speedup(r)
        r["prefill_speedup_kernel_matched"] = km
        if km is not None:
            n_km += 1
        dense_row = _pick_axis_free(_axis_free_rows(
            rows, "dense", r.get("model"), r.get("context_length")))
        native = (None if dense_row is None else
                  _safe_ratio(dense_row.get("prefill_ms_median"),
                              r.get("prefill_ms_median")))
        r["prefill_speedup_native"] = native
        if native is None:
            native_missing.append(
                "%s@%s" % (r.get("model"), r.get("context_length")))
        else:
            n_native += 1
    return {
        "kernel_matched": ("钩子dense臂目的端耗时 / 钩子dcc臂目的端耗时"
                           "（同一算子核；**H2 消费这一列**）"),
        "native": ("dense 行端到端 prefill / dcc 行端到端 prefill"
                   "（异核 + 含构造代价；端到端参考，不得当作机制收益）"),
        "n_rows_with_kernel_matched": n_km,
        "n_rows_with_native": n_native,
        "native_unavailable_rows": native_missing,
        "native_unavailable_reason": ("该 (model, ctx) 上没有唯一的 dense 行；"
                                      "native 列记 None，不影响 kernel-matched "
                                      "列与 H2 判定"),
    }


def dcc_row_modes(a: argparse.Namespace) -> List[str]:
    """`dcc_kv` 行在表里**实际**会写出的 sync_async 值（与 main 的循环同判据）。

    单元素列表有两种可能，必须区分开：

        ["sync", "async"] —— 轴展开（本次真的起了多卡），每个长度两行；
        [SYNC_MODE_NA]    —— 轴**被折叠**（`--ranks 1`），每个长度一行。

    抽成函数而不是在读写两侧各写一遍：两边一旦不同源，就会出现"写行时折叠、
    查行时按 sync 查"的静默不配对（正是 `bee3388` 与 2026-09-21 两次踩到的坑）。
    """
    if sync_axis_applies("dcc_kv", a):
        return list(a.sync_modes)
    return [SYNC_MODE_NA]


def _pp_percent(x: float) -> float:
    """比值 → 百分点，并把 `-0.0` 归一成 `0.0`。

    `-0.0` 在浮点里与 `0.0` 相等，但它会原样写进 JSON/CSV（`-0.0`），读的人
    得停下来判断"是符号搞反了还是本来就为零"。归一化在这里是**报告口径**，
    不是数值修正 —— 判定逻辑不受影响。
    """
    v = float(x) * 100.0
    return 0.0 if v == 0 else v


def _paired_quality_ci(r_dcc: Dict[str, Any], r_ref: Dict[str, Any],
                        a: argparse.Namespace) -> Dict[str, Any]:
    """(Q_DCC − Q_ref) 的配对 95% CI，单位**百分点**。有诊断，不算哑值。

    返回 dict：available / ci_95_low_pp / ci_95_high_pp / mean_diff_pp /
    n_pairs / pairing_keys_match / reason。`available=False` 时
    `reason` 说明**为什么**配不出来 —— 三种原因必须分开写进产物：

        · 两行中有一行没落逐样本数据（老产物 / 该方法是 blocked 行）；
        · 键序列**不等** ⇒ 不是同一批样本或顺序错位（**拒绝配对**）；
        · 两行样本数不等（与上一条独立，单独报，免得被当成同一个原因）。

    ⚠️ 方向口径（实测钉住，别照直觉改）
    ----------------------------------
    `report.paired_bootstrap` 在 `higher_is_better=True` 时把配对差翻了符号
    统一成"越小越好"，于是它返回的 `ci_95_lower/upper` 是
    `−(Q_DCC − Q_ref)` 的区间，**不是**我们要的那个。实测：

        A 比 B 好 20pp ⇒ mean_diff=+0.20、ci=[−0.28, −0.12]（统一方向）
                      ⇒ Q_A−Q_B 的下界 = −ci_upper = **+0.12** = +12pp

    写成 `ci_95_lower` 会把「更好」读成「更差」，而数值符号与量级都毫无异常
    —— 这类错误不会被任何"看起来对不对"的检查抓到，只能靠定向测试。
    """
    out: Dict[str, Any] = {
        "available": False, "ci_95_low_pp": None, "ci_95_high_pp": None,
        "mean_diff_pp": None, "n_pairs": None, "pairing_keys_match": None,
        "ref_method": H.QUALITY_COMPARABLE_REFERENCE, "reason": "",
    }
    sa = r_dcc.get("accuracy_per_sample")
    sb = r_ref.get("accuracy_per_sample")
    if sa is None or sb is None:
        missing = [n for n, v in (("dcc_kv", sa), (H.QUALITY_COMPARABLE_REFERENCE, sb))
                   if v is None]
        out["reason"] = ("缺逐样本数据（%s）：该行没落 accuracy_per_sample"
                         "（旧产物，或该行被阻断/报错而未评测）" % "、".join(missing))
        return out
    if len(sa) != len(sb):
        out["reason"] = ("两行样本数不等（dcc_kv=%d，%s=%d）⇒ 不是同一批样本"
                         % (len(sa), H.QUALITY_COMPARABLE_REFERENCE, len(sb)))
        return out
    ka = r_dcc.get("eval_sample_keys")
    kb = r_ref.get("eval_sample_keys")
    if ka is None or kb is None:
        out["reason"] = ("缺配对身份键（eval_sample_keys）：无法验证两行评的是"
                         "同一批样本、同一顺序 ⇒ 拒绝配对")
        return out
    # **先验键再算数**：键不等就直接拒绝，不进入数值计算。
    if list(ka) != list(kb):
        bad = next((i for i, (x, y) in enumerate(zip(ka, kb)) if x != y), None)
        out["pairing_keys_match"] = False
        out["reason"] = ("配对键序列不一致（第 %s 位起：%r vs %r）⇒ 两行不是同源"
                         "同序的样本，配出来的 CI 无意义"
                         % (bad, ka[bad] if bad is not None else None,
                            kb[bad] if bad is not None else None))
        return out
    out["pairing_keys_match"] = True
    res = R.paired_bootstrap(
        sa, sb, metric_name="accuracy", unit="pp",
        higher_is_better=True, seed=a.seed,
    )
    out.update(
        available=True,
        ci_95_low_pp=_pp_percent(-res.ci_95_upper),   # ← 见 docstring 的方向口径
        ci_95_high_pp=_pp_percent(-res.ci_95_lower),
        mean_diff_pp=_pp_percent(res.mean_diff),      # 原始方向：Q_DCC − Q_ref
        n_pairs=int(res.n_pairs),
        reason="",
    )
    return out


def h2_points_from_rows(a: argparse.Namespace,
                        rows: List[Dict[str, Any]],
                        diagnostics: Optional[Dict[str, Any]] = None,
                        ) -> List[H.H2PerLength]:
    """从主表行数据里抽 H2 的三个原始量，构造逐长度的判据输入。

    配对结构：同一 (model, ctx_len, sync_mode) 下，`dcc_kv` 与
    `kv_budget_shared` 面对同一份提示、同一份评测集，因此两者的差是**配对量**；
    `prefill_speedup` 取 **kernel-matched** 那一列（同核 dense 臂），理由见下。

    为什么 prefill 的分子是 dense 而不是 shared（2026-09-21 修正）
    ---------------------------------------------------------------
    论文 §7.5 把 H2 拆成**互不重叠的三段**：「压缩后的 DCC-KV 与精确注意力
    **不劣**」「DCC-KV 相对共享压缩**更高**」「且 prefill **更快**」。第一段比
    的是 dense，第二段比的是 shared，第三段的「更快」讲的是**系统设计的收益**。
    在 prefill 这一格上，`kv_budget_shared` 同样压过 KV、同样交付 B 长的 KV，
    它不可能比 DCC-KV 慢 10%。原实现写的 `ms_shared / ms_dcc` 得到的是一个结构
    上恒 ≈1 的量：一个恒 ≈1 的比值配 1.10x 的阈值，只能说明它不是论文那句话
    在说的量。
    再从 dense 的端到端数改成 kernel-matched（2026-09-21 同日）
    -------------------------------------------------------
    分子取 dense 解决了"比错对象"，但没解决"用错核"：dense 行走 SDPA 融合核、
    dcc 行走 `attention_kernel` 的显式核，两者的实现差距会整个人进比值里。
    故最终取 **kernel-matched**：分子是**钩子 dense 臂**的目的端耗时（同一个
    显式核、同样带本地块），分母是**钩子 dcc 臂**的目的端耗时，两臂唯一差别
    只剩远端 KV 长度。native 那一列仍会落盘（端到端参考），但**不得**用于 H2。


    三个原始量的来源：
        quality_gain_pp  = (acc_dcc - acc_shared) * 100
        prefill_speedup  = dest_ms_median_dense_kernel / dest_ms_median
                           （= kernel-matched 那一列，见 _kernel_matched_speedup）
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
        `tests/test_adversarial_2026_09_20.py` 的 `test_a4_*` 四条 + 本文件里
        `test_e6_*` 的接线锚点覆盖）。
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
    # dcc_kv 自己那一侧的轴也可能被折叠（`--ranks 1` 时**两个方法都折**），
    # 故它也要按"本次会不会写出该模式"来查，而不是写死 `sync_modes[0]`。
    # 未折叠时 `modes[0] == a.sync_modes[0]`，与旧行为逐位一致。
    modes = dcc_row_modes(a)
    for model in a.models:
        for ctx in a.context_lengths:
            r_dcc = (find_axis_free("dcc_kv", model, ctx) if len(modes) == 1
                     else find("dcc_kv", model, ctx, modes[0]))
            # 基线是单卡方法，其 sync/async 轴同样被折叠 ⇒ 不能按模式查
            # （否则配对恒为空；见 find_axis_free 的说明）。
            r_shr = find_axis_free("kv_budget_shared", model, ctx)
            # ⚠️ dense 行对两个量是**两种身份**，别混（2026-09-21 两度改判）：
            #   · prefill 的分子改成 kernel-matched 后，**不再**需要 dense 行 ——
            #     它的两端都在 dcc_kv 行内测得（原写法 `r_den is None: continue`
            #     会让"dense 行缺席"把 H2 判成 unresolved，是拿无关的行决定另一个
            #     量的可判性）。dense 行只影响 native 那一列。
            #   · 但 quality_comparable 的参照物**就是** dense
            #     （QUALITY_COMPARABLE_REFERENCE），所以它**重新**成为必需项。
            # 两者分开处理：dense 缺席只让前提记 unresolved，不影响 prefill 那一格。
            # dense 是单卡方法（轴恒折叠），但判据仍现算，不写死 ——
            # 若哪天它的 gpu_required 变成 >1，这里会自动跟上。
            if r_dcc is None or r_shr is None:
                continue
            acc_d, acc_s = r_dcc.get("accuracy"), r_shr.get("accuracy")
            sp_km = _kernel_matched_speedup(r_dcc)
            r_den = (find_axis_free(H.QUALITY_COMPARABLE_REFERENCE, model, ctx)
                     if not sync_axis_applies(H.QUALITY_COMPARABLE_REFERENCE, a)
                     else find(H.QUALITY_COMPARABLE_REFERENCE, model, ctx, modes[0]))
            if r_den is None:
                ci: Dict[str, Any] = {
                    "available": False, "ci_95_low_pp": None,
                    "ci_95_high_pp": None, "mean_diff_pp": None, "n_pairs": None,
                    "pairing_keys_match": None,
                    "ref_method": H.QUALITY_COMPARABLE_REFERENCE,
                    "reason": ("参照物行缺席（%s）：「质量相近」前提的参照物就是它，"
                               "无此行则该前提无从判定"
                               % H.QUALITY_COMPARABLE_REFERENCE),
                }
            else:
                ci = _paired_quality_ci(r_dcc, r_den, a)

            # 「质量相近」前提：单侧非劣检验，参照物是 **dense**（不是 shared）——
            # 理由见 hypotheses.QUALITY_COMPARABLE_REFERENCE。它要的是
            # (Q_DCC − Q_dense) 的配对 95% CI 下界，逐样本数据落盘后才算得出。
            comparable = None
            if diagnostics is not None:
                diagnostics.setdefault("lengths", {})[str(ctx)] = dict(
                    ci, model=model,
                    quality_gain_reference="kv_budget_shared",
                )
            if delta is None or floor is None:
                why = ("未声明非劣边界 / 噪声底线（--h2-delta-pp / "
                       "--h2-noise-floor-pp）⇒ 该前提无从判定")
            elif not ci["available"]:
                why = ci["reason"]
            else:
                try:
                    comparable = H.quality_comparable_non_inferior(
                        ci["ci_95_low_pp"], delta_pp=delta,
                        noise_floor_pp=floor, delta_bad_pp=delta_bad)
                    why = ""
                except ValueError as exc:
                    # 参数越出可行区间 ⇒ 本次实验**不具备判定该前提的分辨率**。
                    # 记 unresolved 并**保留原文**，不把它读成「质量确实不相近」
                    # —— 那是两个不同的结论（hypotheses 的 docstring 写死了这条）。
                    comparable = None
                    why = "参数越界（分辨率不足）：%s" % exc
            if comparable is None and diagnostics is not None:
                diagnostics["lengths"][str(ctx)]["unresolved_reason"] = why

            # 量具是否体现压缩收益：由 `_hf` 的契约决定，不在本文件里写死。
            # **并且**这一格必须真的算出了 kernel-matched 那一列：契约声明灵敏、
            # 表里却没有那一列（dcc_kv 行没接钩子 / 同核 dense 臂没跑），同样是
            # 「量具给不出数」。若照旧记 nan 而置 True，`h2_pass` 会把
            # `nan >= 1.10` 判成假 ⇒ **把「没测出来」报成「未达标」**。
            instrument_ok = (bool(_hf.PREFILL_TIMING_CONSUMES_COMPACT_KV)
                             and sp_km is not None)
            # 两个量**各自独立**记录：prefill 那一格没测出，不该把已经测到的
            # 质量差一起抹成 nan（那会让 worst_quality_gain_pp 丢掉一个真实值）。
            points.append(H.H2PerLength(
                length=ctx,
                quality_gain_pp=((acc_d - acc_s) * 100.0
                                 if (acc_d is not None and acc_s is not None)
                                 else float("nan")),
                prefill_speedup=(float("nan") if sp_km is None
                                 else float(sp_km)),
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
    diagnostics: Dict[str, Any] = {}
    points = h2_points_from_rows(a, rows, diagnostics)
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
        "（缺逐样本数据 / 两边键不同源 / 未声明非劣边界与噪声底线 / 参数越界），"
        "记入 h2_unresolved_lengths，**不计入** h2_n_failed —— "
        "分辨率不足不等于未达标。"
        "逐长度的具体原因见 h2_comparable_diagnostics.lengths[<ctx>]"
        ".unresolved_reason；判定量是该长度的 ci_95_low_pp。"
    )
    out["h2_comparable_diagnostics"] = diagnostics
    out["h2_comparable_note"] = (
        "前提的参照物是 dense（hypotheses.QUALITY_COMPARABLE_REFERENCE）："
        "检验量 = Q_DCC − Q_dense 的配对 95% CI 下界（ci_95_low_pp，单位百分点）。"
        "它与 h2_worst_quality_gain_pp 的参照物**不同**"
        "（后者相对 kv_budget_shared，回答的是 H2 前半句的「提升 ≥1.5pp」），"
        "两个数不可互相代入。配对前先逐位比对 eval_sample_keys；"
        "键不一致时拒绝配对（记 unresolved），不产出一个"
        "「数值正常但配错样本」的 CI。"
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
            "压缩带来的 KV 长度收益。分子口径为 **kernel-matched**"
            "（钩子 dense 臂 / 钩子 dcc 臂，同一算子核 "
            "attention_kernel.dcc_kv_attention），"
            "见 _kernel_matched_speedup 的说明。")
    out["h2_prefill_instrument_note"] = (
        "prefill_instrument_valid 取「契约灵敏 **且** 这一格真的算出了 "
        "kernel-matched 那一列」的合取：契约声明灵敏、表里却没有那一列"
        "（dcc_kv 行未接钩子 / 同核 dense 臂没跑）同样是「量具给不出数」。"
        "两者都记 instrument，**不记 failed** —— 若照旧让 nan 参与比较，"
        "「nan >= 1.10」为假，判定会变成「未达标」，即把没测出来报成没有收益。"
        "哪一行的哪一列缺了，看 payload 的 prefill_speedups 声明。"
    )
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
    p.add_argument("--dcc-world", dest="dcc_world", type=int, default=None,
                   help="dcc_kv 行的源端段数：**单卡模拟**「把源序列切成 W 段、"
                        "每段各构造一份紧凑块」，即 W 个源端设备各发一条边。"
                        "**无默认值** —— 它决定每边的预算（per_edge 口径下 "
                        "B_total/world），给个默认值会把「本次模拟了几个源端设备」"
                        "变成没人声明过的假设。单卡下它**不**体现逐边条件化的"
                        "代价与收益（所有段看到同一批 query）。")
    p.add_argument("--dcc-budget-mode", dest="dcc_budget_mode", type=str,
                   default="per_edge", choices=["per_edge", "total"],
                   help="dcc_kv 的每边预算口径：per_edge = B_total/world（主口径）；"
                        "total = 每边都是 B_total（总预算随 world 放大，仅供敏感性"
                        "对照，**不得**用于 H2 的质量主张）")
    p.add_argument("--ranks", type=int, default=1,
                   help="本次实际使用的 rank 数。=1 时**所有**跨设备轴（sync/async）"
                        "都被折叠 —— 单卡下该轴不是自变量，两行必然逐位相同，"
                        "而表里'两行一样'会被读成'异步没有收益'。")
    p.add_argument("--lambda-beta", dest="lambda_beta", type=float, default=None,
                   help="β 拟合的 ridge 正则 λ_β。不给时用 "
                        "dcc_kv_ref.DEFAULT_LAMBDA_BETA（与 src/experiment_metadata.py "
                        "同源），解析后的实际取值会落进每行产物")
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
    if "dcc_kv" in a.methods and a.dcc_world is None:
        print("阻断：--methods 含 dcc_kv，但未给 --dcc-world（源端段数）。")
        print("  --dcc-world 刻意没有默认值：它决定每边的预算（per_edge 口径下")
        print("  B_total/world），给个默认值会把「本次模拟了几个源端设备」变成")
        print("  没人声明过的假设。例：--dcc-world 4 = 源序列切成 4 段。")
        return 2


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
                modes = (a.sync_modes if sync_axis_applies(method, a)
                         else [SYNC_MODE_NA])
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
                            "sync_mode_applicable": sync_axis_applies(method, a),
                            "status": "error", "error_type": type(e).__name__,
                            "error": str(e),
                            "traceback": traceback.format_exc().splitlines()[-8:],
                            "accuracy": None, "prefill_ms_median": None,
                        }
                    rows.append(r)
                    if r["status"] == "ok":
                        acc_str = ("n/a" if r["accuracy"] is None
                                   else f"{r['accuracy']:.4f}")
                        sp_km = _kernel_matched_speedup(r)
                        sp_str = ("km=n/a" if sp_km is None
                                  else f"km={sp_km:.3f}x")
                        print(f"  {model.split('/')[-1]:<28} L={ctx_len:<6} "
                              f"{sync_mode:<5} {method:<17} "
                              f"prefill={r['prefill_ms_median']:9.2f}ms "
                              f"(p95={r['prefill_ms_p95']:9.2f})  "
                              f"{r['tokens_per_s_median']:10.1f} tok/s  "
                              f"peak={r['peak_memory_gb']:5.2f}GB  "
                              f"acc={acc_str}  {sp_str}")
                    else:
                        print(f"  {model.split('/')[-1]:<28} L={ctx_len:<6} "
                              f"{sync_mode:<5} {method:<17} [{r['status']}] "
                              f"{r.get('error_type', '')}")
        del lm
        torch.cuda.empty_cache()

    # 达标性判定
    # 两列加速比必须在 compute_h2 之前补上 —— H2 读的正是 kernel-matched 那列。
    prefill_speedups = attach_prefill_speedups(a, rows)
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
        notes=(f"methods={a.methods} dcc_world={a.dcc_world} "
               f"dcc_budget_mode={a.dcc_budget_mode} ranks={a.ranks}"),
    )
    admissible = _env.gate_guard_for_report(meta)

    payload: Dict[str, Any] = {
        "experiment": "E6",
        "rows": rows,
        "metadata": meta.to_dict(),
        "report_admissibility": admissible,
        "grid": {            "dcc_world": a.dcc_world,
            "dcc_budget_mode": a.dcc_budget_mode,
            "ranks": a.ranks,

            "models": a.models, "context_lengths": a.context_lengths,
            "sync_modes": a.sync_modes, "methods": a.methods,
            "n_points_planned": planned_points(a),
            "n_points_planned_naive_product": (len(a.models) * len(a.context_lengths)
                                               * len(a.sync_modes) * len(a.methods)),
            "sync_axis_expanded_methods": [m for m in a.methods
                                           if sync_axis_applies(m, a)],
            "sync_axis_collapsed_methods": [m for m in a.methods
                                            if not sync_axis_applies(m, a)],
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
            # 与 measure_point 同源：同一份 prompt_split，不在两处各算一遍
            "source_len_by_context": {str(c): _hf.prompt_split(c, a.dest_fraction)[0]
                                      for c in a.context_lengths},
            "dest_len_by_context": {str(c): _hf.prompt_split(c, a.dest_fraction)[1]
                                    for c in a.context_lengths},
            "prefill_speedup_columns": (
                "两列，各自命名、不得互相冒充。"
                "kernel_matched = 钩子dense臂目的端耗时 / 钩子dcc臂目的端耗时"
                "（同一算子核 attention_kernel.dcc_kv_attention，唯一变量是"
                "远端 KV 长度；**H2 消费这一列**）；"
                "native = dense 行端到端 prefill / dcc 行端到端 prefill"
                "（异核：SDPA 融合核 vs 显式核，且含构造代价差异；"
                "只作端到端参考，不得当作机制收益）。"
                "kernel_matched 的两端都在 dcc_kv 行内测得；"
                "native 需要 dense 行，由 attach_prefill_speedups 在收齐行后补。"
            ),
            "meaning": ("True ⇒ 压缩臂的计时窗口里含一次目的端前向，其 past cache "
                        "是裁剪后的 KV，prefill_speedup 可反映压缩收益；"
                        "False ⇒ prefill 一列对压缩方法只含开销、不含收益，"
                        "该长度记 unresolved。"),
        },
        "prefill_speedups": prefill_speedups,
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
