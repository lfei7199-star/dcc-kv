"""2026-09-23 论文架构 ↔ 代码工程一致性审查的回归锚点。

体例与 `test_audit_2026_09_22.py` 一致：每条对应一处**已修**的偏差，
且都先说明「它本可以不被发现」。

本轮的核心发现是一类**比数字错更严重**的偏差：论文声称的机制在代码里
**根本不存在**。公式可以逐式对得上，而 §4.4（唯一原创系统贡献）点名的
三项机制在 79 个 `.py` 里零命中：

  〔A〕三维度存在性 —— 论文 §4.4 写「循环错位的 All-to-Allv 调度：
       第 p 轮向 [s+p]_N 发送」；实测是**单次全局** `all_to_all_single`
       （`src/distributed/comm.py:291`），全仓零命中 `[s+p]`。
  〔B〕论文 §4.4 写「使用独立的通信流与计算流，并以逐块完成事件建立依赖」；
       实测是 `async_op` 句柄 + **块级双缓冲**（`experiments/gpu/_comm.py:296-327`），
       全仓零命中 `torch.cuda.Stream` / `torch.cuda.Event`。
  〔C〕论文 §4.2 写代表 Query 走 `N-1` 轮 `Ring All-Gather`；全仓零命中
       `all_gather`。

**为什么它本可以不被发现**：作者读自己的论文时，「异步」「重叠」「变长」
这些词与代码里的 `async_op`、双缓冲、`send_sizes` 语义**高度神似** ——
`grep async` 会命中，`grep All-to-Allv` 也会命中，唯独三个具体机制名
（错位轮次 / CUDA Stream / all_gather）一个都不命中。这正是
`literature-transplant-fidelity-audit` 技能原先三轴（公式保真 / 证据区间 /
假设对照）**恰好漏掉**的那类：公式忠实、区间正确、假设对齐，
而机制从未被建造。

本轮的处理是**改纸面去描述真实实现**，而非改代码去圆纸面。理由不是省事：
单次集合通信在带宽利用上本就优于 `N-1` 轮环形，且改代码需要 GPU 验证，
而本机无 CUDA ⇒ 无法证明改动正确，那才是真正的不准确。

因此本文件的守卫分三组：

  组 1（λ_v 单一事实源）—— 消除「论文写死的 λ_v」与「实际跑的 λ_v」分叉。
  组 2（论文不再声称未建造的机制）—— 正向锚点 + 禁词锚点，双向锁死。
  组 3（退化前提被显式记录）—— 论文声称的重叠窗口依赖等预算分块，
       而代码在变长预算下**退化为单块**；该退化必须既在代码里可测、
       也在纸面上被写明。
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAPER_DIR = REPO_ROOT / "paper"
PAPER_FILES = (PAPER_DIR / "main.tex",) + tuple(
    sorted((PAPER_DIR / "sections").glob("*.tex"))
)


def _paper_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in PAPER_FILES)


# ===========================================================================
# 组 1：λ_v 的单一事实源
# ===========================================================================

def test_l1_lambda_value_has_a_single_source_constant():
    """λ_v 的默认值必须有一个具名常量，而不是散落的字面量。

    此前 `1e-3` 在 `src/` 与 `experiments/` 中重复出现十余次（2026-09-23
    实测 36 个 `lambda_value` 出现点，其中 12 处是默认值定义）。λ_β 有单一
    事实源（`calibration.DEFAULT_LAMBDA_BETA`）而 λ_v 没有 —— 这个不对称本身
    就是缺陷：论文把两者并列为拟合超参，代码却只对其中一个做了收敛。
    """
    from src.dcc_kv_ref import DEFAULT_LAMBDA_VALUE

    assert DEFAULT_LAMBDA_VALUE == pytest.approx(1e-3)
    assert DEFAULT_LAMBDA_VALUE > 0.0, "λ_v=0 会让 B>M 时的 XᵀX 奇异，不可作运行默认值"


def test_l2_default_value_is_wired_to_the_constant():
    """四个默认值入口必须全部引用同一常量，而非各自写死。"""
    from src.dcc_kv_ref import (
        DEFAULT_LAMBDA_VALUE,
        build_compact_kv,
        fit_compact_value,
    )
    from src.baselines.fast_kv_cpu import FastKVConfig
    from src.distributed.attention_hook import HookConfig
    from src.experiment_metadata import ExperimentMetadata

    assert inspect.signature(build_compact_kv).parameters["lambda_value"].default \
        == DEFAULT_LAMBDA_VALUE
    assert inspect.signature(fit_compact_value).parameters["lambda_reg"].default \
        == DEFAULT_LAMBDA_VALUE
    assert FastKVConfig().lambda_value == DEFAULT_LAMBDA_VALUE
    assert HookConfig(budget=8).lambda_value == DEFAULT_LAMBDA_VALUE
    # experiment_metadata 刻意不 import dcc_kv_ref（元数据模块不应依赖参考实现），
    # 其一致性由这条断言锁死 —— 与 lambda_beta 的同源守卫同模式。
    assert ExperimentMetadata(run_id="t").lambda_value == DEFAULT_LAMBDA_VALUE


def test_l3_no_default_literal_left_in_src():
    """`src/` 内不应再有 `lambda_value: float = 1e-3` 这类字面量默认值。

    注释里提到 1e-3 是允许的（说明历史值），但**签名默认值**必须是常量。
    匹配的是带类型标注的参数默认值，不是任意出现。
    """
    import re

    pattern = re.compile(r"lambda_value\s*:\s*float\s*=\s*1e-3")
    # experiment_metadata.py 刻意不 import dcc_kv_ref（元数据模块不应依赖参考实现），
    # 其字面量的一致性由 test_l2 的等值断言锁死，故此处豁免。
    exempt = {pathlib.Path("src") / "experiment_metadata.py"}
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if path.relative_to(REPO_ROOT) in exempt:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}")
    assert not offenders, (
        "λ_v 的默认值仍有写死的字面量（应引用 DEFAULT_LAMBDA_VALUE）：\n  "
        + "\n  ".join(offenders)
    )


# ===========================================================================
# 组 2：论文不得声称未建造的机制
# ===========================================================================

def test_p1_paper_no_longer_claims_round_robin_ring_shift():
    """§4.4 不得再声称「循环错位 / 环形错位」的按轮调度。

    代码是单次全局集合通信，没有任何按设备对的轮次概念。
    """
    text = _paper_text()
    for banned in ("循环错位", "环形错位"):
        assert banned not in text, (
            f"论文仍出现「{banned}」——代码里没有对应的轮次调度"
            "（`src/distributed/comm.py:291` 是单次 all_to_all_single）"
        )


def test_p2_paper_no_longer_claims_independent_cuda_streams():
    """§4.4 不得再声称「独立的通信流与计算流」或「逐块完成事件」。

    这些术语专指 CUDA Stream / Event；代码用的是 async_op 句柄 + 块级双缓冲。
    用「流」这个词描述双缓冲会让读者（和审稿人）以为存在两套 CUDA 流。
    """
    text = _paper_text()
    for banned in ("通信流与计算流", "通信流则继续接收", "逐块完成事件"):
        assert banned not in text, (
            f"论文仍出现「{banned}」——全仓零命中 `torch.cuda.Stream` / "
            "`torch.cuda.Event`；真实机制是 `_comm.py` 的块级双缓冲"
        )


def test_p3_paper_no_longer_claims_ring_all_gather_for_query_summary():
    """§4.2 不得再把代表 Query 的交换写成已实现的 `Ring All-Gather`。

    全仓零命中 `all_gather`。设计层面的「可用等大小集合通信实现」是允许的，
    但不得读作已接线。
    """
    text = _paper_text()
    assert "Ring All-Gather" not in text, (
        "论文仍在用「Ring All-Gather」描述代表 Query 交换 —— 全仓零命中 "
        "`all_gather`，该通路尚未接线"
    )


def test_p4_paper_documents_the_actually_built_mechanism():
    """正向锚点：纸面必须描述**真实存在**的那个机制。

    只禁词是不够的 —— 删干净了也可以什么都不写。论文必须明确写出
    「一次变长 All-to-Allv（先交换长度与偏移）+ 块级双缓冲」这一组合，
    因为这才是唯一原创贡献的真实内容。
    """
    text = _paper_text()
    assert "块级双缓冲" in text, "论文没有描述实际采用的块级双缓冲流水"
    assert "携带变长分段表" in text, \
        "论文没有说明数据交换由单次携带变长分段表的集合通信完成"


# ===========================================================================
# 组 3：重叠窗口的退化前提必须被显式记录
# ===========================================================================

def test_d1_effective_chunk_count_degenerates_on_variable_budgets():
    """变长预算下 `effective_chunk_count` 必须返回 1（无重叠窗口）。

    这是论文 §4.4 开篇前提（「不同有向边采用不同预算」）与「重叠」之间
    那条**冲突**的代码侧锚点：论文的核心场景关掉了论文的核心机制。
    该断言把这条冲突钉在测试里，使它不可能再被静默忽略。
    """
    from experiments.gpu._comm import effective_chunk_count

    # 逐边预算不同 ⇒ 退化为单块，没有可重叠的窗口
    assert effective_chunk_count([8, 16, 16], [8, 16, 16], n_chunks=8) == 1
    assert effective_chunk_count([16, 16, 16], [8, 8, 16], n_chunks=8) == 1
    # 不对称 All-to-All（两侧各自全等但 send != recv）⇒ 亦退化为单块：
    # 每个块用同一个 [k]*world 同时声明两侧，发 16 收 8 时块数相同而每块行数不同，
    # 声明必然对不上。此条为 2026-09-23 实测发现并修复的缺陷。
    assert effective_chunk_count([16, 16, 16], [8, 8, 8], n_chunks=8) == 1
    # 两侧全等且相等 ⇒ 才存在被切分出的多个块
    assert effective_chunk_count([16, 16, 16], [16, 16, 16], n_chunks=4) > 1
    # n_chunks<=1 或 B<2 亦退化
    assert effective_chunk_count([16, 16], [16, 16], n_chunks=1) == 1
    assert effective_chunk_count([1, 1], [1, 1], n_chunks=8) == 1


def test_d2_paper_documents_the_overlap_degeneration():
    """纸面必须写明退化条件，否则读者会把 `speedup≈1.0` 读成实现缺陷。"""
    text = _paper_text()
    assert "退化为单块" in text or "退化为单块发送" in text, (
        "论文未写明：逐边预算不对齐时流水退化为单块、无重叠窗口"
    )


def test_d3_paper_speedup_bound_is_expressed_in_chunks_not_rounds():
    """§5.4 的加速上界必须按「可重叠单元数」表述，而不是按 $N-1$ 轮。

    原先写 $N-1$ 轮 ⇒ 与 `run_async_pipeline` 的实际循环长度（消息内块数）
    不同源，导致 §6.3 要求的「按式(speedup-bound) 归因实测差距」无法执行：
    分子分母量的不是同一个调度。
    """
    analysis = (PAPER_DIR / "sections" / "05-analysis.tex").read_text(encoding="utf-8")
    assert "可重叠" in analysis, (
        "§5.4 未把加速上界的推导量改成「可重叠单元数」"
    )
