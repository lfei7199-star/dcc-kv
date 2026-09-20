"""G2：异步 All-to-Allv 接**真实前向**的桥接层。

缺口
----
本仓库原先有两块互不相连的东西：

1. `experiments/gpu/_comm.py` 的同步/异步流水 —— 计量骨架齐备，但它的计算侧
   是调用方传进来的 `comp_work` 回调。A5 传的是 `make_comp_work`，那是**规模
   模拟**算子（查询行数由接收元素数换算），算的不是 DCC-KV 的注意力。
2. `src/dcc_kv_ref/attention_kernel.py` 的算子核（G1）—— 算的是真东西，
   但它只认 `CompactKV` 对象，不认通信层收到的**打包行**
   （`[K | β | V]`，见 `_comm.pack_compact_edge`）。

中间缺的正是本模块：把「收到的行」解码回紧凑边、喂进算子核、把各块的部分
结果归并成最终输出，且在同步与异步两种调度下**给出逐位相同的答案**。

两条不变式，强度不同，别混说
------------------------------
**① 调度不变式（逐位）。** 在**固定的** `n_chunks` 下，`mode="sync"` 与
`mode="async"` 的输出逐位相同。理由：`run_sync_pipeline` 与
`run_async_pipeline` 都以"第 0 块、第 1 块、…"的顺序调用 `comp_work`
（异步版是等前一块到达后再算前一块，顺序不变），因此只要 `comp_work` 是
**同一个闭包**（内部按调用顺序 append），两条路径的 partial 列表就逐元素同序，
归并结果自然逐位相同。
实证：`tests/test_forward_pipeline.py::test_sync_and_async_agree_bitwise`
在 n_chunks ∈ {1,2,4,8} 上全部逐位相等。

**② 分块不变式（ULP 级，不是逐位）。** 改变 `n_chunks` **会**改变输出。
原因不是实现瑕疵：每个源边的 softmax 被切成若干子 softmax，再用 lse 归并
（flash-attention 的 combine 步），这与对整块一次 softmax 是**两个不同的
浮点求和**。实测（D_H=8、D_V=6、两侧各 8 行、float32）：

    n_chunks=1 → 2 个 partial（每源一块）
    n_chunks=4 → 8 个 partial
    与 n_chunks=1 的最大绝对差 ≤ 1.79e-7（float32 的 ULP ≈ 5.96e-8）
    ⇒ 1–3 ULP；float64 下同一比较为 2.2e-16（≈1 ULP）

这条与本仓库 G1 的 `query_chunk` 是同一现象（那里沿 query 维分块改变了 GEMM 的
M 维分块核；这里沿 key 维切开了 softmax 的求和集合）。**故 `n_chunks` 是
时延/显存旋钮，不是数值旋钮。** 需要逐位可复现的场合必须固定 `n_chunks`
并把该值写进产物（`ForwardResult.n_chunks_requested` 已落盘）。

为什么①仍必须成立：若连调度都改变答案，A5 的加速比与 E6 的准确率就建立在
两套数值上，测出的差异里混着归并顺序伪影（E0 已证归并顺序在 FP32 下可辨识）。

实现要点
--------
- **部分结果的收集顺序**：`_comm.run_sync_pipeline` 与 `run_async_pipeline`
  都以"第 0 块、第 1 块、…"的顺序调用 `comp_work`（异步版等前一块到达后再算
  前一块，顺序不变）。因此只要 `comp_work` 是**同一个闭包**（内部按调用顺序
  append），两条路径的 partial 列表就逐元素同序 ⇒ 归并结果逐位相同。
- **本地块永远最后归并**：两条路径都在循环结束后追加，位置一致。
- **变长预算退化为单块**：`_comm._split_chunks` 在逐边预算不等时不切块，
  此时异步没有任何可重叠的窗口。这不是失败，是这条轴上没有机会 ——
  `ForwardResult.chunks_effective` 会如实记下 1，免得读到 `speedup≈1.0`
  的人去修一个没坏的实现。
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.gpu import _comm  # noqa: E402

from src.dcc_kv_ref import CompactKV  # noqa: E402
from src.dcc_kv_ref import attention_kernel as K  # noqa: E402


MODES = ("sync", "async")


# =============================================================================
# 行布局
# =============================================================================

@dataclass(frozen=True)
class EdgeLayout:
    """一次 All-to-Allv 的行布局：每段属于哪个源/目的端、每段多少行。

    行布局的口径与 `_comm.all_to_all_v` 一致：`send` 按 **dst 顺序拼接**，
    `recv` 按 **src 顺序拼接**。
    """

    world: int
    d_h: int
    d_v: int
    send_sizes: Tuple[int, ...]
    recv_sizes: Tuple[int, ...]

    @property
    def feature(self) -> int:
        return self.d_h + 1 + self.d_v

    @property
    def is_uniform(self) -> bool:
        return (len(set(self.send_sizes)) == 1
                and len(set(self.recv_sizes)) == 1)

    def total_send_rows(self) -> int:
        return int(sum(self.send_sizes))

    def total_recv_rows(self) -> int:
        return int(sum(self.recv_sizes))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world": self.world,
            "d_h": self.d_h,
            "d_v": self.d_v,
            "feature": self.feature,
            "send_sizes": list(self.send_sizes),
            "recv_sizes": list(self.recv_sizes),
            "is_uniform": self.is_uniform,
        }


def make_uniform_layout(world: int, budget: int, d_h: int, d_v: int) -> EdgeLayout:
    """等预算布局（A2/A4/A5 的默认情形）。"""
    if int(budget) < 1:
        raise ValueError(f"budget 必须 >= 1，得到 {budget}")
    return EdgeLayout(world=int(world), d_h=int(d_h), d_v=int(d_v),
                      send_sizes=tuple([int(budget)] * int(world)),
                      recv_sizes=tuple([int(budget)] * int(world)))


# =============================================================================
# 打包 / 解码
# =============================================================================

def pack_edges(compacts: Sequence[CompactKV], layout: EdgeLayout) -> torch.Tensor:
    """把逐 dst 的紧凑边打成一张 `[sum(send_sizes), d_h+1+d_v]` 的 outbound 张量。

    Raises:
        ValueError: 边数与 world 不符，或某条边的行数与 `send_sizes` 声明不符。
            不静默取实际值 —— 声明与实际不符意味着两侧对预算的理解不同源
            （与 `DistributedComm.all_to_all_v` 的 Step 1b 同一条纪律）。
    """
    if len(compacts) != layout.world:
        raise ValueError(
            f"紧凑边数 {len(compacts)} != world {layout.world}；"
            "outbound payload 必须按 dst 顺序给出每一条边"
        )
    rows = []
    for i, (ck, want) in enumerate(zip(compacts, layout.send_sizes)):
        got = int(ck.keys.shape[0])
        if got != int(want):
            raise ValueError(
                f"第 {i} 条边的行数 {got} 与 send_sizes[{i}]={int(want)} 不符；"
                "各 rank 必须共享同一份预算配置"
            )
        if int(ck.values.shape[0]) != got:
            raise ValueError(
                f"第 {i} 条边的 K/V 行数不一致：{got} 对 {int(ck.values.shape[0])}"
            )
        rows.append(_comm.pack_compact_edge(ck.keys, ck.logit_bias, ck.values))
    if not rows:
        return torch.empty(0, layout.feature)
    return torch.cat(rows, dim=0)


def chunk_source_sizes(rows: int, recv_sizes: Sequence[int], world: int) -> List[int]:
    """某个接收块里，各源各有多少行。

    等预算时 `_split_chunks` 会把每个 dst 的片段各取等长一段，于是块内
    `rows == k * world` ⇒ 均分。变长预算下不做分块（块 = 完整接收缓冲）
    ⇒ 直接用 `recv_sizes`。

    Raises:
        ValueError: 两种情况都不成立（说明布局与块对不上，静默猜测只会
            把行错配到别的源上，且错得很隐蔽）。
    """
    rows = int(rows)
    world = int(world)
    flat = [int(r) for r in recv_sizes]
    if len(flat) != world:
        raise ValueError(f"recv_sizes 长度 {len(flat)} != world {world}")
    if rows == 0:
        return [0] * world
    if rows == sum(flat):
        return flat
    if len(set(flat)) == 1 and rows % world == 0:
        return [rows // world] * world
    raise ValueError(
        f"接收块行数 {rows} 既不是 sum(recv_sizes)={sum(flat)}，"
        f"也不能在 world={world} 下均分 —— 布局与块对不上"
    )


def decode_edges(
    rows: torch.Tensor,
    sizes: Sequence[int],
    d_h: int,
    d_v: int,
) -> List[CompactKV]:
    """把接收到的打包行解码成紧凑边（按 src 顺序）。

    解码规则与 `_comm.pack_compact_edge` 必须严格互逆：列布局 `[K | β | V]`，
    β 在第 `d_h` 列。**β 不能丢**：它是权重的一部分，丢了算的就是"没有 β 的
    注意力"，与 DCC-KV 的实际算子不是同一个（这一条在 `make_comp_work` 上
    已经踩过一次）。

    `selected_indices` 取恒等，且**本层不支持因果掩码**：跨设备只传紧凑表示，
    源块内的原始位置由通信前的约定携带，本层不做位置推断。

    ⚠️ 自查修正（2026-09-18）：原文写「需要因果掩码的调用方应显式传 positions」，
    但 `decode_edges` 与 `pipelined_attention` 都没有 `positions` 形参 ——
    那是一条**无法执行**的引导。需要因果语义请走
    `attention_kernel.dcc_kv_attention`（它接受 `query_positions` 与
    `remote_offsets`，且缺偏移时会显式报错而不是静默按 0 处理）。
    把行序当位置前缀是仓库里犯过的错，见 `attention_kernel` 模块文档第 4 条。

    行数为 0 的边被跳过（空块不该进入算子核，核会直接报错）。
    """
    out: List[CompactKV] = []
    offset = 0
    feature = int(d_h) + 1 + int(d_v)
    if int(rows.shape[0]) != int(sum(sizes)):
        raise ValueError(
            f"块行数 {int(rows.shape[0])} != sum(sizes)={int(sum(sizes))}"
        )
    if int(rows.shape[-1]) != feature:
        raise ValueError(
            f"特征维 {int(rows.shape[-1])} != d_h+1+d_v={feature}；"
            "打包布局与解码参数不同源"
        )
    for size in sizes:
        size = int(size)
        if size == 0:
            continue
        seg = rows[offset:offset + size]
        offset += size
        out.append(CompactKV(
            keys=seg[:, :int(d_h)],
            logit_bias=seg[:, int(d_h)],
            values=seg[:, int(d_h) + 1:],
            selected_indices=torch.arange(size, device=seg.device, dtype=torch.long),
        ))
    return out


# =============================================================================
# 流水前向
# =============================================================================

@dataclass
class ForwardResult:
    """一次流水前向的产出。

    out: 归并后的注意力输出 `[..., Lq, d_v]`。
    timing: `_comm.PipelineTiming`（total_ms / comm_ms / comp_ms）。
    chunks_effective: 实际块数。**1 表示这条轴上没有可重叠的窗口**
        （变长预算、B<2、或 n_chunks<=1），不是实现故障。
    n_partials: 参与归并的部分结果数（= 非空紧凑边数 + 本地块）。
    partial_order: 部分结果的收集顺序描述，供产物留痕。
    """

    out: torch.Tensor
    mode: str
    n_chunks_requested: int
    chunks_effective: int
    n_partials: int
    n_edges: int
    local_included: bool
    partial_order: str
    timing: _comm.PipelineTiming

    def to_dict(self) -> Dict[str, Any]:
        """落盘口径。

        同步/异步的 `comm_ms` **不是同一个量**，不能当同一列读
        ------------------------------------------------------
        异步流水里，每轮循环开头就把本块通信发出去了，然后
        `device_sync()`（= `torch.cuda.synchronize()`，device-wide）在本段的
        收尾把**正在飞的下一块传输**一并等掉。于是下一轮的 `wait()` 立即返回，
        `comm_ms` 只剩「发起开销」；真正的传输时间被折进了 `comp_ms`。
        桩模型实测（单块传输 20ms / 单块计算 5ms / 4 块）：同步臂
        `comm=85.8ms`，异步臂 `comm=20.5ms`（≈ 一块），实际通信总量 80ms。
        拿这两个数作比会得出「异步把通信消掉了 76%」——那是度量产物。

        ⇒ 异步行的 `t_comm_ms` 置 None（不是 0，0 会被读成"通信为零"），
        原值保留在 `t_comm_ms_raw` 供复核；`t_comp_ms` 保留但明确标为上界。
        跨同步模式**只有 `t_total_ms` 可比**。
        """
        sync = self.mode == "sync"
        return {
            "mode": self.mode,
            "n_chunks_requested": self.n_chunks_requested,
            "chunks_effective": self.chunks_effective,
            "overlap_window_available": self.chunks_effective > 1,
            "n_partials": self.n_partials,
            "n_edges": self.n_edges,
            "local_included": self.local_included,
            "partial_order": self.partial_order,
            "t_total_ms": self.timing.total_ms,
            "t_comm_ms": self.timing.comm_ms if sync else None,
            "t_comm_ms_raw": self.timing.comm_ms,
            "t_comm_ms_note": ("同步臂：通信段与计算段由 device_sync 分开，可作拆解用"
                              if sync else
                              "异步臂：此值只剩发起开销（device_sync 已把飞行中的"
                              "下一块等掉），**不是通信时间**；跨同步模式只比 t_total_ms"),
            "t_comp_ms": self.timing.comp_ms,
            "t_comp_ms_note": ("" if sync else
                               "异步臂：上界，吸收了尚未完成的下一块传输"),
            "timing_decomposition_valid": sync,
        }


def pipelined_attention(
    query: torch.Tensor,
    payload: torch.Tensor,
    layout: EdgeLayout,
    *,
    mode: str = "async",
    n_chunks: int = 1,
    local_keys: Optional[torch.Tensor] = None,
    local_values: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> ForwardResult:
    """DCC-KV 的流水前向：远端紧凑边 + 本地精确块 → 一个输出。

    两条路径（`mode="sync"` / `"async"`）**共用同一个 comp 闭包**，因此部分结果
    的收集顺序逐元素相同 ⇒ 在**同一 `n_chunks`** 下归并输出逐位相同。
    见模块文档的两条不变式：调度不变式是逐位的，`n_chunks` 的不变式只是 ULP 级
    —— **分块会改变答案**（实测 float32 下 1–3 ULP），它是时延/显存旋钮而非
    数值旋钮，需要逐位复现时必须固定它。

    Args:
        query: `[..., Lq, d_h]` 目的端 query（两条路径必须传同一张，且不是原地改）。
        payload: `pack_edges(...)` 的产出，按 dst 顺序拼接。
        layout: 行布局（两侧都必须与真实传输一致）。
        mode: "sync" 或 "async"。
        n_chunks: 请求的分块数；实际块数见 `ForwardResult.chunks_effective`。
        local_keys / local_values: 目的端本地块（不压缩）的 K/V。给了就参与归并，
            且**固定为最后一个 partial**。
        scale: 缩放；默认 1/sqrt(d_h)。

    Raises:
        ValueError: mode 非法、payload 行数与 layout 不符、或给了 keys 没给 values。
    """
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一，得到 {mode!r}")
    if int(payload.shape[0]) != layout.total_send_rows():
        raise ValueError(
            f"payload 行数 {int(payload.shape[0])} != sum(send_sizes)="
            f"{layout.total_send_rows()}"
        )
    if (local_keys is None) != (local_values is None):
        raise ValueError("local_keys 与 local_values 必须同时给出或同时省略")

    partials: List[K.PartialAttention] = []
    edge_counts: List[int] = []

    def _comp(recv: torch.Tensor) -> None:
        sizes = chunk_source_sizes(int(recv.shape[0]), layout.recv_sizes, layout.world)
        for ck in decode_edges(recv, sizes, layout.d_h, layout.d_v):
            partials.append(K.compact_kv_attention(
                query, ck, scale=scale, return_lse=True))
            edge_counts.append(int(ck.keys.shape[0]))

    runner = (_comm.run_sync_pipeline if mode == "sync"
              else _comm.run_async_pipeline)
    timing = runner(payload, list(layout.send_sizes), list(layout.recv_sizes),
                    _comp, n_chunks=int(n_chunks))

    local_included = local_keys is not None
    if local_included:
        partials.append(K.dense_attention(
            query, local_keys, local_values,
            causal=False, scale=scale, return_lse=True))
        edge_counts.append(int(local_keys.shape[0]))

    if not partials:
        raise ValueError(
            "没有任何可见的键：远端边全为空且未提供本地块，无法计算注意力"
        )

    out = K.merge_partial_attention(partials)
    return ForwardResult(
        out=out,
        mode=mode,
        n_chunks_requested=int(n_chunks),
        chunks_effective=_comm.effective_chunk_count(
            layout.send_sizes, layout.recv_sizes, int(n_chunks)),
        n_partials=len(partials),
        n_edges=len(edge_counts) - (1 if local_included else 0),
        local_included=local_included,
        partial_order="远端边按块序 → 源序；本地块固定最后",
        timing=timing,
    )


def assert_same_answer(sync: ForwardResult, async_: ForwardResult) -> None:
    """断言同步与异步两路给出**逐位相同**的输出。

    这是 G2 的核心不变式，单独成函数以便在实验脚本里当发布前自检调用
    （不只是测试里用一次）。
    """
    if not torch.equal(sync.out, async_.out):
        diff = float((sync.out - async_.out).abs().max())
        raise AssertionError(
            f"同步与异步的输出不一致（最大绝对差 {diff:.3e}）。"
            "流水改变了答案 —— 归并顺序或部分结果集合被改动了，"
            "此时 A5 的加速比与 E6 的准确率建立在两套数值上，不可混用。"
        )


__all__ = [
    "MODES",
    "EdgeLayout",
    "make_uniform_layout",
    "pack_edges",
    "chunk_source_sizes",
    "decode_edges",
    "ForwardResult",
    "pipelined_attention",
    "assert_same_answer",
]
