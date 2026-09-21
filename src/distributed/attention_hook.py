"""H0 的**方法侧**：把 G1 的紧凑 KV 算子核挂进真实模型的 attention 分派。

这个文件补的是什么
------------------
H0 在两轮里被分成了两半，别混说：

- **量具侧（2026-09-21 已完成）**：`experiments/gpu/_hf.py` 的 `measure_prefill`
  由两段变三段，压缩臂多跑一次目的端前向 ⇒ 紧凑 KV 真的进入计时窗口。
- **方法侧（本文件）**：让 `dcc_kv` 行不再与 `kv_budget_shared` 共用
  `apply_kv_budget`（那是"直接裁缓存"，没有构造链路），改走
  `build_compact_kv`（构造链）+ `attention_kernel.dcc_kv_attention`。

只做量具侧时，`dcc_kv` 与 `kv_budget_shared` 的质量**逐点相同**（两者都只是
topk 裁缓存），质量差恒为 0 ⇒ H2 的第一个合取项无从谈起。这不是"方法没效果"，
而是"方法没跑"。本文件是唯一能让那两条曲线分开的东西。

用法（两段式，阶段与"条件化块从哪来"都必须显式）::

    cfg = HookConfig(budget_ratio=0.05, dcc_world=4)
    with dcc_attention(lm.model, cfg) as st:
        st.source_phase()                 # 源端：只记录 K/V
        lm.model(src_ids, use_cache=True)
        st.destination_phase(compacts="build")    # 目的端：用本次 query 条件化
        lm.model(dst_ids, use_cache=True, position_ids=pos)
        st.destination_phase(compacts="reuse")    # 之后每次复用同一批块
        lm.model(dst_ids, use_cache=True, position_ids=pos)

设计约束（每条都是被上一版坑过才写下来的）
------------------------------------------

**1. 注册**新**键，不覆盖 `"eager"`。**
`ALL_ATTENTION_FUNCTIONS` 是**全局**注册表，`register("eager", fn)` 会把所有
用 eager 的模型一起改掉；更危险的是"退出时 `pop("eager")`"—— 那会把
transformers 自带的实现**删掉**而不是还原。故注册专用键 `HOOK_KEY`，
进入时改写 `config._attn_implementation`；还原 = 把 config 改回原值（幂等）。
`get_interface` 对未知键抛 `KeyError`，所以 config 改回原值后**不可能**还走到钩子。

**2. source / destination 两个阶段必须由调用方显式声明。**
不能靠"`past_key_values` 是不是 None"去猜。猜错的形态是**静默的**：
把目的端当源端 ⇒ 钩子把紧凑块又存了一份进 state，产物看着正常。

**3. `destination_phase` 还要显式声明条件化块从哪来（`build` / `reuse`）。**
这是 2026-09-21 新增的一层，因为**拿"本次前向的 query"去条件化**在评测路径上
是一个泄漏：若本次前向的 query 就是**被打分的 continuation**，那么选键 / β 拟合 /
V 回归都看见了答案（仓库纪律：「评估 Query 绝不能参与选键与拟合」）。
故本模块把这件事做成一个**没有默认值**的必填参数：

    compacts="build"  —— 用本次前向的 query 建块并缓存（评测里应只在"目的端探针"
                         那一次前向上用：它的 query 是目的端自己的提问，不是被打分的 token）
    compacts="reuse"  —— 复用已缓存的块（复用前必须已经 build 过）

两个方向都"响亮"：`reuse` 而缓存为空 ⇒ 抛错；`build` 会**重建并覆盖**缓存，
所以"我以为在复用、其实每次都重建"不会静默发生（重建会把构造耗时留在计时窗口里，
产物里以 `n_builds` 体现）。

**4. 目的端阶段用 state 里的源端 K/V，不用传进来的 key/value。**
`LlamaAttention.forward` 里 `past_key_values.update(...)` 发生在调用 attention
interface **之前**，所以钩子拿到的 key/value 是"完整 cache + 本次新 token"。
拿它当源端会同时犯两个错：把目的端自己的 token 也压掉（源端不该含它），
以及把裁剪后的长度当成源长。

**5. 传入的 cache **只许含目的端本地 KV**；源端一律由 state 携带。**
上一版把本地块取成了"尾部 Lq 个"，隐含假设"传进来的 cache 里除了本次 token
就只有源端"。这在 prefill 计时里恰好成立，但在**评测路径上不成立** ——
目的端有自己的本地上下文（提示词尾段），它既不在 state 里也不该被丢掉；
按旧规则整段本地上下文会被静默排除在注意力之外（形状全对、数值偏）。
现改为：**本地块 = 传入 key/value 的全体**，因果掩码建在**索引空间**
（query 必是本地块的连续后缀，故只需索引，不需要绝对位置）。

反向的错误（把源端也放进 cache）会让源端同时以紧凑块和本地精确块两种形式
进入 softmax，同样"形状对、量级对"。故加一条**指纹守卫**：
若传入 cache 的前 `L_s` 行与 state 里源端 K 的 4 行采样指纹逐位相同，抛错。
它是启发式（4 行采样），不是证明 —— 但足以抓住"复用了同一张张量"这个唯一的真实成因。

**6. 返回值的形状必须是 `[B, Lq, H_q * D_v]`。**
`LlamaAttention.forward` 拿到 `attn_output` 后立刻
`attn_output.reshape(*input_shape, -1)`，而 `input_shape` 是 `[B, Lq]`。
eager 实现在返回前做了 `transpose(1, 2)`，钩子必须与之对齐。
**形状错在这一步不一定报错**：若 `H_q * D_v == Lq * D_v`（例如 `H_q == Lq`），
`reshape` 会静默换个轴读，产物照样是"合理量级"的数。故显式断言。

**7. GQA 的 head 映射走 `h // (H_q / H_kv)`，与 HF 的 `repeat_kv` 同序。**
`repeat_kv` 是 `repeat_interleave`（每个 kv head 连续复制 g 份），不是
`repeat`（整块复制）。用错的话形状完全正确、数值全错。

**8. 钩子不吃 `attention_mask`（显式声明，不是疏忽）。**
它按"源块全可见 + 本地块内部因果"的语义算，这对应"源序列整体位于目的端
query 之前"。但若上游传进来的加性掩码里有**整列被遮蔽**的 key
（padding 或空块），我们忽略它就会把该 key 当真 key 参与 softmax ⇒ 数值错且
无征兆。故做一次廉价检查：存在"对所有 query 都不可见"的 key 列 ⇒ 抛错。
这条检查不完美（前缀 padding 在因果掩码下不呈现整列遮蔽），因此
**调用方仍须保证单序列无 padding** —— 见 `HookConfig.ignores_attention_mask`。

**9. `dcc_world` 是单卡模拟，它**不**体现代价化。**
把源序列切成 W 段、每段各自构造一份紧凑块，模拟的是"W 个源端设备各发一条边"。
但"逐边条件化"的边际价值来自**每条边看到的目的端 query 不同**；单卡下所有段
看到的是同一批 query，因此 `dcc_world > 1` 与"共享压缩"在数值上**仍不可区分**。
本模块不提供 `mode="shared"` 这种假选项，而是把事实记进
`state.summary()["conditionalization_marginal_available"] = False`（附原因）
—— 免得下游把"单卡模拟下 dcc 与 shared 一样"读成"条件化没用"。

**10. 每边预算默认 `B_total / world`，`B_total` 只作敏感性对照。**
放大口径会让 `dcc_world` 越大总预算越大，那是"花更多 KV 换质量"，不是机制收益。
`budget_mode="total"` 的产物**不得**用于 H2 的质量主张（见 `HookConfig`）。

**11. 预算的口径有两种给法：`budget`（绝对）与 `budget_ratio`（相对源长）。**
评测路径上源长逐样本变化（`score_choices` 里每条样本的提示词长度不同），
用绝对预算会让"预算 > 源长"的样本静默退化成不压缩。故引入 `budget_ratio`，
由 `resolve_budget(source_len)` 在**每次构造时**解析，并把解析结果记进
`summary()["resolved_budget_by_source_len"]`。

边界：本模块只声明**接线**，不声明任何结论。README 层面「不得主张任务质量 /
通信性能 / 多卡可扩展性 / 与基线的对比」的约束不受影响。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.dcc_kv_ref import (
    DEFAULT_LAMBDA_BETA,
    CompactKV,
    build_compact_kv,
    identity_compact,
)
from src.dcc_kv_ref import attention_kernel as K


HOOK_KEY = "dcc_kv"
"""注册进 `ALL_ATTENTION_FUNCTIONS` 的键。

刻意不复用 `"eager"` / `"sdpa"`：那两个键是全局共享的，覆盖它们会波及所有模型，
而 `pop` 还原会把 transformers 自带实现删掉（见模块文档第 1 条）。
"""

MASK_BLOCKED_THRESHOLD = -1e9
"""判定"加性掩码里该 key 被遮蔽"的阈值。

不用 `== -inf` 是因为不同后端写 -inf 的方式不同（有的用 dtype 最小值，
如 bf16 的 -3.39e38）。用一个"远小于任何合法 logits"的界更稳。
"""

COMPACTS_MODES = ("build", "reuse")
"""`destination_phase` 的必填参数取值（见模块文档第 3 条）。"""

SOURCE_FINGERPRINT_ROWS = 4
"""指纹守卫采样的行数（见模块文档第 5 条）。采样 4 行而非全量：全量比较是
O(L_s * d_h) 的额外读写，会进到计时窗口里；4 行是 O(4 * d_h)，且"两张不同来源的
张量在这 4 行上逐位相同"实际上不会发生。"""


@dataclass(frozen=True)
class HookConfig:
    """钩子的静态配置。

    Args:
        budget: 总预算 B_total（源端被保留的 key 数，跨所有源块合计）。
            **与 `budget_ratio` 互斥且必须给一个。**
        budget_ratio: 相对**源长**的预算比。适用于源长逐样本变化的场合
            （评测路径）；每样本解析为 `max(1, round(ratio * L_s))`。
        mode: `"dcc_kv"`（走构造链）或 `"dense"`（参照臂：源块用
            `identity_compact`，β≡0、B=L_s ⇒ 应无残差）。
            **刻意没有 `"shared"`**：见模块文档第 9 条。
        dcc_world: 单卡模拟的源端段数（≥1）。1 表示不分段。
        budget_mode: `"per_edge"`（每段 `B_total // world`，主口径）或
            `"total"`（每段都是 `B_total`，仅供敏感性对照）。
        n_repr / projection_dim / lambda_beta / lambda_value / seed:
            构造链（`build_compact_kv`）的超参数，必须随产物落盘。
        build_dtype: 构造链的计算 dtype；`"auto"` = 跟随模型。
        query_chunk: 传给 `compact_kv_attention` 的 query 维分块，用于限制
            远端块的 `[Lq, B]` 中间张量峰值。**它是显存控制手段，不是数值等价
            手段**（实测 1–3 ULP 差异）；本地块走 `dense_attention`，不受它影响。
        layers: 接管哪些层；None = 全部。
    """

    budget: Optional[int] = None
    budget_ratio: Optional[float] = None
    mode: str = "dcc_kv"
    dcc_world: int = 1
    budget_mode: str = "per_edge"
    n_repr: int = 64
    projection_dim: int = 32
    lambda_beta: float = DEFAULT_LAMBDA_BETA
    lambda_value: float = 1e-3
    seed: int = 42
    build_dtype: str = "auto"
    query_chunk: Optional[int] = None
    layers: Optional[Tuple[int, ...]] = None

    ignores_attention_mask: bool = field(default=True, init=False)
    """常量声明（不是配置项）：钩子按"源块全可见 + 本地块因果"算，不消费
    `attention_mask`。放成字段是为了让产物能把这个声明与实现放在一起落盘，
    且测试可以断言它没被偷偷改成 False。"""

    def __post_init__(self) -> None:
        given = (self.budget is not None) + (self.budget_ratio is not None)
        if given != 1:
            raise ValueError(
                "budget 与 budget_ratio 必须**恰好给一个**（当前 "
                f"budget={self.budget!r}, budget_ratio={self.budget_ratio!r}）。"
                "两个都给会让'预算到底按谁算'变成猜；都不给则没有预算可言。"
            )
        if self.budget is not None and int(self.budget) < 1:
            raise ValueError(f"budget 必须 ≥ 1，得到 {self.budget}")
        if self.budget_ratio is not None and not (0.0 < float(self.budget_ratio) <= 1.0):
            raise ValueError(
                f"budget_ratio 必须落在 (0, 1]，得到 {self.budget_ratio}；"
                ">1 不是压缩，≤0 会得到空块。"
            )
        if self.mode not in ("dcc_kv", "dense"):
            raise ValueError(
                f"mode 必须是 'dcc_kv' 或 'dense'，得到 {self.mode!r}。"
                "没有 'shared' —— 单卡模拟下它与 'dcc_kv' 数值相同，"
                "给一个假选项只会制造一个假的对照（见模块文档第 9 条）。"
            )
        if self.budget_mode not in ("per_edge", "total"):
            raise ValueError(
                f"budget_mode 必须是 'per_edge' 或 'total'，得到 {self.budget_mode!r}"
            )
        if int(self.dcc_world) < 1:
            raise ValueError(f"dcc_world 必须 ≥ 1，得到 {self.dcc_world}")
        if self.build_dtype not in ("auto", "float32"):
            raise ValueError(
                f"build_dtype 目前只支持 'auto' / 'float32'，得到 {self.build_dtype!r}"
            )
        if self.query_chunk is not None and int(self.query_chunk) < 1:
            raise ValueError(f"query_chunk 必须 ≥ 1 或为 None，得到 {self.query_chunk}")
        self._check_per_edge_feasible(int(self.budget) if self.budget is not None else None)

    def _check_per_edge_feasible(self, total: Optional[int]) -> None:
        """`per_edge` 口径下每边至少要 1 个 key ⇒ 总预算必须 ≥ world。

        绝对预算在构造时就能查（`total` 有值）；相对预算要到
        `resolve_budget` 拿到源长才可能查（见该方法的说明）。
        """
        if total is None or self.budget_mode != "per_edge":
            return
        if int(total) < int(self.dcc_world):
            raise ValueError(
                f"budget={total} < dcc_world={self.dcc_world}："
                "'per_edge' 口径下每边至少 1 个 key，此时总预算会被抬到 ≥ dcc_world，"
                "与 budget 不再对应。请提高 budget 或降低 dcc_world。"
            )

    def resolve_budget(self, source_len: int) -> int:
        """把预算解析成绝对条数（相对口径在这里落地）。

        每次构造时调用：源长逐样本变化时，绝对预算会让"预算 > 源长"的样本
        静默退化成不压缩，而"退化了多少条样本"在产物里看不出来。

        Raises:
            ValueError: `per_edge` 且解析出的总预算 < world（会让某条边拿 0 个 key）。
        """
        L = int(source_len)
        if L < 1:
            raise ValueError(f"源长为 {L}，无法解析预算")
        if self.budget is not None:
            total = int(self.budget)
        else:
            total = max(1, int(round(float(self.budget_ratio) * L)))
        self._check_per_edge_feasible(total)
        return int(total)

    def edge_budget(self, segment_len: int, source_len: int) -> int:
        """单条源边（一个段）拿到的预算，且不超过该段长度。

        超过段长时返回段长本身 —— 调用方据此改走 `identity_compact`
        （"预算 = 段长"应无残差）。若改用 `build_compact_kv`，它会短路返回
        全部 key，但 **β 仍会拟合**（ridge 解不是恒等），于是"无压缩"这一档
        反而带上残差，性质 1 的检验从此失效。这个分叉是被实测逼出来的。
        """
        total = self.resolve_budget(source_len)
        per = total if self.budget_mode == "total" else total // int(self.dcc_world)
        return int(min(max(per, 1), int(segment_len)))

    def describe(self) -> Dict[str, Any]:
        """配置的落盘口径（不含运行时统计）。"""
        return {
            "budget": None if self.budget is None else int(self.budget),
            "budget_ratio": None if self.budget_ratio is None else float(self.budget_ratio),
            "mode": self.mode,
            "dcc_world": int(self.dcc_world),
            "budget_mode": self.budget_mode,
            "n_repr": int(self.n_repr),
            "projection_dim": int(self.projection_dim),
            "lambda_beta": float(self.lambda_beta),
            "lambda_value": float(self.lambda_value),
            "seed": int(self.seed),
            "build_dtype": self.build_dtype,
            "query_chunk": self.query_chunk,
            "layers": None if self.layers is None else list(self.layers),
        }


@dataclass
class LayerCache:
    """一层的运行态：源端 K/V（RoPE 后、紧凑化之前）。"""

    keys: torch.Tensor    # [B, H_kv, L_s, D_h]
    values: torch.Tensor  # [B, H_kv, L_s, D_v]


def source_partition(length: int, world: int) -> List[Tuple[int, int]]:
    """把源序列均分成 `world` 段，返回半开区间 `[(start, end), ...]`。

    前 `length % world` 段各多 1 个 key —— 这是 `_forward.chunk_source_sizes`
    同款约定，两处必须一致，否则"每段长度"在两套代码里不同值。

    Raises:
        ValueError: `length < world`（会有空段）。空段不是"预算小"，是
            "该设备收到 0 个 key"，它会让归并里出现一个 lse = -inf 的块，
            把"段数"与"有效边数"两个不同的量搞混。
    """
    if int(world) < 1:
        raise ValueError(f"world 必须 ≥ 1，得到 {world}")
    if int(length) < int(world):
        raise ValueError(
            f"源长 {length} < world {world}：会产生空段。"
            "空段的 lse 是 -inf，会被归并当作'该块无可见 key'剔除，"
            "于是'段数'与'有效边数'不再相等 —— 请降低 world。"
        )
    base, rem = divmod(int(length), int(world))
    out: List[Tuple[int, int]] = []
    start = 0
    for i in range(int(world)):
        end = start + base + (1 if i < rem else 0)
        out.append((start, end))
        start = end
    return out


def build_edge_compacts(
    keys: torch.Tensor,        # [L_s, D_h] 单 head（**整个源**，不是一段）
    values: torch.Tensor,      # [L_s, D_v] 单 head
    queries: torch.Tensor,     # [L_r, D_h] 该目的端 head 组（GQA 下是 g*Lq 个）
    cfg: HookConfig,
) -> List[CompactKV]:
    """构造**一条边组**的紧凑块（逐段）。

    `queries` 是"看到该源的这批目的端 query"。在 `dcc_world == 1` 时就是全部；
    在单卡模拟里，无论多少段都是同一批 —— 这正是"条件化的边际价值无法在单卡
    体现"的来源（模块文档第 9 条）。

    ⚠️ `queries` **不得**是"被打分的 continuation"（模块文档第 3 条）：
    选键、β 拟合、V 回归全都吃它。传错了不会有任何报错，只会让质量虚高。
    """
    L_s = int(keys.shape[0])
    if cfg.mode == "dense":
        # 参照臂：单块、B = L_s、β ≡ 0。用 identity_compact 而不是
        # build_compact_kv(budget=L_s)：后者会拟合出非恒等的 β 与 V，
        # 于是"DCC 与 dense 的差"里混进了构造链的残差，参照就失去意义。
        return [identity_compact(keys, values)]

    blocks: List[CompactKV] = []
    for start, end in source_partition(L_s, int(cfg.dcc_world)):
        seg_k = keys[start:end]
        seg_v = values[start:end]
        per = cfg.edge_budget(int(seg_k.shape[0]), L_s)
        if per >= int(seg_k.shape[0]):
            blocks.append(identity_compact(seg_k, seg_v))
            continue
        blocks.append(build_compact_kv(
            source_keys=seg_k,
            source_values=seg_v,
            destination_queries=queries,
            budget=per,
            num_representative_queries=int(cfg.n_repr),
            projection_dim=int(cfg.projection_dim),
            lambda_beta=float(cfg.lambda_beta),
            lambda_value=float(cfg.lambda_value),
            seed=int(cfg.seed),
        ))
    return blocks


def query_groups(query: torch.Tensor, num_kv_heads: int) -> List[torch.Tensor]:
    """按 kv head 把 query 分组，组内合并。

    Args:
        query: `[B, H_q, Lq, D_h]`
    Returns:
        长度 `num_kv_heads` 的列表，第 h 项形状 `[B, g, Lq, D_h]`（g = H_q/H_kv）。

    用 `repeat_interleave` 的语义分组**是硬要求**：HF 的 `repeat_kv` 是
    `[h, h, ..., h]` 连续复制，不是整块复制。分组错了形状照样对，数值全错。
    """
    B, H_q, Lq, D_h = query.shape
    if int(num_kv_heads) < 1 or H_q % int(num_kv_heads) != 0:
        raise ValueError(
            f"num_attention_heads={H_q} 不能被 num_key_value_heads="
            f"{num_kv_heads} 整除，GQA 分组无定义"
        )
    g = H_q // int(num_kv_heads)
    out: List[torch.Tensor] = []
    for h in range(int(num_kv_heads)):
        out.append(query[:, h * g:(h + 1) * g, :, :])
    return out


def _row_fingerprint(t: torch.Tensor, rows: int = SOURCE_FINGERPRINT_ROWS) -> torch.Tensor:
    """沿倒数第 2 维（序列维）均匀采样 `rows` 行，保留前导维。

    `[..., L, D] -> [..., min(rows, L), D]`。用于"传入的 cache 里是不是混进了
    源端"这条守卫：比较 4 行比比较全量便宜两个数量级，而它要抓的成因只有一个
    （调用方把同一张源端张量又传了进来）。
    """
    L = int(t.shape[-2])
    if L == 0:
        return t[..., :0, :]
    k = min(int(rows), L)
    idx = torch.linspace(0, L - 1, k).round().long().to(t.device)
    return t.index_select(-2, idx)


def _assert_local_cache_excludes_source(
    state: "DccAttentionState", layer_idx: int, key: torch.Tensor
) -> None:
    """目的端阶段：传入的 cache 不得含源端 K/V（模块文档第 5 条）。

    混进去的后果是**静默的**：源端会同时以"紧凑块"与"本地精确块"两种形式进入
    softmax，源端的注意力质量被算两遍。形状、dtype、量级全都正常，只有数值偏 ——
    而且偏得不大（源端本来就在上下文里），最难被发现。
    """
    entry = state.source.get(int(layer_idx))
    if entry is None:
        return
    src_len = int(entry.keys.shape[-2])
    if src_len == 0 or int(key.shape[-2]) < src_len:
        return
    a = _row_fingerprint(key[..., :src_len, :])
    b = _row_fingerprint(entry.keys)
    if a.shape == b.shape and bool(torch.equal(a, b)):
        raise ValueError(
            f"第 {layer_idx} 层：目的端阶段传入的 cache 前缀与 state 里的源端 K "
            f"逐位相同（前 {src_len} 行）。本模块的契约是**传入的 cache 只含目的端"
            "本地 KV，源端一律由 state 携带**；把源端也留在 cache 里会让它同时以"
            "紧凑块和本地精确块两种形式进入 softmax（源端质量被算两遍），"
            "而形状与量级都正常、看不出来。请传 past_key_values=None，"
            "或只传目的端自己的那段 KV。"
        )


def _assert_no_fully_blocked_key(attention_mask: Optional[torch.Tensor]) -> None:
    """加性掩码里不允许出现"对所有 query 都不可见"的 key 列。

    钩子不吃 mask（按源块全可见算）。若上游的 mask 里有整列被遮蔽的 key
    （最典型的是 padding），我们仍会把它当有效 key 参与 softmax ⇒ 数值错，
    而且因为形状与量级都正常，从输出看不出问题。故做事前检查。
    """
    if attention_mask is None:
        return
    m = attention_mask
    if m.dim() < 2:
        return
    # [..., Lq, K] 的加性掩码：沿 query 维（-2）看是否有 key 列（-1）全被遮蔽
    blocked = m <= MASK_BLOCKED_THRESHOLD
    all_blocked = blocked.all(dim=-2)
    if bool(all_blocked.any()):
        n = int(all_blocked.sum().item())
        raise ValueError(
            f"attention_mask 里有 {n} 个 key 对所有 query 都不可见（padding 或空块）。"
            "本钩子不消费 attention_mask（按源块全可见计算），这些 key 会被当作"
            "有效 key 参与 softmax ⇒ 数值错误且无征兆。请改用单序列、无 padding 的"
            "输入，或不要把该层交给本钩子。"
        )


class DccAttentionState:
    """一次前向的运行态：源端 K/V、条件化块、分阶段的显式声明。

    阶段是**显式**的（模块文档第 2、3 条）。状态机刻意简单到不会"猜错"：
    `source` 阶段只写入；`destination` 阶段只读取，且要求已写入源端，
    并显式声明条件化块是新建还是复用。
    """

    def __init__(self, cfg: HookConfig) -> None:
        self.cfg = cfg
        self.phase: str = "idle"
        self.compacts_mode: Optional[str] = None
        self.source: Dict[int, LayerCache] = {}
        # (layer, batch, kv_head) -> (建块时的源长, 块列表)
        # 存源长是为了在"复用"时能发现源变了（换了样本却复用旧块，静默错）。
        self.compacts: Dict[Tuple[int, int, int], Tuple[int, List[CompactKV]]] = {}
        self.resolved_budget: Dict[int, int] = {}
        self.n_source_calls: int = 0
        self.n_destination_calls: int = 0
        self.n_builds: int = 0
        self.n_compacts_built: int = 0
        self.n_local_blocks: int = 0

    # ---- 阶段切换 ---------------------------------------------------------

    def source_phase(self) -> "DccAttentionState":
        self.phase = "source"
        self.compacts_mode = None
        return self

    def destination_phase(self, *, compacts: str) -> "DccAttentionState":
        """进入目的端阶段。`compacts` **必填**（见模块文档第 3 条）。

        Args:
            compacts: `"build"` —— 用**本次前向的 query** 条件化建块并覆盖缓存；
                `"reuse"` —— 复用已缓存的块。

        Raises:
            ValueError: `compacts` 取值非法、进入目的端前没有任何源端 K/V、
                或选了 `"reuse"` 但缓存为空。
        """
        if compacts not in COMPACTS_MODES:
            raise ValueError(
                f"compacts 必须是 {COMPACTS_MODES} 之一，得到 {compacts!r}。"
                "刻意没有默认值：评测路径上'用本次 query 条件化'可能是泄漏"
                "（本次 query 就是被打分的 token），'复用'可能是陈旧块 —— "
                "两个方向的猜错都不报错，所以必须显式选。"
            )
        if not self.source:
            raise ValueError(
                "进入目的端阶段前没有任何源端 K/V。"
                "调用方必须先跑一次 source_phase() 下的前向 —— "
                "否则钩子无从构造紧凑块，而「什么都没压」会以全长的形式静默通过。"
            )
        if compacts == "reuse" and not self.compacts:
            raise ValueError(
                "compacts='reuse' 但还没有任何条件化块。"
                "请先用 compacts='build' 跑一次目的端探针前向 —— "
                "复用不存在的块只会回落成'不压缩'。"
            )
        self.compacts_mode = str(compacts)
        self.phase = "destination"
        return self

    # ---- 记录 -------------------------------------------------------------

    def record_source(self, layer_idx: int, keys: torch.Tensor,
                      values: torch.Tensor) -> None:
        self.source[int(layer_idx)] = LayerCache(keys=keys, values=values)
        self.n_source_calls += 1

    def has_layer(self, layer_idx: int) -> bool:
        return int(layer_idx) in self.source

    def layer_kv(self, layer_idx: int) -> LayerCache:
        if int(layer_idx) not in self.source:
            raise KeyError(
                f"层 {layer_idx} 没有源端 K/V。若该层在 source_phase 下被"
                "`cfg.layers` 排除、或模型层数在中途变了，就会到这里。"
            )
        return self.source[int(layer_idx)]

    def summary(self) -> Dict[str, Any]:
        """落盘口径。含**必须与结论一起报**的几条声明。"""
        out = dict(self.cfg.describe())
        out.update({
            "compacts_mode": self.compacts_mode,
            "n_layers_taken_over": len(self.source),
            "n_source_calls": self.n_source_calls,
            "n_destination_calls": self.n_destination_calls,
            "n_builds": self.n_builds,
            "n_compacts_built": self.n_compacts_built,
            "n_local_blocks": self.n_local_blocks,
            "resolved_budget_by_source_len": {
                str(k): int(v) for k, v in sorted(self.resolved_budget.items())
            },
            "ignores_attention_mask": bool(self.cfg.ignores_attention_mask),
            "local_block_contract": (
                "传入的 cache 只含目的端本地 KV；源端由 state 携带"
                "（本地块 = 传入 key/value 的全体，因果掩码建在索引空间）"
            ),
            "conditionalization_marginal_available": False,
            "conditionalization_marginal_reason": (
                "单卡模拟：所有源段看到的是同一批目的端 query，"
                "因此 dcc_kv 与共享压缩在数值上不可区分。"
                "逐边条件化的边际价值只能在多设备路径上测。"
            ),
            "budget_claim_scope": (
                "per_edge：每边 B_total/world，可作主口径"
                if self.cfg.budget_mode == "per_edge" else
                "total：每边 B_total，总预算随 world 放大 —— "
                "**不得**用于 H2 的质量主张，仅作敏感性对照"
            ),
            "build_in_timing_window": (
                "是：条件化块在目的端前向内部构造 ⇒ 该窗口含 T_build（源端侧代价）。"
                "与 dense 臂比较时这是一个额外差异，故 E6 另落一列「复用已建块」的"
                "目的端耗时（kernel-matched），见 _hf.measure_prefill。"
                if self.n_builds > 0 else "否：本次没有构造（复用或纯 dense 臂）"
            ),
        })
        return out


def _resolve_build_dtype(cfg: HookConfig, ref: torch.Tensor) -> torch.dtype:
    if cfg.build_dtype == "float32":
        return torch.float32
    return ref.dtype


def _as_query_dtype(blocks: Sequence[CompactKV], dtype: torch.dtype) -> List[CompactKV]:
    """把紧凑块搬到 query 的 dtype。核内部对 dtype 敏感（-inf 与 exp 都会看它）。"""
    return [
        CompactKV(
            keys=c.keys.to(dtype),
            logit_bias=c.logit_bias.to(dtype),
            values=c.values.to(dtype),
            selected_indices=c.selected_indices,
        )
        for c in blocks
    ]


def _layer_attention(
    state: DccAttentionState,
    layer_idx: int,
    query: torch.Tensor,    # [B, H_q, Lq, D_h]
    key: torch.Tensor,      # [B, H_kv, Lk, D_h]  只含**目的端本地** KV
    value: torch.Tensor,    # [B, H_kv, Lk, D_v]
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """目的端阶段：紧凑源块 + 本地精确块 → `[B, Lq, H_q * D_v]`。"""
    cfg = state.cfg
    cache = state.layer_kv(layer_idx)
    src_k, src_v = cache.keys, cache.values
    _assert_local_cache_excludes_source(state, layer_idx, key)

    B, H_q, Lq, D_h = query.shape
    Lk = int(key.shape[-2])
    D_v = int(value.shape[-1])
    if Lk < Lq:
        raise ValueError(
            f"本地块只有 {Lk} 行，而本次前向有 {Lq} 个 query —— 传入的 cache "
            "里不含本次的 token。钩子在 attention interface 里被调用，"
            "`past_key_values.update` 已在其之前执行，故本次 token 必在末尾；"
            "拿到更短的 cache 说明调用方把 cache 传错了对象。"
        )
    build_dtype = _resolve_build_dtype(cfg, query)
    src_len = int(src_k.shape[-2])

    # 本地块的因果位置建在**索引空间**：query 必是本地块的连续后缀，
    # 故 query i 对应 key 索引 (Lk - Lq + i)。全序列平移不改变因果比较，
    # 所以不需要绝对位置（也就不需要向 HF 要 cache_position）。
    local_qpos = torch.arange(Lk - Lq, Lk, device=query.device)
    g = H_q // int(num_kv_heads)
    groups = query_groups(query, num_kv_heads)
    outs = torch.empty(B, H_q, Lq, D_v, dtype=torch.float32, device=query.device)
    q_scale = float(scale)

    for b in range(B):
        for h in range(int(num_kv_heads)):
            grp = groups[h][b]                       # [g, Lq, D_h]
            key3 = (int(layer_idx), b, h)
            if state.compacts_mode == "reuse":
                entry = state.compacts.get(key3)
                if entry is None:
                    raise KeyError(
                        f"层 {layer_idx} / batch {b} / kv_head {h} 没有可复用的"
                        "条件化块。'reuse' 要求每个 (层, batch, head) 都先 build 过 —— "
                        "缺的那部分会静默退回'不压缩'。"
                    )
                stored_len, blocks = entry
                if int(stored_len) != src_len:
                    raise ValueError(
                        f"层 {layer_idx} / batch {b} / kv_head {h} 的复用块是在"
                        f"源长 {stored_len} 下建的，本次源长是 {src_len} —— "
                        "换了样本却复用旧块：块的 budget、选键与 β 全都对不上，"
                        "而形状与量级都正常。请重新 build（或按样本分 state）。"
                    )
            else:
                blocks = build_edge_compacts(
                    src_k[b, h].to(build_dtype), src_v[b, h].to(build_dtype),
                    grp.reshape(-1, D_h).to(build_dtype), cfg,
                )
                state.compacts[key3] = (src_len, blocks)
                state.n_builds += 1
                state.n_compacts_built += len(blocks)

            out_g = K.dcc_kv_attention(
                grp, _as_query_dtype(blocks, query.dtype),
                local_keys=key[b, h], local_values=value[b, h],
                query_positions=local_qpos, local_offset=0,
                # 源块整体位于目的端 query 之前 ⇒ 全可见；本地块因果。
                # 这两条策略已解耦（见 attention_kernel.dcc_kv_attention）。
                local_causal=True, remote_causal=False,
                scale=q_scale, query_chunk=cfg.query_chunk,
            )
            outs[b, h * g:(h + 1) * g] = out_g.float()
            state.n_local_blocks += 1

    if not torch.isfinite(outs).all():
        raise FloatingPointError(
            "钩子输出里出现非有限值。最常见的原因是某层的紧凑块全空"
            "（预算被夹到 <1）或本地块长度为 0 且源块不可见。"
            "这里显式失败，因为 nan 会一路传播到损失/日志而看不出源头。"
        )
    return outs.to(query.dtype).transpose(1, 2).reshape(B, Lq, H_q * D_v).contiguous()


def _source_attention(
    query: torch.Tensor,     # [B, H_q, Lq, D_h]
    key: torch.Tensor,       # [B, H_kv, Lk, D_h]
    value: torch.Tensor,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """源端阶段：精确、因果。

    ⚠️ **必须带因果**。源端 token 之间的可见性不能省：省掉以后源端的 attention
    输出会变（后续层的 K/V 也就跟着变），于是"dense 参照臂与原生 HF 一致"
    这条断言会**红在源端**，而症状看起来像钩子接线错了。
    参照实现是 `attention_kernel.dense_attention`，与目的端阶段同一个核。
    """
    B, H_q, Lq, D_h = query.shape
    D_v = int(value.shape[-1])
    groups = query_groups(query, num_kv_heads)
    g = H_q // int(num_kv_heads)
    pos = torch.arange(Lq, device=query.device)
    outs = torch.empty(B, H_q, Lq, D_v, dtype=torch.float32, device=query.device)
    for b in range(B):
        for h in range(int(num_kv_heads)):
            # kv head h 只有 **一份** K/V（GQA）；h*g:(h+1)*g 是 query head
            # 维的索引方式，用在 H_kv 维上会取到空片（实测 h=1、g=2 时
            # matmul 报 tensor b (0) at dim 0，症状指向 matmul 而非索引）。
            k_h = key[b, h]
            v_h = value[b, h]
            part = K.dense_attention(
                groups[h][b], k_h, v_h, causal=True, query_positions=pos,
                block_offset=0, scale=float(scale), return_lse=False)
            outs[b, h * g:(h + 1) * g] = part.out.float()
    return outs.to(query.dtype).transpose(1, 2).reshape(B, Lq, H_q * D_v).contiguous()


def make_hook(state: DccAttentionState):
    """造一个绑定到 `state` 的 attention interface。

    绑定而不是用模块级全局：注册表是全局的，但**运行态不是**；把状态藏在
    全局变量里会让"两次前向交错"变成一场猜谜。
    """
    def _hook(module, query, key, value, attention_mask=None, dropout=0.0,
              scaling=None, **kwargs):
        if state.phase == "idle":
            raise RuntimeError(
                "钩子被调用时不在任何阶段。请在 dcc_attention(...) 内用 "
                "source_phase() / destination_phase(compacts=...) 显式声明"
                "（模块文档第 2、3 条）。"
            )
        _assert_no_fully_blocked_key(attention_mask)
        layer_idx = int(getattr(module, "layer_idx", -1))
        cfg = state.cfg
        if cfg.layers is not None and layer_idx not in cfg.layers:
            # 不在接管范围内：按源端方式算（精确因果），但不写 state。
            return _source_attention(
                query, key, value, int(key.shape[1]), scaling), None

        num_kv_heads = int(key.shape[1])
        if state.phase == "source":
            state.record_source(layer_idx, key.detach(), value.detach())
            if cfg.mode == "dcc_kv":
                L_s = int(key.shape[-2])
                state.resolved_budget[L_s] = cfg.resolve_budget(L_s)
            out = _source_attention(query, key, value, num_kv_heads, scaling)
            return out, None

        if state.compacts_mode is None:
            raise RuntimeError(
                "目的端阶段缺少 compacts 声明。请用 "
                "destination_phase(compacts='build' 或 'reuse')。"
            )
        state.n_destination_calls += 1
        out = _layer_attention(state, layer_idx, query, key, value,
                               num_kv_heads, scaling)
        return out, None

    return _hook


@contextlib.contextmanager
def dcc_attention(model, cfg: HookConfig, *, state: Optional[DccAttentionState] = None):
    """把 `model` 的 attention 分派改到 DCC-KV 钩子上，退出时严格还原。

    用法（阶段与条件化块来源都必须显式）::

        with dcc_attention(lm.model, cfg) as st:
            st.source_phase()
            lm.model(src_ids, use_cache=True)
            st.destination_phase(compacts="build")
            lm.model(dst_ids, use_cache=True, position_ids=pos)

    ⚠️ 阶段切换是**普通方法**（返回 self），不是上下文管理器 —— 早先的文档
    写成 `with st.source_phase():` 是错的（`DccAttentionState` 没有
    `__enter__`），照抄会直接 AttributeError。这里刻意不改成 CM：
    前向必须夹在两次切换之间，用 CM 反而容易写成"切换了但没跑前向"。

    还原策略（模块文档第 1 条）：只改写 `config._attn_implementation`，
    注册表里留下专用键 `HOOK_KEY`。若原值与 `HOOK_KEY` 相同（重复进入），
    退出时**不**改动它 —— 否则会把自己前一次的设置抹掉。
    """
    try:
        import transformers  # noqa: F401
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as e:  # pragma: no cover - 依赖缺失时给出可读原因
        raise RuntimeError(
            f"钩子需要 transformers 的 ALL_ATTENTION_FUNCTIONS 注册表，导入失败：{e}"
        ) from e

    target = getattr(model, "config", None)
    if target is None:
        raise ValueError("传入的 model 没有 .config —— 需要是 HF 模型的 config 载体")

    st = state if state is not None else DccAttentionState(cfg)
    previous = getattr(target, "_attn_implementation", None)
    if previous == HOOK_KEY:
        raise ValueError(
            f"该模型的 _attn_implementation 已经是 {HOOK_KEY!r}："
            "说明上一次 dcc_attention 没有正常退出，或出现了嵌套。"
            "嵌套会让两个 state 抢同一个键 —— 直接拒绝，不要猜。"
        )

    ALL_ATTENTION_FUNCTIONS.register(HOOK_KEY, make_hook(st))
    target._attn_implementation = HOOK_KEY
    try:
        yield st
    finally:
        # 还原顺序无所谓：两者都是"把可达路径改回去"，且改完 config 就再也
        # 走不到钩子（get_interface 只按 config 取键）。
        if getattr(target, "_attn_implementation", None) == HOOK_KEY:
            target._attn_implementation = previous


__all__ = [
    "HOOK_KEY",
    "COMPACTS_MODES",
    "SOURCE_FINGERPRINT_ROWS",
    "DccAttentionState",
    "HookConfig",
    "LayerCache",
    "build_edge_compacts",
    "dcc_attention",
    "make_hook",
    "query_groups",
    "source_partition",
]
