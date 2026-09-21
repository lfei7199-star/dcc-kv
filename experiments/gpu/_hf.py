"""真实模型后端（HF）—— 任务指标与 prefill 性能测量。

本模块只在 GPU 上使用（`--backend hf`）。它提供：

1. `load_model()`          —— 加载 HF 因果 LM，锁定精度与 device
2. `apply_kv_budget()`     —— 在 prefill 之后对 KV cache 施加预算约束
3. `score_choices()`       —— 多项选择打分，产出任务准确率
4. `measure_prefill()`     —— prefill 延迟 / 吞吐 / 峰值显存

关于 `apply_kv_budget` 的正确性说明
-----------------------------------
它按"保留 top-B 个位置"的方式压缩缓存：

    k = cache.key_cache[layer]      # [B, H, S, D]
    v = cache.value_cache[layer]
    score = k.float().pow(2).mean(dim=(1, 3))      # [B, S] 跨 head 聚合
    idx = topk(score, B).sort()                     # 保留的位置（保持原顺序）
    cache.key_cache[layer] = k[:, :, idx, :]

这个方法之所以**在数学上是合法的**，而不是一个近似技巧：prefill 阶段每个 key
已经烘焙了自身的绝对位置（RoPE 作用于位置 p 的 key 得到 k_p），保留一个位置
子集并不会改变被保留 key 的表示。解码时新 Query 的绝对位置由其 `cache_position`
决定，因果掩码对该 token 而言是全可见的，因此它仍然按正确的相对距离与
被保留的 key 交互 —— 这正是"稀疏注意力"的语义。

换句话说：这条路径测的是**"删掉一部分 KV 会掉多少点"**，不是
**"用回归出来的紧凑 KV 会掉多少点"**。后者需要 §6.4/§6.3 的压缩构造链路
（Key 选择 + β + Value 回归），其单机版本见 `experiments/cpu/`，
多机版本需要 `src/distributed` 的异步路径。本模块负责提供
**可比的、有意义的任务指标基线**，以及论文 A1/A2/A5/A6 所需的
prefill 计时口径。

关于 `attn_hook`（H0 方法侧，2026-09-21 接入）
----------------------------------------------
`measure_prefill` / `score_choices` / `evaluate` 都有一个可选的 `attn_hook`
参数（`src.distributed.attention_hook.HookConfig`）。给了它，前向就走钩子，
源端由钩子的 state 携带、目的端只带自己的 KV；不给则走本模块原有的
`apply_kv_budget` 路径。两条路径的**测量口径不同，不能混比**：

    attn_hook=None      —— 「直接裁缓存」，源端 KV 留在 cache 里；
    attn_hook=HookConfig —— 「目的端条件化构造 + 紧凑 KV 注意力」，
                            源端**不在** cache 里（目的端前向传 past=None）。

目的端与源端的分工（评测路径）
------------------------------
`score_choices` 按 `dest_fraction` 把提示词切成两段：**前 (1-f) 段是源端**
（被压缩），**后 f 段是目的端自己的本地段**（保持精确、且是唯一合法的条件化
query 来源）。这样切的原因不是形式好看，而是两条硬约束：

1. 目的端在真实系统里**本来就有自己的本地上下文**（提问/指令），压掉它会
   系统性低估准确率，而"整段本地上下文消失"在形状与量级上都看不出来；
2. 构造紧凑块用的 query 会进选键 / β 拟合 / V 回归。若拿**被打分的
   continuation**当条件化 query，就等于让压缩过程看见了答案 —— 仓库纪律
   明确禁止（`E9` 的 oracle 是作弊上界，同款道理）。用目的端本地段作 query
   来源，则越界在结构上不可能发生。

同口径也施加于非钩子基线：`kv_budget_shared` 只裁**源段**、保留同一段目的端
本地段，否则两边的"保留了多少精确 token"不同，质量差里混进了预算不对齐。
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.gpu import _env  # noqa: E402
from src.distributed import attention_hook as _A  # noqa: E402


# =============================================================================
# 模型加载
# =============================================================================

@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    name: str
    precision: str
    device: torch.device
    num_layers: int
    num_kv_heads: int
    head_dim: int


def load_model(
    name_or_path: str,
    precision: str = "bfloat16",
    device: str = "cuda",
    attn_implementation: Optional[str] = None,
) -> LoadedModel:
    """加载因果 LM。

    约定：
    - 一律 `eval()` + `requires_grad_(False)`，避免把训练态的开销算进 prefill；
    - `attn_implementation` 默认让 transformers 自己挑（有 flash-attn 就用）；
      显式传入可用来做 "eager vs flash" 的对照。
    - 需要 device_map 的 70B/72B 场景请自行传 `device_map="auto"` 改造本函数，
      本函数默认单卡放置，便于把 `gpu_count` 语义与元数据对齐。
    """
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _env.dtypes_for(precision)
    print(f"  [hf] 加载 tokenizer: {name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

    kwargs: Dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": True}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    print(f"  [hf] 加载模型: {name_or_path}  dtype={precision}"
          f"  attn={attn_implementation or 'auto'}")
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = model.config
    num_layers = int(getattr(cfg, "num_hidden_layers", 0))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads",
                               getattr(cfg, "num_attention_heads", 0)))
    head_dim = int(getattr(cfg, "head_dim", 0)) or (
        int(getattr(cfg, "hidden_size", 0)) // max(1, int(getattr(cfg, "num_attention_heads", 1)))
    )

    print(f"  [hf] transformers={transformers.__version__}  "
          f"layers={num_layers}  kv_heads={num_kv_heads}  head_dim={head_dim}")
    return LoadedModel(
        model=model, tokenizer=tokenizer, name=name_or_path,
        precision=precision, device=torch.device(device),
        num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )


# =============================================================================
# KV cache 预算约束
# =============================================================================

def _cache_layers(cache: Any) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """统一取出 (layer_index, keys, values) 视图，兼容新旧缓存 API。"""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return [(i, cache.key_cache[i], cache.value_cache[i])
                for i in range(len(cache.key_cache))]
    if hasattr(cache, "layers"):  # transformers >= 4.5x 的 Cache 基类
        out = []
        for i, layer in enumerate(cache.layers):
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is not None and v is not None:
                out.append((i, k, v))
        return out
    # legacy: tuple of (k, v)
    return [(i, layer[0], layer[1]) for i, layer in enumerate(cache)]


def _set_layer(cache: Any, i: int, k: torch.Tensor, v: torch.Tensor) -> None:
    if hasattr(cache, "key_cache"):
        cache.key_cache[i] = k
        cache.value_cache[i] = v
    elif hasattr(cache, "layers"):
        cache.layers[i].keys = k
        cache.layers[i].values = v
    else:
        cache[i] = (k, v)


def _pooled_key_energy(k: torch.Tensor, chunk: int = 8192,
                       mode: str = "topk_rms") -> torch.Tensor:
    """按块累加 Key 的能量谱，返回 [S]。

    为什么不直接 `k.float().pow(2).mean(...)`：那会**瞬时复制整份 K cache**。
    8B 模型 / 32K 上下文 / bf16 的单层 K 为 64 MiB（`1×8×32768×128×2` 字节），
    32 层合计 2.0 GiB；转成 float32 就是再要 4.0 GiB（V 还在旁边），而这一步
    只是要一个 [S] 的排序键 —— 峰值额外显存本应只与 chunk 有关、与 S 无关。

    累加在 float32 中逐块进行。归约沿 `(B, H, d)` 维、分块沿 `S` 维，两者
    **正交**，因此每个 `s` 位置参与归约的元素集合与顺序都不变。实测边界：

    - `chunk >= 2`：结果与"整张转换后一次算完"**逐位相同**
      （覆盖 chunk ∈ {2, 7, 64, 999, 8192, 100000}）；
    - `chunk == 1`：PyTorch 对 `[B, H, 1, d]` 会走另一条归约 kernel，产生
      1 ULP 级差异（bf16 下约 3.6e-07）；
    - **保留位置对任何 chunk 都逐位一致**（含 1）。位置才是影响准确率的量，
      所以显存与数值在这里是可以解耦的。

    见 tests/test_gpu_pipeline.py 的两个等价性锚点。
    """
    S = int(k.shape[-2])
    out = torch.empty(S, dtype=torch.float32, device=k.device)
    for lo in range(0, S, chunk):
        hi = min(S, lo + chunk)
        e = k[..., lo:hi, :].float().pow(2)
        if mode == "topk_norm":
            out[lo:hi] = e.sum(dim=(0, 1, 3))
        else:
            out[lo:hi] = e.mean(dim=(0, 1, 3))
    return out


def apply_kv_budget(
    cache: Any,
    budget: int,
    mode: str = "topk_rms",
    seed: int = 42,
) -> Dict[str, Any]:
    """对 KV cache 施加预算约束，返回保留统计。

    mode:
        identity    —— 不动缓存（baseline，等于精确注意力）
        topk_rms    —— 按跨 head 聚合的 Key 平方均值排序，保留 top-B 位置
        topk_norm   —— 按 Key 的 L2 范数排序（与 RMS 略有差别，作敏感性对照）
        stride      —— 等间隔下采样（无依据的对照组，用于说明"选得对"有价值）
        random      —— 随机保留（下界对照组）

    返回值里的 `keep_ratio` 是实际保留比例，因为不同层的 S 可能不同
    （padding 或滑窗），且 budget 会被夹到 [1, S]。
    """
    kept: List[int] = []
    total: List[int] = []
    per_layer_ratio: List[float] = []

    # 选点用的随机数一律固定在 CPU 上生成：这样同一 seed 在所有 rank 上得到
    # 同一组保留位置，"random" 才是可比的对照，而不是引入了各 rank 差异。
    gen = torch.Generator(device="cpu").manual_seed(seed)

    for i, k, v in _cache_layers(cache):
        S = int(k.shape[-2])
        total.append(S)
        if mode == "identity" or S <= budget:
            kept.append(S)
            per_layer_ratio.append(1.0)
            continue

        B = max(1, min(int(budget), S))
        if mode in ("topk_rms", "topk_norm"):
            # 分块算能量谱：不做整张 k.float()，峰值额外显存与 S 无关
            pooled = _pooled_key_energy(k, mode=mode)        # [S]
        elif mode == "random":
            pooled = torch.rand(S, generator=gen)
        elif mode == "stride":
            idx = torch.linspace(0, S - 1, B).round().long()
            _set_layer(cache, i, k[..., idx, :].contiguous(), v[..., idx, :].contiguous())
            kept.append(B)
            per_layer_ratio.append(B / S)
            continue
        else:
            raise ValueError(f"未知 mode：{mode}")

        idx = torch.topk(pooled, B, largest=True).indices.sort().values
        idx = idx.to(k.device)
        _set_layer(cache, i, k[..., idx, :].contiguous(), v[..., idx, :].contiguous())
        kept.append(B)
        per_layer_ratio.append(B / S)

    return {
        "mode": mode,
        "budget": int(budget),
        "kept_per_layer": kept,
        "total_per_layer": total,
        "keep_ratios": per_layer_ratio,
        "mean_keep_ratio": (sum(per_layer_ratio) / len(per_layer_ratio)) if per_layer_ratio else 0.0,
    }


# =============================================================================
# 多项选择打分
# =============================================================================

@dataclass
class EvalSample:
    prompt: str
    choices: List[str]
    answer: int
    task: str = "unknown"
    length_tag: str = "unknown"


def load_eval_file(path: str, limit: Optional[int] = None) -> List[EvalSample]:
    """读取 JSONL 评测集。

    每行格式：
        {"prompt": "...", "choices": ["...", "..."], "answer": 0,
         "task": "longbench_qa", "length_tag": "32k"}
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评测集不存在：{p}")
    samples: List[EvalSample] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(EvalSample(
                prompt=d["prompt"],
                choices=list(d["choices"]),
                answer=int(d["answer"]),
                task=d.get("task", "unknown"),
                length_tag=d.get("length_tag", "unknown"),
            ))
            if limit and len(samples) >= limit:
                break
    return samples


def prompt_split(prompt_tokens: int, dest_fraction: float) -> Tuple[int, int]:
    """把提示词切成 `(源段长度, 目的端本地段长度)`。

    `dest_fraction = 0` ⇒ 整条提示词都是源端（即本仓库 2026-09-21 之前的口径）；
    `> 0` ⇒ 尾部那一段是目的端自己的本地上下文（见模块文档「目的端与源端的分工」）。

    尾段取 `round(f * S)` 而不是 `int()`：整数截断在 S=4096、f=0.25 上就差 0 个，
    但在 f 很小（如 0.001）时会**恒为 0** —— 于是"目的端本地段"这条轴看起来
    存在、实际从未生效。四舍五入后仍可能为 0（样本太短），此时由调用方决定
    是报错还是退化成"无目的端"。
    """
    S = int(prompt_tokens)
    f = float(dest_fraction)
    if not (0.0 <= f < 1.0):
        raise ValueError(
            f"dest_fraction 必须落在 [0, 1)，得到 {dest_fraction}；"
            "=1 意味着源段为空（无物可压），>1 会算出负长度。"
        )
    if S < 1:
        raise ValueError(f"提示词长度为 {S}，无法切分")
    n_dst = int(round(f * S))
    return S - n_dst, n_dst


@torch.no_grad()
def score_choices(
    lm: LoadedModel,
    sample: EvalSample,
    budget_ratio: Optional[float] = None,
    compaction_mode: str = "identity",
    max_prompt_tokens: Optional[int] = None,
    attn_hook: Optional[Any] = None,
    dest_fraction: float = 0.0,
) -> Dict[str, Any]:
    """对一个样本的每个选项打分，返回正确选项与其得分。

    打分口径：以 "prompt + choice" 的 continuation 平均 logprob 作为分数
    （长度归一），这是长文 QA 评测里的常规做法。所有选项共用**同一份
    prefill 结果**（各自持一份副本），因此选项差异不会污染 prefill 计时。

    四处实现细节是正确性的前提，不要"优化"掉：

    1. **每个选项一份 cache 副本。** 前缀只算一次（prefill 的代价是 O(S²)，
       不能按选项重算），但 `DynamicCache` 在前向中会**就地追加**新 token：
       若所有选项共用同一对象，第 2 个选项就会在"前缀 + 第 1 个选项的
       continuation"之上继续生成，选项之间的比较不再同源。
    2. **必须显式传 position_ids。** 预算裁剪后 cache 长度会短于原文；若留空，
       transformers 会把 continuation 的绝对位置算成"实际张量长度.."而不是
       S..S+T-1，RoPE 的相对距离整体错位，准确率会被系统性低估。
       （保留位置子集本身不改被保留 key 的表示，见模块文档。）
    3. **源段与目的端本地段分开处理**（`dest_fraction > 0` 时）。目的端本地段
       保持精确，且是唯一合法的条件化 query 来源 —— 拿被打分的 continuation
       去条件化等于让压缩看见答案（见模块文档）。
    4. **钩子路径下目的端前向不传源端 cache**（`past_key_values=None`）：
       源端由钩子的 state 携带，传进去会让源端被算两遍，且钩子会抛错拦下。

    Args:
        attn_hook: `attention_hook.HookConfig` 或 None。给了就由钩子接管
            注意力（源端由状态携带），此时 `dest_fraction` 必须 > 0 ——
            否则没有目的端本地段，既无从构造条件化块，也就没有合法的
            条件化 query 来源。
        dest_fraction: 目的端本地段占提示词的比例（见 `prompt_split`）。
    """
    tok = lm.tokenizer
    device = lm.device

    prompt_ids = tok(sample.prompt, return_tensors="pt",
                     add_special_tokens=False).input_ids.to(device)
    if max_prompt_tokens is not None:
        prompt_ids = prompt_ids[:, -max_prompt_tokens:]
    S = int(prompt_ids.shape[1])
    n_src, n_dst = prompt_split(S, dest_fraction)
    src_ids = prompt_ids[:, :n_src]
    dst_ids = prompt_ids[:, n_src:] if n_dst > 0 else None

    if n_src < 1:
        raise ValueError(
            f"dest_fraction={dest_fraction} 在 S={S} 上把源段压成 {n_src} 个 token："
            "源段为空 ⇒ 没有可压缩的内容，'压缩后质量'与'不压缩'不再可比。"
        )

    hook_path = attn_hook is not None
    if hook_path and n_dst < 1:
        raise ValueError(
            f"钩子路径要求目的端本地段非空，而 dest_fraction={dest_fraction} 在 "
            f"S={S} 上算出 {n_dst} 个 token。目的端本地段同时承担两个角色："
            "① 目的端自己的上下文（压掉它会低估准确率）；"
            "② **唯一合法**的条件化 query 来源（用被打分的 continuation 去条件化"
            "等于让压缩看见答案）。两条都不能缺，故这里拒绝执行而不是退化。"
        )
    if hook_path and budget_ratio is not None:
        raise ValueError(
            "钩子路径的预算由 attn_hook（HookConfig.budget / budget_ratio）给出；"
            "同时再传 budget_ratio 会让'预算到底按谁算'变成猜。"
            "请只保留一处声明。"
        )
    if hook_path and compaction_mode != "identity":
        raise ValueError(
            f"钩子路径自己完成构造（build_compact_kv），不接受 compaction_mode="
            f"{compaction_mode!r}：那是 apply_kv_budget 的旋钮，两者作用在同一份 "
            "cache 上会互相覆盖。请传 compaction_mode='identity'。"
        )

    if hook_path:
        with _A.dcc_attention(lm.model, attn_hook) as st:
            # ① 源段：只记录源端 K/V（输出不用）
            st.source_phase()
            lm.model(input_ids=src_ids, use_cache=True)
            # ② 目的端探针：**这一段是 unique 合法的条件化 query 来源**
            dest_pos = torch.arange(n_src, S, device=device).unsqueeze(0)
            st.destination_phase(compacts="build")
            probe = lm.model(input_ids=dst_ids, use_cache=True,
                             position_ids=dest_pos)
            base_logit = probe.logits[0, -1, :]
            base_cache = probe.past_key_values        # 只含目的端本地段
            # ③ 各选项：复用同一批条件化块（不得重建 —— 重建就用上了被打分的 token）
            st.destination_phase(compacts="reuse")
            scores = _score_continuations(
                lm, sample.choices, base_cache, base_logit, S, device)
            hook_summary = st.summary()
    else:
        prefix = lm.model(input_ids=src_ids, use_cache=True)
        cache = prefix.past_key_values
        base_logit = prefix.logits[0, -1, :]
        if budget_ratio is not None:
            # 只裁**源段**：目的端本地段保持精确，与钩子路径同结构（见模块文档）
            apply_kv_budget(cache, budget=max(1, int(round(budget_ratio * n_src))),
                            mode=compaction_mode)
        if dst_ids is not None:
            dest_pos = torch.arange(n_src, S, device=device).unsqueeze(0)
            dest_out = lm.model(input_ids=dst_ids, past_key_values=cache,
                                use_cache=True, position_ids=dest_pos)
            cache = dest_out.past_key_values
            base_logit = dest_out.logits[0, -1, :]
        scores = _score_continuations(
            lm, sample.choices, cache, base_logit, S, device)
        hook_summary = None

    pred = int(max(range(len(scores)), key=lambda i: scores[i]))
    # 预算一律按**源段**长度算：源段才是被压缩的对象。钩子路径的预算来自
    # HookConfig（可能又是相对口径），故这里显式解析一次再落盘 —— 否则产物里
    # 看不出"实际保留了多少条 key"，而那是读这张表的第一件事。
    if hook_path:
        budget_tokens: Optional[int] = int(attn_hook.resolve_budget(n_src))
    elif budget_ratio is not None:
        budget_tokens = max(1, int(round(budget_ratio * n_src)))
    else:
        budget_tokens = None
    return {
        "task": sample.task,
        "length_tag": sample.length_tag,
        "n_choices": len(sample.choices),
        "answer": sample.answer,
        "pred": pred,
        "correct": int(pred == sample.answer),
        "scores": scores,
        "prompt_tokens": S,
        "source_tokens": int(n_src),
        "dest_tokens": int(n_dst),
        "budget_tokens": budget_tokens,
        "compression_path": ("attn_hook" if hook_path else
                             ("apply_kv_budget" if budget_ratio is not None else "none")),
        "attn_hook": hook_summary,
    }


def _score_continuations(lm, choices, cache, base_logit, S, device) -> List[float]:
    """逐选项打分。所有选项都从**同一份前缀**的独立副本出发（见函数文档 ①）。

    `base_logit` 是预测第一个 continuation token 的那个位置的 logits
    （有目的端本地段时它来自目的端那一段，否则来自源段末尾）。
    """
    tok = lm.tokenizer
    scores: List[float] = []
    for choice in choices:
        cont_ids = tok(choice, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(device)
        if cont_ids.numel() == 0:
            scores.append(float("-inf"))
            continue
        T = int(cont_ids.shape[1])
        past = copy.deepcopy(cache)
        position_ids = torch.arange(S, S + T, device=device).unsqueeze(0)
        out = lm.model(input_ids=cont_ids, past_key_values=past,
                       use_cache=True, position_ids=position_ids)
        logits = out.logits[0]                       # [T, V]
        # 第 i 个 continuation token 由第 i-1 个位置的 logits 预测
        prev = torch.cat([base_logit.reshape(1, -1), logits[:-1, :]], dim=0)
        lp = torch.log_softmax(prev.float(), dim=-1)
        tgt = cont_ids[0]
        scores.append(float(lp.gather(-1, tgt[:, None]).mean()))
    return scores


def evaluate(
    lm: LoadedModel,
    samples: Sequence[EvalSample],
    budget_ratio: Optional[float] = None,
    compaction_mode: str = "identity",
    max_prompt_tokens: Optional[int] = None,
    attn_hook: Optional[Any] = None,
    dest_fraction: float = 0.0,
) -> Dict[str, Any]:
    """对样本集评测，返回总体准确率与按任务/长度的分组准确率。

    钩子路径下**每条样本各建一个 state**（`score_choices` 内部开 `dcc_attention`
    上下文）：源长逐样本变化，跨样本复用条件化块会被钩子以"源长变了"拦下 ——
    那不是配置错误，而是"本该每条样本单独条件化"。
    """
    rows = [score_choices(lm, s, budget_ratio, compaction_mode, max_prompt_tokens,
                          attn_hook=attn_hook, dest_fraction=dest_fraction)
            for s in samples]
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else float("nan")

    by_task: Dict[str, List[int]] = {}
    by_len: Dict[str, List[int]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["correct"])
        by_len.setdefault(r["length_tag"], []).append(r["correct"])

    return {
        "n": n,
        "accuracy": acc,
        "by_task": {k: sum(v) / len(v) for k, v in by_task.items()},
        "by_length": {k: sum(v) / len(v) for k, v in by_len.items()},
        "rows": rows,
    }


# =============================================================================
# prefill 计时
# =============================================================================

PREFILL_TIMING_CONSUMES_COMPACT_KV = True
"""`measure_prefill` 的计时窗口里，裁剪后的 KV 是否真的被前向使用过。

**2026-09-21 起为 True**（H0 已接上）。它仍是一个需要被验证的**声明**而不是
注释：`tests/test_adversarial_2026_09_20.py` 的 `test_a1_*` / `test_a3_*` 会用
桩模型观察「有没有任何一次前向拿到了 past cache」以及与它同一轮的 dest 前向，
并把这个常量与观察结果对照。改这个值必须同时改实现，否则测试会红。
（原文此处引用的 `test_h0_prefill_hook` 从未存在 —— 见该测试文件的 I 组。）

接上之前是什么样（2026-09-20 对抗性审查的结论）
------------------------------------------------
`_one()` 原先只做两件事：全长前向 -> `apply_kv_budget` 截断缓存。截断结果
**没有再被任何前向使用**。于是

    T_prefill(压缩方法) = 全长注意力耗时 + 压缩开销
                        >= T_prefill(精确方法)

即 `prefill_speedup <= 1 < 1.10x`，**结构上**不可能满足 H2 的第二个合取项。
以那个状态把 E6 的 `dcc_kv` 直接翻成可测量，H2 会给出一个确定性的假阴性
（「压缩不加速 prefill」），而根源是度量口径，不是方法本身。

H0 接上后的结构（现行为）
------------------------------------------------
`_one()` 做三件事，缺一不可：

    ① 源端构造：对 seq_len 个 token 全长前向 -> KV 进入 cache（两臂共有）
    ② 压缩（仅压缩臂）：把 cache 裁到预算 B
    ③ 目的端：用 cache 里的 KV **再跑一次前向**（dest_len 个 token）

③ 是收益的载体：它看到的 KV 长度在 dense 臂是 S、在压缩臂是 B，其余相同。
`measure_prefill` 把压缩臂的 `dest_len <= 0` 当作**错误**抛出，堵住「忘了开 ③」
这条静默失效通路 —— 那正是本常量此前为 False 的原因。

对照：准确率那条路一直是通的 —— `score_choices` 在截断后用 `deepcopy` 出来的
缓存做 continuation 前向，压缩的收益（更短的上下文）确实进入了打分。现在两条
路的语义一致：都走「只对紧凑 KV 做注意力」。

钩子路径（`attn_hook`，2026-09-21 接入）下的三段
------------------------------------------------
`apply_kv_budget` 那条路把裁剪后的 KV 留在 cache 里，所以 ③ 的 past 就是
裁剪结果。钩子路径不行：源端由钩子的 state 携带（**紧凑块带 β 与 V 回归，
表达不成普通 KV**），因此 ③ 的 past 传 `None`，源端只在钩子里消费。于是：

    ① 源端构造（两臂共有）
    ② 不裁 cache —— 紧凑化发生在钩子内部
    ③ 目的端前向（past=None）：源块由 state 提供，本地块是本次 token

同一路径下本模块**另开一个计时窗口**：钩子把条件化块构造在 ③ 内部，所以窗口
含 `T_build`（源端侧代价，按 `_comm.PipelineTiming` 的分账口径它不该算进
目的端的计算）。第二个窗口先建好块再只计时目的端前向（`compacts="reuse"`），
于是两臂的**唯一变量是目的端注意力消费的 KV 长度**（dense = S，dcc = B）。
E6 用这个窗口算 kernel-matched 的加速比，用第一个窗口算端到端的那个。

边界不变：本常量只声明**量具是否灵敏**，不声明任何结论；README 层面「不得主张
通信性能 / 多卡可扩展性 / 任务质量」的约束不受影响。
"""


def _token_bound(lm: LoadedModel) -> int:
    """生成随机输入 token 的上界（不含）。

    优先用模型/分词器的**真实词表大小**。原实现写死 `randint(0, 1000)`：对
    8B 级模型碰巧成立（词表 ≫ 1000），但在任何 vocab < 1000 的模型上会直接死在
    `F.embedding`，报的是 `IndexError: index out of range` —— 症状指向 embedding，
    真因在输入生成，排查时会往权重/配置上找。取真实词表后，小模型也能跑，
    这条测量路径才谈得上"写完即验证"。
    """
    # 一律用 getattr 取：测试里的桩模型（`_StubLM`）没有 `tokenizer` 属性，
    # 直接写 `lm.tokenizer` 会在**构造候选列表**时就抛 AttributeError，
    # 而"这个桩没有词表"本该由下一行的回退兜住 —— 报错指着 tokenizer、真因
    # 在候选列表求值顺序，症状与真因分离（上一轮 dest_len / --iters 同款形态）。
    candidates = (
        (getattr(lm, "tokenizer", None), "vocab_size"),
        (getattr(getattr(lm, "model", None), "config", None), "vocab_size"),
    )
    for obj, attr in candidates:
        v = getattr(obj, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return 1000


def _peak_gb() -> float:
    """当前设备的最大已分配显存（GB）；无 CUDA 时为 0.0。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1 << 30)


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def measure_prefill(
    lm: LoadedModel,
    seq_len: int,
    batch_size: int = 1,
    warmup: int = 3,
    iters: int = 10,
    budget_ratio: Optional[float] = None,
    compaction_mode: str = "identity",
    max_prompt_tokens: Optional[int] = None,
    dest_len: int = 0,
    attn_hook: Optional[Any] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """测量 prefill 延迟分布、吞吐与峰值显存。

    延迟用 `_env.benchmark_ms` 逐次采样，返回的是**分布**而非单值，
    直接喂给 `experiments.common.report.summarize` 即可得到
    median / p5 / p95 / bootstrap CI。

    三段结构（H0，2026-09-21）
    -------------------------
    输入是 `seq_len + dest_len` 个 token 的**一段连续序列**，按语义切两半：
    前 `seq_len` 个属于源端（构造 KV），后 `dest_len` 个是目的端自己的 query。
    切成一段而不是造两份随机串，是为了让两臂面对完全相同的 token 序列。

    原生路径（`attn_hook=None`）的 `_one()`：
        ① `lm.model(src_ids, use_cache=True)`          —— 源端构造，两臂共有
        ② `apply_kv_budget(cache, B)`                  —— 仅压缩臂
        ③ `lm.model(dst_ids, past_key_values=cache, position_ids=...)   —— 目的端前向

    ③ 让 ① 与 ② 的产物真正进入计时窗口：两臂在这一步的唯一差别是 cache 的
    KV 长度（dense = S，压缩 = B）。① 不可被压缩降低，故它是比值里的公共项
    —— 这正是论文承认的「实测加速比会被不可重叠的部分稀释」。

    ③ 会就地扩展传入的 cache（HF 的正常行为）。因为 ① 每轮都新建 cache，该
    扩展不会跨迭代累积，故各次采样相互独立。

    钩子路径（`attn_hook` 非 None）
    -------------------------------
    ① 源端构造（两臂共有）；② 不动 cache（紧凑化在钩子内部）；③ 目的端前向
    **past=None** —— 源端由钩子 state 携带（紧凑块带 β 与 V 回归，表达不成普通
    KV）。本路径下**开两个计时窗口**：

    - `prefill_ms_*`：① + 构造 + ③，即"端到端含构造"的口径；
    - `dest_ms_*`：**只** ③，复用预先建好的条件化块。两臂在这个窗口里的唯一
      差别是目的端注意力消费的 KV 长度 ⇒ 这是 kernel-matched 的量，
      E6 用它算 `prefill_speedup_kernel_matched`。

    两个窗口都要报：把含构造的那个当"机制收益"会低估，把不含构造的那个当
    "端到端"会高估 —— 而两者的差别（T_build）已经落在产物里（`n_builds`）。

    Args:
        dest_len: 目的端 query 长度。**压缩臂与钩子路径都必须 > 0**。
        attn_hook: `attention_hook.HookConfig` 或 None。给了就走钩子路径，
            此时 **`budget_ratio` 必须为 None**（预算归钩子配置）且
            `compaction_mode` 必须是 `"identity"` —— 两处各写一份预算 /
            两个旋钮作用在同一份 cache 上，都只会产生歧义。
        seed: 生成输入的随机种子。**两臂必须用同一个 seed**：否则 dense 臂与
            dcc 臂看到的 token 序列不同，配对性就没了（而"两臂唯一差别是 KV
            长度"这条断言正是配对性的全部内容）。

    Raises:
        ValueError: 压缩臂/钩子路径而 `dest_len <= 0`（此时 ③ 不会发生，窗口里
            没有任何一次前向使用紧凑 KV ⇒ 加速比结构上恒 <= 1；静默返回会把
            量具缺陷读成方法缺陷，这正是 H0 未接时的失效模式）。
        ValueError: 钩子路径下同时给了 `budget_ratio` / 非 identity 的
            `compaction_mode`。
    """
    hook_path = attn_hook is not None
    if (budget_ratio is not None or hook_path) and int(dest_len) <= 0:
        raise ValueError(
            "压缩臂必须给出 dest_len > 0：否则 ③ 不会发生，"
            "计时窗口里没有任何一次前向使用紧凑 KV，"
            "prefill_speedup 结构上恒 <= 1（H0 未接时的失效模式）。"
        )
    if hook_path and budget_ratio is not None:
        raise ValueError(
            "钩子路径的预算由 attn_hook（HookConfig.budget / budget_ratio）给出；"
            "同时再传 budget_ratio 会让'预算到底按谁算'变成猜。请只保留一处声明。"
        )
    if hook_path and compaction_mode != "identity":
        raise ValueError(
            f"钩子路径自己完成构造（build_compact_kv），不接受 compaction_mode="
            f"{compaction_mode!r}：那是 apply_kv_budget 的旋钮。请传 'identity'。"
        )

    device = lm.device
    total_len = int(seq_len) + max(0, int(dest_len))
    # 显式给种子：两臂要看到**同一串** token，否则"唯一差别是 KV 长度"不成立。
    # 生成固定在 CPU 上再做 device 搬运 —— torch 不允许 CPU generator 生成
    # CUDA 张量（会报 Expected a 'cuda' device type for generator）。
    gen = torch.Generator().manual_seed(int(seed))
    ids = torch.randint(0, _token_bound(lm), (batch_size, total_len),
                        generator=gen).to(device)
    src_ids = ids[:, : int(seq_len)]
    dst_ids = ids[:, int(seq_len):] if int(dest_len) > 0 else None
    dst_pos = (None if dst_ids is None else
               torch.arange(int(seq_len), int(seq_len) + int(dst_ids.shape[1]),
                            device=device).unsqueeze(0).expand(batch_size, -1))

    dest_note = (
        "未测（原生路径：目的端与源端共用同一次前向，无法只对目的端计时）。"
        "本列只在钩子路径下有意义。"
    )
    dest_samples: Optional[List[float]] = None
    dest_peak_gb: Optional[float] = None
    hook_summary: Optional[Dict[str, Any]] = None

    if not hook_path:
        def _one() -> None:
            # ① 源端构造（两臂共有）
            out = lm.model(input_ids=src_ids, use_cache=True)
            cache = out.past_key_values
            # ② 压缩（仅压缩臂）。其产出必须被 ③ 消费，否则不该静默通过。
            if budget_ratio is not None:
                apply_kv_budget(cache,
                                budget=max(1, int(round(budget_ratio * seq_len))),
                                mode=compaction_mode)
            # ③ 目的端：只对 cache 中的 KV 做注意力（H0 的收益载体）
            if dst_ids is not None:
                # position_ids 必须显式给出（2026-09-21 修）。裁剪后 cache 的
                # get_seq_length() 返回的是**实际张量长度 B**（实测确认，不是原长 S），
                # 缺省会让 dst 的 RoPE 位置从 B 起算：实测与正确位置差 4.9e-3，
                # 而 dense 参照臂的路径差只有 9.7e-8 —— 差 5 个数量级。更关键的是
                # dense 臂不裁剪、位置本就是 S..，于是两臂的**目的端位置不一致**，
                # 「两臂唯一差别是 KV 长度」不再成立。这一条与 score_choices 早已
                # 显式给 position_ids 的做法对齐（见其文档要点 ②）。
                lm.model(input_ids=dst_ids, past_key_values=cache, use_cache=True,
                         position_ids=dst_pos)
            del out

        _reset_peak()
        samples = _env.benchmark_ms(_one, warmup=warmup, iters=iters)
        peak_gb = _peak_gb()
        hook_note = ("未接（原生 apply_kv_budget 路径）：源端 KV 留在 cache 里，"
                     "紧凑化不发生，故此行的质量与 dcc_kv 方法论无关。")
    else:
        with _A.dcc_attention(lm.model, attn_hook) as st:
            def _one_full() -> None:
                st.source_phase()
                lm.model(input_ids=src_ids, use_cache=True)
                st.destination_phase(compacts="build")
                lm.model(input_ids=dst_ids, use_cache=True, position_ids=dst_pos)

            _reset_peak()
            samples = _env.benchmark_ms(_one_full, warmup=warmup, iters=iters)
            peak_gb = _peak_gb()

            # 建好块给第二个窗口复用（这一次不计时）。必须重跑一遍而不是
            # 沿用窗口 1 最后一次的块：窗口 1 的每轮都在 ③ 内部重建，末轮的块
            # 与"下一个源端前向"没有关系，而窗口 2 不跑源端前向 —— 直接复用
            # 会让"块对应的源"与"配置声明的源"不是同一次前向。
            st.source_phase()
            lm.model(input_ids=src_ids, use_cache=True)
            st.destination_phase(compacts="build")
            lm.model(input_ids=dst_ids, use_cache=True, position_ids=dst_pos)
            st.destination_phase(compacts="reuse")

            def _one_dest() -> None:
                lm.model(input_ids=dst_ids, use_cache=True, position_ids=dst_pos)

            _reset_peak()
            dest_samples = _env.benchmark_ms(_one_dest, warmup=warmup, iters=iters)
            dest_peak_gb = _peak_gb()
            hook_summary = st.summary()
        dest_note = (
            "只目的端前向，复用预先建好的条件化块 ⇒ 两臂在这个窗口里的唯一差别是"
            "**目的端注意力消费的 KV 长度**（dense 臂 = S，dcc 臂 = B）。"
            "不含 T_build（构造在窗口外）、不含源端构造。"
            "kernel-matched 的加速比必须用这一列算 —— 用含构造的那一列会把"
            "源端侧代价算进目的端。"
        )
        hook_note = ("已接（src/distributed/attention_hook.py）：源端由钩子 state "
                     "携带，目的端前向 past=None，紧凑块带 β 与 V 回归。")

    med = sorted(samples)[len(samples) // 2]
    tokens = batch_size * seq_len
    out: Dict[str, Any] = {
        "seq_len": seq_len,
        "dest_len": int(dest_len),
        "batch_size": batch_size,
        "seed": int(seed),
        "prefill_ms_samples": samples,
        "prefill_ms_median": med,
        "tokens_per_s_median": (tokens / (med / 1000.0)) if med > 0 else float("nan"),
        "peak_memory_gb": peak_gb,
        "budget_ratio": budget_ratio,
        "compaction_mode": compaction_mode,
        "attn_hook": hook_summary,
        "attn_hook_note": hook_note,
        "dest_ms_samples": dest_samples,
        "dest_ms_median": (None if dest_samples is None
                           else sorted(dest_samples)[len(dest_samples) // 2]),
        "dest_peak_memory_gb": dest_peak_gb,
        "dest_window_note": dest_note,
        "budget_tokens": (None if budget_ratio is None
                          else max(1, int(round(budget_ratio * seq_len)))),
        "budget_tokens_resolved": (None if hook_summary is None
                                   else hook_summary.get(
                                       "resolved_budget_by_source_len", {})),
        "kv_bytes_full": 2 * lm.num_layers * lm.num_kv_heads * lm.head_dim
                         * seq_len * batch_size * _env.dtypes_for(lm.precision).itemsize,
    }
    return out


def kv_cache_bytes(lm: LoadedModel, seq_len: int, keep_ratio: float = 1.0,
                   batch_size: int = 1) -> int:
    """KV cache 字节数（口径与论文的通信量讨论一致）。"""
    itemsize = _env.dtypes_for(lm.precision).itemsize
    return int(2 * lm.num_layers * lm.num_kv_heads * lm.head_dim
               * seq_len * batch_size * itemsize * keep_ratio)


__all__ = [
    "LoadedModel",
    "load_model",
    "apply_kv_budget",
    "EvalSample",
    "load_eval_file",
    "prompt_split",
    "score_choices",
    "evaluate",
    "measure_prefill",
    "kv_cache_bytes",
]
