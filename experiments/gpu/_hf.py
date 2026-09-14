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
        kf = k.float()
        if mode in ("topk_rms", "topk_norm"):
            pooled = kf.pow(2).mean(dim=(0, 1, 3))          # [S]
            if mode == "topk_norm":
                pooled = kf.pow(2).sum(dim=(0, 1, 3))
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


@torch.no_grad()
def score_choices(
    lm: LoadedModel,
    sample: EvalSample,
    budget_ratio: Optional[float] = None,
    compaction_mode: str = "identity",
    max_prompt_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """对一个样本的每个选项打分，返回正确选项与其得分。

    打分口径：以 "prompt + choice" 的 continuation 平均 logprob 作为分数
    （长度归一），这是长文 QA 评测里的常规做法。所有选项共用**同一份
    prefill 结果**（各自持一份副本），因此选项差异不会污染 prefill 计时。

    两处实现细节是正确性的前提，不要"优化"掉：

    1. **每个选项一份 cache 副本。** 前缀只算一次（prefill 的代价是 O(S²)，
       不能按选项重算），但 `DynamicCache` 在前向中会**就地追加**新 token：
       若所有选项共用同一对象，第 2 个选项就会在"前缀 + 第 1 个选项的
       continuation"之上继续生成，选项之间的比较不再同源。
    2. **必须显式传 position_ids。** 预算裁剪后 cache 长度为 B < S；若留空，
       transformers 会把 continuation 的绝对位置算成 B..B+T-1 而不是
       S..S+T-1，RoPE 的相对距离整体错位，准确率会被系统性低估。
       （保留位置子集本身不改被保留 key 的表示，见模块文档。）
    """
    tok = lm.tokenizer
    device = lm.device

    prompt_ids = tok(sample.prompt, return_tensors="pt",
                     add_special_tokens=False).input_ids.to(device)
    if max_prompt_tokens is not None:
        prompt_ids = prompt_ids[:, -max_prompt_tokens:]
    S = int(prompt_ids.shape[1])

    prefix = lm.model(input_ids=prompt_ids, use_cache=True)
    cache = prefix.past_key_values
    if budget_ratio is not None:
        apply_kv_budget(cache, budget=max(1, int(round(budget_ratio * S))),
                        mode=compaction_mode)

    scores: List[float] = []
    for choice in sample.choices:
        cont_ids = tok(choice, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(device)
        if cont_ids.numel() == 0:
            scores.append(float("-inf"))
            continue
        T = int(cont_ids.shape[1])
        # 复用前缀：每个选项拿到裁剪后 cache 的**独立副本**（见函数文档 ①），
        # 且必须显式给出绝对位置（见函数文档 ②）
        past = copy.deepcopy(cache)
        position_ids = torch.arange(S, S + T, device=device).unsqueeze(0)
        out = lm.model(input_ids=cont_ids, past_key_values=past,
                       use_cache=True, position_ids=position_ids)
        logits = out.logits[0]                       # [T, V]
        # 第 i 个 continuation token 由第 i-1 个位置的 logits 预测
        prev = torch.cat([prefix.logits[0, -1:, :], logits[:-1, :]], dim=0)
        lp = torch.log_softmax(prev.float(), dim=-1)
        tgt = cont_ids[0]
        scores.append(float(lp.gather(-1, tgt[:, None]).mean()))

    pred = int(max(range(len(scores)), key=lambda i: scores[i]))
    return {
        "task": sample.task,
        "length_tag": sample.length_tag,
        "n_choices": len(sample.choices),
        "answer": sample.answer,
        "pred": pred,
        "correct": int(pred == sample.answer),
        "scores": scores,
        "prompt_tokens": S,
    }


def evaluate(
    lm: LoadedModel,
    samples: Sequence[EvalSample],
    budget_ratio: Optional[float] = None,
    compaction_mode: str = "identity",
    max_prompt_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """对样本集评测，返回总体准确率与按任务/长度的分组准确率。"""
    rows = [score_choices(lm, s, budget_ratio, compaction_mode, max_prompt_tokens)
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
) -> Dict[str, Any]:
    """测量 prefill 延迟分布、吞吐与峰值显存。

    延迟用 `_env.benchmark_ms` 逐次采样，返回的是**分布**而非单值，
    直接喂给 `experiments.common.report.summarize` 即可得到
    median / p5 / p95 / bootstrap CI。
    """
    device = lm.device
    ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    def _one() -> None:
        out = lm.model(input_ids=ids, use_cache=True)
        if budget_ratio is not None:
            apply_kv_budget(out.past_key_values,
                            budget=max(1, int(round(budget_ratio * seq_len))),
                            mode=compaction_mode)
        del out

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    samples = _env.benchmark_ms(_one, warmup=warmup, iters=iters)

    peak_gb = 0.0
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1 << 30)

    med = sorted(samples)[len(samples) // 2]
    tokens = batch_size * seq_len
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "prefill_ms_samples": samples,
        "prefill_ms_median": med,
        "tokens_per_s_median": (tokens / (med / 1000.0)) if med > 0 else float("nan"),
        "peak_memory_gb": peak_gb,
        "budget_ratio": budget_ratio,
        "compaction_mode": compaction_mode,
        "kv_bytes_full": 2 * lm.num_layers * lm.num_kv_heads * lm.head_dim
                         * seq_len * batch_size * _env.dtypes_for(lm.precision).itemsize,
    }


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
    "score_choices",
    "evaluate",
    "measure_prefill",
    "kv_cache_bytes",
]
