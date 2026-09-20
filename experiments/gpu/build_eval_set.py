#!/usr/bin/env python
"""G5：评测集转换器 —— 把 LongBench 转成 `_hf.EvalSample` 的 JSONL。

为什么需要它
------------
E5-A2/A3、E6 与 E7 的**准确率列全部依赖它**：仓库里 `_hf.score_choices` 只认
`{"prompt", "choices", "answer", "task", "length_tag"}` 这种多选题格式，而
LongBench 的原始记录是 `{"input", "context", "answers", ...}`。缺这一环，
租来的卡只能跑出通信量，跑不出任务指标。

口径裁决（2026-09-16，西瓜先生）：**只收原生多选子集**。
即"选项来自数据集自身"的记录才转换；选项需要由我们合成的一律跳过并记账。
理由是这个口径下准确率可与公开数字对照，而合成干扰项会引入评测集构造自由，
使准确率失去可比性。

由此得到三条硬规则
------------------

**规则 1：选项只能来自数据自身的声明。** 两个来源，按优先级：
    ① 记录里显式带 `choices` / `options` 字段；
    ② LongBench 分类任务自带的 `all_classes`（TREC 等任务的标签空间）。
两者都没有的记录**一律跳过**，不猜、不合成。

**规则 2：跳过必须记账，不能静默省略。** 每条被跳过的记录都归入一个原因桶，
最终写出 `*.report.json`。其中**专列一节**报告"负结果条件所需但被跳过"的任务
（见 `E7_REQUIRED_TASKS`）—— 因为按本口径，检索类任务恰恰属于"无原生选项"的
一类，E7 的「强检索任务」条件会因此缺数据。这必须在报告里显式暴露，
否则会变成"E7 少跑一个条件而没人发现"。

**规则 3：提示模板不是官方的，必须声明。** LongBench 官方评测用逐任务
few-shot 模板（`dataset2prompt` / `dataset2maxlen`），本转换器的默认模板是
最小形式 `{context}\n\n{input}\nAnswer:`。用默认模板得到的准确率
**不可**与论文里的 LongBench 数字直接比较；要对照就传 `--prompt-template`
给出与官方一致的模板。这个差异写进 payload，不靠读者自己猜。

用法
----
    # 看一眼转换统计（不写文件）
    python experiments/gpu/build_eval_set.py --source longbench \
        --input /path/to/longbench/trec.jsonl --task trec --dry-run

    # 真正转换
    python experiments/gpu/build_eval_set.py --source longbench \
        --input '/path/to/longbench/*.jsonl' --out data/longbench_mc.jsonl

    # 自检：用内联样例跑完整链路，不需要任何外部数据
    python experiments/gpu/build_eval_set.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.gpu import _hf  # noqa: E402

SCRIPT = "experiments/gpu/build_eval_set.py"

DEFAULT_PROMPT_TEMPLATE = "{context}\n\n{input}\nAnswer:"

# 负结果（E7）需要的任务能力。缺失时报告里会单独警告。
# 注意：按"只收原生多选子集"的口径，检索类任务通常**没有**原生选项，
# 因而会落在 skipped 里 —— 这不是 bug，是口径的直接后果，必须让人看见。
E7_REQUIRED_TASKS: Dict[str, List[str]] = {
    "strong_retrieval": ["passage_retrieval_en", "passage_retrieval_zh"],
    "short_context": ["*"],     # 任何含 <4K 样本的任务都算覆盖
    "low_budget": ["*"],
}

# 通配条件（值为 ["*"]）**不能**靠任务名判定覆盖，得按规则判。
# 自查发现（2026-09-18）：原先 `missing_e7_conditions` 对通配条件直接
# `continue`，注释写"由调用方补判"，但全仓没有任何调用方补判 ——
# 于是 `--require-e7` 名义上卡三个条件，实际只卡 strong_retrieval。
# 现在把两条规则写死在这里：能判的真判，判不了的显式声明判不了。
E7_WILDCARD_RULES: Dict[str, str] = {
    # 转换产物里有 length_tag == "lt4k" 的样本 ⇒ 覆盖
    "short_context": "lt4k_samples_present",
    # 预算轴是实验自变量，不是评测数据的属性 ⇒ 本工具**无法**判定，
    # 不把它算成"已覆盖"，而是单列进 e7_conditions_not_decidable。
    "low_budget": "not_decidable_from_eval_set",
}

# 长度分档（token 数），用于 `length_tag`。边界取 E7 的 4K 与 E6 的 4/8/16/32K。
LENGTH_BUCKETS: List[Tuple[int, str]] = [
    (4096, "lt4k"),
    (8192, "4k-8k"),
    (16384, "8k-16k"),
    (32768, "16k-32k"),
]
LENGTH_OVERFLOW_TAG = "gt32k"
LENGTH_UNKNOWN_TAG = "unknown"


def length_tag_of(n_tokens: Optional[int]) -> str:
    if n_tokens is None or n_tokens <= 0:
        return LENGTH_UNKNOWN_TAG
    for bound, tag in LENGTH_BUCKETS:
        if n_tokens < bound:
            return tag
    return LENGTH_OVERFLOW_TAG


# =============================================================================
# 转换报告
# =============================================================================

@dataclass
class ConversionReport:
    """逐条记账的转换统计。**跳过不是错误，但必须可见。**"""
    source: str
    total: int = 0
    converted: int = 0
    skipped_by_reason: Dict[str, int] = field(default_factory=dict)
    per_task: Dict[str, Dict[str, int]] = field(default_factory=dict)
    examples_skipped: Dict[str, str] = field(default_factory=dict)
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    prompt_template_is_official: bool = False
    length_tag_counts: Dict[str, int] = field(default_factory=dict)

    def note_length_tag(self, tag: str) -> None:
        """记下一条已转换样本的长度分档（供 E7 的 short_context 条件判定）。"""
        self.length_tag_counts[tag] = self.length_tag_counts.get(tag, 0) + 1

    def skip(self, reason: str, task: str, detail: str = "") -> None:
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1
        book = self.per_task.setdefault(task, {"converted": 0, "skipped": 0})
        book["skipped"] += 1
        if reason not in self.examples_skipped and detail:
            self.examples_skipped[reason] = detail[:160]

    def convert(self, task: str) -> None:
        self.converted += 1
        book = self.per_task.setdefault(task, {"converted": 0, "skipped": 0})
        book["converted"] += 1

    def missing_e7_conditions(self) -> Dict[str, List[str]]:
        """哪些 E7 条件在本次转换里**确定**没有拿到数据。

        判定分两类：
        - 具名条件（strong_retrieval）：看是否有任一所需任务转换成功；
        - 通配条件：按 `E7_WILDCARD_RULES` 判。`lt4k_samples_present` 能真判；
          规则里没有的（新加的通配条件忘了配规则）**视为缺失**而不是跳过 ——
          默认"缺失"会吵，但不会静默少报一个条件。
        """
        have = {t for t, b in self.per_task.items() if b["converted"] > 0}
        missing: Dict[str, List[str]] = {}
        for cond, tasks in E7_REQUIRED_TASKS.items():
            if "*" in tasks:
                rule = E7_WILDCARD_RULES.get(cond)
                if rule == "lt4k_samples_present":
                    if self.length_tag_counts.get("lt4k", 0) <= 0:
                        missing[cond] = ["任何含 <4K token 的样本（按 length_tag=lt4k 判）"]
                elif rule == "not_decidable_from_eval_set":
                    continue
                else:
                    missing[cond] = [f"未配置通配规则（E7_WILDCARD_RULES 缺 {cond}）"]
                continue
            if not (set(tasks) & have):
                missing[cond] = tasks
        return missing

    def not_decidable_e7_conditions(self) -> Dict[str, str]:
        """本工具**判不了**的 E7 条件及其原因。

        单列一节而不是静默放过：`--require-e7` 不能用它们拦住租卡，
        但报告必须让人看见"这个条件没有被任何东西检查过"。
        """
        out: Dict[str, str] = {}
        for cond, tasks in E7_REQUIRED_TASKS.items():
            if "*" not in tasks:
                continue
            if E7_WILDCARD_RULES.get(cond) == "not_decidable_from_eval_set":
                out[cond] = (
                    "该条件是**实验自变量**（预算轴），不是评测数据的属性；"
                    "评测集转换器无法判定它是否被覆盖。"
                    "请在 E7 运行配置里显式给出低预算档，别指望本报告替你检查。"
                )
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "total": self.total,
            "converted": self.converted,
            "skipped_total": self.total - self.converted,
            "skipped_by_reason": dict(sorted(self.skipped_by_reason.items(),
                                             key=lambda kv: -kv[1])),
            "per_task": dict(sorted(self.per_task.items())),
            "examples_skipped": self.examples_skipped,
            "prompt_template": self.prompt_template,
            "prompt_template_is_official": self.prompt_template_is_official,
            "e7_conditions_missing": self.missing_e7_conditions(),
            "e7_conditions_not_decidable": self.not_decidable_e7_conditions(),
            "length_tag_counts": dict(sorted(self.length_tag_counts.items())),
            "policy": (
                "只收原生多选子集：选项须来自记录自身的 choices/options 字段，"
                "或分类任务的 all_classes 标签空间。二者皆无则跳过并记账，"
                "不由本工具合成干扰项。"
            ),
        }


# =============================================================================
# LongBench 适配
# =============================================================================

def _render_prompt(record: Dict[str, Any], template: str) -> str:
    return template.format(
        context=str(record.get("context", "")),
        input=str(record.get("input", record.get("question", ""))),
        question=str(record.get("input", record.get("question", ""))),
    )


def _native_choices(record: Dict[str, Any]) -> Tuple[Optional[List[str]], str]:
    """取出记录**自带**的选项集，返回 (choices, 来源说明)。取不到则 (None, 原因)。"""
    for key in ("choices", "options"):
        raw = record.get(key)
        if isinstance(raw, list) and len(raw) >= 2:
            return [str(x) for x in raw], f"field:{key}"
        if isinstance(raw, dict) and len(raw) >= 2:
            # 形如 {"A": "...", "B": "..."} —— 按 key 排序保证确定性
            return [str(raw[k]) for k in sorted(raw)], f"field:{key}"
    classes = record.get("all_classes")
    if isinstance(classes, list) and len(classes) >= 2:
        return [str(x) for x in classes], "field:all_classes"
    return None, "no-native-choices"


def convert_record(
    record: Dict[str, Any],
    *,
    source: str = "longbench",
    task: Optional[str] = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Tuple[Optional[_hf.EvalSample], str, str]:
    """转换单条记录。

    Returns:
        (sample_or_None, task, reason)
        sample 为 None 时 reason 给出跳过的原因键（可用于记账）。
    """
    task_name = str(task or record.get("dataset") or record.get("task") or "unknown")

    answers = record.get("answers", record.get("answer"))
    if answers is None:
        return None, task_name, "no-answer-field"
    if isinstance(answers, (str, int)):
        answer_list = [answers]
    elif isinstance(answers, list) and answers:
        answer_list = answers
    else:
        return None, task_name, "empty-answer"

    choices, why = _native_choices(record)
    if choices is None:
        return None, task_name, why

    # 答案必须能映射到选项。分类任务的 answers 是标签字符串，多选任务是字母。
    gold = answer_list[0]
    idx: Optional[int] = None
    if isinstance(gold, int) and 0 <= gold < len(choices):
        idx = int(gold)
    else:
        g = str(gold).strip()
        if g in choices:
            idx = choices.index(g)
        elif len(g) == 1 and g.isalpha():
            k = ord(g.upper()) - ord("A")
            if 0 <= k < len(choices):
                idx = k
        elif g.isdigit() and int(g) < len(choices):
            idx = int(g)
    if idx is None:
        return None, task_name, "answer-not-in-choices"

    context = str(record.get("context", ""))
    question = str(record.get("input", record.get("question", "")))
    if not context and not question:
        return None, task_name, "empty-prompt"

    n_tokens = record.get("length")
    if not isinstance(n_tokens, int):
        n_tokens = None

    sample = _hf.EvalSample(
        prompt=_render_prompt(record, prompt_template),
        choices=choices,
        answer=idx,
        task=task_name,
        length_tag=length_tag_of(n_tokens),
    )
    return sample, task_name, "ok"


def convert_records(
    records: Iterable[Dict[str, Any]],
    *,
    source: str = "longbench",
    task: Optional[str] = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    prompt_template_is_official: bool = False,
) -> Tuple[List[_hf.EvalSample], ConversionReport]:
    """转换一批记录，并逐条记账。纯函数，不碰磁盘（便于单测）。"""
    rep = ConversionReport(source=source, prompt_template=prompt_template,
                           prompt_template_is_official=prompt_template_is_official)
    out: List[_hf.EvalSample] = []
    for rec in records:
        rep.total += 1
        sample, task_name, reason = convert_record(
            rec, source=source, task=task, prompt_template=prompt_template,
        )
        if sample is None:
            detail = f"answers={rec.get('answers', rec.get('answer'))!r} " \
                     f"keys={sorted(rec.keys())[:8]}"
            rep.skip(reason, task_name, detail)
        else:
            out.append(sample)
            rep.convert(task_name)
            rep.note_length_tag(sample.length_tag)
    return out, rep


# =============================================================================
# 校验与落盘
# =============================================================================

def validate_samples(samples: Sequence[_hf.EvalSample]) -> List[str]:
    """结构与语义校验。返回问题清单（空 = 通过）。"""
    problems: List[str] = []
    if not samples:
        problems.append("样本集为空：不写文件（空评测集会让准确率列静默变成 n/a）")
        return problems
    for i, s in enumerate(samples):
        if not s.prompt.strip():
            problems.append(f"#{i}: prompt 为空")
        if len(s.choices) < 2:
            problems.append(f"#{i}: 选项少于 2 个（无法构成多选题）")
        if not (0 <= s.answer < len(s.choices)):
            problems.append(f"#{i}: answer={s.answer} 越界（choices={len(s.choices)}）")
        if len(set(s.choices)) != len(s.choices):
            problems.append(f"#{i}: 选项有重复项")
        if s.task == "unknown":
            problems.append(f"#{i}: task 未标注（E7 的按任务分组会退化成一个桶）")
    return problems


def write_jsonl(samples: Sequence[_hf.EvalSample], path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for s in samples:
            f.write(json.dumps({
                "prompt": s.prompt, "choices": s.choices, "answer": s.answer,
                "task": s.task, "length_tag": s.length_tag,
            }, ensure_ascii=False) + "\n")


def write_report(rep: ConversionReport, path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False),
                 encoding="utf-8")


def read_records(path: str) -> List[Dict[str, Any]]:
    """读 JSONL。空行跳过；非 JSON 行显式报错（不静默丢弃数据）。"""
    out: List[Dict[str, Any]] = []
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{e}") from e
    return out


# =============================================================================
# 自检样例（不依赖任何外部数据）
# =============================================================================

SELFTEST_RECORDS: List[Dict[str, Any]] = [
    # ① 分类任务：选项来自 all_classes（原生）
    {"dataset": "trec", "input": "What is the capital of France?",
     "context": "Paris is the capital and largest city of France.",
     "answers": ["description"], "length": 5175,
     "all_classes": ["description", "entity", "abbreviation", "human", "numeric", "location"]},
    # ② 显式 choices 字段（原生）
    {"dataset": "custom_mc", "input": "Which city?",
     "context": "Lyon is in France.", "choices": ["Paris", "Lyon", "Nice"],
     "answers": ["B"], "length": 900},
    # ③ 检索任务：无原生选项 ⇒ 必须被跳过并记入 strong_retrieval 缺口
    {"dataset": "passage_retrieval_en", "input": "Find the passage about X",
     "context": "long text ...", "answers": ["3"], "length": 20000},
    # ④ 生成任务：无原生选项 ⇒ 跳过
    {"dataset": "gov_report", "input": "Summarize", "context": "...",
     "answers": ["some summary"], "length": 30000},
    # ⑤ 答案不在选项里 ⇒ 跳过
    {"dataset": "trec", "input": "q", "context": "c", "answers": ["nonexistent"],
     "all_classes": ["description", "entity"], "length": 100},
]


def run_selftest() -> int:
    samples, rep = convert_records(SELFTEST_RECORDS, source="selftest")
    print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    problems = validate_samples(samples)
    print(f"自检：转换 {rep.converted}/{rep.total}，校验问题 {len(problems)} 条")
    for p in problems:
        print("  -", p)
    ok = (rep.converted == 2 and rep.total == 5
          and rep.skipped_by_reason.get("no-native-choices") == 2
          and rep.skipped_by_reason.get("answer-not-in-choices") == 1
          and not problems)
    print("自检结果：", "通过" if ok else "未通过")
    return 0 if ok else 1


# =============================================================================
# main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="G5：LongBench → _hf.EvalSample JSONL（只收原生多选子集）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="longbench", choices=["longbench"])
    p.add_argument("--input", nargs="+", default=None,
                   help="原始 JSONL 路径；支持通配（由 shell 或 --glob 展开）")
    p.add_argument("--glob", dest="use_glob", action="store_true",
                   help="把 --input 当通配符模式展开（Windows 下 shell 不展开时用）")
    p.add_argument("--task", default=None,
                   help="覆盖 task 名（单任务文件建议显式给出，否则取记录的 dataset 字段）")
    p.add_argument("--out", default=None, help="输出 JSONL 路径")
    p.add_argument("--report", default=None,
                   help="转换报告路径；缺省为 <out 同目录>/<out 名>.report.json")
    p.add_argument("--prompt-template", dest="prompt_template", default=DEFAULT_PROMPT_TEMPLATE,
                   help="可用 {context} {input} {question}。默认模板**不是** LongBench 官方模板，"
                        "用它与公开数字比较是无效的")
    p.add_argument("--prompt-template-is-official", dest="prompt_template_is_official",
                   action="store_true",
                   help="声明所给模板与官方一致；不声明时报告里会写明不可比")
    p.add_argument("--limit", type=int, default=None, help="每个输入文件最多取多少条")
    p.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    p.add_argument("--require-e7", dest="require_e7", action="store_true",
                   help="若 E7 的某个条件缺数据则以退出码 4 结束（用于租卡前的自检）")
    p.add_argument("--selftest", action="store_true", help="用内联样例自检，不碰外部数据")
    return p


def main() -> int:
    a = build_parser().parse_args()

    if a.selftest:
        return run_selftest()

    if not a.input:
        print("必须给出 --input（原始 JSONL），或用 --selftest 跑内联自检。", file=sys.stderr)
        return 2

    paths: List[str] = []
    for pat in a.input:
        if a.use_glob or any(ch in pat for ch in "*?["):
            hits = sorted(glob.glob(pat))
            if not hits:
                print(f"[警告] 通配未命中任何文件：{pat}", file=sys.stderr)
            paths.extend(hits)
        else:
            paths.append(pat)
    if not paths:
        print("没有可读的输入文件。", file=sys.stderr)
        return 2

    records: List[Dict[str, Any]] = []
    for path in paths:
        if not pathlib.Path(path).exists():
            print(f"[错误] 文件不存在：{path}", file=sys.stderr)
            return 2
        got = read_records(path)
        if a.limit:
            got = got[: a.limit]
        print(f"  读取 {path}：{len(got)} 条")
        records.extend(got)

    samples, rep = convert_records(
        records, source=a.source, task=a.task,
        prompt_template=a.prompt_template,
        prompt_template_is_official=a.prompt_template_is_official,
    )
    problems = validate_samples(samples)

    print("=" * 78)
    print("G5 评测集转换 —— 报告")
    print("=" * 78)
    print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    print(f"校验问题：{len(problems)} 条")
    for p in problems[:10]:
        print("  -", p)

    missing = rep.missing_e7_conditions()
    if missing:
        print()
        print("[E7 条件缺数据] 按「只收原生多选子集」的口径，以下条件本次没有素材：")
        for cond, tasks in missing.items():
            print(f"  - {cond}：需要 {tasks}")
        print("  这不是转换器的 bug，而是口径的直接后果：检索类任务通常没有原生选项。")
        print("  处置见 docs/writing_scope_and_metrics.md —— 不得静默少报一个条件。")

    not_decidable = rep.not_decidable_e7_conditions()
    if not_decidable:
        print()
        print("[E7 条件本工具判不了] 以下条件**没有被本报告检查过**，"
              "不代表已覆盖：")
        for cond, why in not_decidable.items():
            print(f"  - {cond}：{why}")

    if a.dry_run:
        print("\n--dry-run：未写任何文件。")
        return 4 if (a.require_e7 and missing) else 0

    if not a.out:
        print("必须给出 --out（或加 --dry-run）。", file=sys.stderr)
        return 2
    if problems:
        print("\n[拒绝写盘] 校验未通过 —— 宁可不产出，也不产出会让准确率列静默失真的文件。",
              file=sys.stderr)
        return 3

    write_jsonl(samples, a.out)
    report_path = a.report or str(pathlib.Path(a.out).with_suffix(".report.json"))
    write_report(rep, report_path)
    print(f"\n已写出 {len(samples)} 条 → {a.out}")
    print(f"报告 → {report_path}")

    # 回读校验：写出的文件必须能被 _hf 读回且条数一致
    back = _hf.load_eval_file(a.out)
    print(f"回读校验：{len(back)} 条（与写出条数{'一致' if len(back) == len(samples) else '不一致'}）")
    if len(back) != len(samples):
        return 3
    return 4 if (a.require_e7 and missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
