"""G5 评测集转换器的锚点（纯 CPU，不需要 GPU，也不需要任何外部数据）。

这个文件守的是**口径**而不是功能：转换器真正容易出错的地方不是"能不能转"，
而是"悄悄多转/少转了几条，而下游准确率列照常产出"。

三条被冻成断言的规则：

1. **选项只能来自数据自身**（`choices`/`options`/`all_classes`）。
   任何"猜一个选项集"的路径都必须使转换条数变化时被这里拦住。
2. **跳过必须记账**，且 E7 的「强检索」条件缺数据时要被显式识别 ——
   按本口径检索类任务天然无原生选项，静默少报一个条件是最危险的失败模式。
3. **默认提示模板不是官方的**，报告里必须标 `prompt_template_is_official=False`。
   用非官方模板得到的准确率与论文数字不可比，这一点不能靠读者猜。
"""
from __future__ import annotations

import json
import pathlib

import pytest

from experiments.gpu import build_eval_set as B
from experiments.gpu import _hf


# =============================================================================
# 1. 选项来源
# =============================================================================

def test_all_classes_is_accepted_as_native_choices():
    rec = {"dataset": "trec", "input": "q", "context": "c",
           "answers": ["entity"], "length": 100,
           "all_classes": ["description", "entity", "numeric"]}
    s, task, reason = B.convert_record(rec)
    assert reason == "ok"
    assert task == "trec"
    assert s.choices == ["description", "entity", "numeric"]
    assert s.answer == 1


def test_generative_task_is_skipped_not_fabricated():
    """生成类任务没有可选集 —— 必须跳过，不得合成。"""
    rec = {"dataset": "gov_report", "input": "summarize", "context": "x" * 50,
           "answers": ["a summary"], "length": 30000}
    s, task, reason = B.convert_record(rec)
    assert s is None
    assert reason == "no-native-choices"


def test_retrieval_task_is_skipped_and_surfaces_e7_gap():
    """检索任务被跳过 ⇒ 报告必须点出 E7「强检索」条件缺数据。"""
    recs = [
        {"dataset": "passage_retrieval_en", "input": "find", "context": "x",
         "answers": ["3"], "length": 20000},
        {"dataset": "trec", "input": "q", "context": "c", "answers": ["entity"],
         "all_classes": ["description", "entity"], "length": 100},
    ]
    samples, rep = B.convert_records(recs, source="test")
    assert rep.converted == 1
    assert rep.skipped_by_reason == {"no-native-choices": 1}
    assert "strong_retrieval" in rep.missing_e7_conditions()


def test_answer_must_map_into_choices():
    """答案映射不上就跳过 —— 不能"就近取一个"，那会让准确率虚高。"""
    rec = {"dataset": "t", "input": "q", "context": "c", "answers": ["zzz"],
           "all_classes": ["a", "b"], "length": 10}
    s, _, reason = B.convert_record(rec)
    assert s is None and reason == "answer-not-in-choices"


def test_letter_and_index_answers_both_supported():
    base = {"input": "q", "context": "c", "choices": ["p", "q", "r"], "length": 10}
    s1, _, _ = B.convert_record({**base, "answers": ["B"]})
    s2, _, _ = B.convert_record({**base, "answers": [1]})
    s3, _, _ = B.convert_record({**base, "answers": ["r"]})
    assert (s1.answer, s2.answer, s3.answer) == (1, 1, 2)


def test_options_dict_is_sorted_for_determinism():
    """dict 形式的选项必须按 key 排序，否则同一输入会产出不同的 answer 索引。"""
    rec = {"input": "q", "context": "c", "answers": ["B"],
           "options": {"B": "second", "A": "first", "C": "third"}, "length": 10}
    s, _, reason = B.convert_record(rec)
    assert reason == "ok"
    assert s.choices == ["first", "second", "third"] and s.answer == 1


# =============================================================================
# 2. 记账与提示模板
# =============================================================================

def test_report_counts_every_record():
    samples, rep = B.convert_records(B.SELFTEST_RECORDS, source="selftest")
    assert rep.total == len(B.SELFTEST_RECORDS)
    assert rep.converted == len(samples)
    assert rep.total - rep.converted == sum(rep.skipped_by_reason.values())


def test_default_template_is_declared_non_official():
    _, rep = B.convert_records(B.SELFTEST_RECORDS, source="selftest")
    d = rep.to_dict()
    assert d["prompt_template_is_official"] is False
    assert "{context}" in d["prompt_template"]


def test_custom_template_is_rendered():
    rec = {"dataset": "t", "input": "Q?", "context": "CTX", "answers": ["A"],
           "choices": ["A", "B"], "length": 10}
    s, _, _ = B.convert_record(rec, prompt_template="{context}//{input}//end")
    assert s.prompt == "CTX//Q?//end"


def test_length_tag_buckets():
    """长度分档必须与 E7 的 4K 边界对齐，否则"短上下文"条件无法按标签筛。"""
    assert B.length_tag_of(100) == "lt4k"
    assert B.length_tag_of(4096) == "4k-8k"     # 边界归入上一档（< 判据）
    assert B.length_tag_of(8192) == "8k-16k"
    assert B.length_tag_of(32768) == "gt32k"
    assert B.length_tag_of(None) == "unknown"


# =============================================================================
# 3. 校验与落盘
# =============================================================================

def test_validate_rejects_empty_and_bad_records():
    assert B.validate_samples([])                       # 空集会静默把准确率变 n/a
    bad = _hf.EvalSample(prompt="", choices=["a"], answer=5, task="unknown")
    problems = B.validate_samples([bad])
    assert any("prompt 为空" in p for p in problems)
    assert any("选项少于 2 个" in p for p in problems)
    assert any("answer=5 越界" in p for p in problems)
    assert any("task 未标注" in p for p in problems)


def test_roundtrip_through_hf_loader(tmp_path: pathlib.Path):
    """写出的 JSONL 必须能被 `_hf.load_eval_file` 读回，字段一字不差。"""
    samples, rep = B.convert_records(B.SELFTEST_RECORDS, source="selftest")
    out = tmp_path / "eval.jsonl"
    B.write_jsonl(samples, str(out))
    back = _hf.load_eval_file(str(out))
    assert len(back) == len(samples)
    for a, b in zip(samples, back):
        assert (a.prompt, a.choices, a.answer, a.task, a.length_tag) == \
               (b.prompt, b.choices, b.answer, b.task, b.length_tag)


def test_report_is_written_and_valid_json(tmp_path: pathlib.Path):
    _, rep = B.convert_records(B.SELFTEST_RECORDS, source="selftest")
    p = tmp_path / "eval.report.json"
    B.write_report(rep, str(p))
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["converted"] == rep.converted
    assert "policy" in d


def test_read_records_rejects_malformed_json(tmp_path: pathlib.Path):
    """坏行必须报错并给出行号 —— 静默丢弃会让"少了 3 条"永远查不出来。"""
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2"):
        B.read_records(str(p))


def test_selftest_passes():
    """自检入口本身是租卡前的第一道闸，必须一直返回 0。"""
    assert B.run_selftest() == 0
