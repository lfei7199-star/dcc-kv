"""provenance 戳的守卫（2026-09-22 工程审查 F3）。

背景：CPU 实验的产物原本**没有任何版本标识**，而论文 §6 的全部数字都来自 CPU
落盘。后果不是假想 —— 代表 Query 最远点采样的起点默认值在 09-22 由 `random`
改为 `newest`（服从论文式(11)），而 E10/E11 的落盘停在 09-16（旧口径），
论文因此混用了两套起点口径，事后只能靠人工翻 git 历史才查得出。

本文件把「产物必须能自证哪版代码、哪个口径」变成可执行断言。
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import report as R  # noqa: E402


REQUIRED_KEYS = {
    "written_at_utc",
    "git_commit",
    "git_dirty",
    "fps_start",
    "python_version",
    "numpy_version",
    "source_sha256",
}


# -----------------------------------------------------------------------------
# A. 戳本身
# -----------------------------------------------------------------------------

def test_a1_stamp_has_required_keys():
    prov = R.provenance_stamp()
    assert REQUIRED_KEYS <= set(prov), f"缺少键：{REQUIRED_KEYS - set(prov)}"


def test_a2_fps_start_follows_paper():
    """论文式(11) 明写「以最新 Query 为初始锚点」⇒ 默认起点必须是 `newest`。

    这是 F2 的守卫：改回 `random` 就该红 —— 因为那意味着落盘口径再次脱离论文。
    """
    assert R.provenance_stamp()["fps_start"] == "newest"


def test_a3_all_key_sources_are_hashed():
    """关键源码必须全部可哈希 —— 出现 `missing` 说明路径写错或文件被挪走。"""
    hashes = R.provenance_stamp()["source_sha256"]
    assert hashes, "source_sha256 为空，戳失去意义"
    missing = [k for k, v in hashes.items() if v == "missing"]
    assert not missing, f"以下关键源码未被哈希到：{missing}"
    assert all(len(v) == 16 for v in hashes.values())


def test_a4_stamp_is_cached_but_returned_by_value():
    """同进程内二次调用应一致（缓存），且调用方改不动缓存。"""
    a = R.provenance_stamp()
    a["fps_start"] = "被篡改"
    assert R.provenance_stamp()["fps_start"] == "newest"


# -----------------------------------------------------------------------------
# B. save_summary 的契约
# -----------------------------------------------------------------------------

def test_b1_save_summary_attaches_provenance(tmp_path):
    p = tmp_path / "s.json"
    R.save_summary(str(p), {"a": 1, "b": [2]})
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["a"] == 1 and d["b"] == [2], "原有键必须逐位保留"
    assert REQUIRED_KEYS <= set(d["provenance"])


def test_b2_save_summary_does_not_mutate_caller_dict(tmp_path):
    """调用方的 dict 不能被就地改动 —— 否则同一 payload 复用时会带上旧戳。"""
    payload = {"a": 1}
    R.save_summary(str(tmp_path / "s.json"), payload)
    assert "provenance" not in payload


def test_b3_save_summary_tolerates_non_dict(tmp_path):
    """非 dict 退化为 save_json，不静默篡改形状。"""
    p = tmp_path / "l.json"
    R.save_summary(str(p), [1, 2, 3])
    assert json.loads(p.read_text(encoding="utf-8")) == [1, 2, 3]


def test_b4_save_summary_output_is_standard_json(tmp_path):
    """带戳后仍必须是严格 JSON（NaN/Infinity 已被 json_safe 处理）。"""
    p = tmp_path / "s.json"
    R.save_summary(str(p), {"x": float("nan"), "y": float("inf")})
    text = p.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError(f"非标准 JSON 常量 {c}")))
