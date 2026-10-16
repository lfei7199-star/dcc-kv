"""G6 探针（probe_gpu_construction）守卫测试。

背景（2026-10-16 首日上机实测）：RTX PRO 6000 Blackwell 上探针返回
ok=false，但 checks 里记录的张量设备全是 cuda:0。根因是探针用
`torch.device("cuda")`（index=None）与张量的 `.device`（"cuda:0"，
永远带索引）做 `==` 严格比较——恒为 False，与 torch 版本无关。
构造链路本身无罪（同机 dense 冒烟 200 ms 正常）。

本文件的测试全部可在 CPU 上运行：用假张量 mock build_compact_kv
的返回，即可复现「张量全在 cuda:0」的服务器场景，无需真实 GPU。
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = REPO_ROOT / "experiments" / "gpu" / "_env.py"


class _FakeTensor:
    """只带 .device 的假张量——探针只访问 .device 与 str(.device)。"""

    def __init__(self, device: str):
        self.device = torch.device(device)


def _fake_compact_kv(devices: dict) -> SimpleNamespace:
    """造一个假 CompactKV：四个字段的设备由 devices 指定。"""
    return SimpleNamespace(
        keys=_FakeTensor(devices["keys"]),
        logit_bias=_FakeTensor(devices["logit_bias"]),
        values=_FakeTensor(devices["values"]),
        selected_indices=_FakeTensor(devices["selected_indices"]),
    )


@pytest.fixture
def probe_fn():
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from experiments.gpu import _env

    return _env.probe_gpu_construction


@pytest.fixture
def fake_cuda_randn(monkeypatch):
    """让 CPU 版 torch 的 randn 返回 cuda:0 假张量。

    probe_gpu_construction 内部除 is_available() 外还会
    `torch.randn(L, d, device=dev)` 造 K/V/Q——CPU 版 torch 到这就抛
    "Torch not compiled with CUDA enabled"。patch 掉才能走到判定逻辑。
    """
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: _FakeTensor("cuda:0"))


# ---------------------------------------------------------------------------
# ① 坑的根因：torch.device('cuda') != torch.device('cuda:0')
# ---------------------------------------------------------------------------

def test_device_equality_semantics_is_strict():
    """钉住语义：无索引 device 与带索引 device 的 == 比较为 False。

    这是 2026-10-16 误判的根。若未来某 torch 版本改了这个语义，
    本测试失败，提醒复核探针为何仍然要规范化（规范化比较在两个
    语义下都正确，改语义不影响修复的正确性，但会改变本测试的断言）。
    """
    assert torch.device("cuda").index is None
    assert torch.device("cuda:0").index == 0
    assert torch.device("cuda") != torch.device("cuda:0")
    # 同 type 同 index 才相等
    assert torch.device("cuda:0") == torch.device("cuda:0")
    assert torch.device("cpu") == torch.device("cpu")


# ---------------------------------------------------------------------------
# ② 服务器场景的 CPU 复现：张量全在 cuda:0 ⇒ 必须 ok=True
# ---------------------------------------------------------------------------

def test_probe_passes_when_all_tensors_on_indexed_cuda(
    monkeypatch, probe_fn, fake_cuda_randn
):
    """复现 2026-10-16 服务器实况：四条张量都在 cuda:0 ⇒ ok=True。

    修复前此场景恒返回 ok=False（误判）。修复后 _same_device 允许
    dev 不带索引，'cuda' 匹配 'cuda:0'。
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "src.dcc_kv_ref.build_compact_kv",
        lambda **kw: _fake_compact_kv(
            {k: "cuda:0" for k in ("keys", "logit_bias", "values", "selected_indices")}
        ),
    )
    r = probe_fn("cuda")
    assert r["ok"] is True, f"全 cuda:0 场景必须 PASS，实得: {r}"
    assert r["note"].startswith("构造链路在 CUDA 上可用")
    for case in ("topk", "full_budget"):
        assert r["checks"][case]["all_on_device"] is True


def test_probe_fails_when_a_tensor_stays_on_cpu(
    monkeypatch, probe_fn, fake_cuda_randn
):
    """反向：真有一条张量滞留 CPU ⇒ ok=False，且记账能指名肇事者。

    防止修复把探针改成"永远 PASS"——那比误判更危险。
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "src.dcc_kv_ref.build_compact_kv",
        lambda **kw: _fake_compact_kv(
            {
                "keys": "cuda:0",
                "logit_bias": "cuda:0",
                "values": "cpu",          # ← 肇事者
                "selected_indices": "cuda:0",
            }
        ),
    )
    r = probe_fn("cuda")
    assert r["ok"] is False
    for case in ("topk", "full_budget"):
        c = r["checks"][case]
        assert c["all_on_device"] is False
        assert c["values_device"] == "cpu"      # 记账直接指名
        assert c["keys_device"] == "cuda:0"


def test_checks_record_all_four_devices(monkeypatch, probe_fn, fake_cuda_randn):
    """checks 必须记录四条张量的设备——旧版只记两条，肇事张量无从辨认。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "src.dcc_kv_ref.build_compact_kv",
        lambda **kw: _fake_compact_kv(
            {k: "cuda:0" for k in ("keys", "logit_bias", "values", "selected_indices")}
        ),
    )
    r = probe_fn("cuda")
    for case in ("topk", "full_budget"):
        keys = set(r["checks"][case])
        assert {
            "keys_device", "logit_bias_device", "values_device",
            "selected_indices_device", "all_on_device",
        } <= keys, f"{case} 记账字段不全: {sorted(keys)}"


def test_probe_cpu_device_path(monkeypatch, probe_fn, fake_cuda_randn):
    """--device cpu 路径：dev=cpu，张量全 cpu ⇒ ok=True（不被误判）。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "src.dcc_kv_ref.build_compact_kv",
        lambda **kw: _fake_compact_kv(
            {k: "cpu" for k in ("keys", "logit_bias", "values", "selected_indices")}
        ),
    )
    r = probe_fn("cpu")
    assert r["ok"] is True


def test_probe_no_cuda_available_skips(monkeypatch, probe_fn):
    """无 CUDA ⇒ skipped=True，不是失败（硬件闸门语义）。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = probe_fn("cuda")
    assert r["ok"] is False
    assert r["skipped"] is True
    assert "无可用 CUDA 设备" in r["reason"]


# ---------------------------------------------------------------------------
# ③ 源码守卫：裸比较不得回归
# ---------------------------------------------------------------------------

def test_source_guard_no_bare_device_equality():
    """AST 守卫：probe_gpu_construction 函数体内不得有裸 `t.device == dev`。

    修复的正确形式是 _same_device helper（type + index 规范化）。
    若有人手改回裸比较，CPU 上的场景测试仍可能过（取决于 mock 的
    device），故加源码级守卫双保险。
    """
    src = _ENV_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # a) helper 必须存在
    helpers = {
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_same_device"
    }
    assert "_same_device" in helpers, "_same_device helper 被删除"

    # b) probe 函数体内不得有 `<attr>.device == <name dev>` 形状的比较
    probe = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "probe_gpu_construction"
    )
    for node in ast.walk(probe):
        if not isinstance(node, ast.Compare):
            continue
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            continue
        left, right = node.left, node.comparators[0]
        left_is_device_attr = (
            isinstance(left, ast.Attribute) and left.attr == "device"
        )
        right_is_dev_name = isinstance(right, ast.Name) and right.id == "dev"
        assert not (left_is_device_attr and right_is_dev_name), (
            "probe_gpu_construction 内发现裸 `X.device == dev` 比较"
            f"（第 {node.lineno} 行）——这正是 2026-10-16 误判的根，"
            "请改用 _same_device"
        )

    # c) probe 内必须出现 _same_device 调用
    calls = {
        n.func.id for n in ast.walk(probe)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_same_device" in calls, "probe_gpu_construction 未调用 _same_device"
