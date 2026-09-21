"""2026-09-20 对抗性审查发现的回归锚点。

与 `test_selfcheck_2026_09_18.py` 同一体例：每条对应一个**已修**的缺陷，
且每条都先证明「它本可以不被发现」。逐条：

  A1-A4 prefill 量具与口径。**2026-09-20 版**锁的是「计时窗口里只有全长前向、
         裁剪后的 KV 从未被使用 ⇒ 加速比结构上恒 <= 1」；**2026-09-21 接上 H0
         后改写**（目的端前向进入窗口，它看到的 KV 由 S 变 B）。同组锚点现锁：
         目的端看到的 KV 必须等于预算 B、两臂唯一差别就是那个长度、漏给
         dest_len 必须抛错、加速比的分子必须是 dense。量具失效时判定程序
         必须**拒判**（unresolved / instrument）而不是报「未达标」。
  B1-B2 异步臂的 comm_ms 不是通信时间，却作为普通字段落盘
         ⇒ 会被读成「异步消掉了通信」。
  C1-C3 重复次数（warmup/iters）不进产物，论文 §6.4 的「每点 >=10 次 run」
         无从校验，且 `--iters 1` 的冒烟结果与合规结果在 JSON 上不可区分；
         另有元数据里 λ_β 默认值与项目默认不同源。
  D1-D2 设备索引锚点只拦一种写法，而 `tests/gpu/` 里同族写法一直活着。
  E1-E2 `assert n["syncs"] >= 4` 是单侧的，「多同步一次」这种把 overlap
         抹平的回归恰好落在余量里。
  F     E8 产物完全没有运行元数据（E5/E6/E7 都有）。
  G     save_json 写出裸 `NaN`，不是合法 JSON。
  I     写进源码/文档的测试路径必须真实存在 —— 本仓库**两次**指向从未存在
         的测试文件（`_hf.py` 的 H0 说明、`e6_main_table.py` 的 H2 配对说明），
         读者会据此认为覆盖已经有了。
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.common import report as R  # noqa: E402
from experiments.gpu import _comm, _env, _forward, _hf  # noqa: E402


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# 声明为「必须默认运行」的锚点。它们审的是 gpu 相关代码，但自己**不需要**
# GPU —— conftest 按名字自动打 marker 时极易把它们误标成 gpu 而默认排除。
MUST_RUN_BY_DEFAULT = (
    "test_d1_no_global_rank_as_device_index",
    "test_d2_device_binding_goes_through_the_shared_helper",
    "test_d3_local_rank_of_semantics",
)

# 必须**保持**被排除的（真需要 GPU）
MUST_STAY_GPU_ONLY = (
    "test_nccl_2proc_all_reduce",
    "test_async_vs_sync_overlap",
)


def _collect(extra):
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", *extra,
                        "--collect-only", "-q"],
                       cwd=str(REPO), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout + r.stderr


# =============================================================================
# A. prefill 量具：计时窗口里到底有没有用裁剪后的 KV
# =============================================================================

class _StubOut:
    def __init__(self, cache):
        self.past_key_values = cache


class _StubModel:
    """记录每次前向看到的 (输入长度, 传入的 past 长度)。"""

    def __init__(self):
        self.calls: list = []

    def __call__(self, input_ids=None, past_key_values=None, use_cache=False,
                 **kw):
        lin = int(input_ids.shape[1])
        past = None
        if past_key_values is not None:
            past = int(past_key_values[0][0].shape[-2])
        self.calls.append((lin, past))
        total = lin + (past or 0)
        cache = [(torch.zeros(1, 4, total, 8), torch.zeros(1, 4, total, 8))
                 for _ in range(2)]
        return _StubOut(cache)


class _StubLM:
    def __init__(self):
        self.model = _StubModel()
        self.device = torch.device("cpu")
        self.precision = "bfloat16"
        self.num_layers = 2
        self.num_kv_heads = 4
        self.head_dim = 8


_REAL_APPLY_KV_BUDGET = _hf.apply_kv_budget
"""真身必须在**导入期**抓一次。

踩过的坑：若在包裹函数里写 `orig = _hf.apply_kv_budget`，那么同一个测试里
第二次安装 spy 时，`orig` 会捕获到**第一个 spy**，两次计数串成嵌套链 ——
第一段（精确路径）会凭空多出第二段的调用次数。这正是「桩把被测行为吃掉」
的反面：桩把**不该有的调用**算进来了。
"""


def _run_prefill(monkeypatch, budget_ratio, seq_len=512, dest_len=64):
    """跑一次 measure_prefill，返回 (桩模型, 压缩被调用的次数)。

    计数用包裹真实实现的 spy（而不是 stub 成 no-op）：这样「压缩到底做没做」
    有真实副作用可查，不会因为桩把行为抹掉而变成永真断言。

    `dest_len` 从 2026-09-21 起是必给的：压缩臂不给会被 `measure_prefill`
    直接拒跑（见 A3）。
    """
    lm = _StubLM()
    calls = {"n": 0}

    def counting(cache, budget, mode="topk_rms", seed=42):
        calls["n"] += 1
        return _REAL_APPLY_KV_BUDGET(cache, budget, mode, seed)

    monkeypatch.setattr(_hf, "apply_kv_budget", counting)
    _hf.measure_prefill(lm, seq_len=seq_len, batch_size=1, warmup=1, iters=3,
                        budget_ratio=budget_ratio, compaction_mode="topk_rms",
                        dest_len=dest_len)
    return lm, calls


def test_a1_prefill_timing_consumes_the_compacted_cache(monkeypatch):
    """计时窗口里必须出现「目的端前向」，且它看到的 KV 是**裁剪后**的。

    本条由 2026-09-20 版 `test_a1_prefill_timing_never_uses_the_compacted_cache`
    **改写**而来（H0 接上后原断言被推翻）。它防的东西没有变 —— 仍是「量具失敏」，
    只是判据从「有没有前向拿到 past」升级为「拿到 past 的那次，past 有多长」：
    必须等于预算 B，而不是全长 S。只查 past 是否为 None 已经不够了。

    事实基础见 `_hf.PREFILL_TIMING_CONSUMES_COMPACT_KV` 的说明。
    """
    seq_len, dest_len = 512, 64
    B = max(1, int(round(0.05 * seq_len)))
    lm, calls = _run_prefill(monkeypatch, budget_ratio=0.05,
                             seq_len=seq_len, dest_len=dest_len)

    src = [(lin, past) for lin, past in lm.model.calls if lin == seq_len]
    dst = [(lin, past) for lin, past in lm.model.calls if lin == dest_len]
    assert len(src) == 4, "源端构造：warmup(1) + iters(3)"
    assert len(dst) == 4, "目的端前向：每轮一次（H0 的载体）"
    assert all(past is None for _, past in src), \
        "源端构造不该带 past —— 它是第一段，没有更早的 KV"
    assert {past for _, past in dst} == {B}, (
        f"目的端看到的 KV 长度必须等于预算 B={B}，实测 "
        f"{sorted({p for _, p in dst})}；若等于全长 S={seq_len}，"
        "说明裁剪结果又没进计时窗口（H0 回退）"
    )
    assert calls["n"] >= 1, "压缩确实被施加了"

    # 契约常量必须与观察到的行为一致（防「改常量不改实现」）
    assert _hf.PREFILL_TIMING_CONSUMES_COMPACT_KV is True


def test_a1b_compaction_shortens_what_the_destination_attends_to(monkeypatch):
    """两臂的**唯一**差别必须是目的端看到的 KV 长度：S vs B。

    这是「prefill_speedup > 1 从哪来」的机器可读形式：源端构造在两臂上
    长度相同、次数相同，故在比值里是公共项；收益只可能来自目的端那一步。
    本条由 2026-09-20 版的「压缩路径是精确路径的严格超集」改写而来 ——
    那个论断在 H0 接上后已不成立（压缩臂的 attention 现在**更短**）。
    """
    seq_len, dest_len = 512, 64
    B = max(1, int(round(0.05 * seq_len)))
    lm_d, calls_d = _run_prefill(monkeypatch, budget_ratio=None,
                                 seq_len=seq_len, dest_len=dest_len)
    lm_c, calls_c = _run_prefill(monkeypatch, budget_ratio=0.05,
                                 seq_len=seq_len, dest_len=dest_len)

    # 源端构造：两臂逐位一致（公共项）
    assert ([c for c in lm_d.model.calls if c[0] == seq_len]
            == [c for c in lm_c.model.calls if c[0] == seq_len])
    # 目的端：输入长度相同，可见的 KV 长度不同
    assert ([c[0] for c in lm_d.model.calls if c[0] == dest_len]
            == [c[0] for c in lm_c.model.calls if c[0] == dest_len])
    assert {c[1] for c in lm_d.model.calls if c[0] == dest_len} == {seq_len}
    assert {c[1] for c in lm_c.model.calls if c[0] == dest_len} == {B}
    assert B < seq_len, "预算必须真的小于全长，否则这条锚点没有判别力"
    # 压缩只发生在压缩臂
    assert calls_d["n"] == 0, "精确路径不该做压缩"
    assert calls_c["n"] >= 1, "压缩路径多做了压缩这一步"


def _e6_ns(**kw):
    import argparse
    base = dict(models=["m"], context_lengths=[4096],
                sync_modes=["sync", "async"], h2_delta_pp=None,
                h2_noise_floor_pp=None, h2_delta_bad_pp=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _h2_rows(prefill_shared=130.0, prefill_dcc=100.0, prefill_dense=120.0):
    """H2 配对用的一批行。

    `dense` 与 `shared` 的取值**刻意不同**（120 vs 130）：这样 A4 才能判别
    加速比的分子到底取了谁 —— 若两者相同，取错分子也测不出来。
    """
    from experiments.gpu import e6_main_table as E
    return [
        {"method": "dcc_kv", "model": "m", "context_length": 4096,
         "sync_async": "sync", "accuracy": 0.70,
         "prefill_ms_median": prefill_dcc},
        {"method": "kv_budget_shared", "model": "m", "context_length": 4096,
         "sync_async": E.SYNC_MODE_NA, "accuracy": 0.60,
         "prefill_ms_median": prefill_shared},
        {"method": "dense", "model": "m", "context_length": 4096,
         "sync_async": E.SYNC_MODE_NA, "accuracy": 0.80,
         "prefill_ms_median": prefill_dense},
    ]


def test_a2_h2_separates_instrument_trouble_from_resolution(monkeypatch):
    """量具有效 ⇒ 未定原因是 `resolution`；量具失效 ⇒ `instrument`。

    两个原因必须能区分：它们的处置完全不同（前者去测噪声底线，后者去修量具），
    而且 `instrument` 绝不能落成「未达标」。H0 接上后默认走前者；后者由本用例
    显式把契约翻回假来验证 —— 这样「钩子接好没接好」在产物里始终看得出来。
    """
    from experiments.gpu import e6_main_table as E
    assert _hf.PREFILL_TIMING_CONSUMES_COMPACT_KV is True

    pts = E.h2_points_from_rows(_e6_ns(), _h2_rows())
    assert len(pts) == 1
    assert pts[0].prefill_instrument_valid is True

    out = E.compute_h2(_e6_ns(), _h2_rows())
    assert out["h2_unresolved_reasons"] == {"4096": "resolution"}
    assert out["h2_n_failed"] == 0, "前提未定不等于未达标"
    assert out["h2_prefill_instrument_valid"] is True

    # 反向：量具契约被翻回假时，同一批行必须改判为 instrument
    monkeypatch.setattr(_hf, "PREFILL_TIMING_CONSUMES_COMPACT_KV", False)
    out2 = E.compute_h2(_e6_ns(), _h2_rows())
    assert out2["h2_unresolved_reasons"] == {"4096": "instrument"}
    assert out2["h2_n_failed"] == 0, "量具无效不等于未达标"
    assert out2["h2_prefill_instrument_valid"] is False


def test_a3_compressed_arm_refuses_to_run_without_a_destination_forward():
    """压缩臂漏给 `dest_len` 时必须**抛错**，不能静默产出一个恒 <=1 的加速比。

    这条是 **H0 的回退守卫**。若有人把目的端前向去掉（或忘了传 dest_len），
    量具会悄悄重新失敏，而产物看起来一切正常 —— 这正是 D1 当初的形态。
    把「失效」变成「响亮失败」是唯一的防法。

    精确臂（budget_ratio=None）不受此限：它本来就不压缩，没有收益可丢。
    """
    lm = _StubLM()
    with pytest.raises(ValueError, match="dest_len"):
        _hf.measure_prefill(lm, seq_len=512, batch_size=1, warmup=0, iters=1,
                            budget_ratio=0.05, compaction_mode="topk_rms")

    lm2 = _StubLM()
    _hf.measure_prefill(lm2, seq_len=512, batch_size=1, warmup=0, iters=1,
                        budget_ratio=None)
    assert lm2.model.calls, "精确臂应能照常跑"


def test_a4_speedup_is_derived_dense_over_dcc_not_the_other_way():
    """方向锚点：`prefill_speedup = prefill_dense / prefill_dcc`。

    **2026-09-21 修正**：本用例原先断言 `prefill_shared / prefill_dcc`，
    与它自己的名字（dense_over_dcc）不符 —— 是上一轮留下的命名/口径错配。
    论文 §7.5 把 H2 拆成三段（与精确注意力**不劣** / 相对共享压缩**更高** /
    prefill **更快**），第三段的对照物是精确注意力；而 `shared` 同样压过 KV、
    同样交付 B 长的 KV，拿它做分子得到的比值结构上恒 ≈1，配 1.10x 的阈值
    不自洽。故改判据、并让本用例真正按名字去测。

    反了的话，压缩越慢反而报出越高的加速比 —— 纯粹的符号错，不会报错。
    `_h2_rows` 里 dense(120) 与 shared(130) 取值不同，因此若将来有人把分子
    又换回 shared，这里会立刻红（1.3 != 1.2），而不是静静通过。
    """
    from experiments.gpu import e6_main_table as E
    slow = E.h2_points_from_rows(
        _e6_ns(), _h2_rows(prefill_dense=120.0, prefill_dcc=100.0))
    fast = E.h2_points_from_rows(
        _e6_ns(), _h2_rows(prefill_dense=100.0, prefill_dcc=120.0))
    assert slow[0].prefill_speedup == pytest.approx(1.2)
    assert fast[0].prefill_speedup == pytest.approx(100.0 / 120.0)


# =============================================================================
# B. 异步臂的计时拆解
# =============================================================================

def _fake_result(mode: str) -> _forward.ForwardResult:
    return _forward.ForwardResult(
        out=torch.zeros(1, 2), mode=mode, n_chunks_requested=4,
        chunks_effective=4, n_partials=2, n_edges=1, local_included=True,
        partial_order="test",
        timing=_comm.PipelineTiming(total_ms=10.0, comm_ms=7.5, comp_ms=2.5),
    )


def test_b1_async_row_does_not_offer_a_communication_time():
    """异步行的 `t_comm_ms` 必须是 None —— 0 会被读成「通信为零」。"""
    d = _fake_result("async").to_dict()
    assert d["t_comm_ms"] is None
    assert d["t_comm_ms_raw"] == pytest.approx(7.5), "原值不能丢"
    assert d["timing_decomposition_valid"] is False
    assert "不是通信时间" in d["t_comm_ms_note"]
    assert "上界" in d["t_comp_ms_note"]
    assert d["t_total_ms"] == pytest.approx(10.0)


def test_b2_sync_row_keeps_the_decomposition_usable():
    d = _fake_result("sync").to_dict()
    assert d["t_comm_ms"] == pytest.approx(7.5)
    assert d["timing_decomposition_valid"] is True
    assert d["t_comp_ms_note"] == ""


def test_b3_a4_async_cell_carries_nan_and_a_validity_flag():
    """A4 的二维格是 A2 vs A5 能否分开引用的依据，格子里的数必须能自证口径。"""
    from experiments.common import hypotheses as H
    ok = H.A4Cell(budget_ratio=0.05, budget=64, sync_mode="sync",
                  t_build_ms=1.0, t_comm_ms=8.0, t_comp_ms=2.0,
                  t_total_ms=11.0, p50_ms=11.0,
                  timing_decomposition_valid=True, t_comm_ms_raw=8.0)
    bad = H.A4Cell(budget_ratio=0.05, budget=64, sync_mode="async",
                   t_build_ms=1.0, t_comm_ms=float("nan"), t_comp_ms=9.0,
                   t_total_ms=11.0, p50_ms=11.0,
                   timing_decomposition_valid=False, t_comm_ms_raw=0.4,
                   timing_decomposition_note="异步臂：不是通信时间")
    d = bad.to_dict()
    assert d["t_comm_ms"] != d["t_comm_ms"], "必须是 NaN（自反为假）"
    assert d["timing_decomposition_valid"] is False
    assert d["t_comm_ms_raw"] == pytest.approx(0.4)
    assert ok.to_dict()["timing_decomposition_valid"] is True
    assert "timing_decomposition_valid" in ok.to_dict()


# =============================================================================
# C. 重复次数与默认值同源
# =============================================================================

def test_c1_metadata_lambda_beta_shares_the_project_default():
    """元数据里的 λ_β 默认值必须与项目默认同源（E11 定档 3e-2）。"""
    from src.experiment_metadata import ExperimentMetadata
    from src.dcc_kv_ref.calibration import DEFAULT_LAMBDA_BETA
    assert ExperimentMetadata(run_id="t").lambda_beta == DEFAULT_LAMBDA_BETA
    assert DEFAULT_LAMBDA_BETA == pytest.approx(3e-2)


def test_c2_metadata_records_repetition_counts():
    meta = _env.build_metadata(run_id="t", warmup=1, iters=2)
    d = meta.to_dict()
    assert d["warmup"] == 1 and d["iters"] == 2


def test_c3_e6_payload_records_repetitions_and_the_instrument():
    """源级锚点：产物里必须有 repetitions 与 instrument 两块。

    实测（2026-09-20）此前只有 stdout 打印 iters<10 的警告，产物里没有任何
    地方能区分 `--iters 1` 的冒烟结果与合规结果 —— 而论文 §6.4 声称
    「每点 >= 10 次 run」。
    """
    src = _read("experiments/gpu/e6_main_table.py")
    assert '"repetitions"' in src and '"below_norm"' in src
    assert '"norm_required_iters"' in src
    assert '"instrument"' in src
    assert '"prefill_timing_consumes_compact_kv"' in src
    # 元数据调用必须把实际重复次数带上（否则等于没记）
    assert "warmup=a.warmup, iters=a.iters," in src


def test_c4_e5_and_e7_also_pass_repetition_counts():
    for rel in ("experiments/gpu/e5_gpu_ablation.py",
                "experiments/gpu/e7_negative_results.py"):
        assert 'warmup=a["warmup"], iters=a["iters"],' in _read(rel) or \
            "warmup=a.warmup, iters=a.iters," in _read(rel), rel


# =============================================================================
# D. 设备索引：用 AST 锚点，注释免疫
# =============================================================================

def test_h_1_guards_are_selected_by_default():
    """锚点必须真的跑起来。

    「测试全绿」与「守卫在跑」是两件事：一条被 marker 排除的用例会让全量测试
    依然全绿，却什么也没守。这条元守卫把「静默排除」变成会失败的断言。
    """
    out = _collect([])
    for name in MUST_RUN_BY_DEFAULT:
        assert name in out, f"{name} 没有被默认收集到 ⇒ 守卫是空的"
    for name in MUST_STAY_GPU_ONLY:
        assert name not in out, f"{name} 不该在无 GPU 的默认运行里出现"


def test_h_2_accelerator_only_cases_keep_their_marker():
    """反向确认：需要 GPU 的仍在 `-m gpu` 集合里，不需要的不在。"""
    gpu_out = _collect(["-m", "gpu"])
    for name in MUST_RUN_BY_DEFAULT:
        assert name not in gpu_out, f"{name} 被误标为 gpu 用例"
    for name in MUST_STAY_GPU_ONLY:
        assert name in gpu_out, f"{name} 丢掉了 gpu marker"


def test_h_3_every_case_in_this_file_runs_by_default():
    """本文件里的每一个 test_ 函数都必须被默认收集到。

    这一条是**自覆盖**的：以后往本文件加锚点时，若名字里带上了 gpu / nccl /
    end_to_end 之类的关键词而被 conftest 自动排除，这里会直接失败 ——
    不必记得那条坑。名字列表由 AST 从源码里读，不靠人工维护。
    """
    src = _read("tests/test_adversarial_2026_09_20.py")
    defs = [n.name for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    assert len(defs) >= 20, f"只找到 {len(defs)} 个用例，解析可能出错"

    out = _collect([])
    missing = [d for d in defs if d not in out]
    assert missing == [], (
        "本文件里有用例被默认排除了（conftest 按名字打了 gpu marker）：\n  "
        + "\n  ".join(missing))


def _device_violations(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "set_device" \
                    and node.args and isinstance(node.args[0], ast.Name) \
                    and node.args[0].id == "rank":
                bad.append(f"{path}: set_device(rank) 第 {node.lineno} 行")
        if isinstance(node, ast.JoinedStr):
            has_cuda = any(isinstance(v, ast.Constant)
                           and isinstance(v.value, str) and "cuda" in v.value
                           for v in node.values)
            uses_rank = any(isinstance(v, ast.FormattedValue)
                            and isinstance(v.value, ast.Name)
                            and v.value.id == "rank" for v in node.values)
            if has_cuda and uses_rank:
                bad.append(f'{path}: f"cuda:{{rank}}" 第 {node.lineno} 行')
    return bad


@pytest.mark.parametrize("sub", ["experiments", "tests_gpu_tree"])
# 参数 id 刻意不写 "gpu"：conftest 按名字自动打 marker，写进去就可能被
# 默认排除（见 conftest 的说明）。这一条本身就是那个坑的活样本。
def test_d1_no_global_rank_as_device_index(sub):
    """多节点下全局 rank 8 在第 2 个 8 卡节点上是 cuda:0。

    用 AST 而不是字符串匹配：字符串锚点拦不住 `set_device(rank)`、
    `f'cuda:{rank}'`（单引号）、`device_map=f"cuda:{rank}"` 等等价写法 ——
    实测原锚点只拦得住 1/6 种。AST 也不受注释里出现这个模式的影响。
    """
    root = REPO / ("experiments/gpu" if sub == "experiments" else "tests/gpu")
    files = sorted(p for p in root.rglob("*.py"))
    assert files, f"{root} 下没有 .py，锚点是空的"
    bad = []
    for p in files:
        bad.extend(_device_violations(p))
    assert bad == [], "设备索引必须走 _env.local_rank_of：\n" + "\n".join(bad)


def test_d2_device_binding_goes_through_the_shared_helper():
    """修了调用点还不够：必须走同一个 helper，否则下次又会各写一遍。"""
    for rel in ("tests/gpu/test_nccl_basic.py",
                "tests/gpu/test_async_overlap.py",
                "tests/gpu/test_end_to_end_8b.py",
                "tests/gpu/profiling/torch_profiler_runner.py"):
        src = _read(rel)
        assert "local_rank_of" in src, rel
        assert "_local_rank(" in src, rel


def test_d3_local_rank_of_semantics(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert _env.local_rank_of(8) == 3
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert _env.local_rank_of(8) == 8


# =============================================================================
# E. 同步计数：双侧且由机制推导
# =============================================================================

def _device_sync_callsites(fn_name: str) -> int:
    """数 `_comm` 里某函数体内部的 `_env.device_sync()` 静态调用点。

    这给出运行期计数的**结构性上界**：调用点写死了几个，跑起来就不可能多。
    """
    tree = ast.parse(_read("experiments/gpu/_comm.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            n = 0
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                        and sub.func.attr == "device_sync":
                    n += 1
            return n
    raise AssertionError(f"没有找到函数 {fn_name}")


def test_e1_async_pipeline_has_exactly_two_sync_callsites():
    """异步流水只允许两处 device_sync：循环内每个 comp 段一次、末轮一次。

    多一处（尤其 `wait()` 之后那处）就会把正在飞的下一块等掉 ⇒ 加速比恒 1.0。
    这条是运行期精确计数断言（见 test_gpu_pipeline.py）的结构性依据。
    """
    assert _device_sync_callsites("run_async_pipeline") == 2
    assert _device_sync_callsites("run_sync_pipeline") == 2


def test_e2_the_old_one_sided_assertion_is_gone():
    """`>= 4` 的余量恒为 2，抓不到「多同步一次」。"""
    src = _read("tests/test_gpu_pipeline.py")
    assert 'n["syncs"] >= 4' not in src
    assert "n_sync == 2 * eff" in src and "n_async == eff" in src


# =============================================================================
# F. E8 的元数据
# =============================================================================

def test_f1_e8_payload_carries_run_metadata():
    """E5/E6/E7 都写了 metadata，只有 E8 没有 —— 而低精度实验最需要它。"""
    src = _read("experiments/gpu/e8_low_precision.py")
    assert '"metadata"' in src
    assert "_env.build_metadata(" in src
    assert '"low-precision-numerics"' in src


# =============================================================================
# G. 产物必须是合法 JSON
# =============================================================================

def test_g1_save_json_emits_standard_json(tmp_path):
    """裸 `NaN` / `Infinity` 不是合法 JSON（RFC 8259），严格解析器会报错。"""
    p = tmp_path / "x.json"
    big = {"a": float("nan"), "b": [1.0, float("inf"), float("-inf")],
           "c": {"d": float("nan")}}
    R.save_json(str(p), big)
    text = p.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    # 严格解析（拒绝 NaN/Infinity 字面量）
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError(f"非标准 JSON 常量 {c}")))
    assert json.loads(text) == {"a": None, "b": [1.0, None, None],
                                "c": {"d": None}}


def test_g2_json_safe_does_not_touch_finite_values():
    """只换非有限浮点，其余逐位不动 —— 禁止顺手「整理」数据。"""
    vals = [0.0, -0.0, 1e-300, 1e300, 0.1, -2.5, 3.0]
    got = R.json_safe({"v": vals})
    assert got["v"] == vals
    assert [repr(x) for x in got["v"]] == [repr(x) for x in vals]
    assert R.json_safe("s") == "s"
    assert R.json_safe(7) == 7
    assert R.json_safe(None) is None


# =============================================================================
# I. 引用完整性：写进源码/文档的测试路径必须真实存在
# =============================================================================

# 只认带 tests/ 前缀的路径。刻意不认裸文件名 —— 同目录互指是允许的，
# 本文件开头就那样引用了 test_selfcheck_2026_09_18.py。
# 用 [.] 而不是反斜杠转义，少一层转义就少一个静默出错的机会。
_REF_TEST_PATH = re.compile(r'tests/(?:gpu/|profiling/)?test_[A-Za-z0-9_]+[.]py')

# 扫描面 = 源码 + docs。**不含 .github/** —— 那里的 PR 模板拿一个虚构的
# 测试文件名当占位符示例，它是示例不是引用，收进来会造成永久误报。
_REF_SCAN_GLOBS = ('experiments/**/*.py', 'src/**/*.py', 'tests/**/*.py',
                   'docs/*.md')


def test_i1_every_referenced_test_file_exists():
    """源码/文档里写下的测试路径必须真实存在。

    实证依据：本仓库**两次**出现「引用了从未存在的测试文件」——
    `experiments/gpu/_hf.py` 的 H0 契约说明指向一个不存在的 H0 锚点文件，
    `experiments/gpu/e6_main_table.py` 的 H2 配对说明指向一个不存在的配对锚点
    文件（两者都见第 31 条；本 docstring 刻意不写全路径，否则这条守卫会绊住
    自己 —— 这正是它要防的那类错）。

    危害不是笔误那么轻：读者据此会认为覆盖已经有了，于是不去补。
    **「声称有测试」比「没写测试」更坏** —— 与 D11（守卫被静默排除、全量照样
    全绿）同源：都是「缺失不报错」。
    """
    referenced = {}
    for pattern in _REF_SCAN_GLOBS:
        for path in sorted(REPO.glob(pattern)):
            text = path.read_text(encoding='utf-8', errors='replace')
            for m in _REF_TEST_PATH.finditer(text):
                referenced.setdefault(m.group(0), []).append(
                    path.relative_to(REPO).as_posix())

    assert referenced, '扫描面没命中任何引用 —— 正则或 glob 写错了'

    missing = {rel: sorted(set(where))
               for rel, where in referenced.items()
               if not (REPO / rel).exists()}
    assert not missing, (
        '引用了不存在的测试文件（会被读成「已有覆盖」）：' + repr(missing))

