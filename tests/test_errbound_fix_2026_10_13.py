"""式(37) 误差界修正的回归锚点（2026-10-13）。

背景：2026-10-12 全面核查发现论文 §5 的误差界（`eq:err-bound`）初版可被
**构造反例**击穿——两块归并，O₁=0、O₂=e₁、M₁=M₂=1（S=2）、δM₁=0.1、
δO₁=6e₁：前置条件 Σ|δM|=0.1 < S/2=1 满足，而 LHS=3.119 > RHS=3.100。

**为什么它本可以不被发现**：E13 的网格搜索给出「5760 个 Query 零反例」，
与论文陈述一致；初版界失效的构型（某块精确输出为零而紧凑输出非零、
且 δM 同向）**合成压缩器不产生**，网格因此从未暴露它。栅格加密在这个
命题上无效——揭示缺口靠的是反例构造。这是「零反例」与「命题成立」之别
的又一实例，也是本文件把反例本身钉成测试的原因。

本轮修正：分子扰动 ΔN = Σ_s(M_s·δO_s + δM_s·Ô_s) 中的第二项（质量误差
作用在紧凑输出上）此前未被计入。修正界为

    LHS ≤ [ε_out + ε_mass·max_s(‖Ô_s‖ + ‖O_s‖)] / Ŝ ≤ 2·[同左] / S

守卫分四组：

  组 1（反例回归）——旧界在反例上必须违反（哨兵：证明反例有杀伤力），
       新界两种形式必须成立。
  组 2（对抗搜索哨兵）——固定种子的随机多块搜索下，新界在 1e-9 相对
       容差外零真反例，而旧界被击穿 > 0 次（证明搜索有击穿能力，
       不是「搜不到」）。
  组 3（实现 ↔ 论文学术对齐）——E13 脚本的 RHS 必须与论文式(37) 同式；
       论文三节必须携带本轮数字且不残留旧轮次排他数字。
  组 4（落盘产物复算）——论文引用的汇总数字必须能从
       results/cpu/e13/e13_results.json 逐位复算。
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAPER_05 = REPO_ROOT / "paper" / "sections" / "05-analysis.tex"
PAPER_06 = REPO_ROOT / "paper" / "sections" / "06-experiment.tex"
PAPER_07 = REPO_ROOT / "paper" / "sections" / "07-discussion.tex"
E13_SCRIPT = REPO_ROOT / "experiments" / "cpu" / "e13_bound_tightness.py"
E13_RESULTS = REPO_ROOT / "results" / "cpu" / "e13" / "e13_results.json"
E13_OLDBOUND = REPO_ROOT / "results" / "cpu" / "e13" / "e13_results_oldbound_20260922.json"


# ===========================================================================
# 界的两种实现（与 _onetime_scripts/_verify_eq37_fix_20261012.py 同式）
# ===========================================================================


def _bound_terms(M, O, dM, dO):
    """给定各块 (M_s, O_s, δM_s, δO_s)，返回 LHS 与两种界的 RHS。"""
    M, O, dM, dO = M.double(), O.double(), dM.double(), dO.double()
    S = M.sum()
    Mh = M + dM
    Sh = Mh.sum()
    Oh = O + dO

    N = (M.unsqueeze(1) * O).sum(0)
    Nh = (Mh.unsqueeze(1) * Oh).sum(0)
    lhs = (Nh / Sh - N / S).norm()

    eps_mass = dM.abs().sum()
    eps_out = (M * (O - Oh).norm(dim=1)).sum()
    o_max = O.norm(dim=1).max()
    oh_max = Oh.norm(dim=1).max()

    rhs_old = eps_out / S + 2.0 * eps_mass * o_max / S
    bracket = eps_out + eps_mass * (oh_max + o_max)
    return lhs, rhs_old, bracket / Sh, 2.0 * bracket / S, eps_mass, S


def _counterexample():
    """击穿初版界的构造反例：O₁=0、O₂=e₁、M₁=M₂=1、δM₁=0.1、δO₁=6e₁。"""
    d = 4
    M = torch.tensor([1.0, 1.0])
    O2 = torch.zeros(d)
    O2[0] = 1.0
    O = torch.stack([torch.zeros(d), O2])
    dO1 = torch.zeros(d)
    dO1[0] = 6.0
    dO = torch.stack([dO1, torch.zeros(d)])
    dM = torch.tensor([0.1, 0.0])
    return M, O, dM, dO


def _random_case(rng: torch.Generator):
    """随机一个多块归并场景，覆盖各类对抗结构。"""
    n_blk = int(torch.randint(1, 9, (1,), generator=rng))
    d = int(torch.randint(1, 17, (1,), generator=rng))
    log_m = torch.randn(n_blk, generator=rng) * 3.0
    M = torch.exp(log_m) + 1e-6
    S = M.sum()
    O = torch.randn(n_blk, d, generator=rng)
    O[torch.rand(n_blk, generator=rng) < 0.15] = 0.0
    O = O * (10.0 ** torch.randn(n_blk, 1, generator=rng))
    dM = torch.randn(n_blk, generator=rng)
    dM = dM / (dM.abs().sum() + 1e-300) * 0.49 * S * torch.rand(
        1, generator=rng).item()
    if torch.rand(1, generator=rng).item() < 0.5:
        dM = -dM.abs()
    scale = 10.0 ** torch.randn(1, generator=rng).item() * 2
    dO = torch.randn(n_blk, d, generator=rng) * scale
    if n_blk > 1 and torch.rand(1, generator=rng).item() < 0.3:
        i = int(torch.randint(0, n_blk, (1,), generator=rng))
        if O[i].norm() > 0:
            dO[i] = O[i] / O[i].norm() * dO[i].norm()
    return M, O, dM, dO


# ===========================================================================
# 组 1：反例回归
# ===========================================================================


def test_g1_old_bound_is_violated_by_the_counterexample():
    """初版界必须在该反例上违反——这是修正的出发点，也是哨兵。

    若这条不过，说明反例本身失效（或有人把界改回了错的形式），
    组 2 的「零反例」将失去对照意义。
    """
    M, O, dM, dO = _counterexample()
    lhs, rhs_old, _, _, eps_mass, S = _bound_terms(M, O, dM, dO)
    assert eps_mass < S / 2, "前置条件必须满足，否则界本无保证"
    assert rhs_old < lhs, (
        f"初版界未违反：LHS={lhs:.6f} RHS_old={rhs_old:.6f}；"
        "反例失效或界已被改动")


def test_g2_corrected_bound_holds_on_the_counterexample():
    """修正界（Ŝ 形式与 /S 形式）必须在同一反例上成立。"""
    M, O, dM, dO = _counterexample()
    lhs, rhs_old, rhs_hat, rhs_s, _, _ = _bound_terms(M, O, dM, dO)
    assert rhs_hat >= lhs
    assert rhs_s >= lhs
    # 修正界比旧界大（修正项补在旧界失控处），但不至于宽松到无信息
    assert rhs_old < rhs_hat
    assert rhs_hat / lhs < 1.05, "修正界在反例上应 barely 成立，不应过度宽松"


def test_g3_dM_zero_recovers_equality_case():
    """δM≡0 时修正界退化为 ε_out/Ŝ = ε_out/S，与旧界的取等情形一致。

    这条锁「修正不丢旧界已正确的部分」：旧界在 δM≡0 时解析恒等，
    修正界必须保持该性质，否则就是拿一个新错误换一个旧错误。
    """
    M, O, dM, dO = _counterexample()
    lhs, rhs_old, rhs_hat, _, _, _ = _bound_terms(M, O, torch.zeros_like(dM), dO)
    assert rhs_old == pytest.approx(rhs_hat, rel=1e-12)


# ===========================================================================
# 组 2：对抗搜索哨兵
# ===========================================================================


@pytest.mark.parametrize("n_trials", [5000])
def test_g4_adversarial_search_cannot_break_corrected_bound(n_trials):
    """固定种子随机搜索：新界在 1e-9 相对容差外零真反例。

    样本量取 5000（全量 20 万次的一次筛）：既保持秒级耗时，
    又足够覆盖随机空间。与全量脚本的口径一致——
    取等号处的浮点噪声按 1e-9 相对容差剔除，超过容差才算真反例。
    """
    rng = torch.Generator().manual_seed(20261013)
    true_viol = 0
    old_viol = 0
    for _ in range(n_trials):
        M, O, dM, dO = _random_case(rng)
        lhs, rhs_old, rhs_hat, rhs_s, eps_mass, S = _bound_terms(M, O, dM, dO)
        if eps_mass >= S / 2 or lhs <= 0:
            continue
        if rhs_old < lhs:
            old_viol += 1
        for rhs in (rhs_hat, rhs_s):
            if rhs < lhs * (1.0 - 1e-9):
                true_viol += 1
    assert true_viol == 0, f"新界被击穿 {true_viol} 次，不能落稿"
    assert old_viol > 0, (
        "旧界在 5000 次搜索中零违反——搜索失去击穿能力，"
        "本测试的「零反例」结论不可信")


# ===========================================================================
# 组 3：实现 ↔ 论文学术对齐
# ===========================================================================


def test_g5_paper_equation_carries_the_corrected_form():
    """论文 §5 的方程必须是修正形式（Ŝ 归一化 + 紧凑输出项）。

    文本锚点：方程含 \\widehat{S}（紧凑侧分母）与 O^{\\ast}（或等价的
    max_s(‖Ô_s‖+‖O_s‖) 展开）。旧界以 S 归一化且无紧凑输出项，
    两个锚点同时缺失即回退。
    """
    text = PAPER_05.read_text(encoding="utf-8")
    eq_start = text.index("\\label{eq:err-bound}")
    eq = text[eq_start:eq_start + 1200]
    assert "\\widehat{S}(q)" in eq, "方程未以紧凑侧分母 Ŝ 归一化（旧界形式）"
    assert ("O^{\\ast}" in eq) or ("\\|\\widehat{O}_s" in eq), (
        "方程未计入紧凑输出范数项（漏 δM_s·Ô_s 通路的旧界形式）")


def test_g6_e13_script_implements_the_corrected_rhs():
    """E13 脚本的 RHS 必须与论文式(37) 同式。

    锚点取 `eps_mass * (o_hat_max + o_max) / S_hat_q`：三项
    （紧凑输出项 + 精确输出项 + Ŝ 归一化）缺一不可。
    """
    src = E13_SCRIPT.read_text(encoding="utf-8")
    assert "eps_mass * (o_hat_max + o_max) / S_hat_q" in src, (
        "E13 的 RHS 不是修正界形式；若这是有意的回退，"
        "必须先有新的对抗验证结论")
    assert "S_hat_q" in src and "o_hat_max" in src


def test_g7_paper_carries_this_round_numbers_only():
    """论文三节必须携带本轮（2026-10-13）E13 数字，且不残留旧轮次排他数字。

    排他数字清单（只可能属于旧跑）：2.10/3.51（9-16 轮）、
    2.08/3.61/199（9-22 轮）、189（修正前试跑）、6.2\\% 与 4/60
    （本轮曾误用 aggregated 中位口径，已改 14.6\\% 与 22/60）。
    """
    text = "\n".join(
        p.read_text(encoding="utf-8") for p in (PAPER_05, PAPER_06, PAPER_07)
    )
    for token in ("2.23", "7.37", "9.56", "14.6\\%", "190"):
        assert token in text, f"论文缺少本轮数字 {token}"
    for token in ("2.10", "3.51", "2.08", "3.61", "6.2\\%", "4/60"):
        assert token not in text, f"论文残留旧轮次排他数字 {token}"
    # 199/189 只在与浮点违反计数相邻的上下文里排他；宽松起见查孤立出现
    for token in ("199 例", "189 例"):
        assert token not in text, f"论文残留旧轮次浮点违反数 {token}"


# ===========================================================================
# 组 4：落盘产物复算
# ===========================================================================


@pytest.fixture(scope="module")
def e13():
    if not E13_RESULTS.exists():
        pytest.skip(f"E13 产物缺失: {E13_RESULTS}")
    return json.loads(E13_RESULTS.read_text(encoding="utf-8"))


def test_g8_paper_tightness_numbers_reproduce_from_disk(e13):
    """紧致度汇总（中位 2.23 / 95 分位 7.37 / 最大 9.56 / 最小 1.00）可复算。"""
    rows = e13["rows"]
    rm = sorted(r["ratio_median"] for r in rows)
    p95 = rm[min(len(rm) - 1, int(0.95 * (len(rm) - 1) + 0.5))]
    assert st.median(rm) == pytest.approx(2.23, abs=5e-3)
    assert p95 == pytest.approx(7.37, abs=5e-3)
    assert max(rm) == pytest.approx(9.56, abs=5e-3)
    assert min(rm) == pytest.approx(1.00, abs=5e-3)


def test_g9_no_true_violation_on_disk_and_float_noise_is_localized(e13):
    """落盘产物：超容差违反为 0；浮点噪声全部位于取等格子内。"""
    rows = e13["rows"]
    assert sum(r["n_violating_strict"] for r in rows) == 0
    n_viol = sum(r["n_violating"] for r in rows)
    assert n_viol == 190
    eq_rows = [r for r in rows if r["budget"] == 256 and r["beta_mode"] == "none"]
    assert len(eq_rows) == 10
    assert sum(r["n_queries"] for r in eq_rows) == 480
    assert sum(r["n_violating"] for r in eq_rows) == 190
    assert sum(r["n_violating"] for r in rows
               if not (r["budget"] == 256 and r["beta_mode"] == "none")) == 0


def test_g10_precondition_stats_reproduce_from_disk(e13):
    """前置条件：β 全量最高 14.6%（无格子超半数）、不加 β 最高 100%（22/60 超半数）。"""
    rows = e13["rows"]
    full = [r for r in rows if r["beta_mode"] == "full"]
    none = [r for r in rows if r["beta_mode"] == "none"]
    assert max(r["precondition_violation_frac"] for r in full) == \
        pytest.approx(0.1458, abs=1e-3)
    assert max(r["precondition_violation_frac"] for r in none) == \
        pytest.approx(1.0, abs=1e-9)
    assert sum(1 for r in full if r["precondition_violation_frac"] > 0.5) == 0
    assert sum(1 for r in none if r["precondition_violation_frac"] > 0.5) == 22


def test_g11_oldbound_archive_is_preserved():
    """旧界产物的留档副本必须存在。

    它承载「旧界为什么错」的证据（对照物）：删了就无法向复核查说明
    修界前的数据形态。按项目删除边界，承载结论的对照物一律保留。
    """
    assert E13_OLDBOUND.exists(), (
        "旧界留档副本缺失；它是「式(37) 为何被修正」的对照证据")
