# 诊断产物（diagnostics）

**这里的 JSON 不是实验落盘**（实验落盘在 `results/cpu/`），而是**一次性诊断**的结论摘要。

## 为什么它们会被入库

这些诊断的原始产物原本只在**仓库外**（`_diag_out/`），见 `docs/commit_log.md`
第 406–407 行的「仓库外一次性产物，不入库」声明 —— 这是有意选择（不把一次性脚本塞进仓库）。
代价是：**仓库内的文档引用了这些数字，但 clone 仓库的人无法审计它们**
（2026-09-22 工程审查 **F6**）。

本目录即为此而设：把**被文档引用的结论**（几 KB）随库落盘，让数字有落点。
快照时间：**2026-09-22**。

| 文件 | 来源脚本（仓库外） | 支撑 | 关键结论 |
|---|---|---|---|
| `box_constraint_summary.json`<br>`box_constraint_full.log` | `_onetime_scripts/checks/_diag_box_constraint.py` | 偏离台账 **D1** | 默认 $\lambda_\beta=3\times10^{-2}$ 下箱约束**中位绑定率 = 0**、单格最大 $\lvert\Delta\rvert \approx 0.038\%$；而 $\lambda_\beta=0$ 时绑定率升至 **6.25%** ⇒ 它是「正则缺席时承压的第二道防线」，不是摆设 |
| `fps_start_summary.json`<br>`fps_start_full.log` | `_onetime_scripts/checks/_diag_fps_start.py` | 审查项 **F2** | 180 格对照。`newest` 相对 `random` 的偏移 = 起点噪声**自身波动**的 **0.21–0.28 倍**（mix 0.255 / out 0.208 / abs_mass 0.283）⇒ **结论层面不翻转** |
| `fps_equiv_summary.json` | `_onetime_scripts/checks/_diag_fps_equiv.py` | F2 的管道互校 | A：`random` 模式改动前后**逐位相同**（6/6 例）；B：`newest` 的首个锚点确为最新 Query（`first_is_last_index` 全 true）；C：复杂度不退化 |
| `lam0_summary.json` | `_onetime_scripts/checks/_diag_lambda_zero.py` | `commit_log` 第 34 条 | $\lambda_\beta=0$ 的锚点诊断：绑定率中位 **6.25%**、$\beta$ 标准差中位 **1.20** ⇒ $\lambda_\beta=0$ 会让偏置发散 |

## ⚠️ 读这些数字时的三条边界

1. **它们保的是「结论」，不是「逐格数字」。** `fps_start` 的 `max_abs` 达
   `mix 0.3159` / `out 0.4662` —— 池化中位数会把这些抵消，但表里报的是**逐条件中位数**。
   这正是 F2 最终**选择重跑**、而不是「靠这段论证说服读者」的原因。
2. **`gate2_xcheck_vs_e11` 的 `max_abs_diff = 0` 只在旧落盘（`random` 口径）下成立。**
   2026-09-22 之后 `results/cpu/` 已整体重跑为 `newest` 口径，**该互校闸不再对应当前落盘**；
   它证明的是「当时的落盘确实是 random 口径」这一历史事实。
3. **来源脚本不在仓库内。** 想重跑请先索取 `_onetime_scripts/`；本目录只保证
   **被引用的数字有落点**，不承诺脚本可用性（这正是 F6 的原状）。

## 相关

- 旧落盘（`random` 口径）快照：仓库外 `_archive/results_cpu_random_start_2026-09-22/`
- 判决与出处：`docs/author_kit27_deviation_log.md`（D1）、`docs/commit_log.md` 第 34/36 条
