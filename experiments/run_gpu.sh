#!/usr/bin/env bash
# 运行 GPU 实验（E5–E8）。**必须有真实 GPU。**
#
# 用法：
#   bash experiments/run_gpu.sh e8 --print-env       # 只体检环境，任何机器可跑
#   bash experiments/run_gpu.sh e6 --plan            # 只打印网格/条件清单，任何机器可跑
#   bash experiments/run_gpu.sh e8                   # E8（单卡即可）
#   bash experiments/run_gpu.sh e5 --nproc 4         # E5（多卡，NCCL）
#   bash experiments/run_gpu.sh e6 --model <path> --eval-file <jsonl>
#   bash experiments/run_gpu.sh e7 --model <path> --eval-file <jsonl>
#   bash experiments/run_gpu.sh all                  # 按资源就绪顺序依次跑
#
# E5 需要 ≥2 卡 + NCCL；E8 单卡即可（但必须 GPU，除非显式 --allow-cpu）。
# E6/E7 单卡即可，但需要真实模型权重。
#
# 退出码：0 = 全部通过（或 -h/--help 主动查看用法）；3 = 环境闸门未通过（无可用 GPU 或缺 torch，**不是实验失败**）；
#         2 = 参数层拒绝（**含无参数与未知实验名** —— 用法错误不是成功）；127 = 缺少命令（如 torchrun）；1 = 实验真失败。
# 环境自举：默认用 PATH 里的 python；可用 PYTHON=<解释器> 覆盖（本机默认 python
#           若无 torch，脚本会以退出码 3 报错并给出指引，而不是 import 阶段崩掉）。
# 多卡分支的 torchrun 也自举：PATH 找不到时回落到 $PY 同目录（Windows venv 的
#           Scripts/ 布局，未激活 venv 时不在 PATH 上），见下方探测块。
#
# --print-env / --plan **必须跟在实验名后面**（如 `run_gpu.sh e6 --plan`）。
# 原因：本脚本把 $1 无条件当作实验名（见下方 EXP="$1"），所以
# `run_gpu.sh --plan` 会把 --plan 当成实验名并落到「未知实验」分支，
# 而下方那个"体检/计划短路"循环根本遍历不到它。这是 2026-09-16 审计发现的
# 用法与实现不符之处 —— 已按实现修正文档，未改逻辑。
# （2026-10-14 起该分支 exit 2：旧实现里 usage 恒 exit 0，用法错误在 CI 里
#  看起来像成功。）
# 例外：**-h/--help 放第一个参数也认**（直接出帮助 exit 0，见下方专门分支）——
# 它是帮助请求而非实验模式开关，不该被位置陷阱咬到。
#
# 各脚本对两个开关的支持并不齐（实现差异，不是笔误）：
#   --print-env : E5 / E6 / E7 / E8 均支持
#   --plan      : 仅 E6 / E7 支持；E5 / E8 没有该开关，会报 unrecognized arguments

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"

# --- 环境自举（2026-10-14 补）--------------------------------------------------
# 原先这里直接把 PATH 里的 `python` 当解释器用：本机默认 python 没有 torch 时，
# 内层脚本在 import 阶段就 ModuleNotFoundError，**环境闸门（退出码 3）根本
# 够不着** —— README 承诺的「3 = 环境闸门未通过」在最常用入口上不可达。
# 现在先探测两件事，失败都带明确指引：
#   解释器不存在        -> 127（找不到命令）
#   解释器在但缺 torch  -> 3（环境闸门未通过：GPU 实验的底座不完整）
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "错误：找不到 Python 解释器：$PY" >&2
  echo "      请用 PYTHON=<带 torch 的解释器> 指定环境，例如：" >&2
  echo "      PYTHON=<repo>/dcc-kv/venv/Scripts/python.exe bash experiments/run_gpu.sh ..." >&2
  exit 127
fi
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "错误：解释器 $PY 无法 import torch —— GPU 实验的底座不完整。" >&2
  echo "      请用 PYTHON=<带 torch 的解释器> 指定环境（venv 或 repo-tests 均可）：" >&2
  echo "      PYTHON=<repo>/dcc-kv/venv/Scripts/python.exe bash experiments/run_gpu.sh ..." >&2
  echo "      torch 缺失属于环境闸门未通过，退出码 3（不是实验失败）。" >&2
  exit 3
fi

# --- torchrun 探测（2026-10-14 补）------------------------------------------------
# 多卡分支需要 torchrun。原先只查 PATH：Windows venv 布局里 torchrun 与解释器
# 同在 Scripts/ 下，**未激活 venv 时不在 PATH 上** —— 于是"装好了 PyTorch 却
# 没有激活 venv"会以 127 收场，症状指向"缺命令"而非"没激活"。
# 现在 PATH 找不到时回落到 $PY 同目录（解释器所在目录）。找不到不在这里退出，
# 由 run_one 按 127 收尾（单卡路径根本不需要它）。
TORCHRUN=""
_pybin="$(dirname "$(command -v "$PY")")"
if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN="$(command -v torchrun)"
else
  for _cand in "$_pybin/torchrun" "$_pybin/torchrun.exe"; do
    if [ -x "$_cand" ]; then TORCHRUN="$_cand"; break; fi
  done
fi

usage() {
  # $1 = 退出码（缺省 0）。显式 -h/--help 传 0（用户主动要看帮助）；
  # 无参数、未知实验名传 2（参数层拒绝）。旧实现恒 exit 0：用法错误会被 CI
  # 读成"成功"，2026-10-14 修（见 test_run_gpu_entry_2026_10_14.py）。
  sed -n '2,21p' "$0"
  echo
  echo "可用实验：e5  e6  e7  e8  all"
  exit "${1:-0}"
}

if [ $# -eq 0 ]; then usage 2; fi

# -h/--help 作为**第一个**参数时直接出帮助并 exit 0。旧实现把 $1 无条件当作
# 实验名，于是 `run_gpu.sh -h` 落到「未知实验」分支 exit 2 —— 与「-h 是要看
# 帮助」的普遍预期相反（2026-10-14 修）。
# 注意这与下方 --print-env/--plan 的位置陷阱**不是**一回事：那两个是实验模式
# 开关、位置有意义；-h 永远是帮助请求，与位置无关。
case "${1:-}" in
  -h|--help) usage ;;
esac

EXP="$1"; shift
NPROC=1
PASSTHRU=()

# 注意：下面向 run_one 传透传参数时用的是 "${PASSTHRU[@]}"，**不能**写成
# "${PASSTHRU[@]:-}"。带 `:-` 的形式在数组为空时会展开成**一个空字符串**，
# 于是 `python <脚本> ""` 会让 argparse 直接报
# "unrecognized arguments:" —— 也就是说，任何"不带透传参数"的正常调用
# （例如 `bash experiments/run_gpu.sh e8`）都会在参数解析阶段就失败。

while [ $# -gt 0 ]; do
  case "$1" in
    --nproc) NPROC="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done

# 统一的体检/计划短路：这些模式不需要 GPU，也不该被 torchrun 包起来
for arg in "${PASSTHRU[@]}"; do
  case "$arg" in
    --print-env|--plan)
      echo "[模式] $arg —— 不经 torchrun，直接在单进程下运行"
      NPROC=0
      ;;
  esac
done

run_one() {
  local name="$1" file="$2" nproc="$3"; shift 3
  echo
  echo "############################################################"
  echo "# $name   (${nproc} 进程)"
  echo "############################################################"

  if [ "$nproc" -le 1 ]; then
    "$PY" "$file" "$@"
    return $?
  fi

  if [ -z "$TORCHRUN" ]; then
    echo "错误：未找到 torchrun。多卡实验需要它（PyTorch ≥1.10 附带）。" >&2
    echo "      已试过：PATH，以及 $PY 同目录（Windows venv 的 Scripts/ 布局）。" >&2
    echo "      也可直接用：torchrun --nproc_per_node=$nproc $file $*" >&2
    return 127
  fi

  "$TORCHRUN" --nproc_per_node="$nproc" --standalone "$file" "$@"
}

# 退出码汇总规则（2026-10-14 修）：取所有实验里**最大的非零退出码**。
# 优先级：3（环境闸门未通过）> 2（参数/用法拒绝）> 127（缺命令）> 1（实验失败）。
# 旧实现是 `|| FAILED=1`：内层 gate 的退出码 3 被无条件压成 1，
# README 承诺的「退出码 3 = 环境闸门未通过，不是实验失败」经本入口
# **永远不可达**，自动化/CI 无法区分「没显卡」与「实验失败」。
FINAL=0
track() {  # track <rc>：把单个实验的退出码并入 FINAL（只升不降）
  if [ "$1" -gt "$FINAL" ]; then FINAL="$1"; fi
}
case "$EXP" in
  e5)
    RC=0
    run_one "E5 消融 A1/A2/A3/A5" experiments/gpu/e5_gpu_ablation.py "$NPROC" \
      "${PASSTHRU[@]}" || RC=$?
    track "$RC"
    ;;
  e6)
    RC=0
    run_one "E6 主表与可扩展性" experiments/gpu/e6_main_table.py "$NPROC" \
      "${PASSTHRU[@]}" || RC=$?
    track "$RC"
    ;;
  e7)
    RC=0
    run_one "E7 负结果与适用边界" experiments/gpu/e7_negative_results.py "$NPROC" \
      "${PASSTHRU[@]}" || RC=$?
    track "$RC"
    ;;
  e8)
    RC=0
    run_one "E8 低精度数值稳定性" experiments/gpu/e8_low_precision.py "$NPROC" \
      "${PASSTHRU[@]}" || RC=$?
    track "$RC"
    ;;
  all)
    # 按 tests/gpu/README.md 的资源就绪顺序：先最省资源的，再最费的
    RC=0
    run_one "E8 低精度数值稳定性（单卡）" experiments/gpu/e8_low_precision.py 1 \
      "${PASSTHRU[@]}" || RC=$?
    track "$RC"
    RC=0
    run_one "E7 负结果与适用边界（单卡+模型）" experiments/gpu/e7_negative_results.py 1 \
      "${PASSTHRU[@]}" || RC=$?
    track "$RC"
    RC=0
    run_one "E6 主表（单卡+模型）" experiments/gpu/e6_main_table.py 1 \
      "${PASSTHRU[@]}" || RC=$?
    track "$RC"
    RC=0
    run_one "E5 消融 A1/A2/A5（多卡+NCCL）" experiments/gpu/e5_gpu_ablation.py \
      "${NPROC:-4}" "${PASSTHRU[@]}" || RC=$?
    track "$RC"
    ;;
  *)
    echo "未知实验：$EXP" >&2
    usage 2
    ;;
esac

echo
echo "############################################################"
case "$FINAL" in
  0)
    echo "# 完成。"
    ;;
  3)
    echo "# 环境闸门未通过（退出码 3）：没有可用 GPU / 环境不完整 —— 不是实验失败。" >&2
    ;;
  2)
    echo "# 有实验在参数层被拒绝（退出码 2）：检查命令行参数（如 e6 必须显式给 --dcc-world）。" >&2
    ;;
  127)
    echo "# 缺少命令（退出码 127，例如 torchrun）：参见上方错误信息。" >&2
    ;;
  *)
    echo "# 有实验失败（退出码 $FINAL）。" >&2
    ;;
esac
echo "############################################################"
exit "$FINAL"
