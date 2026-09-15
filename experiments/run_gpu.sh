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
# --print-env / --plan **必须跟在实验名后面**（如 `run_gpu.sh e6 --plan`）。
# 原因：本脚本把 $1 无条件当作实验名（见下方 EXP="$1"），所以
# `run_gpu.sh --plan` 会把 --plan 当成实验名并落到「未知实验」分支，
# 而下方那个"体检/计划短路"循环根本遍历不到它。这是 2026-09-16 审计发现的
# 用法与实现不符之处 —— 已按实现修正文档，未改逻辑。
#
# 各脚本对两个开关的支持并不齐（实现差异，不是笔误）：
#   --print-env : E5 / E6 / E7 / E8 均支持
#   --plan      : 仅 E6 / E7 支持；E5 / E8 没有该开关，会报 unrecognized arguments

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"

usage() {
  sed -n '2,16p' "$0"
  echo
  echo "可用实验：e5  e6  e7  e8  all"
  exit 0
}

if [ $# -eq 0 ]; then usage; fi

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

  if ! command -v torchrun >/dev/null 2>&1; then
    echo "错误：未找到 torchrun。多卡实验需要它（PyTorch ≥1.10 附带）。" >&2
    echo "      也可直接用：torchrun --nproc_per_node=$nproc $file $*" >&2
    return 127
  fi

  torchrun --nproc_per_node="$nproc" --standalone "$file" "$@"
}

FAILED=0
case "$EXP" in
  e5)
    run_one "E5 消融 A1/A2/A3/A5" experiments/gpu/e5_gpu_ablation.py "$NPROC" \
      "${PASSTHRU[@]}" || FAILED=1
    ;;
  e6)
    run_one "E6 主表与可扩展性" experiments/gpu/e6_main_table.py "$NPROC" \
      "${PASSTHRU[@]}" || FAILED=1
    ;;
  e7)
    run_one "E7 负结果与适用边界" experiments/gpu/e7_negative_results.py "$NPROC" \
      "${PASSTHRU[@]}" || FAILED=1
    ;;
  e8)
    run_one "E8 低精度数值稳定性" experiments/gpu/e8_low_precision.py "$NPROC" \
      "${PASSTHRU[@]}" || FAILED=1
    ;;
  all)
    # 按 tests/gpu/README.md 的资源就绪顺序：先最省资源的，再最费的
    run_one "E8 低精度数值稳定性（单卡）" experiments/gpu/e8_low_precision.py 1 \
      "${PASSTHRU[@]}" || FAILED=1
    run_one "E7 负结果与适用边界（单卡+模型）" experiments/gpu/e7_negative_results.py 1 \
      "${PASSTHRU[@]}" || FAILED=1
    run_one "E6 主表（单卡+模型）" experiments/gpu/e6_main_table.py 1 \
      "${PASSTHRU[@]}" || FAILED=1
    run_one "E5 消融 A1/A2/A5（多卡+NCCL）" experiments/gpu/e5_gpu_ablation.py \
      "${NPROC:-4}" "${PASSTHRU[@]}" || FAILED=1
    ;;
  *)
    echo "未知实验：$EXP" >&2
    usage
    ;;
esac

echo
echo "############################################################"
if [ "$FAILED" -eq 0 ]; then
  echo "# 完成。"
else
  echo "# 有实验未通过（注意：退出码 3 = 环境闸门未通过，不是实验失败）。" >&2
fi
echo "############################################################"
exit "$FAILED"
