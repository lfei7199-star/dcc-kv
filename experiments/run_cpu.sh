#!/usr/bin/env bash
# 运行全部可在纯 CPU 上复现的实验（E0–E4 + A3 机制级）。
#
# 用法：
#   bash experiments/run_cpu.sh            # 完整档（默认参数）
#   bash experiments/run_cpu.sh --quick    # 快速档，用于冒烟
#   bash experiments/run_cpu.sh -o results/cpu/run1
#
# 退出码：任一实验非零即整体非零。

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
QUICK=""
OUT="results/cpu"

while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK="--quick"; shift ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

FAILED=0
run() {
  local name="$1"; shift
  echo
  echo "############################################################"
  echo "# $name"
  echo "############################################################"
  if ! "$PY" "$@"; then
    echo "!!! $name 失败" >&2
    FAILED=1
  fi
}

echo "仓库根目录：$REPO_ROOT"
echo "Python：$PY  ($($PY -c 'import sys;print(sys.version.split()[0])'))"
echo "输出目录：$OUT"
echo
echo "以下实验均不需要 GPU、不需要模型权重、不需要 NCCL。"

run "E0 顺序无关性"          experiments/cpu/e0_order_invariance.py    $QUICK --out "$OUT/e0"
run "E1 构造接口与形状不变量" experiments/cpu/e1_interface_shapes.py   $QUICK --out "$OUT/e1"
run "E2 压缩保真度曲线"       experiments/cpu/e2_fidelity_curve.py      $QUICK --out "$OUT/e2"
run "E3 边级条件化 vs 共享压缩" experiments/cpu/e3_edge_conditioning.py $QUICK --out "$OUT/e3"
run "E4 分布式等价性"         experiments/cpu/e4_dist_equivalence.py    $QUICK --out "$OUT/e4"
run "A3 机制级组件消融"       experiments/cpu/e5a_mechanism_ablation.py $QUICK --out "$OUT/a3"

echo
echo "############################################################"
if [ "$FAILED" -eq 0 ]; then
  echo "# 全部通过。结果在 $OUT/"
else
  echo "# 有实验失败，见上方 !!! 行。" >&2
fi
echo "############################################################"
exit "$FAILED"
