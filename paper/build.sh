#!/usr/bin/env bash
# =============================================================================
# DCC-KV 论文一键构建
#
# 用途：先生成 4 张矢量图，再编译正文并处理参考文献。
# 依赖：TeX Live（xelatex、bibtex）。含中文，必须用 xelatex（不能用 pdflatex）。
#
# 用法：
#   bash build.sh            # 完整构建（图表 + 正文）
#   bash build.sh figures    # 只生成图表
#   bash build.sh clean      # 清理中间文件
#   bash build.sh distclean  # 清理中间文件 + 生成物（main.pdf 与 figures/*.pdf）
#
# 字体：图表与正文默认使用 SimSun / SimHei / Microsoft YaHei。
#       在 Linux/macOS 上请修改 main.tex 与 figures/preamble.tex 中的字体名。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")" || exit 1

FIG_DIR="figures"
FIGS=(fig1_architecture fig2_async_pipeline fig3_error_decomposition fig4_comm_scaling fig5_lambda_curve)

build_figures() {
    echo "==> 生成矢量图（$FIG_DIR/）"
    for f in "${FIGS[@]}"; do
        printf '    %-28s' "$f"
        if (cd "$FIG_DIR" && xelatex -interaction=nonstopmode "$f.tex" >/dev/null 2>&1); then
            echo "OK"
        else
            echo "FAIL"
            echo "    详见 $FIG_DIR/$f.log" >&2
            return 1
        fi
    done
}

build_pdf() {
    echo "==> 编译正文"
    xelatex -interaction=nonstopmode main.tex >/dev/null
    bibtex main >/dev/null 2>&1 || echo "    (bibtex 警告：检查是否有未解析引用)"
    xelatex -interaction=nonstopmode main.tex >/dev/null
    xelatex -interaction=nonstopmode main.tex >/dev/null
    echo "    -> main.pdf"
    if grep -q "Warning: Citation.*undefined" main.log 2>/dev/null; then
        echo "    ⚠️  存在未解析的引用，请检查 refs.bib" >&2
    fi
}

clean() {
    echo "==> 清理中间文件"
    rm -f ./*.aux ./*.bbl ./*.blg ./*.log ./*.out ./*.toc ./*.synctex.gz
    rm -f "$FIG_DIR"/*.aux "$FIG_DIR"/*.log "$FIG_DIR"/*.out
}

distclean() {
    clean
    echo "==> 清理生成物"
    rm -f main.pdf
    for f in "${FIGS[@]}"; do rm -f "$FIG_DIR/$f.pdf"; done
}

case "${1:-all}" in
    figures)  build_figures ;;
    pdf)      build_pdf ;;
    clean)    clean ;;
    distclean) distclean ;;
    all)      build_figures && build_pdf ;;
    *)        echo "用法: bash build.sh [all|figures|pdf|clean|distclean]" >&2; exit 1 ;;
esac
