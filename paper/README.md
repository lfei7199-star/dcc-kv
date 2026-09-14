# paper/ — DCC-KV 论文

本目录存放论文正文、参考文献与图表。

## 目录结构

```
paper/
├── main.tex                     全文入口（第 1-4 章 + \input 后续章节）
├── sections/
│   ├── 05-analysis.tex          第 5 章 复杂度与误差分析
│   ├── 06-experiment.tex        第 6 章 实验方案
│   └── 07-discussion.tex        第 7 章 讨论 + 第 8 章 结论 + 可复现性说明
├── refs.bib                     参考文献（17 条）
├── figures/
│   ├── preamble.tex             图表共享前言（中文字体 + TikZ 样式）
│   ├── fig1_architecture.tex    DCC-KV 总体架构
│   ├── fig2_async_pipeline.tex  异步 All-to-Allv 流水时序
│   ├── fig3_error_decomposition.tex  误差来源与误差界
│   ├── fig4_comm_scaling.tex    通信量解析对比
│   └── fig5_lambda_curve.tex    λ_β 的归并误差曲线（数据取自 results/cpu/e11）
├── build.sh                     一键构建脚本
└── fetch_refs.sh                按 refs.bib 拉取 arXiv PDF（可选）
```

## 构建

需要 TeX Live（`xelatex` + `bibtex`）。**含中文，必须用 `xelatex`，不能用 `pdflatex`。**

```bash
bash build.sh              # 完整构建：先生成图表，再编译正文
bash build.sh figures      # 只生成 5 张矢量图
bash build.sh clean        # 清理中间文件
bash build.sh distclean    # 清理中间文件 + 生成物
```

产出 `main.pdf`。图表 PDF 由 `figures/*.tex` 生成，已加入 `.gitignore`
（只提交源码，避免二进制文件入库；`docs/release_checklist.md` 第 5 节要求的
vector PDF 由构建流程保证）。

## 字体

默认使用 Windows 简体中文环境自带字体：`SimSun`（正文）、`SimHei`（粗体）、
`Microsoft YaHei`（sans）。在其他平台构建时，请修改 `main.tex` 与
`figures/preamble.tex` 中的 `\setCJKmainfont` 等设置为系统已安装的 CJK 字体，
例如 `Noto Serif CJK SC`。

## 参考文献

`refs.bib` 中每条目的标题、作者与 arXiv 编号均经原文首页核对。要重新下载
文献 PDF 到本地 `literature/`（该目录不入库）：

```bash
bash fetch_refs.sh
```

## 写作约束

正文写作受以下边界约束，修改时请保持：

- `README.md` 中「不主张的内容」：不得主张任务质量结果、通信性能结果、
  多 GPU 可扩展性、与基线的对比优势。
- §4.3 的紧凑 KV 构造机制（Key 选择 / 质量偏置 β / Value 回归）沿用
  Attention Matching（*Fast KV Compaction via Attention Matching*, MIT CSAIL），
  **不是本文原创贡献**，正文已显式声明。
- §6 的 E3 标注为「仅量级验证」：`tests/test_dist_equivalence.py` 中的
  `test_dcc_kv_lower_error_than_shared` 只断言两种方法的误差各自有界，
  并未断言 DCC-KV 更优，因此不构成 H2 的证据。
- 图 4 为解析结果，图注已标注「非实测数据」。
- 图 5 的坐标取自 `results/cpu/e11/summary.json` 的 180 格中位数（箱约束
  `[-3,3]` 固定打开），头部注释已标明数据来源；改数须同步改图与
  `§6.3.7` 表 `tab:lambda-ladder`。
