#!/usr/bin/env bash
# =============================================================================
# 下载 DCC-KV 论文参考文献的 arXiv PDF 到本地 literature/ 目录
#
# 用途：本地离线阅读与核对（尤其用于核对 §4.3 与 Attention Matching 的一致性）
# 说明：literature/ 已加入 .gitignore，PDF 不入库（体量大且受各自许可约束）
# 用法：bash fetch_refs.sh
# =============================================================================

set -u
cd "$(dirname "$0")" || exit 1
OUT="literature"
mkdir -p "$OUT"

# key|arxiv_id|短名（用于文件名，便于与 refs.bib 对应）
REFS="
liu2023ring|2310.01889|RingAttention
brandon2023striped|2311.09431|StripedAttention
li2024distflash|2310.03294|DistFlashAttn
fang2024usp|2405.07719|USP
acharya2024star|2411.17116|StarAttention
huang2025apb|2502.12085|APB
zhang2023h2o|2306.14048|H2O
liu2023scissorhands|2305.17118|Scissorhands
xiao2024streamingllm|2309.17453|StreamingLLM
li2024snapkv|2404.14469|SnapKV
cai2024pyramidkv|2406.02069|PyramidKV
tang2024quest|2406.10774|Quest
zweiger2026attentionmatching|2602.16284|AttentionMatching
borzunov2023petals|2209.01188|Petals
dao2022flashattention|2205.14135|FlashAttention
dao2023flashattention2|2307.08691|FlashAttention2
milakov2018online|1805.02867|OnlineSoftmax
"

OK=0; FAIL=0
: > "$OUT/_download_log.txt"

echo "$REFS" | while IFS='|' read -r key arxid short; do
    [ -z "$key" ] && continue
    fname="${key}_${arxid}_${short}.pdf"
    dest="$OUT/$fname"
    if [ -s "$dest" ]; then
        echo "SKIP  $fname (already present)"
        echo "SKIP|$key|$arxid|$fname" >> "$OUT/_download_log.txt"
        continue
    fi
    url="https://arxiv.org/pdf/${arxid}"
    # -L 跟随重定向；--fail 让 HTTP 错误返回非零码；重试 2 次
    if curl -sSL --fail --retry 2 --retry-delay 2 \
            -A "Mozilla/5.0 (compatible; dcc-kv-ref-fetch/1.0)" \
            -o "$dest" "$url" 2>/dev/null; then
        size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
        # arXiv 有时返回 HTML 错误页，体积很小或非 PDF
        head -c 4 "$dest" | grep -q "%PDF" || { rm -f "$dest"; printf 'BADPDF %s (not a PDF)\n' "$fname"; echo "BADPDF|$key|$arxid|$fname" >> "$OUT/_download_log.txt"; continue; }
        printf 'OK    %-52s %8s bytes\n' "$fname" "$size"
        echo "OK|$key|$arxid|$fname" >> "$OUT/_download_log.txt"
    else
        printf 'FAIL  %-52s (arXiv:%s)\n' "$fname" "$arxid"
        echo "FAIL|$key|$arxid|$fname" >> "$OUT/_download_log.txt"
    fi
    sleep 2   # 尊重 arXiv 速率限制
done

echo ""
echo "=== 结果汇总 ==="
printf 'OK   : %s\n' "$(grep -c '^OK|' "$OUT/_download_log.txt" 2>/dev/null || echo 0)"
printf 'SKIP : %s\n' "$(grep -c '^SKIP|' "$OUT/_download_log.txt" 2>/dev/null || echo 0)"
printf 'FAIL : %s\n' "$(grep -c '^FAIL|' "$OUT/_download_log.txt" 2>/dev/null || echo 0)"
printf 'BAD  : %s\n' "$(grep -c '^BADPDF|' "$OUT/_download_log.txt" 2>/dev/null || echo 0)"
echo ""
echo "--- 失败清单 ---"
grep -E '^(FAIL|BADPDF)\|' "$OUT/_download_log.txt" 2>/dev/null || echo "(无失败)"
exit 0
