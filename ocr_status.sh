#!/usr/bin/env bash
# Show OCR progress and estimated time remaining.
#     bash ocr_status.sh
cd "$(dirname "$0")"
OUT=hindi/data/raw/ocr/ocr.jsonl
TOTAL=16371   # pages across all 72 PDFs

if ! pgrep -f run_ocr.py > /dev/null; then
    echo "OCR is NOT running."
    [ -f "$OUT" ] && echo "Pages done: $(wc -l < "$OUT")"
    exit 0
fi

A=$(wc -l < "$OUT" 2>/dev/null || echo 0)
echo "OCR running. Measuring rate over 60s..."
sleep 60
B=$(wc -l < "$OUT")
RATE=$((B - A))

echo "pages done : $B / $TOTAL"
echo "rate       : $RATE pages/min"
if [ "$RATE" -gt 0 ]; then
    echo "remaining  : $(( (TOTAL - B) / RATE )) min"
else
    echo "remaining  : no progress in the last minute"
fi
