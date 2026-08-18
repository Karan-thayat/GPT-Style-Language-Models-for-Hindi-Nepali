#!/usr/bin/env bash
# Show news-scraping progress for both languages.
#
#     bash scrape_status.sh
#
# Reports pages and words collected so far, and how far that is toward the 52.8M
# words each language needs for the assignment's 20% manual-collection floor.

cd "$(dirname "$0")"

for pair in "hindi:hindi/data/raw/scraped/news_crawled.jsonl" \
            "nepali:nepali/data/raw/scraped/news.jsonl"; do
    LANG="${pair%%:*}"
    FILE="${pair##*:}"

    if pgrep -f "scrape_only.py --config $LANG" > /dev/null; then
        STATUS="running"
    else
        STATUS="stopped"
    fi

    if [ -f "$FILE" ]; then
        PAGES=$(wc -l < "$FILE")
        WORDS=$(.venv/bin/python - "$FILE" <<'PY'
import json, sys
total = 0
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        try:
            total += len(json.loads(line)["text"].split())
        except Exception:
            pass
print(total)
PY
)
        PCT=$(.venv/bin/python -c "print(f'{100*$WORDS/52800000:.1f}')")
        printf "%-7s %-8s %7s pages  %12s words  %5s%% of the 52.8M needed\n" \
            "$LANG" "$STATUS" "$(printf "%'d" "$PAGES")" "$(printf "%'d" "$WORDS")" "$PCT"
    else
        printf "%-7s %-8s no output yet\n" "$LANG" "$STATUS"
    fi
done
