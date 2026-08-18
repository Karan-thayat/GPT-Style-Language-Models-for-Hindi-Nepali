#!/usr/bin/env python3
"""Run only the manual scrape sources, writing straight to a JSONL file.

Why this is separate from ``build_corpus.py``: sources there are consumed in
manifest order, so a large downloaded corpus is exhausted before any crawler
starts. Crawling is rate-limited and takes hours, so it has to run *alongside*
the corpus build rather than queued behind it.

Output lands in the language's ``scraped/`` directory, where a ``LocalFileSource``
picks it up on the next corpus build - already labelled ``manual``, which is what
makes it count toward the assignment's 20% manual-collection requirement.

Usage::

    python scripts/scrape_only.py --config nepali/configs/sources.yaml \
        --output nepali/data/raw/scraped/news.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lma.data.sources import build_source, load_manifest
from lma.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Source manifest YAML")
    parser.add_argument("--output", required=True, type=Path, help="Destination JSONL")
    parser.add_argument("--only", default="scrape", help="Source type to run (default: scrape)")
    parser.add_argument("--report-every", type=int, default=100, help="Progress interval in pages")
    return parser.parse_args()


def main() -> int:
    """Crawl every enabled scrape source and append the results to one file."""
    args = parse_args()
    logger = setup_logging(None, name="scrape_only")

    specs = [s for s in load_manifest(args.config) if s.get("type") == args.only]
    logger.info(
        "Running %d %s source(s): %s",
        len(specs), args.only, ", ".join(s["source_id"] for s in specs),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Append rather than truncate: a re-run should add to what earlier runs
    # collected, and the page cache means already-fetched URLs replay for free.
    total_pages = 0
    total_words = 0
    started = time.monotonic()

    with args.output.open("a", encoding="utf-8") as handle:
        for spec in specs:
            source = build_source(spec)
            source_pages = 0
            logger.info("--- %s ---", spec["source_id"])
            try:
                for doc in source.stream(limit=spec.get("max_pages")):
                    handle.write(
                        json.dumps(
                            {
                                "text": doc.text,
                                "source_id": doc.source_id,
                                "collection": doc.collection,
                                "url": doc.url,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    source_pages += 1
                    total_pages += 1
                    total_words += len(doc.text.split())
                    if total_pages % args.report_every == 0:
                        # Flush periodically so progress survives a kill and is
                        # visible to anything watching the file.
                        handle.flush()
                        rate = total_pages / max(1e-9, time.monotonic() - started)
                        logger.info(
                            "  %s pages, %s words, %.2f pages/s",
                            f"{total_pages:,}", f"{total_words:,}", rate,
                        )
            except KeyboardInterrupt:
                logger.warning("Interrupted; %s pages kept so far", f"{total_pages:,}")
                return 130
            except Exception as exc:  # noqa: BLE001 - one bad site must not end the run
                logger.warning("  %s failed: %s: %s", spec["source_id"], type(exc).__name__, exc)
            logger.info("  %s: %s pages", spec["source_id"], f"{source_pages:,}")

    logger.info(
        "Done: %s pages, %s words in %.1f min",
        f"{total_pages:,}", f"{total_words:,}", (time.monotonic() - started) / 60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
