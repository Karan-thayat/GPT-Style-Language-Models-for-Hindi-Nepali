#!/usr/bin/env python3
"""OCR scanned PDFs/books into JSONL for the manual-collection portion of the corpus.

Put PDFs in ``<language>/data/raw/pdf/`` and run this script; the JSONL lands in
``<language>/data/raw/ocr/``, where the ``manual_ocr_*`` source in the manifest
picks it up automatically on the next corpus build.

Usage::

    python scripts/run_ocr.py --language hi
    python scripts/run_ocr.py --language ne --pdf-dir nepali/data/raw/pdf --per-page
    python scripts/run_ocr.py --language ne --check   # dependency check only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lma.data.ocr import OcrConfig, check_dependencies, ingest_directory, install_hint
from lma.utils.logging_utils import setup_logging, write_json

#: Default directory layout per language.
LANGUAGE_DIRS = {"hi": "hindi", "ne": "nepali"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True, choices=["hi", "ne"], help="Target language")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="Directory of PDFs (default: <lang>/data/raw/pdf)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL (default: <lang>/data/raw/ocr/ocr.jsonl)")
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI for true OCR")
    parser.add_argument("--per-page", action="store_true", help="Emit one document per page instead of per book")
    parser.add_argument("--no-embedded", action="store_true", help="Force OCR even when a text layer exists")
    parser.add_argument("--check", action="store_true", help="Only report dependency availability and exit")
    return parser.parse_args()


def main() -> int:
    """Run OCR ingestion for one language."""
    args = parse_args()
    lang_dir = REPO_ROOT / LANGUAGE_DIRS[args.language]
    logger = setup_logging(lang_dir / "logs", name=f"run_ocr_{args.language}")

    available = check_dependencies(need_ocr=True)
    logger.info("OCR dependency check:")
    for tool, path in available.items():
        logger.info("  %-10s %s", tool, path or "NOT FOUND")
    if not all(available.values()):
        logger.warning("Missing tools. Install with:\n  %s", install_hint(args.language))
    if args.check:
        return 0 if all(available.values()) else 1

    pdf_dir = args.pdf_dir or lang_dir / "data/raw/pdf"
    output = args.output or lang_dir / "data/raw/ocr/ocr.jsonl"

    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Created empty %s - place scanned PDFs there and re-run.", pdf_dir)
        return 0

    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found under %s; nothing to do.", pdf_dir)
        return 0
    logger.info("Found %d PDF(s) under %s", len(pdfs), pdf_dir)

    config = OcrConfig(language=args.language, dpi=args.dpi, prefer_embedded_text=not args.no_embedded)
    summary = ingest_directory(pdf_dir, output, config, one_document_per_page=args.per_page)

    logger.info("--- OCR complete ---")
    logger.info("Documents written: %s", f"{summary['documents']:,}")
    logger.info("Characters:        %s", f"{summary['characters']:,}")
    logger.info("Output:            %s", output)
    for entry in summary["files"]:
        logger.info("  %s -> %d doc(s), %s chars",
                    Path(entry["path"]).name, entry["documents"], f"{entry['characters']:,}")

    write_json(lang_dir / "reports/ocr_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
