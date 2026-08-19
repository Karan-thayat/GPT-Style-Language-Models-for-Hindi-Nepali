#!/usr/bin/env python3
"""Count exact token totals per split and optionally pre-tokenize to disk.

Why this is a separate stage
---------------------------
The Phase 1 requirement is stated in *tokens after tokenization*, which cannot be
known while the corpus is being built - the tokenizer does not exist yet. This
script closes the loop: it runs the trained tokenizer over every split and
reports the exact token counts that the dataset statistics and the report must
cite, alongside the shortfall against the 500M target.

With ``--write-bin`` it also serializes token ids to a flat uint16/uint32 memmap
array, which is the format Phase 2's data loader will consume. That is optional
here so Phase 1 stays about data and tokenizers.

Usage::

    python scripts/tokenize_corpus.py --config hindi/configs/pipeline.yaml
    python scripts/tokenize_corpus.py --config nepali/configs/pipeline.yaml --write-bin
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from lma.data.splits import SPLIT_NAMES, iter_jsonl
from lma.tokenizer.train_tokenizer import BOS_ID, EOS_ID, TokenizerTrainer
from lma.utils.logging_utils import setup_logging, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Pipeline YAML for one language")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Override the .model path")
    parser.add_argument("--write-bin", action="store_true", help="Also write token ids as a flat memmap array")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=500_000_000,
        help="Requirement to compare the achieved training token count against",
    )
    return parser.parse_args()


def find_tokenizer(tokenizer_dir: Path, language: str) -> Path:
    """Locate the trained tokenizer model for a language.

    Raises:
        SystemExit: When no model file is present.
    """
    candidates = sorted(Path(tokenizer_dir).glob(f"{language}_*.model"))
    if not candidates:
        raise SystemExit(f"No tokenizer model found in {tokenizer_dir}. Run scripts/train_tokenizer.py first.")
    # Prefer the most recently modified, which is the final (non-sweep) model.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    """Count tokens for every split of one language."""
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    language = config["language"]
    paths = config["paths"]

    logger = setup_logging(REPO_ROOT / paths["logs_dir"], name=f"tokenize_corpus_{language}")

    processed_dir = REPO_ROOT / paths["processed_dir"]
    stats_dir = REPO_ROOT / paths["stats_dir"]
    model_file = args.tokenizer or find_tokenizer(REPO_ROOT / paths["tokenizer_dir"], language)
    logger.info("Tokenizing %s corpus with %s", language, model_file)

    processor = TokenizerTrainer.load(model_file)
    vocab_size = processor.get_piece_size()
    # uint16 suffices below 65536 pieces and halves the on-disk size.
    dtype = np.uint16 if vocab_size < 2**16 else np.uint32

    results: dict[str, dict] = {}
    for split in SPLIT_NAMES:
        split_file = processed_dir / f"{language}_{split}.jsonl"
        if not split_file.exists():
            logger.warning("Split file missing, skipping: %s", split_file)
            continue

        started = time.monotonic()
        documents = 0
        tokens = 0
        characters = 0
        utf8_bytes = 0
        # The manual-collection requirement is stated in tokens, not words, and
        # manual prose does not tokenize at the same rate as downloaded web text.
        # Each document carries the collection it came from, so the split is
        # accumulated here rather than approximated from the word-based figure.
        tokens_by_collection: dict[str, int] = {}
        buffer: list[np.ndarray] = [] if args.write_bin else []

        for row in iter_jsonl(split_file):
            text = row["text"]
            # Each document is framed with BOS/EOS, which is how Phase 2 will
            # pack sequences; counting them here keeps the two stages consistent.
            ids = [BOS_ID, *processor.encode(text, out_type=int), EOS_ID]
            documents += 1
            tokens += len(ids)
            characters += len(text)
            utf8_bytes += len(text.encode("utf-8"))
            collection = row.get("collection", "unknown")
            tokens_by_collection[collection] = tokens_by_collection.get(collection, 0) + len(ids)
            if args.write_bin:
                buffer.append(np.asarray(ids, dtype=dtype))

        elapsed = time.monotonic() - started
        entry = {
            "documents": documents,
            "tokens": tokens,
            "characters": characters,
            "utf8_bytes": utf8_bytes,
            "chars_per_token": characters / tokens if tokens else 0.0,
            "bytes_per_token": utf8_bytes / tokens if tokens else 0.0,
            "tokens_per_document": tokens / documents if documents else 0.0,
            "by_collection": dict(sorted(tokens_by_collection.items())),
            "seconds": elapsed,
        }

        if args.write_bin and buffer:
            bin_path = processed_dir / f"{language}_{split}.bin"
            flat = np.concatenate(buffer)
            flat.tofile(bin_path)
            entry["bin_file"] = str(bin_path)
            entry["bin_dtype"] = np.dtype(dtype).name
            entry["bin_bytes"] = int(flat.nbytes)
            logger.info("Wrote %s (%s tokens, %.1f MiB)", bin_path, f"{flat.size:,}", flat.nbytes / 1024**2)

        results[split] = entry
        logger.info(
            "%-11s %s docs, %s tokens, %.2f chars/token (%.1fs)",
            split, f"{documents:,}", f"{tokens:,}", entry["chars_per_token"], elapsed,
        )

    train_tokens = results.get("train", {}).get("tokens", 0)
    total_tokens = sum(entry["tokens"] for entry in results.values())
    shortfall = max(0, args.target_tokens - train_tokens)

    # The assignment's ">= 20% manual" floor applies to the training tokens, so it
    # is computed on the training split alone.
    by_collection = results.get("train", {}).get("by_collection", {})
    manual_tokens = by_collection.get("manual", 0)

    report = {
        "language": language,
        "language_name": config["language_name"],
        "model": config["model"],
        "tokenizer_model": str(model_file),
        "vocab_size": vocab_size,
        "per_split": results,
        "train_tokens": train_tokens,
        "total_tokens": total_tokens,
        "by_collection": dict(sorted(by_collection.items())),
        "manual_token_fraction": manual_tokens / train_tokens if train_tokens else None,
        "target_train_tokens": args.target_tokens,
        "target_met": train_tokens >= args.target_tokens,
        "shortfall_tokens": shortfall,
        "achieved_fraction_of_target": train_tokens / args.target_tokens if args.target_tokens else None,
    }
    write_json(stats_dir / "token_counts.json", report)

    logger.info("--- Exact token counts (%s) ---", language)
    logger.info("Training tokens: %s", f"{train_tokens:,}")
    logger.info("All splits:      %s", f"{total_tokens:,}")
    if report["manual_token_fraction"] is not None:
        logger.info(
            "Manual share:    %.2f%% of training tokens (%s manual, %s downloaded)",
            100 * report["manual_token_fraction"],
            f"{manual_tokens:,}",
            f"{by_collection.get('downloaded', 0):,}",
        )
    logger.info(
        "Target %s: %s (%.1f%% achieved)",
        f"{args.target_tokens:,}",
        "MET" if report["target_met"] else f"SHORT by {shortfall:,}",
        100 * (report["achieved_fraction_of_target"] or 0),
    )
    if not report["target_met"]:
        logger.warning(
            "Training tokens are below target. For the graded run, either enable more "
            "sources in the manifest or document and justify the shortfall in the report."
        )
    logger.info("Wrote %s", stats_dir / "token_counts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
