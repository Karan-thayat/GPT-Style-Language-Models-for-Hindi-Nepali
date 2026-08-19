#!/usr/bin/env python3
"""Ingest a pre-tokenized corpus by decoding it back to text and reprocessing it.

Use this when a corpus arrives as a token dump (``*.bin`` of uint16 ids plus a
byte-level BPE ``vocab.json``) rather than as text. Byte-level BPE is losslessly
invertible, so the original text is recovered exactly and then run through the
project's own pipeline: normalization, quality filtering, deduplication, and
deterministic split assignment.

Why decode rather than use the tokens directly
----------------------------------------------
The token ids belong to *someone else's* tokenizer. Phase 1 requires a tokenizer
trained from scratch on this project's own corpus, and it requires corpus
statistics that can only be computed over text. Decoding puts the data back into
the form the rest of the pipeline - and the rest of the assignment - operates on.

What is unavoidably lost
------------------------
Per-document provenance. A flat token stream records which tokens follow which,
not which website or book each document came from, so every recovered document is
attributed to one synthetic source id and counted as ``downloaded``. Manually
collected text must therefore come from the manifest sources, which is exactly
why ``--merge-existing`` keeps the previously built corpus rather than replacing
it - that corpus carries the ``manual`` provenance the 20% requirement needs.

Usage::

    python scripts/ingest_token_dump.py \
        --config hindi/configs/pipeline.yaml \
        --dump <dump-dir>/train_tokens_500M.bin \
        --vocab <dump-dir>/tokenizer/vocab.json \
        --merge-existing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lma.data.checkpoint import CheckpointManager, IngestState, truncate_jsonl_to
from lma.data.clean import CleanerConfig, DocumentCleaner
from lma.data.detokenize import ByteLevelBpeDecoder
from lma.data.normalize import IndicNormalizer, get_profile
from lma.data.splits import SPLIT_NAMES, SplitAssigner, SplitConfig, SplitWriter, iter_jsonl
from lma.stats.corpus_stats import CorpusStatistics
from lma.utils.config import load_pipeline_config
from lma.utils.logging_utils import setup_logging, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Pipeline YAML for the language")
    parser.add_argument("--profile", default="full", help="Profile from the pipeline config")
    parser.add_argument("--dump", required=True, type=Path, help="uint16 token dump (.bin)")
    parser.add_argument("--vocab", required=True, type=Path, help="Byte-level BPE vocab.json for the dump")
    parser.add_argument(
        "--source-id",
        default="pretokenized_dump",
        help="Source id recorded for every recovered document",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Keep the documents already in the processed split files and add to them. "
             "Preserves manually collected text and its provenance.",
    )
    parser.add_argument("--limit-tokens", type=int, default=None, help="Decode only the first N tokens")
    parser.add_argument("--chunk-tokens", type=int, default=4_000_000, help="Tokens decoded per chunk")
    parser.add_argument("--report-every", type=int, default=100_000, help="Progress interval in documents")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint directory")
    parser.add_argument("--checkpoint-every", type=int, default=200_000, help="Checkpoint every N kept docs; 0 disables")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint")
    parser.add_argument("--restart", action="store_true", help="Discard any checkpoint and start over")
    return parser.parse_args()


def load_existing(processed_dir: Path, language: str, logger) -> list[dict]:
    """Read the documents already present in the split files.

    These are re-emitted through the same cleaner so that deduplication sees them,
    which means a document appearing in both the existing corpus and the token dump
    is kept once rather than twice.
    """
    existing: list[dict] = []
    for split in SPLIT_NAMES:
        path = processed_dir / f"{language}_{split}.jsonl"
        if not path.exists():
            continue
        rows = list(iter_jsonl(path))
        existing.extend(rows)
        logger.info("  %s: %s existing documents", path.name, f"{len(rows):,}")
    return existing


def main() -> int:
    """Decode a token dump and rebuild the corpus from it."""
    args = parse_args()
    config = load_pipeline_config(args.config, args.profile)
    language = config["language"]
    paths = config["paths"]

    logger = setup_logging(REPO_ROOT / paths["logs_dir"], name=f"ingest_dump_{language}")
    processed_dir = REPO_ROOT / paths["processed_dir"]
    stats_dir = REPO_ROOT / paths["stats_dir"]

    logger.info("Ingesting token dump for %s", config["language_name"])
    logger.info("  dump : %s", args.dump)
    logger.info("  vocab: %s", args.vocab)

    decoder = ByteLevelBpeDecoder(args.vocab)
    total_tokens = args.dump.stat().st_size // 2
    logger.info("  %s tokens in dump, vocabulary of %s pieces",
                f"{total_tokens:,}", f"{decoder.vocab_size:,}")

    normalizer = IndicNormalizer(get_profile(language))
    cleaner = DocumentCleaner(CleanerConfig(language=language, **config["cleaning"]))
    assigner = SplitAssigner(SplitConfig(**config["splits"]))
    statistics = CorpusStatistics(language=language)

    # --- carry over the existing corpus ---------------------------------------
    existing_docs: list[dict] = []
    if args.merge_existing:
        logger.info("Reading existing corpus to merge:")
        existing_docs = load_existing(processed_dir, language, logger)
        logger.info("  %s documents carried over", f"{len(existing_docs):,}")

    checkpoint_dir = args.checkpoint_dir or (processed_dir / "checkpoints_dump")
    manager = CheckpointManager(checkpoint_dir, language)
    if args.restart and manager.exists():
        manager.clear()
        logger.info("Cleared previous checkpoint")

    state = IngestState(language=language, profile=f"dump:{args.profile}")
    resuming = False
    if args.resume and manager.exists():
        loaded = manager.load(expected_profile=f"dump:{args.profile}")
        if loaded is not None:
            state, dedup_payload = loaded
            resuming = True
            logger.info("Resuming: %s tokens decoded, %s docs kept",
                        f"{state.source_positions.get('tokens', 0):,}", f"{state.kept:,}")
            for split_name, expected in state.split_summary.get("documents", {}).items():
                truncate_jsonl_to(processed_dir / f"{language}_{split_name}.jsonl", expected)
            cleaner.restore_stats({
                "seen": state.seen, "kept": state.kept, "rejected": state.rejected,
                "kept_chars": state.kept_chars, "kept_words": state.kept_words,
            })
            if dedup_payload is not None:
                cleaner.load_dedup_state(dedup_payload)
            if state.corpus_stats_state:
                statistics.load_state_dict(state.corpus_stats_state)

    state.segments += 1
    previous_elapsed = state.elapsed_seconds
    started = time.monotonic()
    tokens_done = state.source_positions.get("tokens", 0)

    def save_checkpoint(summary: dict) -> None:
        """Persist ingest state, including how far through the dump we are."""
        state.seen = cleaner.stats.seen
        state.kept = cleaner.stats.kept
        state.rejected = dict(cleaner.stats.rejected)
        state.kept_chars = cleaner.stats.kept_chars
        state.kept_words = cleaner.stats.kept_words
        state.split_summary = summary
        state.corpus_stats_state = statistics.state_dict()
        state.source_positions["tokens"] = tokens_done
        state.elapsed_seconds = previous_elapsed + (time.monotonic() - started)
        manager.save(state, cleaner.dedup_state())

    def emit(text: str, source_id: str, collection: str) -> None:
        """Normalize, filter, and write one document."""
        kept = cleaner.process(normalizer.normalize(text))
        if kept is None:
            return
        split = assigner.assign(kept, source_id)
        writer.write(split, kept, source_id, collection)
        statistics.add(kept, source_id, collection)

    # The existing corpus is rewritten from scratch here, so the writer truncates
    # unless we are resuming a partially completed dump ingest.
    with SplitWriter(processed_dir, language, append=resuming) as writer:
        if resuming and state.split_summary:
            writer.restore_counts(state.split_summary)

        # Existing documents go in first, so their manual/scraped provenance is
        # preserved and they win the deduplication race against any dump copy.
        if existing_docs and not resuming:
            logger.info("Writing %s carried-over documents", f"{len(existing_docs):,}")
            for row in existing_docs:
                emit(row["text"], row.get("source_id", "existing"), row.get("collection", "downloaded"))
            logger.info("  %s kept after filtering", f"{cleaner.stats.kept:,}")

        logger.info("Decoding dump from token %s", f"{tokens_done:,}")
        last_checkpoint = cleaner.stats.kept
        try:
            for document in decoder.stream_documents(
                args.dump,
                chunk_tokens=args.chunk_tokens,
                limit_tokens=args.limit_tokens,
            ):
                # Approximate the dump position from characters consumed; exact
                # per-document token counts are not recoverable after decoding.
                tokens_done = min(total_tokens, tokens_done + max(1, len(document) // 3))
                emit(document, args.source_id, "downloaded")

                if checkpoint_every := args.checkpoint_every:
                    if cleaner.stats.kept - last_checkpoint >= checkpoint_every:
                        writer.flush()
                        save_checkpoint(writer.summary())
                        last_checkpoint = cleaner.stats.kept
                        logger.info("checkpoint at %s kept docs", f"{cleaner.stats.kept:,}")

                if cleaner.stats.seen % args.report_every == 0:
                    rate = (cleaner.stats.seen - state.seen) / max(1e-9, time.monotonic() - started)
                    logger.info(
                        "seen=%s kept=%s (%.1f%%) words=%s %.0f docs/s",
                        f"{cleaner.stats.seen:,}", f"{cleaner.stats.kept:,}",
                        100 * cleaner.stats.kept / max(1, cleaner.stats.seen),
                        f"{cleaner.stats.kept_words:,}", rate,
                    )
        except KeyboardInterrupt:
            writer.flush()
            save_checkpoint(writer.summary())
            logger.warning("Interrupted; checkpoint saved. Re-run with --resume.")
            return 130

        split_summary = writer.summary()

    elapsed = previous_elapsed + (time.monotonic() - started)
    cleaning_stats = cleaner.stats.as_dict()
    corpus_stats = statistics.as_dict()

    write_json(stats_dir / "corpus_manifest.json", {
        "language": language,
        "language_name": config["language_name"],
        "model": config["model"],
        "profile": {"name": f"dump:{args.profile}"},
        "elapsed_seconds": elapsed,
        "segments": state.segments,
        "config": {"cleaning": config["cleaning"], "splits": config["splits"]},
        "sources": [{
            "source_id": args.source_id,
            "type": "pretokenized_dump",
            "collection": "downloaded",
            "dump_file": str(args.dump),
            "vocab_file": str(args.vocab),
            "dump_tokens": total_tokens,
        }],
        "merged_existing_documents": len(existing_docs),
        "cleaning_stats": cleaning_stats,
        "split_summary": split_summary,
        "corpus_stats": corpus_stats,
    })
    write_json(stats_dir / "cleaning_stats.json", cleaning_stats)
    write_json(stats_dir / "corpus_stats.json", corpus_stats)

    logger.info("--- Ingest complete in %.1f min ---", elapsed / 60)
    logger.info("Documents seen : %s", f"{cleaning_stats['seen']:,}")
    logger.info("Documents kept : %s (%.1f%%)", f"{cleaning_stats['kept']:,}", 100 * cleaning_stats["keep_rate"])
    logger.info("Words kept     : %s", f"{corpus_stats['totals']['words']:,}")
    manual = corpus_stats["collection_split"]["manual_word_fraction"]
    logger.info("Manual share   : %.2f%% of words", 100 * manual)
    if manual < 0.20:
        logger.warning(
            "Manual share is %.2f%%, below the 20%% requirement. The decoded dump counts as "
            "downloaded text; add scraped/OCR'd material via the source manifest to close the gap.",
            100 * manual,
        )
    for split, count in split_summary["documents"].items():
        logger.info("  %-11s %s docs, %s chars", split, f"{count:,}", f"{split_summary['characters'][split]:,}")

    if args.checkpoint_every:
        manager.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
