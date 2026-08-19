#!/usr/bin/env python3
"""Build one language's corpus: collect, normalize, clean, deduplicate, split.

The whole stage is streaming - documents are filtered as they arrive and written
straight into their split file - so peak memory is independent of corpus size and
a 500M-token ingest never needs the raw text on disk.

Usage::

    python scripts/build_corpus.py --config hindi/configs/pipeline.yaml \
        --sources hindi/configs/sources.yaml --profile sampled

    # Production run targeting ~500M training tokens:
    python scripts/build_corpus.py --config nepali/configs/pipeline.yaml \
        --sources nepali/configs/sources.yaml --profile full
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
from lma.data.normalize import IndicNormalizer, get_profile
from lma.data.sources import iter_documents, load_manifest
from lma.data.splits import SplitAssigner, SplitConfig, SplitWriter
from lma.stats.corpus_stats import CorpusStatistics
from lma.utils.config import load_pipeline_config
from lma.utils.logging_utils import setup_logging, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Pipeline YAML for one language")
    parser.add_argument("--sources", required=True, type=Path, help="Source manifest YAML")
    parser.add_argument("--profile", default="sampled", help="Profile from the pipeline config (sampled|full)")
    parser.add_argument("--limit-docs", type=int, default=None, help="Override per-source document limit")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Override the training-token target; ingest stops once reached",
    )
    parser.add_argument(
        "--manual-fraction",
        type=float,
        default=0.20,
        help="Required share of training tokens from manual collection. The downloaded "
             "budget is the remainder, so 0.20 caps downloaded text at 80%% of the target.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Estimate used to convert the token target into a character budget before the tokenizer exists",
    )
    parser.add_argument("--report-every", type=int, default=20_000, help="Progress logging interval in documents")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for resume checkpoints (default: <processed_dir>/checkpoints). "
             "On Colab, point this at mounted Google Drive so it survives the VM.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25_000,
        help="Save a checkpoint every N kept documents. 0 disables checkpointing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint in --checkpoint-dir if one exists",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete any existing checkpoint and start this profile from scratch",
    )
    return parser.parse_args()


def main() -> int:
    """Run the corpus build for one language."""
    args = parse_args()
    config = load_pipeline_config(args.config, args.profile)
    language = config["language"]
    paths = config["paths"]
    profile = config["profile"]

    logs_dir = REPO_ROOT / paths["logs_dir"]
    logger = setup_logging(logs_dir, name=f"build_corpus_{language}")
    logger.info("Building corpus for %s (%s), profile=%s", config["language_name"], language, profile["name"])

    processed_dir = REPO_ROOT / paths["processed_dir"]
    stats_dir = REPO_ROOT / paths["stats_dir"]

    # --- assemble the pipeline components ------------------------------------
    normalizer = IndicNormalizer(get_profile(language))
    cleaner = DocumentCleaner(CleanerConfig(language=language, **config["cleaning"]))
    assigner = SplitAssigner(SplitConfig(**config["splits"]))
    statistics = CorpusStatistics(language=language)

    specs = load_manifest(args.sources)
    logger.info("Loaded %d enabled source(s): %s", len(specs), ", ".join(s["source_id"] for s in specs))

    doc_limit = args.limit_docs if args.limit_docs is not None else profile.get("per_source_doc_limit")
    target_tokens = args.target_tokens if args.target_tokens is not None else profile.get("target_train_tokens")
    # The tokenizer does not exist yet, so the token target is enforced through a
    # character budget using an estimated compression rate. The achieved token
    # count is measured exactly later, in scripts/tokenize_corpus.py.
    # Separate budgets per collection. A single shared budget cannot express the
    # assignment's coupled requirements - whichever source streams first would
    # consume it and the manual ratio would fall out arbitrarily.
    #
    # Manual text is deliberately uncapped: it is the scarce side, so taking all of
    # it can only improve the ratio, and capping it would throw away the very data
    # the requirement is about.
    char_budget = int(target_tokens * args.chars_per_token) if target_tokens else None
    downloaded_budget = (
        int(char_budget * (1.0 - args.manual_fraction)) if char_budget else None
    )
    if char_budget:
        logger.info(
            "Target %s train tokens -> %s chars total (at %.2f chars/token)",
            f"{target_tokens:,}", f"{char_budget:,}", args.chars_per_token,
        )
        logger.info(
            "  downloaded budget: %s chars (%.0f%% of target = %s tokens)",
            f"{downloaded_budget:,}", 100 * (1 - args.manual_fraction),
            f"{int(target_tokens * (1 - args.manual_fraction)):,}",
        )
        logger.info(
            "  manual: uncapped, target %s tokens (%.0f%%) - the scarce side, so all of it is taken",
            f"{int(target_tokens * args.manual_fraction):,}", 100 * args.manual_fraction,
        )

    # --- checkpoint setup -----------------------------------------------------
    checkpoint_dir = args.checkpoint_dir or (processed_dir / "checkpoints")
    manager = CheckpointManager(checkpoint_dir, language)
    if args.restart and manager.exists():
        logger.info("--restart given; deleting existing checkpoint in %s", checkpoint_dir)
        manager.clear()

    state = IngestState(language=language, profile=profile["name"])
    resuming = False
    if args.resume and manager.exists():
        loaded = manager.load(expected_profile=profile["name"])
        if loaded is not None:
            state, dedup_payload = loaded
            resuming = True
            logger.info(
                "Resuming from checkpoint: segment %d, %s docs seen, %s kept, %s train chars",
                state.segments + 1, f"{state.seen:,}", f"{state.kept:,}", f"{state.train_chars:,}",
            )
            # Reconcile the output files with the checkpoint before appending. A
            # process killed mid-write can leave more lines on disk than the last
            # checkpoint counted; appending on top of those would double-count them
            # in a corpus whose statistics claim otherwise.
            for split_name, expected in state.split_summary.get("documents", {}).items():
                path = processed_dir / f"{language}_{split_name}.jsonl"
                actual = truncate_jsonl_to(path, expected)
                if actual != expected:
                    logger.warning(
                        "  %s: checkpoint expected %s lines but file had %s; "
                        "the checkpoint is ahead of the data, so some documents will be re-ingested",
                        path.name, f"{expected:,}", f"{actual:,}",
                    )
                else:
                    logger.info("  %s: reconciled at %s lines", path.name, f"{actual:,}")

            cleaner.restore_stats({
                "seen": state.seen, "kept": state.kept, "rejected": state.rejected,
                "kept_chars": state.kept_chars, "kept_words": state.kept_words,
            })
            if dedup_payload is not None:
                cleaner.load_dedup_state(dedup_payload)
                logger.info("  restored deduplication index")
            if state.corpus_stats_state:
                statistics.load_state_dict(state.corpus_stats_state)
    elif args.resume:
        logger.info("--resume given but no checkpoint found in %s; starting fresh", checkpoint_dir)

    state.segments += 1
    previous_elapsed = state.elapsed_seconds
    started = time.monotonic()
    train_chars = state.train_chars
    # Per-collection character counts, so the budget and the reported ratio both
    # reflect what each side actually contributed.
    chars_by_collection = dict(state.chars_by_collection)
    # Set once the downloaded cap is hit. Downloaded documents are skipped from then
    # on while manual sources keep streaming, so this is a per-collection latch
    # rather than a signal to stop the run.
    downloaded_full = False
    checkpoint_every = args.checkpoint_every

    def save_checkpoint(writer_summary: dict) -> None:
        """Persist the full ingest state, flushing output files first."""
        state.seen = cleaner.stats.seen
        state.kept = cleaner.stats.kept
        state.rejected = dict(cleaner.stats.rejected)
        state.kept_chars = cleaner.stats.kept_chars
        state.kept_words = cleaner.stats.kept_words
        state.train_chars = train_chars
        state.chars_by_collection = dict(chars_by_collection)
        state.split_summary = writer_summary
        state.corpus_stats_state = statistics.state_dict()
        state.elapsed_seconds = previous_elapsed + (time.monotonic() - started)
        manager.save(state, cleaner.dedup_state())

    with SplitWriter(processed_dir, language, append=resuming) as writer:
        if resuming and state.split_summary:
            writer.restore_counts(state.split_summary)

        last_checkpoint_at = cleaner.stats.kept
        try:
            for document in iter_documents(
                specs,
                per_source_limit=doc_limit,
                skip_positions=state.source_positions,
                completed_sources=state.completed_sources,
                on_progress=lambda source_id, position: (
                    state.completed_sources.append(source_id)
                    if position == -1 and source_id not in state.completed_sources
                    else state.source_positions.__setitem__(source_id, position)
                ),
            ):
                # The downloaded budget is a cap on one collection, not a stop
                # condition for the whole ingest. Skipping here rather than breaking
                # lets the manual sources - which are listed after the large
                # downloaded corpus and are the scarce side of the ratio - still be
                # read once downloaded text is full.
                if (
                    downloaded_budget is not None
                    and document.collection == "downloaded"
                    and chars_by_collection.get("downloaded", 0) >= downloaded_budget
                ):
                    continue

                normalized = normalizer.normalize(document.text)
                kept = cleaner.process(normalized)
                if kept is None:
                    continue

                split = assigner.assign(kept, document.source_id)
                writer.write(split, kept, document.source_id, document.collection)
                statistics.add(kept, document.source_id, document.collection)

                if split == "train":
                    train_chars += len(kept)
                    chars_by_collection[document.collection] = (
                        chars_by_collection.get(document.collection, 0) + len(kept)
                    )
                    downloaded_chars = chars_by_collection.get("downloaded", 0)
                    # Only downloaded text is capped. Manual text keeps flowing, so
                    # a source that yields more of it than expected improves the
                    # ratio instead of being cut off.
                    if (
                        downloaded_budget is not None
                        and downloaded_chars >= downloaded_budget
                        and not downloaded_full
                    ):
                        downloaded_full = True
                        logger.info(
                            "Downloaded budget reached (%s downloaded chars, %s manual); "
                            "skipping further downloaded text, continuing with manual sources",
                            f"{downloaded_chars:,}",
                            f"{chars_by_collection.get('manual', 0):,}",
                        )

                if checkpoint_every and cleaner.stats.kept - last_checkpoint_at >= checkpoint_every:
                    writer.flush()
                    save_checkpoint(writer.summary())
                    last_checkpoint_at = cleaner.stats.kept
                    logger.info(
                        "checkpoint saved at %s kept docs (%s train chars)",
                        f"{cleaner.stats.kept:,}", f"{train_chars:,}",
                    )

                if cleaner.stats.seen % args.report_every == 0:
                    rate = (cleaner.stats.seen - state.seen) / max(1e-9, time.monotonic() - started)
                    if downloaded_budget:
                        dl = chars_by_collection.get("downloaded", 0)
                        man = chars_by_collection.get("manual", 0)
                        share = 100 * man / max(1, dl + man)
                        pct = f" (dl {100 * dl / downloaded_budget:.1f}% of budget, manual {share:.1f}%)"
                    else:
                        pct = ""
                    logger.info(
                        "seen=%s kept=%s (%.1f%%) train_chars=%s%s %.0f docs/s",
                        f"{cleaner.stats.seen:,}", f"{cleaner.stats.kept:,}",
                        100 * cleaner.stats.kept / max(1, cleaner.stats.seen),
                        f"{train_chars:,}", pct, rate,
                    )
        except KeyboardInterrupt:
            # Save before exiting so a deliberate stop loses nothing.
            writer.flush()
            save_checkpoint(writer.summary())
            logger.warning(
                "Interrupted; checkpoint saved at %s kept docs. Re-run with --resume to continue.",
                f"{cleaner.stats.kept:,}",
            )
            return 130

        split_summary = writer.summary()

    elapsed = previous_elapsed + (time.monotonic() - started)
    cleaning_stats = cleaner.stats.as_dict()
    corpus_stats = statistics.as_dict()

    manifest = {
        "language": language,
        "language_name": config["language_name"],
        "model": config["model"],
        "profile": profile,
        "elapsed_seconds": elapsed,
        "budget_reached": downloaded_full,
        "segments": state.segments,
        "resumed": resuming,
        "config": {"cleaning": config["cleaning"], "splits": config["splits"]},
        "sources": [{k: v for k, v in spec.items() if k != "notes"} for spec in specs],
        "cleaning_stats": cleaning_stats,
        "split_summary": split_summary,
        "corpus_stats": corpus_stats,
    }

    write_json(stats_dir / "corpus_manifest.json", manifest)
    write_json(stats_dir / "cleaning_stats.json", cleaning_stats)
    write_json(stats_dir / "corpus_stats.json", corpus_stats)

    logger.info("--- Corpus build complete for %s in %.1fs ---", language, elapsed)
    logger.info("Documents seen: %s", f"{cleaning_stats['seen']:,}")
    logger.info("Documents kept: %s (%.1f%% keep rate)", f"{cleaning_stats['kept']:,}", 100 * cleaning_stats["keep_rate"])
    logger.info("Words kept: %s", f"{corpus_stats['totals']['words']:,}")
    manual_fraction = corpus_stats["collection_split"]["manual_word_fraction"]
    logger.info("Manual collection share: %.2f%% of words (requirement: >=20%%)", 100 * manual_fraction)
    if manual_fraction < 0.20:
        logger.warning(
            "Manual share is below the 20%% requirement. Enable more scrape/OCR sources "
            "in the manifest, or reduce downloaded sources, before the graded run."
        )
    for split, count in split_summary["documents"].items():
        logger.info("  %-11s %s docs, %s chars", split, f"{count:,}", f"{split_summary['characters'][split]:,}")
    logger.info("Top rejection reasons: %s", list(cleaning_stats["rejected"].items())[:6])
    logger.info("Wrote statistics to %s", stats_dir)
    if state.segments > 1:
        logger.info("Completed across %d segments, %.1f min total", state.segments, elapsed / 60)

    # The run finished, so retire the checkpoint. Leaving it would make a later
    # re-run "resume" from the end of a completed corpus and add nothing.
    if checkpoint_every:
        manager.clear()
        logger.info("Cleared checkpoint (run complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
