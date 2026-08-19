#!/usr/bin/env python3
"""Train and evaluate one language's tokenizer, including the vocabulary sweep.

Stages:

1. **Sweep** (optional, ``--sweep``) - train candidate tokenizers across
   vocabulary sizes and both algorithms, measure each on held-out validation
   text, and select the best by fertility subject to an unknown-rate ceiling.
2. **Final training** - fit the chosen configuration on the training split.
3. **Evaluation** - fertility, unknown and byte-fallback rate, compression,
   piece statistics, token frequencies, and tokenization examples, all on
   held-out text.

``--eval-only`` runs stage 3 against the tokenizer already in the language's
tokenizer directory. This exists because a tokenizer may have been fitted in a
different session (a long training run on Colab, say) while the evaluation that
the report needs was never written; re-running the full script would refit the
vocabulary and invalidate every artifact keyed to the current model file's hash.

Usage::

    python scripts/train_tokenizer.py --config hindi/configs/pipeline.yaml --sweep
    python scripts/train_tokenizer.py --config nepali/configs/pipeline.yaml \
        --sweep --use-sweep-best
    python scripts/train_tokenizer.py --config hindi/configs/pipeline.yaml \
        --profile full --eval-only
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lma.stats.corpus_stats import iter_split_texts
from lma.stats.tokenizer_stats import (
    MAX_VOCAB_SIZE,
    MIN_VOCAB_SIZE,
    TokenizerEvaluator,
    select_best_config,
    sweep_vocabulary_sizes,
)
from lma.tokenizer.train_tokenizer import TokenizerConfig, train_from_jsonl
from lma.utils.config import load_pipeline_config
from lma.utils.logging_utils import setup_logging, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Pipeline YAML for one language")
    parser.add_argument("--profile", default="sampled", help="Profile from the pipeline config")
    parser.add_argument("--sweep", action="store_true", help="Run the vocabulary-size / algorithm sweep")
    parser.add_argument(
        "--use-sweep-best",
        action="store_true",
        help="Train the final tokenizer with the sweep's winning config instead of the config file's",
    )
    parser.add_argument("--heldout-docs", type=int, default=3000, help="Validation documents used for evaluation")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override the final vocabulary size")
    parser.add_argument("--model-type", default=None, choices=["unigram", "bpe"], help="Override the algorithm")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the tokenizer already in the tokenizer directory instead of fitting a new one",
    )
    parser.add_argument(
        "--sweep-max-lines",
        type=int,
        default=None,
        help="Training lines per sweep candidate, overriding the profile's sweep_max_train_lines",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file, so a report can be tied to the exact model it measured."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_existing_tokenizer(tokenizer_dir: Path, language: str, model_type: str, vocab_size: int) -> Path:
    """Find the trained ``.model`` file to evaluate in ``--eval-only`` mode.

    The configured name is preferred so that evaluation matches the configuration
    the rest of the pipeline assumes. A single unambiguous alternative is accepted
    as a fallback; two or more are refused rather than guessed, because silently
    measuring the wrong vocabulary would put wrong numbers in the report.

    Args:
        tokenizer_dir: The language's tokenizer directory.
        language: ISO code.
        model_type: Configured algorithm.
        vocab_size: Configured vocabulary size.

    Returns:
        Path to the ``.model`` file.
    """
    expected = tokenizer_dir / f"{language}_{model_type}_{vocab_size}.model"
    if expected.exists():
        return expected

    candidates = sorted(tokenizer_dir.glob(f"{language}_*.model"))
    if not candidates:
        raise SystemExit(
            f"--eval-only found no tokenizer in {tokenizer_dir}. Expected {expected.name}; "
            "run this script without --eval-only to train one."
        )
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise SystemExit(
            f"--eval-only found several tokenizers in {tokenizer_dir} ({names}) and none matches "
            f"the configured {expected.name}. Pass --vocab-size/--model-type to disambiguate."
        )
    return candidates[0]


def evaluate_and_report(
    manifest: dict,
    heldout: list[str],
    config: dict,
    stats_dir: Path,
    logger,
    sweep_ran: bool,
) -> int:
    """Measure a trained tokenizer on held-out text and write ``tokenizer_stats.json``.

    This is stage 3, shared by the training path and ``--eval-only`` so both write
    exactly the same report structure.

    Args:
        manifest: Training manifest, or the equivalent record for a tokenizer
            trained in an earlier run.
        heldout: Held-out validation documents, never seen by the tokenizer.
        config: Resolved pipeline configuration for this language.
        stats_dir: Directory to write the report into.
        logger: Logger for the human-readable summary.
        sweep_ran: Whether a vocabulary sweep was part of this run.

    Returns:
        Process exit code.
    """
    language = config["language"]
    model_file = Path(manifest["model_file"])
    # Recording the hash is what makes a stale report detectable: if the tokenizer
    # is refitted later, these metrics no longer describe the model on disk, and
    # scripts/check_phase1.py can say so instead of the report quietly drifting.
    manifest.setdefault("model_sha256", file_sha256(model_file))
    evaluator = TokenizerEvaluator(model_file, language)
    metrics = evaluator.evaluate(heldout)
    piece_stats = evaluator.piece_statistics()
    frequencies = evaluator.token_frequency(heldout)
    examples = evaluator.examples(heldout, max_examples=12)

    report = {
        "language": language,
        "language_name": config["language_name"],
        "model": config["model"],
        "training_manifest": manifest,
        "heldout_documents": len(heldout),
        "metrics": metrics.as_dict(),
        "piece_statistics": piece_stats,
        "token_frequency": frequencies,
        "examples": examples,
        "sweep_ran": sweep_ran,
    }
    write_json(stats_dir / "tokenizer_stats.json", report)

    logger.info("--- Held-out tokenizer metrics (%s) ---", language)
    logger.info("Vocabulary size:       %d", metrics.vocab_size)
    logger.info("Fertility:             %.4f tokens/word", metrics.fertility)
    if metrics.fertility <= 1.0:
        # Fertility <= 1 means words are not being split into subwords at all,
        # which the assignment rules out: sub-word tokenization is the point.
        logger.warning(
            "Fertility %.4f is not > 1.0 - this tokenizer is not segmenting into subwords.",
            metrics.fertility,
        )
    logger.info("Unknown rate:          %.3e", metrics.unk_rate)
    logger.info("Byte-fallback rate:    %.4f%% (script only: %.4f%%)",
                100 * metrics.byte_fallback_rate, 100 * metrics.script_byte_fallback_rate)
    logger.info("Chars per token:       %.3f", metrics.chars_per_token)
    logger.info("Bytes per token:       %.3f", metrics.bytes_per_token)
    logger.info("Single-token words:    %.1f%%", 100 * metrics.single_token_word_rate)
    logger.info("Vocab utilization:     %.1f%% of pieces seen in held-out text",
                100 * frequencies.get("vocab_utilization", 0.0))
    roundtrip_failures = sum(1 for ex in examples if not ex["roundtrip_ok"])
    logger.info("Round-trip failures:   %d of %d examples", roundtrip_failures, len(examples))
    logger.info("Wrote %s", stats_dir / "tokenizer_stats.json")
    return 0


def main() -> int:
    """Train, sweep, and evaluate the tokenizer for one language."""
    args = parse_args()
    config = load_pipeline_config(args.config, args.profile)
    language = config["language"]
    paths = config["paths"]
    profile = config["profile"]
    tokenizer_cfg = config["tokenizer"]

    logger = setup_logging(REPO_ROOT / paths["logs_dir"], name=f"train_tokenizer_{language}")
    logger.info("Tokenizer stage for %s (%s), profile=%s", config["language_name"], language, profile["name"])

    processed_dir = REPO_ROOT / paths["processed_dir"]
    tokenizer_dir = REPO_ROOT / paths["tokenizer_dir"]
    stats_dir = REPO_ROOT / paths["stats_dir"]

    train_file = processed_dir / f"{language}_train.jsonl"
    val_file = processed_dir / f"{language}_validation.jsonl"
    for required in (train_file, val_file):
        if not required.exists():
            raise SystemExit(f"Missing {required}. Run scripts/build_corpus.py first.")

    # Held-out text is loaded once and reused for every evaluation so all
    # candidates are scored on exactly the same data.
    heldout = list(iter_split_texts(val_file, limit=args.heldout_docs))
    if not heldout:
        raise SystemExit(f"Validation split {val_file} is empty; cannot evaluate on held-out text.")
    logger.info("Loaded %d held-out validation documents", len(heldout))

    chosen_vocab = args.vocab_size or tokenizer_cfg["vocab_size"]
    chosen_type = args.model_type or tokenizer_cfg["model_type"]
    sweep_results: list[dict] = []

    # The 5-10K vocabulary band is a course requirement, not a preference: the
    # ~25M Phase 2 parameter budget counts the embedding matrix, so a larger
    # vocabulary spends the budget on embeddings instead of on the model. Warn
    # loudly rather than failing, so a deliberate out-of-band experiment is still
    # possible - but never silently.
    if not MIN_VOCAB_SIZE <= chosen_vocab <= MAX_VOCAB_SIZE:
        logger.warning(
            "Vocabulary size %d is outside the required %d-%d band; embeddings would "
            "consume %.1f%% of the 25M budget at d_model=384.",
            chosen_vocab, MIN_VOCAB_SIZE, MAX_VOCAB_SIZE, 100 * chosen_vocab * 384 / 25_000_000,
        )

    # --- stage 1: sweep --------------------------------------------------------
    if args.sweep:
        sizes = tokenizer_cfg["sweep_vocab_sizes"]
        types = tokenizer_cfg["sweep_model_types"]
        sweep_lines = args.sweep_max_lines or profile.get("sweep_max_train_lines")
        logger.info("Sweeping vocab sizes %s over model types %s on %s training lines",
                    sizes, types, f"{sweep_lines:,}" if sweep_lines else "all")
        started = time.monotonic()
        sweep_results = sweep_vocabulary_sizes(
            train_files=[train_file],
            heldout_texts=heldout,
            language=language,
            vocab_sizes=sizes,
            model_types=types,
            max_train_lines=sweep_lines,
        )
        logger.info("Sweep finished in %.1fs", time.monotonic() - started)

        for result in sweep_results:
            if "error" in result:
                logger.warning("  %-8s vocab=%-6s FAILED: %s",
                               result["model_type"], result["vocab_size"], result["error"])
            else:
                logger.info(
                    "  %-8s vocab=%-6s fertility=%.3f unk=%.2e byte_fb=%.4f%% chars/tok=%.2f",
                    result["model_type"], result["vocab_size"], result["fertility"],
                    result["unk_rate"], 100 * result["byte_fallback_rate"], result["chars_per_token"],
                )

        best = select_best_config(sweep_results)
        if best is None:
            logger.warning("No viable sweep candidate; keeping configured tokenizer settings")
        else:
            logger.info(
                "Sweep winner: %s vocab=%d (fertility=%.3f)",
                best["model_type"], best["vocab_size"], best["fertility"],
            )
            if args.use_sweep_best:
                chosen_vocab = best.get("requested_vocab_size", best["vocab_size"])
                chosen_type = best["model_type"]
                logger.info("Using sweep winner for the final tokenizer")

        write_json(stats_dir / "tokenizer_sweep.json", {
            "language": language,
            "training_lines_per_candidate": sweep_lines,
            "heldout_documents": len(heldout),
            "results": sweep_results,
            "selected": best,
            "selection_rule": (
                "Best held-out fertility subject to (a) vocabulary size within the mandated "
                "5,000-10,000 band, (b) fertility > 1.0 so the tokenizer is genuinely "
                "sub-word, (c) unk_rate <= 1e-4, and (d) embedding parameters <= 40% of the "
                "25M Phase 2 budget at d_model=384; then step down to any smaller vocabulary "
                "within 3% fertility. Fertility alone would always select the largest "
                "candidate, since it falls monotonically with vocab size."
            ),
            "selection_parameters": {
                "min_vocab_size": MIN_VOCAB_SIZE,
                "max_vocab_size": MAX_VOCAB_SIZE,
                "min_fertility": 1.0,
                "max_unk_rate": 1e-4,
                "model_dim": 384,
                "param_budget": 25_000_000,
                "max_embedding_share": 0.40,
                "fertility_tolerance": 0.03,
            },
        })

    # --- stage 2: final training (skipped when only evaluating) -----------------
    if args.eval_only:
        model_file = locate_existing_tokenizer(tokenizer_dir, language, chosen_type, chosen_vocab)
        manifest = {
            "model_file": str(model_file),
            "vocab_file": str(model_file.with_suffix(".vocab")),
            "model_sha256": file_sha256(model_file),
            "trained_in_this_run": False,
            "corpus_files": [str(train_file)],
            "config": {
                "language": language,
                "model_type": chosen_type,
                "vocab_size": chosen_vocab,
                "character_coverage": tokenizer_cfg["character_coverage"],
                "byte_fallback": tokenizer_cfg["byte_fallback"],
                "split_digits": tokenizer_cfg["split_digits"],
                "normalization_rule_name": tokenizer_cfg["normalization_rule_name"],
                "seed": tokenizer_cfg["seed"],
            },
        }
        logger.info("Evaluating existing tokenizer %s (sha256 %s)",
                    model_file.name, manifest["model_sha256"][:12])
        return evaluate_and_report(
            manifest, heldout, config, stats_dir, logger, sweep_ran=False
        )

    final_config = TokenizerConfig(
        language=language,
        model_type=chosen_type,
        vocab_size=chosen_vocab,
        character_coverage=tokenizer_cfg["character_coverage"],
        byte_fallback=tokenizer_cfg["byte_fallback"],
        split_digits=tokenizer_cfg["split_digits"],
        normalization_rule_name=tokenizer_cfg["normalization_rule_name"],
        seed=tokenizer_cfg["seed"],
        # Memory controls. SentencePiece loads every training sentence into RAM
        # before it samples, so on a corpus of this size the defaults
        # (4M sentences, 8 threads) grow past 5GB and the trainer is OOM-killed.
        # These are read from the pipeline config so the cap can be tuned per
        # machine without editing code.
        input_sentence_size=tokenizer_cfg.get("input_sentence_size", 1_000_000),
        num_threads=tokenizer_cfg.get("num_threads", 4),
    )
    logger.info("Training final tokenizer: %s, vocab=%d", final_config.model_type, final_config.vocab_size)
    started = time.monotonic()
    manifest = train_from_jsonl(
        [train_file], final_config, tokenizer_dir, max_lines=profile.get("tokenizer_max_train_lines")
    )
    logger.info("Trained on %s lines in %.1fs -> %s",
                f"{manifest['training_lines']:,}", time.monotonic() - started, manifest["model_file"])

    # --- stage 3: evaluation ---------------------------------------------------
    return evaluate_and_report(
        manifest, heldout, config, stats_dir, logger, sweep_ran=bool(sweep_results)
    )


if __name__ == "__main__":
    raise SystemExit(main())
