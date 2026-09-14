#!/usr/bin/env python3
"""Generate one language's synthetic comparative-reasoning dataset.

The assignment requires the finetuning set to be built programmatically rather
than downloaded, so ground-truth labels are known by construction, and requires
the generation scripts to be released so the dataset is reproducible. Running
this with the same ``--seed`` reproduces the corpus exactly.

Usage::

    python scripts/generate_reasoning_data.py --language hi
    python scripts/generate_reasoning_data.py --language ne --examples 20000

Writes, under ``<language>/data/reasoning/``:

* ``<lang>_reasoning_{train,validation,test}.jsonl`` - the splits
* and, under ``<language>/reports/``, ``reasoning_dataset.json`` - the statistics
  and leakage evidence the report cites.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lma.reasoning.generate import (  # noqa: E402
    GeneratorConfig,
    ReasoningDatasetBuilder,
    dataset_statistics,
    write_jsonl,
)
from lma.utils.logging_utils import setup_logging, write_json  # noqa: E402

#: Language code -> (directory, model label), matching every other script.
LANGUAGES = {"hi": ("hindi", "Model H"), "ne": ("nepali", "Model L")}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--language", required=True, choices=sorted(LANGUAGES))
    parser.add_argument(
        "--examples",
        type=int,
        default=20_000,
        help="Total examples across all splits (default: 20000)",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the output directory for the JSONL splits",
    )
    return parser.parse_args()


def main() -> int:
    """Generate, write, and verify one language's reasoning dataset."""
    args = parse_args()
    directory, label = LANGUAGES[args.language]

    logs_dir = REPO_ROOT / directory / "logs"
    logger = setup_logging(logs_dir, name=f"reasoning_data_{args.language}")

    config = GeneratorConfig(
        language=args.language, total_examples=args.examples, seed=args.seed
    )
    builder = ReasoningDatasetBuilder(config)

    logger.info("Generating %s reasoning data (%s)", label, args.language)
    logger.info(
        "  entity pools: %d shared, %d evaluation-only distractors",
        len(builder.shared_entities),
        len(builder.eval_only_entities),
    )

    examples = builder.build()
    stats = dataset_statistics(examples, builder)

    out_dir = args.out_dir or (REPO_ROOT / directory / "data" / "reasoning")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        rows = [e for e in examples if e.split == split]
        path = out_dir / f"{args.language}_reasoning_{split}.jsonl"
        write_jsonl(rows, path)
        logger.info("  wrote %-12s %6d examples  %s", split, len(rows), path.name)

    stats["output_dir"] = str(out_dir.relative_to(REPO_ROOT))
    stats["files"] = {
        split: f"{args.language}_reasoning_{split}.jsonl"
        for split in ("train", "validation", "test")
    }

    reports_dir = REPO_ROOT / directory / "reports"
    write_json(reports_dir / "reasoning_dataset.json", stats)

    checks = stats["leakage_checks"]
    logger.info("\n--- Leakage control ---")
    logger.info("  chain-signature overlap (train vs eval): %d", checks["chain_signature_overlap"])
    logger.info("  prompt overlap (train vs eval):          %d", checks["prompt_overlap_train_eval"])
    logger.info("  duplicate prompts:                       %d", checks["duplicate_prompts"])
    logger.info("  distractor answers in eval:              %d", checks["distractor_answers_in_eval"])
    logger.info(
        "  distractors appearing in training:       %d",
        len(checks["distractors_in_training_examples"]),
    )

    # These are the properties the report claims. Failing loudly here is better
    # than shipping a dataset whose evaluation split silently overlaps training.
    violations = [
        name
        for name, value in (
            ("chain_signature_overlap", checks["chain_signature_overlap"]),
            ("prompt_overlap_train_eval", checks["prompt_overlap_train_eval"]),
            ("duplicate_prompts", checks["duplicate_prompts"]),
            ("distractor_answers_in_eval", checks["distractor_answers_in_eval"]),
            (
                "distractors_in_training_examples",
                len(checks["distractors_in_training_examples"]),
            ),
        )
        if value
    ]
    if violations:
        logger.error("LEAKAGE CHECKS FAILED: %s", ", ".join(violations))
        return 1

    logger.info("\nAll leakage checks passed.")
    logger.info("Wrote %s", (reports_dir / "reasoning_dataset.json").relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
