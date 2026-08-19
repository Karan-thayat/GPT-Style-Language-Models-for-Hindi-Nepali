#!/usr/bin/env python3
"""Verify that the two corpora and tokenizers are genuinely independent.

The project's hard constraint is that Model H and Model L share no data,
vocabulary, or weights. That is easy to claim and easy to violate silently -
especially here, where the two languages share a script and are sourced from
overlapping data providers. This script checks it as a property of the artifacts
on disk rather than trusting the pipeline's intent:

1. **No shared documents** - the exact-text hash sets of the two corpora must be
   disjoint, checked across every split.
2. **No cross-split leakage within a language** - train must not share documents
   with validation or test, or Phase 2's held-out metrics would be meaningless.
3. **Separate tokenizer files** - distinct model files with distinct checksums.
4. **Non-identical vocabularies** - beyond being separate files, the vocabularies
   must actually differ; identical content would indicate one was copied.
5. **Language purity** - a sample of each corpus is re-checked with the language
   identifier, so contamination is quantified rather than assumed absent.

Exits non-zero if any check fails, so it can gate a commit.

Usage::

    python scripts/validate_independence.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lma.data.language_id import IndicLanguageIdentifier
from lma.data.splits import SPLIT_NAMES, iter_jsonl
from lma.utils.logging_utils import setup_logging, write_json

#: Language code -> (directory, human-readable model name). Insertion order is
#: Model H then Model L, and the checks below unpack it rather than naming codes
#: literally, so the language pair can change in one place.
LANGUAGES = {"hi": ("hindi", "Model H (Hindi)"), "ne": ("nepali", "Model L (Nepali)")}

_FAILURES: list[str] = []


def record(logger, name: str, passed: bool, detail: str = "") -> bool:
    """Log one check result and remember failures."""
    if passed:
        logger.info("  PASS  %s", name)
    else:
        logger.error("  FAIL  %s  %s", name, detail)
        _FAILURES.append(f"{name}: {detail}")
    return passed


def document_hashes(path: Path) -> set[bytes]:
    """Return the set of whitespace-normalized text hashes in a JSONL split."""
    hashes: set[bytes] = set()
    for row in iter_jsonl(path):
        hashes.add(hashlib.sha256(" ".join(row["text"].split()).encode("utf-8")).digest())
    return hashes


def file_checksum(path: Path) -> str:
    """Return a file's SHA-256 hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Run every independence check and report the outcome."""
    logger = setup_logging(None, name="validate_independence")
    logger.info("=" * 68)
    logger.info("Corpus and tokenizer independence validation")
    logger.info("=" * 68)

    report: dict = {"checks": {}, "corpus": {}, "tokenizers": {}}

    # --- gather per-language document hashes ---------------------------------
    logger.info("\n[1] Document overlap between the two corpora")
    per_language: dict[str, dict[str, set[bytes]]] = {}
    for language, (directory, label) in LANGUAGES.items():
        processed = REPO_ROOT / directory / "data/processed"
        splits: dict[str, set[bytes]] = {}
        for split in SPLIT_NAMES:
            path = processed / f"{language}_{split}.jsonl"
            splits[split] = document_hashes(path) if path.exists() else set()
        per_language[language] = splits
        total = sum(len(v) for v in splits.values())
        logger.info(
            "  %s: %s unique documents (%s)",
            label, f"{total:,}",
            ", ".join(f"{s}={len(v):,}" for s, v in splits.items()),
        )
        report["corpus"][language] = {s: len(v) for s, v in splits.items()}

    # Keyed off LANGUAGES rather than literal codes, so changing the language pair
    # cannot leave this comparing a language that is no longer in the project.
    first, second = LANGUAGES
    first_all = set().union(*per_language[first].values())
    second_all = set().union(*per_language[second].values())
    shared = first_all & second_all
    record(
        logger,
        "corpora share no documents",
        not shared,
        f"{len(shared):,} documents appear in BOTH corpora",
    )
    report["checks"]["shared_documents"] = len(shared)

    # --- cross-split leakage within each language ----------------------------
    logger.info("\n[2] Train/held-out leakage within each language")
    for language, (_, label) in LANGUAGES.items():
        splits = per_language[language]
        train = splits["train"]
        for held in ("validation", "test"):
            overlap = train & splits[held]
            record(
                logger,
                f"[{language}] train disjoint from {held}",
                not overlap,
                f"{len(overlap):,} documents in both train and {held}",
            )
            report["checks"][f"{language}_train_{held}_overlap"] = len(overlap)
        val_test = splits["validation"] & splits["test"]
        record(logger, f"[{language}] validation disjoint from test", not val_test,
               f"{len(val_test):,} shared")
        report["checks"][f"{language}_validation_test_overlap"] = len(val_test)

    # --- tokenizer separation ------------------------------------------------
    logger.info("\n[3] Tokenizer and vocabulary separation")
    models: dict[str, Path] = {}
    vocabs: dict[str, Path] = {}
    for language, (directory, label) in LANGUAGES.items():
        tokenizer_dir = REPO_ROOT / directory / "tokenizer"
        found = sorted(tokenizer_dir.glob(f"{language}_*.model"))
        if not found:
            record(logger, f"[{language}] tokenizer model exists", False, f"none in {tokenizer_dir}")
            continue
        model = max(found, key=lambda p: p.stat().st_mtime)
        models[language] = model
        vocab = model.with_suffix(".vocab")
        if vocab.exists():
            vocabs[language] = vocab
        logger.info("  %s: %s", label, model.relative_to(REPO_ROOT))
        report["tokenizers"][language] = {
            "model_file": str(model.relative_to(REPO_ROOT)),
            "model_sha256": file_checksum(model),
        }

    if len(models) == 2:
        record(logger, "tokenizer model files are distinct paths",
               models[first].resolve() != models[second].resolve(), "same file used for both")
        checksums = {lang: report["tokenizers"][lang]["model_sha256"] for lang in models}
        record(logger, "tokenizer models differ in content",
               checksums[first] != checksums[second], "identical checksums - one is a copy")

    if len(vocabs) == 2:
        pieces = {
            lang: {line.split("\t")[0] for line in path.read_text(encoding="utf-8").splitlines() if line}
            for lang, path in vocabs.items()
        }
        record(logger, "vocabularies are not identical", pieces[first] != pieces[second],
               "the two vocabularies contain exactly the same pieces")

        # Hindi and Nepali share the Devanagari script, so unlike a cross-script
        # pair some overlap is *expected* - both vocabularies will contain common
        # Devanagari characters and frequent shared pieces. What must not happen is
        # near-identity, which would mean one vocabulary was copied from the other
        # rather than fitted on its own corpus. The threshold is therefore a
        # near-duplicate test, not an "overlap must be zero" test.
        overlap = len(pieces[first] & pieces[second])
        union = len(pieces[first] | pieces[second])
        jaccard = overlap / union if union else 0.0
        logger.info(
            "  Vocabulary piece overlap: %s pieces, Jaccard %.3f "
            "(the two languages share a script, so partial overlap is expected; "
            "a value near 1.0 would indicate one vocabulary was copied)",
            f"{overlap:,}", jaccard,
        )
        record(logger, "vocabularies are not near-duplicates", jaccard < 0.95, f"Jaccard {jaccard:.3f}")
        report["checks"]["vocab_jaccard"] = jaccard

    # --- language purity -----------------------------------------------------
    logger.info("\n[4] Language purity of each corpus (sampled)")
    identifier = IndicLanguageIdentifier()
    for language, (directory, label) in LANGUAGES.items():
        path = REPO_ROOT / directory / "data/processed" / f"{language}_train.jsonl"
        if not path.exists():
            continue
        # Every label the identifier can return, so a document identified as a
        # language outside this project's pair is counted rather than crashing.
        counts = {code: 0 for code in ("hi", "ne", "bn", "as", "unknown")}
        for index, row in enumerate(iter_jsonl(path)):
            if index >= 2000:
                break
            verdict = identifier.identify(row["text"]).language
            counts[verdict] = counts.get(verdict, 0) + 1
        total = sum(counts.values()) or 1
        correct = counts[language] / total
        logger.info("  %s: %.1f%% identified as %s (%s)", label, 100 * correct, language, counts)
        record(logger, f"[{language}] corpus is >=95% the intended language", correct >= 0.95,
               f"only {100 * correct:.1f}%")
        report["checks"][f"{language}_purity"] = correct

    # --- summary -------------------------------------------------------------
    logger.info("\n" + "=" * 68)
    if _FAILURES:
        logger.error("INDEPENDENCE VALIDATION FAILED (%d issue(s)):", len(_FAILURES))
        for failure in _FAILURES:
            logger.error("  - %s", failure)
    else:
        logger.info("All independence checks passed.")

    report["passed"] = not _FAILURES
    report["failures"] = _FAILURES
    write_json(REPO_ROOT / "report/independence_validation.json", report)
    logger.info("Wrote report/independence_validation.json")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
