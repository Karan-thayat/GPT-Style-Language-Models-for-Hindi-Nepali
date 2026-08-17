#!/usr/bin/env python3
"""Check every Phase 1 deliverable against what is actually on disk.

The assignment lists seven deliverables and several hard constraints. This script
verifies each one against real artifacts rather than intent, and prints a
pass/fail table. It is meant to be run before pushing to the `phase-1` branch.

Usage::

    python scripts/check_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lma.utils.logging_utils import read_json

#: Language code -> (directory, model label).
LANGUAGES = {"bn": ("bengali", "Model H"), "as": ("assamese", "Model L")}

#: The manual-collection floor stated in the assignment.
MANUAL_REQUIREMENT = 0.20

#: The per-language training-token target stated in the assignment.
TOKEN_TARGET = 500_000_000


class Checklist:
    """Accumulates named check results for a final report."""

    def __init__(self) -> None:
        """Start with no results."""
        self.results: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        """Record one check."""
        self.results.append((name, passed, detail))

    @property
    def failures(self) -> list[tuple[str, bool, str]]:
        """Return only the failed checks."""
        return [r for r in self.results if not r[1]]

    def render(self) -> str:
        """Return a printable table of every check."""
        width = max(len(name) for name, _, _ in self.results) + 2
        lines = []
        for name, passed, detail in self.results:
            mark = "PASS" if passed else "FAIL"
            line = f"  [{mark}]  {name.ljust(width)}"
            if detail:
                line += f" {detail}"
            lines.append(line)
        return "\n".join(lines)


def check_deliverables(checklist: Checklist) -> None:
    """Verify that each of the seven listed deliverables exists."""
    print("\n[Deliverables]")

    code_files = {
        "1. Dataset collection scripts": [
            "lma/data/sources.py", "scripts/build_corpus.py", "scripts/run_ocr.py",
        ],
        "2. Dataset preprocessing pipelines": [
            "lma/data/normalize.py", "lma/data/clean.py", "lma/data/language_id.py",
        ],
        "5. Tokenizer training code": [
            "lma/tokenizer/train_tokenizer.py", "scripts/train_tokenizer.py",
        ],
    }
    for label, paths in code_files.items():
        missing = [p for p in paths if not (REPO_ROOT / p).exists()]
        checklist.add(label, not missing, f"missing: {missing}" if missing else "")

    for language, (directory, model) in LANGUAGES.items():
        reports = REPO_ROOT / directory / "reports"
        stats = reports / "corpus_stats.json"
        checklist.add(
            f"3. Dataset statistics ({model}, {language})",
            stats.exists(),
            "" if stats.exists() else f"missing {stats.relative_to(REPO_ROOT)}",
        )

        processed = REPO_ROOT / directory / "data/processed"
        splits = {s: processed / f"{language}_{s}.jsonl" for s in ("train", "validation", "test")}
        missing = [s for s, p in splits.items() if not p.exists() or p.stat().st_size == 0]
        checklist.add(
            f"4. Train/val/test splits ({model}, {language})",
            not missing,
            f"missing or empty: {missing}" if missing else "",
        )

        tokenizer_dir = REPO_ROOT / directory / "tokenizer"
        vocabs = list(tokenizer_dir.glob(f"{language}_*.vocab"))
        models = list(tokenizer_dir.glob(f"{language}_*.model"))
        checklist.add(f"6. Vocabulary file ({model}, {language})", bool(vocabs),
                      "" if vocabs else f"no .vocab in {tokenizer_dir.relative_to(REPO_ROOT)}")
        checklist.add(f"7. Tokenizer model file ({model}, {language})", bool(models),
                      "" if models else f"no .model in {tokenizer_dir.relative_to(REPO_ROOT)}")


def check_requirements(checklist: Checklist) -> None:
    """Verify the quantitative requirements: manual share and token counts."""
    print("\n[Quantitative requirements]")

    for language, (directory, model) in LANGUAGES.items():
        reports = REPO_ROOT / directory / "reports"

        stats_path = reports / "corpus_stats.json"
        if stats_path.exists():
            stats = read_json(stats_path)
            manual = stats["collection_split"]["manual_word_fraction"]
            checklist.add(
                f"Manual collection >= 20% ({model}, {language})",
                manual >= MANUAL_REQUIREMENT,
                f"{100 * manual:.1f}% of words",
            )

        tokens_path = reports / "token_counts.json"
        if tokens_path.exists():
            tokens = read_json(tokens_path)
            train_tokens = tokens["train_tokens"]
            met = train_tokens >= TOKEN_TARGET
            checklist.add(
                f"Training tokens ~500M ({model}, {language})",
                met,
                f"{train_tokens:,} tokens ({100 * train_tokens / TOKEN_TARGET:.2f}% of target)"
                + ("" if met else " - shortfall must be reported and justified"),
            )
        else:
            checklist.add(
                f"Training tokens ~500M ({model}, {language})",
                False,
                "token_counts.json missing - run scripts/tokenize_corpus.py",
            )


def check_constraints(checklist: Checklist) -> None:
    """Verify the assignment's hard constraints."""
    print("\n[Hard constraints]")

    # No pretrained models or HuggingFace Transformer classes anywhere in the code.
    banned = ("transformers", "AutoTokenizer", "AutoModel", "from_pretrained")
    offenders: list[str] = []
    for path in list(REPO_ROOT.glob("lma/**/*.py")) + list(REPO_ROOT.glob("scripts/*.py")):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            # Only flag real usage, not prose in comments and docstrings explaining
            # that these are deliberately not used.
            for line in text.splitlines():
                stripped = line.strip()
                if token in line and not stripped.startswith("#") and "no pretrained" not in line.lower():
                    if f"import {token}" in line or f"{token}(" in line or f"{token}." in line:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}: {stripped[:60]}")
    checklist.add("No pretrained models / HF Transformer classes", not offenders,
                  f"{offenders[:3]}" if offenders else "")

    # Independence: verified separately, but surfaced here too.
    validation_path = REPO_ROOT / "report/independence_validation.json"
    if validation_path.exists():
        validation = read_json(validation_path)
        checklist.add("Corpora and tokenizers are independent", validation.get("passed", False),
                      "" if validation.get("passed") else f"{validation.get('failures')}")
    else:
        checklist.add("Corpora and tokenizers are independent", False,
                      "run scripts/validate_independence.py")

    # The report itself.
    report = REPO_ROOT / "report/phase1_report.md"
    checklist.add("Phase 1 report present", report.exists(),
                  "" if report.exists() else "run scripts/make_report.py")
    figures = list((REPO_ROOT / "report/figures").glob("*.png"))
    checklist.add("Report figures present", len(figures) >= 5, f"{len(figures)} figure(s)")

    readme = REPO_ROOT / "README.md"
    checklist.add("README with reproduction steps", readme.exists())


def main() -> int:
    """Run every Phase 1 check and print the summary."""
    print("=" * 72)
    print("Phase 1 deliverable and constraint check")
    print("=" * 72)

    checklist = Checklist()
    check_deliverables(checklist)
    check_requirements(checklist)
    check_constraints(checklist)

    print("\n" + checklist.render())
    print("\n" + "=" * 72)
    failures = checklist.failures
    passed = len(checklist.results) - len(failures)
    print(f"{passed}/{len(checklist.results)} checks passed")

    if failures:
        print("\nOutstanding:")
        for name, _, detail in failures:
            print(f"  - {name}{': ' + detail if detail else ''}")
        print(
            "\nNote: a token-count shortfall is acceptable if reported and justified "
            "(the assignment allows this explicitly for the lower-resource language)."
        )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
