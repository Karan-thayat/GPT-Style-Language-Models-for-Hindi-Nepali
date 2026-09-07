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
LANGUAGES = {"hi": ("hindi", "Model H"), "ne": ("nepali", "Model L")}

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

        # The split *files* are large artifacts, which the submission instructions
        # require to live on Google Drive rather than in git. What has to be in
        # the repository is the definition that reproduces them: the ratios, the
        # seed, and the assignment rule. Check for that first, and treat the
        # local files as a convenience rather than the deliverable - otherwise
        # this reports a failure for doing exactly what the instructions ask.
        processed = REPO_ROOT / directory / "data/processed"
        splits = {s: processed / f"{language}_{s}.jsonl" for s in ("train", "validation", "test")}
        present_locally = [s for s, p in splits.items() if p.exists() and p.stat().st_size > 0]

        manifest_path = REPO_ROOT / directory / "reports" / "corpus_manifest.json"
        defined = False
        detail = "no split definition in corpus_manifest.json"
        if manifest_path.exists():
            config = read_json(manifest_path).get("config", {})
            ratios = config.get("splits", {})
            defined = {"train", "validation", "test"} <= set(ratios)
            if defined:
                detail = (
                    f"train {ratios['train']:.0%} / val {ratios['validation']:.0%} / "
                    f"test {ratios['test']:.0%}, seed {ratios.get('seed')}"
                )
                if len(present_locally) < 3:
                    detail += " (files on Drive, per the submission instructions)"

        checklist.add(
            f"4. Train/val/test splits ({model}, {language})",
            defined,
            detail,
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

        # The assignment states the floor in tokens: "at least 20% of the final
        # training tokens must come from manual collection". Measure it that way.
        # The two differ - manual prose tokenizes at a different rate to the
        # downloaded web text - so the word-based figure is only a fallback for
        # when the corpus has not been tokenized yet.
        tokens_path = reports / "token_counts.json"
        manual_fraction = None
        unit = ""
        if tokens_path.exists():
            counts = read_json(tokens_path)
            by_collection = counts.get("by_collection") or {}
            total = counts.get("train_tokens") or sum(by_collection.values())
            if total:
                manual_fraction = by_collection.get("manual", 0) / total
                unit = "of training tokens"
        if manual_fraction is None:
            stats_path = reports / "corpus_stats.json"
            if stats_path.exists():
                stats = read_json(stats_path)
                manual_fraction = stats["collection_split"]["manual_word_fraction"]
                unit = "of words (corpus not tokenized yet; the requirement is in tokens)"
        if manual_fraction is not None:
            checklist.add(
                f"Manual collection >= 20% ({model}, {language})",
                manual_fraction >= MANUAL_REQUIREMENT,
                f"{100 * manual_fraction:.1f}% {unit}",
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


def check_tokenizer_report(checklist: Checklist) -> None:
    """Verify the tokenizer reporting the assignment asks for in §1.3.

    Existence of a ``.model`` file is not evidence that anything was reported
    about it: §1.3 requires vocabulary size, token-frequency statistics, average
    characters per token, tokenization examples, and unknown-token statistics. A
    tokenizer trained in a different session can leave the model on disk with no
    measurements behind it, and the report then renders empty cells while every
    file-existence check still passes - which is exactly what happened once.
    """
    print("\n[Tokenizer reporting]")

    import hashlib

    for language, (directory, model) in LANGUAGES.items():
        stats_path = REPO_ROOT / directory / "reports/tokenizer_stats.json"
        if not stats_path.exists():
            checklist.add(
                f"Tokenizer metrics reported ({model}, {language})",
                False,
                "tokenizer_stats.json missing - run scripts/train_tokenizer.py --eval-only",
            )
            continue

        stats = read_json(stats_path)
        metrics = stats.get("metrics", {})
        missing = [
            field
            for field in ("vocab_size", "fertility", "unk_rate", "chars_per_token")
            if metrics.get(field) is None
        ]
        if not stats.get("examples"):
            missing.append("examples")
        if not stats.get("token_frequency", {}).get("top_tokens"):
            missing.append("token_frequency")
        checklist.add(
            f"Tokenizer metrics reported ({model}, {language})",
            not missing,
            f"missing: {missing}" if missing else
            f"fertility {metrics['fertility']:.3f}, unk {metrics['unk_rate']:.1e}, "
            f"{metrics['chars_per_token']:.2f} chars/token",
        )

        # Sub-word tokenization is a stated requirement, not a nicety: a fertility
        # at or below 1.0 means words are not being split at all.
        if metrics.get("fertility"):
            checklist.add(
                f"Fertility > 1 (sub-word) ({model}, {language})",
                metrics["fertility"] > 1.0,
                f"{metrics['fertility']:.3f} tokens/word",
            )

        # The metrics must describe the tokenizer that is actually shipped.
        recorded = stats.get("training_manifest", {}).get("model_sha256")
        model_file = Path(stats.get("training_manifest", {}).get("model_file", ""))
        if recorded and model_file.exists():
            digest = hashlib.sha256()
            with model_file.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            matches = digest.hexdigest() == recorded
            checklist.add(
                f"Metrics match the shipped tokenizer ({model}, {language})",
                matches,
                "" if matches else "tokenizer was refitted after these metrics were measured",
            )

    # Finally, the rendered report must not still be showing the placeholders.
    report = REPO_ROOT / "report/phase1_report.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        placeholders = [
            phrase
            for phrase in ("_Tokenizer has not been trained._", "_Sweep has not been run._")
            if phrase in text
        ]
        checklist.add(
            "Report has no unrendered tokenizer sections",
            not placeholders,
            f"still present: {placeholders}" if placeholders else "",
        )


def check_constraints(checklist: Checklist) -> None:
    """Verify the assignment's hard constraints."""
    print("\n[Hard constraints]")

    # No pretrained models or HuggingFace Transformer classes anywhere in the code.
    banned = ("transformers", "AutoTokenizer", "AutoModel", "from_pretrained")
    offenders: list[str] = []

    # Parse rather than grep. A textual scan cannot tell a real import from the
    # same name written inside a string - the Phase 2 checker holds the literal
    # "nn.Transformer" precisely in order to search for it, and a grep flags that
    # as a violation. Walking the AST looks only at code that actually executes.
    import ast

    for path in list(REPO_ROOT.glob("lma/**/*.py")) + list(REPO_ROOT.glob("scripts/*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, ast.Name):
                names = [node.id]

            for name in names:
                for token in banned:
                    if name == token or name.startswith(f"{token}."):
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}"
                        )
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
    check_tokenizer_report(checklist)
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
