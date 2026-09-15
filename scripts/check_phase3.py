#!/usr/bin/env python3
"""Check the Phase 3 deliverables and constraints against the assignment.

    python scripts/check_phase3.py

Mirrors ``check_phase1.py`` and ``check_phase2.py``: it walks the six
deliverables section 3 lists, the requirements, and the hard constraints, and
reports what is actually on disk rather than asserting the phase is finished.
Exits non-zero while anything is outstanding, so it can gate a push.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LANGUAGES = {"hi": ("hindi", "Model H"), "ne": ("nepali", "Model L")}


class Checklist:
    """Collects pass/fail results and prints them in one block."""

    def __init__(self) -> None:
        """Start with no results."""
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        """Record one check."""
        self.rows.append((name, ok, detail))

    def report(self) -> int:
        """Print every check and return the process exit code."""
        print("Phase 3 deliverable and constraint check")
        print("=" * 72)
        print()
        for name, ok, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}]  {name:<48} {detail}")

        passed = sum(1 for _, ok, _ in self.rows if ok)
        print()
        print("=" * 72)
        print(f"{passed}/{len(self.rows)} checks passed")

        outstanding = [(n, d) for n, ok, d in self.rows if not ok]
        if outstanding:
            print("\nOutstanding:")
            for name, detail in outstanding:
                print(f"  - {name}{': ' + detail if detail else ''}")
        return 0 if not outstanding else 1


def read_json(path: Path) -> dict:
    """Read a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def check_deliverables(checklist: Checklist) -> None:
    """The six items section 3 lists as deliverables."""
    for name, path in (
        ("1. Reasoning data generation script", "scripts/generate_reasoning_data.py"),
        ("1. Finetuning script", "scripts/finetune_reasoning.py"),
        ("1. Reasoning evaluation script", "scripts/evaluate_reasoning.py"),
        ("1. Reasoning data generator module", "lma/reasoning/generate.py"),
        ("1. Finetuning module", "lma/reasoning/finetune.py"),
    ):
        exists = (REPO_ROOT / path).exists()
        checklist.add(name, exists, "" if exists else f"{path} missing")

    for code, (directory, label) in LANGUAGES.items():
        # The splits are deliberately not committed: they are JSONL, the
        # submission instructions send large artifacts to Drive, and the corpus
        # is a pure function of the seed. So a fresh clone legitimately has no
        # data directory, and demanding the files would fail every checkout.
        # What must hold is that the dataset is *reproducible*: either the files
        # are present, or the generator and its recorded statistics are.
        data_dir = REPO_ROOT / directory / "data" / "reasoning"
        splits = [
            data_dir / f"{code}_reasoning_{split}.jsonl"
            for split in ("train", "validation", "test")
        ]
        present = [p for p in splits if p.exists()]
        generator = REPO_ROOT / "scripts" / "generate_reasoning_data.py"
        statistics = REPO_ROOT / directory / "reports" / "reasoning_dataset.json"
        reproducible = generator.exists() and statistics.exists()

        if len(present) == 3:
            detail = "3 splits on disk"
        elif reproducible:
            detail = (
                f"not generated here - run: python scripts/generate_reasoning_data.py "
                f"--language {code}"
            )
        else:
            detail = f"{len(present)}/3 splits, and the generator is missing"

        checklist.add(
            f"1. Reasoning splits ({label})",
            len(present) == 3 or reproducible,
            detail,
        )

        config = REPO_ROOT / directory / "configs" / "finetune.yaml"
        checklist.add(
            f"1. Finetuning config ({label})",
            config.exists(),
            "" if config.exists() else f"{config.relative_to(REPO_ROOT)} missing",
        )

        record = REPO_ROOT / directory / "reports" / "finetune_record.json"
        detail = ""
        ok = record.exists()
        if ok:
            data = read_json(record)
            detail = f"{data.get('steps', 0):,} steps, best val {data.get('best_val_loss', 0):.4f}"
        checklist.add(f"3. Finetuning log ({label})", ok, detail or "not run yet")

        evaluation = REPO_ROOT / directory / "reports" / "phase3_evaluation.json"
        checklist.add(
            f"3. Reasoning metrics ({label})",
            evaluation.exists(),
            "" if evaluation.exists() else "phase3_evaluation.json missing",
        )

    for code, (_, label) in LANGUAGES.items():
        figures = REPO_ROOT / "report" / "figures"
        pre = figures / f"reasoning_attention_pretrained_{code}.png"
        post = figures / f"reasoning_attention_finetuned_{code}.png"
        checklist.add(
            f"4. Pretrain vs finetune attention ({label})",
            pre.exists() and post.exists(),
            "" if pre.exists() and post.exists() else "heatmaps missing",
        )

    report = REPO_ROOT / "report" / "phase3_report.md"
    checklist.add(
        "6. Final consolidated report",
        report.exists(),
        "" if report.exists() else "run scripts/make_phase3_report.py",
    )


def check_requirements(checklist: Checklist) -> None:
    """Dataset properties and reasoning metrics the assignment requires."""
    for code, (directory, label) in LANGUAGES.items():
        dataset = REPO_ROOT / directory / "reports" / "reasoning_dataset.json"
        if dataset.exists():
            stats = read_json(dataset)
            checks = stats.get("leakage_checks", {})

            clean = (
                checks.get("chain_signature_overlap", 1) == 0
                and checks.get("prompt_overlap_train_eval", 1) == 0
                and checks.get("duplicate_prompts", 1) == 0
                and checks.get("distractor_answers_in_eval", 1) == 0
            )
            checklist.add(f"Train/test leakage controlled ({label})", clean,
                          "" if clean else str(checks))

            splits = stats.get("by_split", {})
            has_splits = all(
                splits.get(name, 0) > 0 for name in ("train", "validation", "test")
            )
            checklist.add(
                f"Train/validation/test splits ({label})",
                has_splits,
                ", ".join(f"{k}={v:,}" for k, v in sorted(splits.items())),
            )

            variety = stats.get("template_variety", {})
            checklist.add(
                f"Template variety reported ({label})",
                bool(variety),
                f"{variety.get('dimensions', 0)} dimensions, "
                f"{len(variety.get('chain_lengths', []))} chain lengths"
                if variety
                else "",
            )
        else:
            checklist.add(f"Train/test leakage controlled ({label})", False,
                          "reasoning_dataset.json missing")

        evaluation = REPO_ROOT / directory / "reports" / "phase3_evaluation.json"
        if not evaluation.exists():
            continue
        results = read_json(evaluation)

        comparison = results.get("comparison", {}).get("exact_match", {})
        has_both = "pretrained" in comparison and "finetuned" in comparison
        checklist.add(
            f"Pretrained vs finetuned accuracy ({label})",
            has_both,
            f"{comparison.get('pretrained', 0):.3f} -> {comparison.get('finetuned', 0):.3f}"
            if has_both
            else "",
        )

        checklist.add(
            f"Chance baseline reported ({label})",
            "chance_baseline" in results,
            f"{results.get('chance_baseline', {}).get('overall', 0):.3f}",
        )
        # The error analysis must cover every mistake, not a stored sample: an
        # earlier revision examined 40 predictions and reported zeros for a model
        # whose handful of errors fell outside them.
        finetuned = results.get("finetuned", {})
        analysis = results.get("error_analysis", {}).get("finetuned", {})
        covered = (
            "errors" in finetuned
            and analysis.get("errors_examined") == finetuned.get("errors")
        )
        checklist.add(
            f"Error analysis covers every error ({label})",
            covered,
            f"{analysis.get('errors_examined', 0)} of {finetuned.get('errors', '?')} errors"
            if "errors" in finetuned else "re-run scripts/evaluate_reasoning.py",
        )

        samples = finetuned.get("successes") or []
        failures = finetuned.get("failures")
        checklist.add(
            f"Qualitative examples recorded ({label})",
            bool(samples) and (failures is not None),
            f"{len(samples)} successes, {len(failures or [])} failures",
        )

        attention = results.get("attention", {})
        prompts = attention.get("finetuned", {}).get("prompts", 0)
        checklist.add(
            f"Attention compared before and after ({label})",
            "pretrained" in attention and "finetuned" in attention and prompts >= 50,
            f"averaged over {prompts} prompts",
        )
        answer_figure = REPO_ROOT / "report" / "figures" / f"reasoning_answer_attention_{code}.png"
        checklist.add(
            f"Answer-attention analysis ({label})",
            "answer_attention" in attention.get("finetuned", {}) and answer_figure.exists(),
            "" if answer_figure.exists() else "figure missing",
        )


def check_constraints(checklist: Checklist) -> None:
    """Independence, provenance, and the Drive links the instructions require."""
    records = {}
    for code, (directory, _) in LANGUAGES.items():
        path = REPO_ROOT / directory / "reports" / "finetune_record.json"
        if path.exists():
            records[code] = read_json(path)

    if len(records) == 2:
        hi, ne = records["hi"], records["ne"]
        checklist.add(
            "Finetuned from separate pretrained checkpoints",
            hi.get("pretrained_checkpoint") != ne.get("pretrained_checkpoint"),
        )
        checklist.add(
            "Identical architecture (data is the variable)",
            hi.get("model_config") == ne.get("model_config"),
        )

    # Reasoning data must not be shared between the two languages.
    texts = {}
    for code, (directory, _) in LANGUAGES.items():
        path = REPO_ROOT / directory / "data" / "reasoning" / f"{code}_reasoning_train.jsonl"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                texts[code] = {json.loads(line)["text"] for line in handle if line.strip()}
    if len(texts) == 2:
        shared = texts["hi"] & texts["ne"]
        checklist.add(
            "No reasoning examples shared between languages",
            not shared,
            f"{len(shared)} shared" if shared else "",
        )

    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    rows = re.findall(r"\|\s*Model [HL][^|]*finetuned checkpoint[^\n]*", text, re.I)
    has_links = bool(rows) and all("drive.google.com" in row for row in rows)
    checklist.add(
        "Finetuned checkpoint Drive links in README",
        has_links,
        "" if has_links else "add links for the finetuned checkpoints",
    )

    tests = REPO_ROOT / "scripts" / "test_reasoning.py"
    checklist.add("Component test suite", tests.exists())


def main() -> int:
    """Run every Phase 3 check and print the summary."""
    checklist = Checklist()
    check_deliverables(checklist)
    check_requirements(checklist)
    check_constraints(checklist)
    return checklist.report()


if __name__ == "__main__":
    raise SystemExit(main())
