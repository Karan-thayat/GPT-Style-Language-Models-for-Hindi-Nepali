"""Check the Phase 2 deliverables and constraints against the assignment.

    python scripts/check_phase2.py

Mirrors ``check_phase1.py``: it walks the eight deliverables section 2 lists, the
quantitative requirements, and the hard constraints, and reports what is present
rather than asserting the phase is finished.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Roughly 25M trainable parameters per model, per section 2.1.
PARAMETER_TARGET = 25_000_000
PARAMETER_TOLERANCE = 0.15

LANGUAGES = {"hi": ("hindi", "Model H"), "ne": ("nepali", "Model L")}


class Checklist:
    """Collects pass/fail results and prints them in one block."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, ok, detail))

    def report(self) -> int:
        print("Phase 2 deliverable and constraint check")
        print("=" * 72)
        print()
        for name, ok, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}]  {name:<46} {detail}")

        passed = sum(1 for _, ok, _ in self.rows if ok)
        print()
        print("=" * 72)
        print(f"{passed}/{len(self.rows)} checks passed")

        outstanding = [(n, d) for n, ok, d in self.rows if not ok]
        if outstanding:
            print("\nOutstanding:")
            for name, detail in outstanding:
                print(f"  - {name}: {detail}")
        return 0 if not outstanding else 1


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_deliverables(checklist: Checklist) -> None:
    """The eight items section 2 lists as deliverables."""
    model_src = REPO_ROOT / "lma" / "model" / "transformer.py"
    source = model_src.read_text(encoding="utf-8") if model_src.exists() else ""

    checklist.add(
        "1. Transformer implementation (MHA, positions, mask)",
        all(k in source for k in ("CausalSelfAttention", "position_embedding", "causal_mask")),
        "" if source else "lma/model/transformer.py missing",
    )

    # The architecture must be built from primitives, not assembled from
    # pre-built blocks - the constraint section 2.1 is most explicit about.
    banned = [p for p in ("nn.Transformer", "nn.MultiheadAttention", "transformers.") if p in source]
    checklist.add(
        "   built from primitive layers only",
        not banned,
        f"found {banned}" if banned else "",
    )

    for code, (directory, label) in LANGUAGES.items():
        config = REPO_ROOT / directory / "configs" / "model.yaml"
        checklist.add(f"2. Model configuration ({label})", config.exists(),
                      "" if config.exists() else f"{config.relative_to(REPO_ROOT)} missing")

    for code, (directory, label) in LANGUAGES.items():
        history = REPO_ROOT / directory / "reports" / "training_history.json"
        detail = ""
        ok = history.exists()
        if ok:
            data = read_json(history)
            detail = f"{data.get('step', 0):,} steps, {len(data.get('history', []))} evaluations"
            ok = bool(data.get("history"))
        checklist.add(f"4. Training log ({label})", ok, detail or "missing")

    curves = REPO_ROOT / "report" / "figures" / "phase2_loss_curves.png"
    checklist.add("4. Loss curves", curves.exists(),
                  "" if curves.exists() else "report/figures/phase2_loss_curves.png missing")

    checklist.add(
        "5. Evaluation scripts",
        (REPO_ROOT / "scripts" / "evaluate_model.py").exists(),
        "",
    )

    for code, (directory, label) in LANGUAGES.items():
        evaluation = REPO_ROOT / directory / "reports" / "phase2_evaluation.json"
        checklist.add(f"5. Evaluation results ({label})", evaluation.exists(),
                      "" if evaluation.exists() else "phase2_evaluation.json missing")

    # Section 2.3 asks for heatmaps as figures, not only as tables.
    for code, (_, label) in LANGUAGES.items():
        heatmap = REPO_ROOT / "report" / "figures" / f"attention_heatmaps_{code}.png"
        checklist.add(f"7. Attention heatmaps ({label})", heatmap.exists(),
                      "" if heatmap.exists() else f"attention_heatmaps_{code}.png missing")

    report = REPO_ROOT / "report" / "phase2_report.md"
    checklist.add("8. Phase 2 report", report.exists(), "")


def check_requirements(checklist: Checklist) -> None:
    """Parameter budget and the metrics section 2.3 requires."""
    for code, (directory, label) in LANGUAGES.items():
        path = REPO_ROOT / directory / "reports" / "phase2_evaluation.json"
        if not path.exists():
            continue
        data = read_json(path)

        params = data.get("parameters", 0)
        within = abs(params - PARAMETER_TARGET) / PARAMETER_TARGET <= PARAMETER_TOLERANCE
        checklist.add(f"~25M parameters ({label})", within, f"{params:,}")

        lm = data.get("language_modeling", {})
        checklist.add(
            f"Perplexity and BPB reported ({label})",
            "perplexity" in lm and "bits_per_byte" in lm,
            f"ppl {lm.get('perplexity', 0):.2f}, BPB {lm.get('bits_per_byte', 0):.4f}",
        )

        generation = data.get("generation", {})
        expected = {"greedy", "temperature_0.5", "temperature_1.0", "temperature_1.5"}
        checklist.add(
            f"All four decoding settings ({label})",
            expected <= set(generation),
            f"{len(generation)} settings",
        )

        first = next(iter(generation.values()), {})
        checklist.add(
            f"BLEU, chrF, ROUGE-L, diversity ({label})",
            all(k in first for k in ("bleu", "chrf", "rouge_l", "repetition_rate",
                                     "distinct_1", "distinct_2")),
            "",
        )

        attention = data.get("attention", {})
        checklist.add(
            f"Attention entropy and distance ({label})",
            "layer_mean_entropy" in attention and "layer_mean_distance" in attention,
            f"{len(attention.get('layer_mean_entropy', []))} layers",
        )


def check_constraints(checklist: Checklist) -> None:
    """Independence, and the Drive links the submission instructions require."""
    hi = REPO_ROOT / "hindi" / "reports" / "phase2_evaluation.json"
    ne = REPO_ROOT / "nepali" / "reports" / "phase2_evaluation.json"
    if hi.exists() and ne.exists():
        a, b = read_json(hi), read_json(ne)
        checklist.add(
            "Models trained independently",
            a.get("checkpoint") != b.get("checkpoint"),
            "separate checkpoints",
        )
        checklist.add(
            "Identical architecture (data is the variable)",
            a.get("model_config") == b.get("model_config"),
            "",
        )

    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    # The instructions require Drive links for the pretrained checkpoints.
    has_links = bool(re.search(r"drive\.google\.com\S*\)", text)) and "checkpoint" in text.lower()
    unfilled = text.count("_pending_") + text.count("_to be added_")
    checklist.add(
        "Checkpoint Drive links in README",
        has_links and unfilled == 0,
        f"{unfilled} placeholder(s) remain" if unfilled else "",
    )

    tests = REPO_ROOT / "scripts" / "test_model.py"
    checklist.add("Component test suite", tests.exists(), "")


def main() -> int:
    checklist = Checklist()
    check_deliverables(checklist)
    check_requirements(checklist)
    check_constraints(checklist)
    return checklist.report()


if __name__ == "__main__":
    raise SystemExit(main())
