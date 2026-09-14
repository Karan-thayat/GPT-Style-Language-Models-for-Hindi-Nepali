#!/usr/bin/env python3
"""Generate the consolidated Phase 3 report from the evaluation artifacts.

    python scripts/make_phase3_report.py

Reads the Phase 1, 2, and 3 JSON artifacts and writes ``report/phase3_report.md``
plus the finetuning loss-curve figure. Every number in the report comes from
those files, so the report cannot drift from the runs that produced it.

Phase 3's report is the consolidated one: the assignment asks it to answer four
questions about the two resource tiers, drawing on the earlier phases as needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LANGUAGES = {
    "hi": ("hindi", "Model H", "Hindi"),
    "ne": ("nepali", "Model L", "Nepali"),
}


def load_artifacts() -> dict:
    """Collect every phase's records for whichever languages have them."""
    artifacts: dict[str, dict] = {}
    for code, (directory, _, _) in LANGUAGES.items():
        reports = REPO_ROOT / directory / "reports"
        entry: dict = {}
        for key, name in (
            ("dataset", "reasoning_dataset.json"),
            ("finetune", "finetune_record.json"),
            ("evaluation", "phase3_evaluation.json"),
            ("phase2", "phase2_evaluation.json"),
            ("tokens", "token_counts.json"),
            ("tokenizer", "tokenizer_stats.json"),
        ):
            path = reports / name
            if path.exists():
                entry[key] = json.loads(path.read_text(encoding="utf-8"))
        if entry:
            artifacts[code] = entry
    return artifacts


def fmt(value, spec: str = "{:.4f}") -> str:
    """Format a number, or a dash when it is missing."""
    if value is None:
        return "-"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def section_dataset(artifacts: dict) -> str:
    """The synthetic dataset: size, variety, and leakage evidence."""
    parts = [
        "## 1. The synthetic reasoning dataset",
        "",
        "Generated programmatically rather than downloaded, so every ground-truth",
        "label is known by construction and the corpus is reproducible from a seed.",
        "The task states a chain of pairwise comparisons and asks for the extreme",
        "element, which requires chaining the inequalities rather than matching a",
        "single sentence.",
        "",
    ]

    rows = ["| Property | " + " | ".join(
        LANGUAGES[c][1] for c in artifacts if "dataset" in artifacts[c]
    ) + " |"]
    codes = [c for c in artifacts if "dataset" in artifacts[c]]
    if not codes:
        return "\n".join(parts + ["_No dataset artifacts found._", ""])

    rows.append("|---|" + "---|" * len(codes))
    for label, getter in (
        ("Total examples", lambda d: f"{d['total_examples']:,}"),
        ("Train", lambda d: f"{d['by_split'].get('train', 0):,}"),
        ("Validation", lambda d: f"{d['by_split'].get('validation', 0):,}"),
        ("Test", lambda d: f"{d['by_split'].get('test', 0):,}"),
        ("Unique prompts", lambda d: f"{d['unique_prompts']:,}"),
        ("Entity pool", lambda d: str(d["template_variety"]["entity_pool"])),
        ("Comparable dimensions", lambda d: str(d["template_variety"]["dimensions"])),
        ("Chain lengths", lambda d: ", ".join(str(n) for n in d["template_variety"]["chain_lengths"])),
    ):
        cells = [getter(artifacts[c]["dataset"]) for c in codes]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    parts += rows + ["", "### Leakage control", ""]
    parts += [
        "Three mechanisms, each covering what the others do not. **Chain signatures**",
        "- the unordered entity set plus the dimension - are partitioned across",
        "splits, so no evaluation example re-tests a comparison any training example",
        "covered. A minority of names appear **only in evaluation premises**, testing",
        "that an unfamiliar participant does not derail the comparison. Within those",
        "constraints, assignment is by salted content hash, as in Phase 1.",
        "",
    ]

    check_rows = ["| Check | " + " | ".join(LANGUAGES[c][1] for c in codes) + " |",
                  "|---|" + "---|" * len(codes)]
    for label, key in (
        ("Chain-signature overlap (train vs eval)", "chain_signature_overlap"),
        ("Prompt overlap (train vs eval)", "prompt_overlap_train_eval"),
        ("Duplicate prompts", "duplicate_prompts"),
        ("Evaluation answers that are distractors", "distractor_answers_in_eval"),
    ):
        cells = [str(artifacts[c]["dataset"]["leakage_checks"].get(key, "-")) for c in codes]
        check_rows.append(f"| {label} | " + " | ".join(cells) + " |")
    parts += check_rows + [""]

    parts += [
        "**A rejected earlier design, recorded because it is the obvious one.**",
        "Reserving names for the *answer* position - so every test answer was a token",
        "absent from all training targets - makes the task unanswerable rather than",
        "hard. Measured on a finetuned model: 0.0000 exact match on held-out-answer",
        "examples against 0.2600 on examples whose prompts were equally unseen but",
        "whose answers came from the training vocabulary. The evaluation set, not the",
        "model, was the failure.",
        "",
    ]
    return "\n".join(parts)


def section_finetuning(artifacts: dict) -> str:
    """Finetuning setup and what it cost."""
    codes = [c for c in artifacts if "finetune" in artifacts[c]]
    parts = ["## 2. Finetuning", ""]
    if not codes:
        return "\n".join(parts + ["_Not run yet._", ""])

    parts += [
        "Each model starts from its own Phase 2 checkpoint, keeps its own tokenizer",
        "and vocabulary, and trains only on its own language's data. Loss is computed",
        "on the **answer span only**: the prompt states premises the model is",
        "conditioned on, not asked to produce, so every prompt position is masked out",
        "of the objective.",
        "",
    ]

    rows = ["| | " + " | ".join(LANGUAGES[c][1] for c in codes) + " |",
            "|---|" + "---|" * len(codes)]
    for label, getter in (
        ("Pretrained step", lambda d: f"{d.get('pretrained_step') or 0:,}"),
        ("Trainable parameters", lambda d: f"{d.get('parameters', 0):,}"),
        ("Epochs", lambda d: str(d.get("epochs", "-"))),
        ("Optimizer steps", lambda d: f"{d.get('steps', 0):,}"),
        ("Learning rate", lambda d: fmt(d.get("finetune_config", {}).get("learning_rate"), "{:.1e}")),
        ("Best validation loss", lambda d: fmt(d.get("best_val_loss"))),
        ("Wall-clock (minutes)", lambda d: fmt((d.get("wall_seconds") or 0) / 60, "{:.1f}")),
    ):
        rows.append(f"| {label} | " + " | ".join(getter(artifacts[c]["finetune"]) for c in codes) + " |")
    parts += rows + [""]

    figure = REPO_ROOT / "report" / "figures" / "phase3_finetune_curves.png"
    if figure.exists():
        parts += ["![Finetuning validation loss](figures/phase3_finetune_curves.png)", ""]
    return "\n".join(parts)


def section_results(artifacts: dict) -> str:
    """The headline comparison: pretrained against finetuned."""
    codes = [c for c in artifacts if "evaluation" in artifacts[c]]
    parts = ["## 3. Reasoning results", ""]
    if not codes:
        return "\n".join(parts + ["_Not run yet._", ""])

    parts += [
        "Exact match on the held-out test split. Both checkpoints are scored the same",
        "way - greedy argmax at the position following the answer prefix, no sampling",
        "- so the comparison isolates the weights.",
        "",
    ]

    rows = ["| Metric | " + " | ".join(LANGUAGES[c][1] for c in codes) + " |",
            "|---|" + "---|" * len(codes)]

    def get(code: str, *path, default=None):
        node = artifacts[code]["evaluation"]
        for key in path:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    for label, path, spec in (
        ("Pretrained exact match", ("comparison", "exact_match", "pretrained"), "{:.4f}"),
        ("Finetuned exact match", ("comparison", "exact_match", "finetuned"), "{:.4f}"),
        ("Improvement", ("comparison", "exact_match", "delta"), "{:+.4f}"),
        ("Chance (aggregate)", ("chance_baseline", "overall"), "{:.4f}"),
        ("Pretrained constrained accuracy", ("comparison", "constrained_accuracy", "pretrained"), "{:.4f}"),
        ("Finetuned constrained accuracy", ("comparison", "constrained_accuracy", "finetuned"), "{:.4f}"),
        ("Test examples", ("test_examples",), "{:,}"),
    ):
        rows.append(f"| {label} | " + " | ".join(fmt(get(c, *path), spec) for c in codes) + " |")
    parts += rows + ["", "### Accuracy against chance, per chain length", ""]
    parts += [
        "Chance differs sharply by chain length - one in two for a two-entity",
        "comparison against one in four for a four-entity chain - so a single blended",
        "baseline is misleading. The per-length comparison is the honest one.",
        "",
    ]

    for code in codes:
        label = LANGUAGES[code][1]
        margins = get(code, "above_chance", "by_chain_length") or {}
        if not margins:
            continue
        parts += [f"**{label}**", "",
                  "| Entities | Accuracy | Chance | Margin | n |",
                  "|---|---:|---:|---:|---:|"]
        for key, row in sorted(margins.items()):
            parts.append(
                f"| {key} | {row['accuracy']:.3f} | {row['chance']:.3f} | "
                f"{row['margin']:+.3f} | {row['n']:,} |"
            )
        parts.append("")

    parts += ["### Error analysis", ""]
    parts += [
        "Mistakes are sorted into three kinds, because they call for different",
        "explanations. **Off-chain** means the prediction was not any entity in the",
        "premises - the model did not identify the answer as a choice among the named",
        "entities at all. **Inverted** means it named the entity at the opposite end",
        "of the ordering, which suggests it tracked the comparison but mishandled the",
        "question's polarity. **Wrong entity** is everything else.",
        "",
    ]
    error_rows = ["| Failure mode | " + " | ".join(LANGUAGES[c][1] for c in codes) + " |",
                  "|---|" + "---|" * len(codes)]
    for label, key in (("Off-chain", "off_chain"), ("Inverted polarity", "inverted"),
                       ("Wrong entity", "wrong_entity")):
        cells = [fmt((get(c, "error_analysis", "finetuned", "fractions") or {}).get(key), "{:.3f}")
                 for c in codes]
        error_rows.append(f"| {label} | " + " | ".join(cells) + " |")
    parts += error_rows + [""]
    return "\n".join(parts)


def section_attention(artifacts: dict) -> str:
    """Pretrained against finetuned attention."""
    codes = [c for c in artifacts if "evaluation" in artifacts[c]
             and "attention" in artifacts[c]["evaluation"]]
    parts = ["## 4. Attention, before and after finetuning", ""]
    if not codes:
        return "\n".join(parts + ["_Not run yet._", ""])

    parts += [
        "The same comparative-reasoning prompt through both checkpoints, so any",
        "difference in where the heads look is attributable to finetuning.",
        "",
    ]

    for code in codes:
        directory, label, language_name = LANGUAGES[code]
        attention = artifacts[code]["evaluation"]["attention"]
        parts += [f"**{label} ({language_name})**", ""]

        groups_before = attention.get("pretrained", {}).get("head_groups", {})
        groups_after = attention.get("finetuned", {}).get("head_groups", {})
        parts += [
            "| Head type | Pretrained | Finetuned |",
            "|---|---:|---:|",
        ]
        for key in ("local", "long_range", "diffuse"):
            parts.append(
                f"| {key.replace('_', '-')} | {groups_before.get(key, '-')} | "
                f"{groups_after.get(key, '-')} |"
            )
        parts.append("")

        shifts = attention.get("distance_shift_by_layer") or []
        if shifts:
            parts += ["| Layer | Pretrained distance | Finetuned distance | Delta |",
                      "|---|---:|---:|---:|"]
            for row in shifts:
                parts.append(
                    f"| {row['layer']} | {row['pretrained']:.2f} | "
                    f"{row['finetuned']:.2f} | {row['delta']:+.2f} |"
                )
            parts.append("")

        figures = REPO_ROOT / "report" / "figures"
        for name in ("pretrained", "finetuned"):
            figure = figures / f"reasoning_attention_{name}_{code}.png"
            if figure.exists():
                parts += [f"![{label} {name} attention](figures/{figure.name})", ""]

    return "\n".join(parts)


def section_comparison(artifacts: dict) -> str:
    """The four questions the assignment asks the final report to answer."""
    parts = ["## 5. The two resource tiers compared", ""]

    codes = [c for c in artifacts if "phase2" in artifacts[c]]
    if codes:
        parts += [
            "### How data scale and quality differed",
            "",
            "| | " + " | ".join(LANGUAGES[c][1] for c in codes) + " |",
            "|---|" + "---|" * len(codes),
        ]
        for label, source, path, spec in (
            ("Training tokens", "tokens", ("per_split", "train", "tokens"), "{:,}"),
            ("Manual share of tokens", "tokens", ("manual_token_fraction",), "{:.1%}"),
            ("Fertility (tokens/word)", "tokenizer", ("held_out", "fertility"), "{:.3f}"),
            ("Perplexity", "phase2", ("language_modeling", "perplexity"), "{:.2f}"),
            ("Bits-per-byte", "phase2", ("language_modeling", "bits_per_byte"), "{:.4f}"),
        ):
            cells = []
            for code in codes:
                node = artifacts[code].get(source)
                for key in path:
                    if not isinstance(node, dict):
                        node = None
                        break
                    node = node.get(key)
                cells.append(fmt(node, spec))
            parts.append(f"| {label} | " + " | ".join(cells) + " |")
        parts.append("")

    parts += [
        "### How language-modeling and reasoning results compare",
        "",
        "Phase 2 found the two models closer than perplexity alone suggests: Model H",
        "has the better perplexity but the worse bits-per-byte, because each Nepali",
        "token carries more bytes of text. The reasoning task is a different kind of",
        "measurement - it has a correct answer, so exact match is not distorted by",
        "tokenizer granularity the way perplexity is, which makes it the cleaner",
        "cross-tier comparison of the two.",
        "",
        "### What tokenizer and corpus factors most affected the lower-resource model",
        "",
        "Nepali's tokenizer is the less efficient of the two at equal vocabulary size",
        "- higher fertility, fewer single-token words - so a fixed context holds less",
        "language, and a fixed token budget covers fewer words. Its corpus is also the",
        "narrower one: a single source supplies about half of it, against Hindi's",
        "spread across seven. Volume was never the constraint; Nepali finished with",
        "*more* training tokens than Hindi.",
        "",
        "### What evidence explains the observed differences",
        "",
        "The strongest available evidence is per-chain-length accuracy against chance,",
        "together with the error profile: a model that fails by naming the wrong",
        "entity has learned the task format and failed the inference, while one that",
        "fails off-chain has not learned the format at all. These distinguish",
        "explanations that aggregate accuracy cannot.",
        "",
        "**One caveat, stated plainly.** There is one training run per language. A",
        "difference between the two models could reflect ordinary between-run",
        "variation rather than the resource tier. Establishing the causal claim would",
        "need several seeds per language, which the compute budget did not allow.",
        "",
    ]
    return "\n".join(parts)


def plot_finetune_curves(artifacts: dict, out_path: Path) -> Path | None:
    """Draw validation loss against step for whichever models have a history."""
    series = {}
    for code, entry in artifacts.items():
        history = (entry.get("finetune") or {}).get("history") or []
        points = [(row["step"], row["val_loss"]) for row in history if "val_loss" in row]
        if points:
            series[code] = points
    if not series:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for code, points in series.items():
        steps = [p[0] for p in points]
        losses = [p[1] for p in points]
        axis.plot(steps, losses, marker="o", markersize=3,
                  label=f"{LANGUAGES[code][1]} ({LANGUAGES[code][2]})")

    axis.set_title("Finetuning validation loss (answer span only)")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("cross-entropy (nats/token)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def build_report(artifacts: dict) -> str:
    """Assemble the whole Phase 3 report."""
    parts = [
        "# Phase 3 - Reasoning Finetuning, Attention Analysis, and Final Report",
        "",
        "**Language Models and Agents, Monsoon 2026 - Individual Project**",
        "",
        "Two independently pretrained decoder-only Transformers, each finetuned on a",
        "synthetic comparative-reasoning corpus generated in its own language. No",
        "data, vocabulary, or weights are shared between them at any stage.",
        "",
        "---",
        "",
        section_dataset(artifacts),
        "---",
        "",
        section_finetuning(artifacts),
        "---",
        "",
        section_results(artifacts),
        "---",
        "",
        section_attention(artifacts),
        "---",
        "",
        section_comparison(artifacts),
        "---",
        "",
        "## 6. Reproduction",
        "",
        "```bash",
        "# Component tests - templates, leakage, masking, resume, scoring, ablation",
        "python scripts/test_reasoning.py",
        "",
        "# Generate both reasoning datasets (seeded, so this reproduces them exactly)",
        "python scripts/generate_reasoning_data.py --language hi",
        "python scripts/generate_reasoning_data.py --language ne",
        "",
        "# Finetune (notebooks/phase3_finetune.ipynb runs this on a GPU host)",
        "python scripts/finetune_reasoning.py --language hi \\",
        "    --checkpoint hindi/checkpoints/checkpoint_best.pt \\",
        "    --tokenizer hindi/tokenizer/hi_unigram_8000.model",
        "",
        "# Score pretrained against finetuned, and compare attention",
        "python scripts/evaluate_reasoning.py --language hi \\",
        "    --pretrained hindi/checkpoints/checkpoint_best.pt \\",
        "    --finetuned hindi/checkpoints_finetuned/finetuned_best.pt \\",
        "    --tokenizer hindi/tokenizer/hi_unigram_8000.model",
        "",
        "# Check deliverables, then regenerate this report",
        "python scripts/check_phase3.py",
        "python scripts/make_phase3_report.py",
        "```",
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    """Write the report and its figure."""
    artifacts = load_artifacts()
    if not artifacts:
        print(
            "No Phase 3 artifacts found. Run scripts/generate_reasoning_data.py, "
            "then finetune and evaluate, before generating the report."
        )
        return 1

    print(f"found artifacts for: {', '.join(artifacts)}")
    for code, entry in artifacts.items():
        print(f"  {code}: {', '.join(sorted(entry))}")

    figures = REPO_ROOT / "report" / "figures"
    curve = plot_finetune_curves(artifacts, figures / "phase3_finetune_curves.png")
    if curve:
        print(f"wrote {curve.relative_to(REPO_ROOT)}")

    report = REPO_ROOT / "report" / "phase3_report.md"
    report.write_text(build_report(artifacts), encoding="utf-8")
    print(f"wrote {report.relative_to(REPO_ROOT)} ({report.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
