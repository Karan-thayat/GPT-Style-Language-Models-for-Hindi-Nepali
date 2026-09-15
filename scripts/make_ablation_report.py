#!/usr/bin/env python3
"""Compare the no-positional-embeddings ablation with the standard model.

    python scripts/make_ablation_report.py                 # whichever language has results
    python scripts/make_ablation_report.py --language hi

Reads the standard model's Phase 2 artifacts and the ablation's, and writes
``report/bonus_ablation.md`` with two figures. Every number in the report comes
from those files, so it cannot drift from the runs that produced it.

Expected inputs, per language directory:

* ``reports/phase2_evaluation.json`` and ``reports/training_history.json`` - the
  standard model, from Phase 2.
* ``reports/bonus_nopos_evaluation.json`` and ``reports/training_history_nopos.json``
  - the ablation, from ``notebooks/bonus_nopos_evaluate.ipynb``.
* ``report/figures/attention_{heatmaps,summary}_<code>_nopos.png`` - its figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

LANGUAGES = {"hi": ("hindi", "Model H", "Hindi"), "ne": ("nepali", "Model L", "Nepali")}
DECODING = (("greedy", "Greedy"), ("temperature_0.5", "T = 0.5"),
            ("temperature_1.0", "T = 1.0"), ("temperature_1.5", "T = 1.5"))


def read(path: Path) -> dict | None:
    """Load a JSON artifact, or None when it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load(root: Path, code: str) -> dict:
    """Collect the standard and ablated artifacts for one language."""
    reports = root / LANGUAGES[code][0] / "reports"
    return {
        "standard": read(reports / "phase2_evaluation.json"),
        "standard_history": read(reports / "training_history.json") or {},
        "ablated": read(reports / "bonus_nopos_evaluation.json"),
        "ablated_history": read(reports / "training_history_nopos.json") or {},
    }


def fmt(value, spec: str) -> str:
    """Format a number, or a dash when it is missing."""
    return "-" if value is None else spec.format(value)


def validation_losses(history: dict) -> dict[int, float]:
    """Validation loss by step, excluding the duplicate end-of-run evaluation."""
    return {row["step"]: row["val_loss"] for row in history.get("history", [])
            if "val_loss" in row and not row.get("final")}


def plot_loss_curves(data: dict, code: str, out_path: Path) -> Path | None:
    """Validation loss against step for the standard and ablated runs."""
    standard, ablated = validation_losses(data["standard_history"]), validation_losses(data["ablated_history"])
    if not (standard and ablated):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for series, label, style in ((standard, "standard (learned positional embeddings)", "o-"),
                                 (ablated, "ablated (no positional embeddings)", "s--")):
        steps = sorted(series)
        axis.plot(steps, [series[s] for s in steps], style, markersize=3, label=label)
    axis.set_title(f"Validation loss with and without positional embeddings - {LANGUAGES[code][1]}")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("validation cross-entropy (nats/token)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def plot_attention_profile(data: dict, code: str, out_path: Path) -> Path:
    """Per-layer mean attention distance and entropy, standard against ablated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    standard, ablated = data["standard"]["attention"], data["ablated"]["attention"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, key, name in ((axes[0], "layer_mean_distance", "mean attention distance (positions)"),
                            (axes[1], "layer_mean_entropy", "mean attention entropy (nats)")):
        layers = list(range(len(standard[key])))
        axis.plot(layers, standard[key], "o-", label="standard")
        axis.plot(layers, ablated[key], "s--", label="no positional embeddings")
        axis.set_title(name.capitalize() + " by layer")
        axis.set_xlabel("layer")
        axis.set_ylabel(name)
        axis.set_xticks(layers)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(f"Attention profile with and without positional embeddings - {LANGUAGES[code][1]}")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def build(data: dict, code: str, root: Path) -> str:
    """Assemble the ablation report for one language."""
    directory, label, language_name = LANGUAGES[code]
    standard, ablated = data["standard"], data["ablated"]
    standard_history, ablated_history = data["standard_history"], data["ablated_history"]
    figures = root / "report" / "figures"

    planned = (ablated_history.get("train_config") or {}).get("max_steps", 12_000)
    ablated_step = ablated.get("training_step") or ablated_history.get("step") or 0
    standard_step = standard.get("training_step") or standard_history.get("step") or 0
    complete = ablated_step >= planned

    # The two runs must differ in the positional flag and nothing that affects
    # training; a config written before the flag existed means "positions on".
    def model_value(config: dict, key: str):
        return config.get(key, True) if key == "use_positional_embeddings" else config.get(key)

    model_keys = set(standard["model_config"]) | set(ablated["model_config"])
    model_differences = sorted(k for k in model_keys
                               if model_value(standard["model_config"], k) != model_value(ablated["model_config"], k))
    train_standard = standard_history.get("train_config") or {}
    train_ablated = ablated_history.get("train_config") or {}
    train_differences = sorted(k for k in set(train_standard) | set(train_ablated)
                               if train_standard.get(k) != train_ablated.get(k))

    parts = [
        f"# Bonus ablation - {label} ({language_name}) without positional embeddings",
        "",
        "**Language Models and Agents, Monsoon 2026 - Individual Project**",
        "",
        "## 1. Setup",
        "",
        f"The {language_name} model was pretrained a second time with the positional-embedding",
        "table removed (`ModelConfig(use_positional_embeddings=False)`) and then put through the",
        "full Phase 2 evaluation suite, on the same held-out text, prompts, decoding settings and",
        "attention sample as the standard model.",
        "",
        "| | Standard | No positional embeddings |",
        "|---|---:|---:|",
        f"| Trainable parameters | {fmt(standard.get('parameters'), '{:,}')} | {fmt(ablated.get('parameters'), '{:,}')} |",
        f"| Optimizer steps | {standard_step:,} | {ablated_step:,} |",
        f"| Tokens seen | {fmt(standard.get('tokens_seen'), '{:,}')} | {fmt(ablated.get('tokens_seen'), '{:,}')} |",
        f"| Best validation loss | {fmt(standard_history.get('best_val_loss'), '{:.4f}')} | {fmt(ablated_history.get('best_val_loss'), '{:.4f}')} |",
        "",
        "Model-configuration differences: " + (", ".join(f"`{k}`" for k in model_differences) or "none") + ".",
        "Training-configuration differences: " + (", ".join(f"`{k}`" for k in train_differences) or "none") + ".",
        "",
    ]
    if not complete:
        parts += [
            f"> **The ablation stopped at step {ablated_step:,} of {planned:,}.** The validation losses",
            "> compared at matched steps below are like for like. The evaluation-suite metrics that",
            f"> follow set a {ablated_step:,}-step ablated model against the {standard_step:,}-step",
            "> standard model, so part of any gap there is simply less training.",
            "",
        ]

    standard_losses = validation_losses(standard_history)
    ablated_losses = validation_losses(ablated_history)
    common = sorted(set(standard_losses) & set(ablated_losses))
    parts += ["## 2. Validation loss at matched steps", ""]
    gaps: list[tuple[int, float]] = []
    if common:
        shown = sorted({common[0], common[-1], *common[:: max(1, len(common) // 6)]})
        parts += ["| Step | Standard | No positional embeddings | Difference |", "|---:|---:|---:|---:|"]
        for step in shown:
            gap = ablated_losses[step] - standard_losses[step]
            parts.append(f"| {step:,} | {standard_losses[step]:.4f} | {ablated_losses[step]:.4f} | {gap:+.4f} |")
        gaps = [(step, ablated_losses[step] - standard_losses[step]) for step in common]
        parts.append("")
    else:
        parts += ["_No steps are recorded in both training histories._", ""]
    if plot_loss_curves(data, code, figures / f"bonus_ablation_loss_{code}.png"):
        parts += [f"![Validation loss with and without positional embeddings](figures/bonus_ablation_loss_{code}.png)", ""]

    lm_standard, lm_ablated = standard["language_modeling"], ablated["language_modeling"]
    parts += ["## 3. Language-modeling metrics", "",
              "| Metric | Standard | No positional embeddings |", "|---|---:|---:|"]
    for name, key, spec in (("Cross-entropy (nats/token)", "cross_entropy", "{:.4f}"),
                            ("Perplexity", "perplexity", "{:.2f}"),
                            ("Bits-per-byte", "bits_per_byte", "{:.4f}"),
                            ("Held-out tokens", "tokens", "{:,}")):
        parts.append(f"| {name} | {fmt(lm_standard.get(key), spec)} | {fmt(lm_ablated.get(key), spec)} |")
    parts.append("")

    parts += ["## 4. Generation quality", "",
              "Each cell reads standard → no positional embeddings.", "",
              "| Decoding | chrF | ROUGE-L | BLEU-4 | Repetition | Distinct-2 |", "|---|---|---|---|---|---|"]
    for key, name in DECODING:
        s, a = standard["generation"].get(key, {}), ablated["generation"].get(key, {})
        cells = [f"{fmt(s.get(m), spec)} → {fmt(a.get(m), spec)}"
                 for m, spec in (("chrf", "{:.2f}"), ("rouge_l", "{:.2f}"), ("bleu", "{:.2f}"),
                                 ("repetition_rate", "{:.3f}"), ("distinct_2", "{:.3f}"))]
        parts.append(f"| {name} | " + " | ".join(cells) + " |")
    parts.append("")
    for key, name in (("greedy", "Greedy"), ("temperature_1.0", "T = 1.0")):
        s_samples = (standard.get("samples") or {}).get(key) or []
        a_samples = (ablated.get("samples") or {}).get(key) or []
        if s_samples and a_samples:
            parts += [f"*{name}, standard:*", "", f"> {s_samples[0].strip()}", "",
                      f"*{name}, no positional embeddings:*", "", f"> {a_samples[0].strip()}", ""]

    att_standard, att_ablated = standard["attention"], ablated["attention"]
    n_layer = len(att_standard["layer_mean_distance"])
    parts += ["## 5. Attention", "",
              "| Layer | Distance, standard | Distance, ablated | Entropy, standard | Entropy, ablated |",
              "|---:|---:|---:|---:|---:|"]
    for layer in range(n_layer):
        parts.append(f"| {layer} | {att_standard['layer_mean_distance'][layer]:.2f} | "
                     f"{att_ablated['layer_mean_distance'][layer]:.2f} | "
                     f"{att_standard['layer_mean_entropy'][layer]:.3f} | {att_ablated['layer_mean_entropy'][layer]:.3f} |")
    groups_standard, groups_ablated = att_standard.get("head_groups", {}), att_ablated.get("head_groups", {})
    parts += ["", "| Head type | Standard | No positional embeddings |", "|---|---:|---:|"]
    for key, name in (("local", "Local"), ("long_range", "Long-range"), ("diffuse", "Diffuse")):
        parts.append(f"| {name} | {groups_standard.get(key, '-')} | {groups_ablated.get(key, '-')} |")
    parts.append("")
    plot_attention_profile(data, code, figures / f"bonus_ablation_attention_{code}.png")
    parts += [f"![Attention profile](figures/bonus_ablation_attention_{code}.png)", ""]
    for filename, caption in ((f"attention_heatmaps_{code}.png", "Standard model attention heatmaps"),
                              (f"attention_heatmaps_{code}_nopos.png", "Ablated model attention heatmaps"),
                              (f"attention_summary_{code}_nopos.png", "Ablated model per-head statistics")):
        if (figures / filename).exists():
            parts += [f"![{caption}](figures/{filename})", ""]

    # --- What breaks, computed from the numbers above ---
    ppl_s, ppl_a = lm_standard.get("perplexity"), lm_ablated.get("perplexity")
    bpb_s, bpb_a = lm_standard.get("bits_per_byte"), lm_ablated.get("bits_per_byte")
    parts += ["## 6. What breaks without position information", ""]
    if None not in (ppl_s, ppl_a, bpb_s, bpb_a):
        ratio = ppl_a / ppl_s
        size = ("no cost" if ratio <= 1.0 else "a modest cost" if ratio < 1.25
                else "a substantial cost" if ratio < 2.0 else "a severe cost")
        parts += [
            f"**Language modeling.** Removing the positional table changed test perplexity from "
            f"{ppl_s:.2f} to {ppl_a:.2f} (×{ratio:.2f}) and bits-per-byte from {bpb_s:.4f} to "
            f"{bpb_a:.4f}: {size}.",
            "",
        ]
    parts += [
        "**What removing the table does and does not remove.** Attention with no positional signal is order-blind only one",
        "layer at a time: a single layer sees the tokens before each position as an unordered set.",
        "Under the causal mask, though, every position's state summarizes a prefix of a different",
        "length, so a second layer can recover order - `scripts/test_reasoning.py` verifies both",
        "halves of that on small models. What the ablation measures is how much of the positional",
        "signal a 12-layer stack reconstructs for itself, and at what price.",
        "",
    ]
    if len(gaps) >= 2:
        (first_step, first_gap), (last_step, last_gap) = gaps[0], gaps[-1]
        trend = ("narrowed" if abs(last_gap) < abs(first_gap) else "widened")
        parts += [
            f"**Over training,** the validation-loss gap {trend} from {first_gap:+.4f} nats at step "
            f"{first_step:,} to {last_gap:+.4f} at step {last_step:,}"
            + (" - the stack keeps recovering order it was not given." if trend == "narrowed"
               else " - the missing signal is not being recovered with more training."),
            "",
        ]
    d0_s, d0_a = att_standard["layer_mean_distance"][0], att_ablated["layer_mean_distance"][0]
    e0_s, e0_a = att_standard["layer_mean_entropy"][0], att_ablated["layer_mean_entropy"][0]
    parts += [
        f"**Attention.** Layer 0 is where the loss of position should show first, because it is the",
        f"only layer with no earlier layer to reconstruct order from. There, mean attention distance",
        f"went from {d0_s:.2f} to {d0_a:.2f} positions and entropy from {e0_s:.3f} to {e0_a:.3f}; across",
        f"all heads, local heads went from {groups_standard.get('local', '-')} to "
        f"{groups_ablated.get('local', '-')} and long-range heads from "
        f"{groups_standard.get('long_range', '-')} to {groups_ablated.get('long_range', '-')}.",
        "",
    ]
    greedy_s, greedy_a = standard["generation"].get("greedy", {}), ablated["generation"].get("greedy", {})
    if "repetition_rate" in greedy_s and "repetition_rate" in greedy_a:
        parts += [
            f"**Generation.** Greedy repetition went from {greedy_s['repetition_rate']:.3f} to "
            f"{greedy_a['repetition_rate']:.3f}, and chrF at T = 1.0 from "
            f"{standard['generation'].get('temperature_1.0', {}).get('chrf', math.nan):.2f} to "
            f"{ablated['generation'].get('temperature_1.0', {}).get('chrf', math.nan):.2f}.",
            "",
        ]
    parts += [
        "**Caveat.** One run with and one without positional embeddings: differences of the size",
        "that ordinary run-to-run variation produces should not be read as effects of the ablation.",
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    """Write the ablation report for the language that has results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", choices=sorted(LANGUAGES), default=None)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()

    codes = [args.language] if args.language else list(LANGUAGES)
    code = next((c for c in codes if load(args.root, c)["ablated"]), None)
    if code is None:
        print("No ablation results found. Place bonus_nopos_evaluation.json and "
              "training_history_nopos.json in <language>/reports/ first.")
        return 1
    data = load(args.root, code)
    if data["standard"] is None:
        print(f"Missing the standard model's phase2_evaluation.json for {code}.")
        return 1

    report = args.root / "report" / "bonus_ablation.md"
    report.write_text(build(data, code, args.root), encoding="utf-8")
    print(f"wrote {report} for {LANGUAGES[code][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
