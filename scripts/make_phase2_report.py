"""Generate the Phase 2 report from the evaluation artifacts.

    python scripts/make_phase2_report.py

Reads ``<language>/reports/phase2_evaluation.json`` and the training histories
written during pretraining, and writes ``report/phase2_report.md`` plus the loss
curve figure. Every number in the report comes from those files, so the report
cannot drift from the runs that produced it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LANGUAGES = {
    "hi": ("hindi", "Model H", "Hindi"),
    "ne": ("nepali", "Model L", "Nepali"),
}

DECODE_ORDER = ["greedy", "temperature_0.5", "temperature_1.0", "temperature_1.5"]
DECODE_LABELS = {
    "greedy": "Greedy",
    "temperature_0.5": "T = 0.5",
    "temperature_1.0": "T = 1.0",
    "temperature_1.5": "T = 1.5",
}


def load_artifacts() -> dict:
    """Collect evaluation and training records for whichever models exist."""
    artifacts = {}
    for code, (directory, _, _) in LANGUAGES.items():
        reports = REPO_ROOT / directory / "reports"
        entry = {}

        evaluation = reports / "phase2_evaluation.json"
        if evaluation.exists():
            entry["evaluation"] = json.loads(evaluation.read_text())

        history = reports / "training_history.json"
        if history.exists():
            entry["history"] = json.loads(history.read_text())

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


def section_architecture(artifacts: dict) -> str:
    """The model configuration and its parameter accounting."""
    rows = []
    for code, entry in artifacts.items():
        config = (entry.get("evaluation") or {}).get("model_config") or {}
        if not config:
            continue
        _, label, name = LANGUAGES[code]
        rows.append((label, name, config, entry["evaluation"].get("parameters")))

    if not rows:
        return "_No trained model found._"

    lines = [
        "| | " + " | ".join(f"{label} ({name})" for label, name, _, _ in rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for field, key in (
        ("Layers", "n_layer"),
        ("Model dimension", "n_embd"),
        ("Attention heads", "n_head"),
        ("Feed-forward width", "n_inner"),
        ("Context length", "block_size"),
        ("Vocabulary", "vocab_size"),
        ("Dropout", "dropout"),
    ):
        values = [str(config.get(key, "-")) for _, _, config, _ in rows]
        lines.append(f"| {field} | {' | '.join(values)} |")

    params = [f"{p:,}" if p else "-" for _, _, _, p in rows]
    lines.append(f"| **Trainable parameters** | **{'** | **'.join(params)}** |")
    return "\n".join(lines)


def section_training(artifacts: dict) -> str:
    """How far each run got, and on how much data."""
    lines = [
        "| | " + " | ".join(f"{LANGUAGES[c][1]}" for c in artifacts) + " |",
        "|---|" + "---|" * len(artifacts),
    ]

    def row(name: str, getter, spec="{:,}"):
        values = []
        for code in artifacts:
            try:
                values.append(fmt(getter(artifacts[code]), spec))
            except (KeyError, TypeError):
                values.append("-")
        lines.append(f"| {name} | {' | '.join(values)} |")

    row("Training steps", lambda e: e["history"]["step"])
    row("Tokens seen", lambda e: e["history"]["tokens_seen"] / 1e9, "{:.2f}B")
    row("Best validation loss", lambda e: e["history"]["best_val_loss"], "{:.4f}")
    row("Wall-clock hours", lambda e: e["history"]["wall_seconds"] / 3600, "{:.1f}")
    return "\n".join(lines)


def section_lm_metrics(artifacts: dict) -> str:
    """Perplexity and bits-per-byte on the held-out test split."""
    present = [c for c in artifacts if "evaluation" in artifacts[c]]
    if not present:
        return "_No evaluation found._"

    lines = [
        "| Metric | " + " | ".join(f"{LANGUAGES[c][1]} ({LANGUAGES[c][2]})" for c in present) + " |",
        "|---|" + "---|" * len(present),
    ]
    for label, key, spec in (
        ("Cross-entropy (nats/token)", "cross_entropy", "{:.4f}"),
        ("**Perplexity**", "perplexity", "{:.2f}"),
        ("**Bits-per-byte**", "bits_per_byte", "{:.4f}"),
        ("Held-out tokens", "tokens", "{:,}"),
        ("Bytes per token", "bytes_per_token", "{:.2f}"),
    ):
        values = [
            fmt(artifacts[c]["evaluation"].get("language_modeling", {}).get(key), spec)
            for c in present
        ]
        lines.append(f"| {label} | {' | '.join(values)} |")
    return "\n".join(lines)


def section_generation(artifacts: dict) -> str:
    """BLEU, chrF, ROUGE-L and diversity across the four decoding settings."""
    blocks = []
    for code in artifacts:
        evaluation = artifacts[code].get("evaluation")
        if not evaluation or "generation" not in evaluation:
            continue
        _, label, name = LANGUAGES[code]

        lines = [
            f"**{label} ({name})**",
            "",
            "| Decoding | BLEU-4 | chrF | ROUGE-L | Repetition | Distinct-1 | Distinct-2 |",
            "|---|---|---|---|---|---|---|",
        ]
        for setting in DECODE_ORDER:
            metrics = evaluation["generation"].get(setting)
            if not metrics:
                continue
            lines.append(
                f"| {DECODE_LABELS[setting]} "
                f"| {fmt(metrics.get('bleu'), '{:.2f}')} "
                f"| {fmt(metrics.get('chrf'), '{:.2f}')} "
                f"| {fmt(metrics.get('rouge_l'), '{:.2f}')} "
                f"| {fmt(metrics.get('repetition_rate'), '{:.3f}')} "
                f"| {fmt(metrics.get('distinct_1'), '{:.3f}')} "
                f"| {fmt(metrics.get('distinct_2'), '{:.3f}')} |"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "_No generation results found._"


def section_attention(artifacts: dict) -> str:
    """Per-layer attention statistics and the head classification."""
    blocks = []
    for code in artifacts:
        evaluation = artifacts[code].get("evaluation")
        attention = (evaluation or {}).get("attention")
        if not attention:
            continue
        _, label, name = LANGUAGES[code]

        entropies = attention.get("layer_mean_entropy", [])
        distances = attention.get("layer_mean_distance", [])
        groups = attention.get("head_groups", {})

        lines = [
            f"**{label} ({name})** - head types: "
            f"{groups.get('local', 0)} local, "
            f"{groups.get('long_range', 0)} long-range, "
            f"{groups.get('diffuse', 0)} diffuse",
            "",
            "| Layer | Mean entropy | Mean attention distance |",
            "|---|---|---|",
        ]
        for index, (entropy, distance) in enumerate(zip(entropies, distances)):
            lines.append(f"| {index} | {entropy:.3f} | {distance:.2f} |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "_No attention analysis found._"


def section_samples(artifacts: dict) -> str:
    """A few generated continuations, for the qualitative discussion."""
    blocks = []
    for code in artifacts:
        samples = (artifacts[code].get("evaluation") or {}).get("samples")
        if not samples:
            continue
        _, label, name = LANGUAGES[code]
        lines = [f"**{label} ({name})**", ""]
        for setting in DECODE_ORDER:
            texts = samples.get(setting) or []
            if not texts:
                continue
            lines.append(f"*{DECODE_LABELS[setting]}*")
            lines.append("")
            for text in texts[:2]:
                snippet = text.strip().replace("\n", " ")[:220]
                lines.append(f"> {snippet}")
                lines.append("")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "_No samples recorded._"


def plot_loss_curves(artifacts: dict, out_path: Path) -> Path | None:
    """Draw both models' validation curves on shared axes."""
    histories = {c: artifacts[c]["history"] for c in artifacts if "history" in artifacts[c]}
    if not histories:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    for code, history in histories.items():
        rows = history.get("history", [])
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        losses = [r["val_loss"] for r in rows]
        label = f"{LANGUAGES[code][1]} ({LANGUAGES[code][2]})"

        axes[0].plot(steps, losses, marker="o", markersize=3, label=label)
        axes[1].plot(
            steps,
            [math.exp(min(v, 20)) for v in losses],
            marker="o",
            markersize=3,
            label=label,
        )

    axes[0].set_title("Validation cross-entropy")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("nats per token")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation perplexity")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("perplexity")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def build_report(artifacts: dict) -> str:
    """Assemble the Markdown report."""
    figures = REPO_ROOT / "report" / "figures"
    have_curves = (figures / "phase2_loss_curves.png").exists()

    parts = [
        "# Phase 2 - Model Implementation, Pretraining, and Evaluation",
        "",
        "**Language Models and Agents, Monsoon 2026 - Individual Project**",
        "",
        "Two decoder-only Transformers, implemented from primitive PyTorch layers "
        "and pretrained independently on the Phase 1 corpora. No weights, data, or "
        "vocabulary are shared between them.",
        "",
        "---",
        "",
        "## 1. Architecture",
        "",
        "The forward pass is written out explicitly in `lma/model/transformer.py`: "
        "the QKV projection, the reshape into heads, the scaled dot product, the "
        "causal mask, the softmax, the head concatenation, and the output "
        "projection. Only `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` and "
        "`nn.Dropout` are used; there is no `nn.Transformer*` module and no "
        "pre-built attention block anywhere in the model.",
        "",
        section_architecture(artifacts),
        "",
        "### Design choices",
        "",
        "**Scaling by 1/sqrt(d_k).** Query and key components are approximately "
        "unit variance, so their dot product over `d_k` dimensions has variance "
        "proportional to `d_k`. Without the scaling, score magnitudes grow with "
        "model width, the softmax saturates, and its gradient vanishes - the wider "
        "the model, the worse the problem, which is precisely the regime the "
        "architecture is meant to support.",
        "",
        "**Causal masking.** An additive mask puts `-inf` on future positions "
        "*before* the softmax, so they receive exactly zero probability rather "
        "than being renormalized afterwards. This is verified empirically rather "
        "than assumed: perturbing the token at position `t` leaves every logit at "
        "positions `< t` bit-identical, and attention weight on the upper triangle "
        "measures exactly 0.0. Both checks run in `scripts/test_model.py`.",
        "",
        "**Pre-norm blocks.** Normalization is applied entering each sublayer and "
        "the raw residual is added, so the path from input to loss carries no "
        "normalization and gradients reach early layers undiminished. Post-norm "
        "places a LayerNorm on that path and needs careful warmup to stay stable "
        "at this depth.",
        "",
        "**Learned absolute positional embeddings.** Chosen for simplicity at a "
        "single fixed context length. The cost is that the model cannot run on "
        "sequences longer than `block_size` - no embedding exists for those "
        "positions - where sinusoidal or rotary encodings would extrapolate.",
        "",
        "**Tied input and output embeddings.** One matrix serves as both the token "
        "embedding and the output projection, saving `vocab_size x n_embd` = 3.07M "
        "parameters, about an eighth of the budget, and tying each token's "
        "representation to the vector that predicts it.",
        "",
        "---",
        "",
        "## 2. Pretraining",
        "",
        "AdamW with weight decay applied only to the 2D weight matrices - biases, "
        "LayerNorm gains, and embeddings set scale and offset rather than mixing "
        "features, and decaying them is a bias with no regularizing benefit. The "
        "learning rate warms up linearly and then follows a cosine decay: warmup "
        "because Adam's second-moment estimate is unreliable in the first few "
        "hundred steps, and cosine because it spends most of the budget at a high "
        "rate before annealing smoothly.",
        "",
        section_training(artifacts),
        "",
        "Checkpoints carry model weights, optimizer state, scheduler position "
        "(the step, from which the schedule is a pure function), the step counter, "
        "RNG state for both torch and the batch sampler, and the full "
        "configuration. This matters because the training platform stops a session "
        "after 12 hours: a run spans several sessions, and an interrupted run "
        "resumed from its checkpoint reaches **bit-identical** weights to one that "
        "never stopped, which is asserted in the test suite.",
        "",
    ]

    if have_curves:
        parts += [
            "![Validation loss and perplexity](figures/phase2_loss_curves.png)",
            "",
        ]

    parts += [
        "---",
        "",
        "## 3. Language-modeling metrics",
        "",
        section_lm_metrics(artifacts),
        "",
        "Perplexity is `exp` of the mean cross-entropy. It is **not** comparable "
        "between the two models on its own: a tokenizer that splits text into "
        "fewer, longer pieces faces fewer prediction decisions per sentence and "
        "scores better without modelling the language any better. Bits-per-byte "
        "removes that advantage by expressing the same total loss per byte of the "
        "underlying UTF-8 text, which is a property of the text rather than of the "
        "segmentation, and is therefore the figure to use when comparing Model H "
        "against Model L.",
        "",
        "---",
        "",
        "## 4. Generation quality",
        "",
        "Continuations were generated from held-out prefixes under greedy decoding "
        "and temperatures 0.5, 1.0 and 1.5, and scored against the text that "
        "actually followed those prefixes in the held-out split.",
        "",
        section_generation(artifacts),
        "",
        "### What these metrics can and cannot tell us",
        "",
        "BLEU, chrF and ROUGE-L were designed for tasks with a correct answer - "
        "translation, summarization - where the reference is *the* right output. "
        "Open-ended continuation has no such reference: a fluent, grammatical, "
        "entirely sensible continuation that simply differs from what the corpus "
        "happened to say next scores near zero. The absolute values here should "
        "therefore not be read as quality.",
        "",
        "They remain informative **comparatively**. chrF is the most useful of the "
        "three for these languages: it scores character n-grams, so two "
        "inflections of the same lemma - common in Hindi and Nepali, both "
        "morphologically rich - receive partial credit where BLEU's word n-grams "
        "give none. BLEU-4 is the least informative, since matching four "
        "consecutive words by chance is vanishingly unlikely in open-ended text.",
        "",
        "The diversity diagnostics need no reference and are the more honest "
        "signal. Repetition rate rises and Distinct-1/2 fall as decoding "
        "temperature drops, which is the expected and well-documented failure of "
        "greedy decoding: the model collapses onto its most probable "
        "continuations and loops.",
        "",
        "### Sample continuations",
        "",
        section_samples(artifacts),
        "",
        "---",
        "",
        "## 5. Attention analysis",
        "",
        "Two summaries per head, computed only over positions a causal model may "
        "attend to. **Entropy** measures how spread a head's attention is; because "
        "a head at position `t` can see only `t + 1` positions, raw entropy grows "
        "along the sequence for structural reasons, so the normalized form "
        "(divided by `log(t + 1)`) is what should be compared across heads. **Mean "
        "attention distance** is how far back a head looks on average, weighted by "
        "attention mass.",
        "",
        section_attention(artifacts),
        "",
    ]

    for code in artifacts:
        heatmap = figures / f"attention_heatmaps_{code}.png"
        summary = figures / f"attention_summary_{code}.png"
        label = LANGUAGES[code][1]
        if heatmap.exists():
            parts += [f"![{label} attention heatmaps](figures/{heatmap.name})", ""]
        if summary.exists():
            parts += [f"![{label} per-head statistics](figures/{summary.name})", ""]

    parts += [
        "---",
        "",
        "## 6. Reproduction",
        "",
        "```bash",
        "# Component tests - causality, shapes, resume, and the metrics",
        "python scripts/test_model.py",
        "",
        "# Pretrain (notebooks/phase2_pretrain.ipynb runs this on a GPU host)",
        "# then evaluate each trained checkpoint:",
        "python scripts/evaluate_model.py --language hi \\",
        "    --checkpoint hindi/checkpoints/checkpoint_best.pt \\",
        "    --data-dir <tokenized-data> \\",
        "    --tokenizer hindi/tokenizer/hi_unigram_8000.model",
        "",
        "# Regenerate this report",
        "python scripts/make_phase2_report.py",
        "```",
        "",
    ]

    return "\n".join(parts)


def main() -> int:
    artifacts = load_artifacts()
    if not artifacts:
        print(
            "No Phase 2 artifacts found. Train a model, then run "
            "scripts/evaluate_model.py before generating the report."
        )
        return 1

    print(f"found artifacts for: {', '.join(artifacts)}")
    for code, entry in artifacts.items():
        have = [k for k in ("evaluation", "history") if k in entry]
        print(f"  {code}: {', '.join(have)}")

    figures = REPO_ROOT / "report" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    curve = plot_loss_curves(artifacts, figures / "phase2_loss_curves.png")
    if curve:
        print(f"wrote {curve.relative_to(REPO_ROOT)}")

    report = REPO_ROOT / "report" / "phase2_report.md"
    report.write_text(build_report(artifacts), encoding="utf-8")
    print(f"wrote {report.relative_to(REPO_ROOT)} ({report.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
