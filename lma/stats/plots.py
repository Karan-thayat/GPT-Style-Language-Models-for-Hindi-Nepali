"""Plotting for the Phase 1 report.

Kept strictly separate from computation: every function here takes already-computed
statistics dicts and returns a saved figure path. Nothing in this module reads a
corpus or trains a tokenizer.

The project requires each plot to carry a title, axis labels, and a legend where
applicable, so :func:`_finalize` applies those uniformly and every function goes
through it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # Headless: figures are written to disk, never displayed.
import matplotlib.pyplot as plt

#: One consistent colour per language across every figure in the report.
LANGUAGE_COLORS = {"hi": "#1f77b4", "ne": "#d62728", "as": "#9467bd", "bn": "#2ca02c"}
LANGUAGE_LABELS = {
    "hi": "Hindi (Model H)",
    "ne": "Nepali (Model L)",
    "as": "Assamese",
    "bn": "Bengali",
}

#: Fonts covering the Indic scripts used here. Devanagari and Bengali need
#: different faces, and figures may mix both, so all are offered to the
#: fallback chain and matplotlib picks per glyph.
_INDIC_FONTS = (
    "Noto Sans Devanagari",
    "Noto Serif Devanagari",
    "Lohit Devanagari",
    "Samyak Devanagari",
    "Noto Sans Bengali",
    "Noto Serif Bengali",
    "Lohit Bengali",
    "Mukti Narrow",
    "FreeSans",
)

#: Latin-capable fallbacks. Required because the Noto Indic faces contain *no*
#: Latin glyphs at all - every English axis label, legend entry, and numeral would
#: render as a tofu box if an Indic font were used alone.
_LATIN_FONTS = ("DejaVu Sans", "Liberation Sans", "FreeSans", "Arial")


def configure_fonts() -> str:
    """Install a font fallback chain that renders both Indic and Latin text.

    Matplotlib resolves ``font.family`` as an ordered list, falling through to the
    next entry per missing glyph. Figures here mix Indic tick labels with English
    axis titles, and no single available font covers both: DejaVu Sans has no
    Indic coverage, and the Noto Indic faces have no Latin coverage. Listing a
    Latin font first and the Indic faces after it means Latin text uses the Latin
    font and Indic characters fall through to whichever face covers them.

    Returns:
        A human-readable description of the resolved chain, for logging.
    """
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    latin = [name for name in _LATIN_FONTS if name in available]
    indic = [name for name in _INDIC_FONTS if name in available]

    chain = latin[:1] + indic + latin[1:]
    if not chain:
        return plt.rcParams["font.family"][0]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = chain
    # Minus signs come from the Indic font otherwise, which lacks U+2212.
    plt.rcParams["axes.unicode_minus"] = False
    return " -> ".join(chain) if indic else f"{chain[0]} (no Indic font found)"


def _finalize(fig, ax, title: str, xlabel: str, ylabel: str, output_path: Path, legend: bool = False) -> Path:
    """Apply required labelling, save the figure, and close it.

    Args:
        fig: Matplotlib figure.
        ax: Primary axes.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        output_path: Destination image path.
        legend: Whether to draw a legend.

    Returns:
        The path the figure was written to.
    """
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    if legend:
        ax.legend(fontsize=9)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_corpus_size_comparison(stats_by_language: dict[str, dict], output_path: Path) -> Path:
    """Grouped bars comparing documents, words, and characters across languages."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [("documents", "Documents"), ("words", "Words"), ("characters", "Characters")]

    for ax, (key, label) in zip(axes, metrics):
        languages = list(stats_by_language)
        values = [stats_by_language[lang]["totals"][key] for lang in languages]
        colors = [LANGUAGE_COLORS.get(lang, "#7f7f7f") for lang in languages]
        bars = ax.bar([LANGUAGE_LABELS.get(l, l) for l in languages], values, color=colors)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Model / language", fontsize=9)
        ax.set_ylabel(f"{label} (count)", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Corpus size comparison: Hindi (higher-resource) vs Nepali (lower-resource)",
                 fontsize=12, fontweight="bold")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_document_length_distribution(stats_by_language: dict[str, dict], output_path: Path) -> Path:
    """Quantile curves of document length in words, one line per language."""
    fig, ax = plt.subplots(figsize=(8, 5))
    quantile_keys = ["min", "p10", "p25", "median", "p75", "p90", "p99", "max"]
    positions = [0, 10, 25, 50, 75, 90, 99, 100]

    for language, stats in stats_by_language.items():
        lengths = stats["document_length_words"]
        values = [lengths[key] for key in quantile_keys]
        ax.plot(
            positions,
            values,
            marker="o",
            label=LANGUAGE_LABELS.get(language, language),
            color=LANGUAGE_COLORS.get(language, "#7f7f7f"),
        )

    ax.set_yscale("log")
    return _finalize(
        fig, ax,
        title="Document length distribution by quantile",
        xlabel="Quantile (percentile of documents)",
        ylabel="Document length (words, log scale)",
        output_path=output_path,
        legend=True,
    )


def plot_source_composition(stats: dict, language: str, output_path: Path) -> Path:
    """Horizontal bars of word counts per source, coloured by collection method."""
    by_source = stats["by_source"]["words"]
    if not by_source:
        by_source = {"(none)": 0}
    sources = list(by_source)[:15][::-1]
    values = [by_source[s] for s in sources]

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.45 * len(sources) + 1.5)))
    bars = ax.barh(sources, values, color=LANGUAGE_COLORS.get(language, "#7f7f7f"))
    for bar, value in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:,}", va="center", fontsize=8)

    ax.set_xscale("log")
    return _finalize(
        fig, ax,
        title=f"Corpus composition by source - {LANGUAGE_LABELS.get(language, language)}",
        xlabel="Words contributed (log scale)",
        ylabel="Source identifier",
        output_path=output_path,
    )


def plot_manual_fraction(stats_by_language: dict[str, dict], output_path: Path, requirement: float = 0.20) -> Path:
    """Stacked bars of manual vs downloaded token share against the 20% requirement."""
    languages = list(stats_by_language)
    manual = []
    downloaded = []
    for language in languages:
        words = stats_by_language[language]["collection_split"]["words"]
        total = sum(words.values()) or 1
        manual.append(words.get("manual", 0) / total * 100)
        downloaded.append(sum(v for k, v in words.items() if k != "manual") / total * 100)

    labels = [LANGUAGE_LABELS.get(l, l) for l in languages]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, manual, label="Manually collected", color="#2ca02c")
    ax.bar(labels, downloaded, bottom=manual, label="Downloaded (public corpora)", color="#aec7e8")
    ax.axhline(
        requirement * 100,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Requirement: {requirement:.0%} manual",
    )

    for index, value in enumerate(manual):
        ax.text(index, value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 100)
    return _finalize(
        fig, ax,
        title="Manual vs downloaded share of training tokens",
        xlabel="Model / language",
        ylabel="Share of training words (%)",
        output_path=output_path,
        legend=True,
    )


def plot_cleaning_funnel(cleaning_stats: dict[str, dict], output_path: Path) -> Path:
    """Bars showing how many documents each filter rejected, per language."""
    languages = list(cleaning_stats)
    reasons: list[str] = []
    for language in languages:
        for reason in cleaning_stats[language].get("rejected", {}):
            if reason not in reasons:
                reasons.append(reason)
    reasons.sort(key=lambda r: -sum(cleaning_stats[l].get("rejected", {}).get(r, 0) for l in languages))
    reasons = reasons[:12]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / max(1, len(languages))
    positions = range(len(reasons))

    for index, language in enumerate(languages):
        rejected = cleaning_stats[language].get("rejected", {})
        values = [rejected.get(reason, 0) for reason in reasons]
        offsets = [p + index * width for p in positions]
        ax.bar(
            offsets, values, width=width,
            label=LANGUAGE_LABELS.get(language, language),
            color=LANGUAGE_COLORS.get(language, "#7f7f7f"),
        )

    ax.set_xticks([p + width * (len(languages) - 1) / 2 for p in positions])
    ax.set_xticklabels(reasons, rotation=40, ha="right", fontsize=8)
    ax.set_yscale("symlog")
    return _finalize(
        fig, ax,
        title="Documents rejected by each cleaning filter",
        xlabel="Rejection reason (first filter to fire)",
        ylabel="Documents rejected (symlog scale)",
        output_path=output_path,
        legend=True,
    )


def plot_vocab_sweep(sweep_by_language: dict[str, Sequence[dict]], output_path: Path) -> Path:
    """Fertility and byte-fallback rate against vocabulary size for every candidate."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for language, results in sweep_by_language.items():
        base_color = LANGUAGE_COLORS.get(language, "#7f7f7f")
        for model_type, marker, style in (("unigram", "o", "-"), ("bpe", "s", "--")):
            points = sorted(
                (r for r in results if "error" not in r and r.get("model_type") == model_type),
                key=lambda r: r["vocab_size"],
            )
            if not points:
                continue
            sizes = [r["vocab_size"] for r in points]
            label = f"{LANGUAGE_LABELS.get(language, language)} - {model_type}"
            axes[0].plot(sizes, [r["fertility"] for r in points],
                         marker=marker, linestyle=style, color=base_color, label=label)
            # Prefer the script-only rate: the raw rate is dominated by newline
            # byte tokens, which measure document formatting rather than coverage.
            axes[1].plot(
                sizes,
                [r.get("script_byte_fallback_rate", r["byte_fallback_rate"]) * 100 for r in points],
                marker=marker, linestyle=style, color=base_color, label=label,
            )

    for ax, title, ylabel in (
        (axes[0], "Fertility vs vocabulary size", "Fertility (tokens per word)"),
        (axes[1], "Script byte-fallback vs vocabulary size", "Non-whitespace byte tokens (% of tokens)"),
    ):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Vocabulary size (pieces)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)

    fig.suptitle("Tokenizer vocabulary selection on held-out text", fontsize=12, fontweight="bold")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_token_frequency(freq_by_language: dict[str, dict], output_path: Path, top_k: int = 30) -> Path:
    """Zipf-style log-log rank/frequency curves plus cumulative coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for language, freq in freq_by_language.items():
        color = LANGUAGE_COLORS.get(language, "#7f7f7f")
        label = LANGUAGE_LABELS.get(language, language)
        tokens = freq.get("top_tokens", [])
        if tokens:
            ranks = range(1, len(tokens) + 1)
            axes[0].plot(list(ranks), [t["share"] * 100 for t in tokens], marker=".", color=color, label=label)

        coverage = freq.get("coverage", {})
        if coverage:
            points = sorted((int(k.split("_")[1]), v) for k, v in coverage.items())
            axes[1].plot([p[0] for p in points], [p[1] * 100 for p in points],
                         marker="o", color=color, label=label)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Token frequency by rank (held-out text)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Token rank (log scale)", fontsize=10)
    axes[0].set_ylabel("Share of all tokens (%, log scale)", fontsize=10)

    axes[1].set_xscale("log")
    axes[1].set_title("Cumulative coverage by vocabulary prefix", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Top-N most frequent tokens (log scale)", fontsize=10)
    axes[1].set_ylabel("Cumulative share of tokens (%)", fontsize=10)

    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)

    fig.suptitle("Token-frequency statistics per language", fontsize=12, fontweight="bold")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_piece_length_histogram(piece_stats_by_language: dict[str, dict], output_path: Path) -> Path:
    """Grouped bars of vocabulary piece lengths in characters."""
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = [str(i) for i in range(1, 10)] + ["10+"]
    languages = list(piece_stats_by_language)
    width = 0.8 / max(1, len(languages))

    for index, language in enumerate(languages):
        histogram = piece_stats_by_language[language].get("piece_length_histogram", {})
        total = sum(histogram.values()) or 1
        values = [histogram.get(key, 0) / total * 100 for key in keys]
        offsets = [i + index * width for i in range(len(keys))]
        ax.bar(offsets, values, width=width,
               label=LANGUAGE_LABELS.get(language, language),
               color=LANGUAGE_COLORS.get(language, "#7f7f7f"))

    ax.set_xticks([i + width * (len(languages) - 1) / 2 for i in range(len(keys))])
    ax.set_xticklabels(keys)
    return _finalize(
        fig, ax,
        title="Vocabulary piece-length distribution",
        xlabel="Piece length (characters, excluding word-boundary marker)",
        ylabel="Share of vocabulary (%)",
        output_path=output_path,
        legend=True,
    )


def plot_split_sizes(split_stats_by_language: dict[str, dict], output_path: Path, unit: str = "characters") -> Path:
    """Grouped bars of split sizes per train/validation/test split.

    Args:
        split_stats_by_language: ``{language: {split: {"totals": {"words": n}}}}``.
            The ``words`` key carries whatever quantity `unit` names.
        output_path: Destination image path.
        unit: Name of the plotted quantity, used in the axis label so the figure
            never misstates what it shows.
    """
    splits = ["train", "validation", "test"]
    fig, ax = plt.subplots(figsize=(9, 5))
    languages = list(split_stats_by_language)
    width = 0.8 / max(1, len(languages))

    for index, language in enumerate(languages):
        per_split = split_stats_by_language[language]
        values = [per_split.get(split, {}).get("totals", {}).get("words", 0) for split in splits]
        offsets = [i + index * width for i in range(len(splits))]
        bars = ax.bar(offsets, values, width=width,
                      label=LANGUAGE_LABELS.get(language, language),
                      color=LANGUAGE_COLORS.get(language, "#7f7f7f"))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}",
                    ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks([i + width * (len(languages) - 1) / 2 for i in range(len(splits))])
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_yscale("log")
    return _finalize(
        fig, ax,
        title="Train / validation / test split sizes",
        xlabel="Split",
        ylabel=f"{unit.capitalize()} in split (log scale)",
        output_path=output_path,
        legend=True,
    )
