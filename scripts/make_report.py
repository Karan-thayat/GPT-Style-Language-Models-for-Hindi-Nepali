#!/usr/bin/env python3
"""Assemble the Phase 1 report: figures, comparison tables, and Markdown.

Reads the JSON artifacts written by the earlier stages and produces
``report/phase1_report.md`` plus every figure under ``report/figures/``. This
script computes nothing new about the corpora - it only formats what the
pipeline measured, so the report can never disagree with the recorded statistics.

Usage::

    python scripts/make_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lma.stats import plots
from lma.utils.logging_utils import read_json, setup_logging

#: Language code -> directory, in report order (higher-resource first).
LANGUAGES = {"bn": "bengali", "as": "assamese"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "report", help="Output directory")
    return parser.parse_args()


def load_artifacts() -> dict[str, dict]:
    """Load every per-language JSON artifact that exists.

    Missing files are tolerated so the report can be regenerated after a partial
    run; the corresponding sections simply note that the stage has not run.
    """
    artifacts: dict[str, dict] = {}
    for language, directory in LANGUAGES.items():
        reports = REPO_ROOT / directory / "reports"
        entry: dict = {}
        for key, filename in (
            ("corpus", "corpus_stats.json"),
            ("cleaning", "cleaning_stats.json"),
            ("manifest", "corpus_manifest.json"),
            ("tokenizer", "tokenizer_stats.json"),
            ("sweep", "tokenizer_sweep.json"),
            ("tokens", "token_counts.json"),
        ):
            path = reports / filename
            if path.exists():
                entry[key] = read_json(path)
        artifacts[language] = entry
    return artifacts


def load_independence() -> dict | None:
    """Load the independence-validation report, if it has been run."""
    path = REPO_ROOT / "report/independence_validation.json"
    return read_json(path) if path.exists() else None


def section_independence(validation: dict | None) -> str:
    """Render the corpus/tokenizer independence verification results."""
    if validation is None:
        return (
            "_Not yet run. Execute `python scripts/validate_independence.py` to verify "
            "the independence constraint against the artifacts on disk._"
        )

    checks = validation.get("checks", {})
    status = "**PASSED**" if validation.get("passed") else "**FAILED**"
    rows = [
        f"Overall: {status}",
        "",
        "| Verified property | Result |",
        "|---|---|",
        f"| Documents shared between the two corpora | {fmt(checks.get('shared_documents'))} |",
    ]
    for language, label in (("bn", "Bengali"), ("as", "Assamese")):
        for held in ("validation", "test"):
            key = f"{language}_train_{held}_overlap"
            if key in checks:
                rows.append(f"| {label}: train ∩ {held} documents | {fmt(checks[key])} |")
    if "vocab_jaccard" in checks:
        rows.append(
            f"| Vocabulary piece overlap (Jaccard) | {checks['vocab_jaccard']:.3f} "
            "(overlap expected — shared script; ≈1.0 would indicate reuse) |"
        )
    for language, label in (("bn", "Bengali"), ("as", "Assamese")):
        key = f"{language}_purity"
        if key in checks:
            rows.append(f"| {label} corpus identified as {language} | {100 * checks[key]:.1f}% |")

    if not validation.get("passed"):
        rows += ["", "Failures:", ""] + [f"- {failure}" for failure in validation.get("failures", [])]
    return "\n".join(rows)


def fmt(value: float | int | None, spec: str = ",") -> str:
    """Format a number for a Markdown table, rendering None as an em dash."""
    if value is None:
        return "—"
    return f"{value:{spec}}"


def build_figures(artifacts: dict[str, dict], figures_dir: Path, logger) -> list[tuple[str, Path]]:
    """Generate every figure the report references.

    Returns:
        A list of ``(caption, path)`` pairs for figures that were produced.
    """
    figures: list[tuple[str, Path]] = []
    font = plots.configure_fonts()
    logger.info("Plot font: %s", font)

    corpus_stats = {lang: a["corpus"] for lang, a in artifacts.items() if "corpus" in a}
    cleaning_stats = {lang: a["cleaning"] for lang, a in artifacts.items() if "cleaning" in a}

    if corpus_stats:
        figures.append((
            "Corpus size across the two resource tiers (documents, words, characters; log scale).",
            plots.plot_corpus_size_comparison(corpus_stats, figures_dir / "corpus_size_comparison.png"),
        ))
        figures.append((
            "Document length distribution by quantile, showing how the two corpora differ in shape as well as size.",
            plots.plot_document_length_distribution(corpus_stats, figures_dir / "document_length_distribution.png"),
        ))
        figures.append((
            "Manual versus downloaded share of training words, against the 20% manual-collection requirement.",
            plots.plot_manual_fraction(corpus_stats, figures_dir / "manual_fraction.png"),
        ))
        for language, stats in corpus_stats.items():
            figures.append((
                f"Per-source composition of the {stats['language']} corpus by word count.",
                plots.plot_source_composition(stats, language, figures_dir / f"source_composition_{language}.png"),
            ))

    if cleaning_stats:
        figures.append((
            "Documents rejected by each cleaning filter, attributed to the first filter that fired.",
            plots.plot_cleaning_funnel(cleaning_stats, figures_dir / "cleaning_funnel.png"),
        ))

    # Split sizes are plotted in characters: the SplitWriter tracks characters per
    # split exactly, whereas words are only aggregated corpus-wide. Characters are
    # a faithful size proxy and the axis is labelled accordingly.
    split_stats = {}
    for language, entry in artifacts.items():
        manifest = entry.get("manifest")
        if not manifest:
            continue
        characters = manifest.get("split_summary", {}).get("characters", {})
        if characters:
            split_stats[language] = {
                split: {"totals": {"words": chars, "characters": chars}}
                for split, chars in characters.items()
            }
    if split_stats:
        figures.append((
            "Train/validation/test split sizes in characters (log scale).",
            plots.plot_split_sizes(split_stats, figures_dir / "split_sizes.png"),
        ))

    sweeps = {lang: a["sweep"]["results"] for lang, a in artifacts.items() if "sweep" in a}
    if sweeps:
        figures.append((
            "Vocabulary selection: held-out fertility and byte-fallback rate against vocabulary size, for both algorithms.",
            plots.plot_vocab_sweep(sweeps, figures_dir / "vocab_sweep.png"),
        ))

    frequencies = {
        lang: a["tokenizer"]["token_frequency"]
        for lang, a in artifacts.items()
        if "tokenizer" in a and a["tokenizer"].get("token_frequency")
    }
    if frequencies:
        figures.append((
            "Token-frequency distribution and cumulative coverage on held-out text.",
            plots.plot_token_frequency(frequencies, figures_dir / "token_frequency.png"),
        ))

    piece_stats = {
        lang: a["tokenizer"]["piece_statistics"]
        for lang, a in artifacts.items()
        if "tokenizer" in a and a["tokenizer"].get("piece_statistics")
    }
    if piece_stats:
        figures.append((
            "Vocabulary piece-length distribution for the two tokenizers.",
            plots.plot_piece_length_histogram(piece_stats, figures_dir / "piece_length_histogram.png"),
        ))

    return figures


def section_dataset_table(artifacts: dict[str, dict]) -> str:
    """Render the headline dataset-statistics comparison table."""
    rows = [
        "| Statistic | Bengali (Model H) | Assamese (Model L) |",
        "|---|---:|---:|",
    ]

    def cell(language: str, getter) -> str:
        """Safely extract and format one table cell."""
        entry = artifacts.get(language, {})
        try:
            return getter(entry)
        except (KeyError, TypeError, ZeroDivisionError):
            return "—"

    specs = [
        ("Documents kept", lambda e: fmt(e["corpus"]["totals"]["documents"])),
        ("Documents seen (pre-filter)", lambda e: fmt(e["cleaning"]["seen"])),
        ("Keep rate", lambda e: f"{100 * e['cleaning']['keep_rate']:.1f}%"),
        ("Words", lambda e: fmt(e["corpus"]["totals"]["words"])),
        ("Characters", lambda e: fmt(e["corpus"]["totals"]["characters"])),
        ("Sentences", lambda e: fmt(e["corpus"]["totals"]["sentences"])),
        ("Mean words/document", lambda e: f"{e['corpus']['averages']['words_per_document']:.1f}"),
        ("Mean chars/word", lambda e: f"{e['corpus']['averages']['characters_per_word']:.2f}"),
        ("Median document length (words)", lambda e: f"{e['corpus']['document_length_words']['median']:.0f}"),
        ("Word types (distinct)", lambda e: fmt(e["corpus"]["lexical_diversity"]["word_types"])),
        ("Type/token ratio", lambda e: f"{e['corpus']['lexical_diversity']['type_token_ratio']:.4f}"),
        ("Bengali-script char ratio", lambda e: f"{e['corpus']['script_composition']['bengali_char_ratio']:.3f}"),
        ("Manual collection share (words)", lambda e: f"{100 * e['corpus']['collection_split']['manual_word_fraction']:.1f}%"),
        ("Distinct characters", lambda e: fmt(e["corpus"]["distinct_characters"])),
        ("**Training tokens (exact)**", lambda e: f"**{fmt(e['tokens']['train_tokens'])}**"),
        ("Total tokens (all splits)", lambda e: fmt(e["tokens"]["total_tokens"])),
        ("Achieved vs 500M target", lambda e: f"{100 * e['tokens']['achieved_fraction_of_target']:.2f}%"),
    ]

    for label, getter in specs:
        rows.append(f"| {label} | {cell('bn', getter)} | {cell('as', getter)} |")
    return "\n".join(rows)


def section_tokenizer_table(artifacts: dict[str, dict]) -> str:
    """Render the held-out tokenizer-metrics comparison table."""
    rows = [
        "| Tokenizer metric (held-out) | Bengali (Model H) | Assamese (Model L) |",
        "|---|---:|---:|",
    ]

    def cell(language: str, getter) -> str:
        """Safely extract and format one table cell."""
        entry = artifacts.get(language, {})
        try:
            return getter(entry)
        except (KeyError, TypeError, ZeroDivisionError):
            return "—"

    specs = [
        ("Algorithm", lambda e: e["tokenizer"]["metrics"]["model_type"]),
        ("Vocabulary size", lambda e: fmt(e["tokenizer"]["metrics"]["vocab_size"])),
        ("**Fertility (tokens/word)**", lambda e: f"**{e['tokenizer']['metrics']['fertility']:.3f}**"),
        ("Unknown-token rate", lambda e: f"{e['tokenizer']['metrics']['unk_rate']:.2e}"),
        ("Byte-fallback rate (all)", lambda e: f"{100 * e['tokenizer']['metrics']['byte_fallback_rate']:.4f}%"),
        ("**Byte-fallback rate (script only)**",
         lambda e: f"**{100 * e['tokenizer']['metrics']['script_byte_fallback_rate']:.4f}%**"),
        ("Characters per token", lambda e: f"{e['tokenizer']['metrics']['chars_per_token']:.3f}"),
        ("Bytes per token", lambda e: f"{e['tokenizer']['metrics']['bytes_per_token']:.3f}"),
        ("Single-token words", lambda e: f"{100 * e['tokenizer']['metrics']['single_token_word_rate']:.1f}%"),
        ("Vocabulary utilization", lambda e: f"{100 * e['tokenizer']['token_frequency']['vocab_utilization']:.1f}%"),
        ("Byte-fallback pieces in vocab", lambda e: fmt(e["tokenizer"]["piece_statistics"]["byte_fallback_pieces"])),
        ("Bengali-script pieces in vocab", lambda e: fmt(e["tokenizer"]["piece_statistics"]["bengali_script_pieces"])),
        ("Mean piece length (chars)", lambda e: f"{e['tokenizer']['piece_statistics']['mean_piece_length_chars']:.2f}"),
        ("Held-out documents evaluated", lambda e: fmt(e["tokenizer"]["heldout_documents"])),
    ]

    for label, getter in specs:
        rows.append(f"| {label} | {cell('bn', getter)} | {cell('as', getter)} |")
    return "\n".join(rows)


def section_sweep_table(artifacts: dict[str, dict]) -> str:
    """Render the vocabulary sweep results for both languages."""
    blocks = []
    for language, entry in artifacts.items():
        sweep = entry.get("sweep")
        if not sweep:
            continue
        name = "Bengali (Model H)" if language == "bn" else "Assamese (Model L)"
        lines = [
            f"**{name}**",
            "",
            "| Algorithm | Vocab size | Fertility | Unk rate | Byte-fallback | Chars/token |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for result in sweep["results"]:
            if "error" in result:
                lines.append(
                    f"| {result['model_type']} | {fmt(result['vocab_size'])} | failed | — | — | — |"
                )
                continue
            lines.append(
                f"| {result['model_type']} | {fmt(result['vocab_size'])} | "
                f"{result['fertility']:.3f} | {result['unk_rate']:.1e} | "
                f"{100 * result['byte_fallback_rate']:.4f}% | {result['chars_per_token']:.2f} |"
            )
        selected = sweep.get("selected")
        if selected:
            lines += [
                "",
                f"Selected: **{selected['model_type']}, vocab {selected['vocab_size']:,}** "
                f"(fertility {selected['fertility']:.3f}).",
            ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "_Sweep has not been run._"


def section_examples(artifacts: dict[str, dict]) -> str:
    """Render tokenization examples for both languages."""
    blocks = []
    for language, entry in artifacts.items():
        tokenizer = entry.get("tokenizer")
        if not tokenizer or not tokenizer.get("examples"):
            continue
        name = "Bengali (Model H)" if language == "bn" else "Assamese (Model L)"
        lines = [f"**{name}**", ""]
        for example in tokenizer["examples"][:5]:
            text = example["text"][:150]
            pieces = " ".join(example["pieces"][:40])
            lines += [
                f"- Text: `{text}`",
                f"  - Pieces ({example['n_tokens']} tokens, {example['n_words']} words, "
                f"fertility {example['fertility']:.2f}): `{pieces}`",
                f"  - Round-trip decode matches: {example['roundtrip_ok']}",
            ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "_Tokenizer has not been trained._"


def section_top_tokens(artifacts: dict[str, dict], top_k: int = 20) -> str:
    """Render the most frequent tokens per language side by side."""
    columns: dict[str, list[str]] = {}
    for language, entry in artifacts.items():
        tokenizer = entry.get("tokenizer")
        if not tokenizer:
            continue
        tokens = tokenizer.get("token_frequency", {}).get("top_tokens", [])[:top_k]
        columns[language] = [f"`{t['piece']}` ({100 * t['share']:.2f}%)" for t in tokens]
    if not columns:
        return "_Tokenizer has not been trained._"

    rows = ["| Rank | Bengali (Model H) | Assamese (Model L) |", "|---:|---|---|"]
    depth = max(len(v) for v in columns.values())
    for index in range(depth):
        bn = columns.get("bn", [])
        asm = columns.get("as", [])
        rows.append(
            f"| {index + 1} | {bn[index] if index < len(bn) else '—'} | {asm[index] if index < len(asm) else '—'} |"
        )
    return "\n".join(rows)


def section_sources(artifacts: dict[str, dict]) -> str:
    """Render the per-source contribution tables."""
    blocks = []
    for language, entry in artifacts.items():
        corpus = entry.get("corpus")
        if not corpus:
            continue
        name = "Bengali (Model H)" if language == "bn" else "Assamese (Model L)"
        words = corpus["by_source"]["words"]
        docs = corpus["by_source"]["documents"]
        total_words = sum(words.values()) or 1
        lines = [
            f"**{name}**",
            "",
            "| Source | Collection | Documents | Words | Share of words |",
            "|---|---|---:|---:|---:|",
        ]
        manifest_sources = {s["source_id"]: s for s in entry.get("manifest", {}).get("sources", [])}
        for source_id, word_count in words.items():
            collection = manifest_sources.get(source_id, {}).get("collection", "—")
            lines.append(
                f"| `{source_id}` | {collection} | {fmt(docs.get(source_id, 0))} | "
                f"{fmt(word_count)} | {100 * word_count / total_words:.1f}% |"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "_No corpus statistics found._"


def section_cleaning(artifacts: dict[str, dict]) -> str:
    """Render the cleaning-filter rejection breakdown."""
    reasons: list[str] = []
    for entry in artifacts.values():
        for reason in entry.get("cleaning", {}).get("rejected", {}):
            if reason not in reasons:
                reasons.append(reason)
    if not reasons:
        return "_No cleaning statistics found._"

    rows = ["| Rejection reason | Bengali (Model H) | Assamese (Model L) |", "|---|---:|---:|"]
    for reason in reasons:
        bn = artifacts.get("bn", {}).get("cleaning", {}).get("rejected", {}).get(reason, 0)
        asm = artifacts.get("as", {}).get("cleaning", {}).get("rejected", {}).get(reason, 0)
        rows.append(f"| `{reason}` | {fmt(bn)} | {fmt(asm)} |")

    for label, key in (("**Total kept**", "kept"), ("**Total seen**", "seen")):
        bn = artifacts.get("bn", {}).get("cleaning", {}).get(key, 0)
        asm = artifacts.get("as", {}).get("cleaning", {}).get(key, 0)
        rows.append(f"| {label} | {fmt(bn)} | {fmt(asm)} |")
    return "\n".join(rows)


def build_markdown(
    artifacts: dict[str, dict],
    figures: list[tuple[str, Path]],
    report_dir: Path,
    validation: dict | None = None,
) -> str:
    """Compose the full Phase 1 report as Markdown."""
    profile = artifacts.get("bn", {}).get("manifest", {}).get("profile", {}).get("name", "unknown")

    figure_block = "\n\n".join(
        f"![{caption}](figures/{path.name})\n\n*Figure {index}: {caption}*"
        for index, (caption, path) in enumerate(figures, start=1)
    )

    return f"""# Phase 1 — Data Collection and Tokenizer Construction

**Language Models and Agents, Monsoon 2026 — Individual Project**

| | |
|---|---|
| **Model H** (higher-resource) | Bengali (`bn`) |
| **Model L** (lower-resource) | Assamese (`as`) |
| Run profile | `{profile}` |
| Independence | Separate corpora, separate tokenizers, separate vocabularies |

---

## 1. Language Selection and Justification

### Model H — Bengali

Bengali is a non-English Indian language with substantial public text. It is the
second most-spoken language in India (roughly 97 million first-language speakers
per the 2011 Census, and about 237 million worldwide), it has an active Wikipedia,
and it is well represented in openly available Indic web corpora such as AI4Bharat
Sangraha. That combination is what makes a ~500M-token monolingual corpus
realistic for the higher-resource tier.

### Model L — Assamese

Assamese is on the project's permitted lower-resource list. It has roughly 15
million speakers and official status in Assam, yet its openly available digital
corpora are one to two orders of magnitude smaller than Bengali's. That gap is the
point: it produces a genuine resource-tier contrast rather than two comfortable
corpora of similar size.

### Why this pair is a good experimental design

Bengali and Assamese share the Bengali-Assamese script, which isolates the
variable under study. Because the writing system, character set, and
normalization requirements are held constant, differences observed in Phase 2 and
Phase 3 between Model H and Model L are attributable to **data scale and quality**
rather than to script or tokenizer-family effects.

The shared script also creates the project's main data-integrity hazard, which is
addressed explicitly in §3: the two languages are easy to confuse
programmatically, so preventing cross-contamination requires a positive
language-identity test rather than a script check.

---

## 2. Corpus Construction Pipeline

Documents flow through six stages, each independently configured and counted:

```
sources (HF datasets / scraping / OCR)
  -> Unicode normalization      (lma/data/normalize.py)
  -> quality filtering          (lma/data/clean.py)
  -> exact dedup (SHA-256)      (lma/data/clean.py)
  -> near dedup (MinHash+LSH)   (lma/data/clean.py)
  -> deterministic splitting    (lma/data/splits.py)
  -> per-split JSONL + statistics
```

The pipeline is **streaming end to end**: documents are filtered as they arrive
and appended directly to their split file, so peak memory does not depend on
corpus size and the raw text never has to be staged on disk.

### 2.1 Unicode normalization

The normalizer is deliberately **script-aware and language-specific**, because the
obvious implementation is wrong here. A generic Indic normalizer folds
U+09F0 (`ৰ`) into U+09B0 (`র`), which silently rewrites Assamese as Bengali and
destroys the very distinction this project studies. Instead:

- The **Assamese** profile preserves U+09F0 (`ৰ`) and U+09F1 (`ৱ`).
- The **Bengali** profile folds them to `র`/`ব`, since in a Bengali corpus their
  presence indicates OCR noise or mis-tagged text.

Beyond that, normalization applies NFC, unifies the two-part vowel signs
(`U+09C7 + U+09BE` → `U+09CB`) and the nukta letters, strips invisible characters
while **preserving ZWNJ/ZWJ** (which control Indic conjunct formation), folds
punctuation variants and Bengali digits, and tidies whitespace.

The normalizer is **idempotent**, verified by test. This required care: NFC
*decomposes* U+09DC/U+09DD/U+09DF into base letter plus nukta, so composing them
must be followed by a second NFC pass. Without it, normalizing an
already-normalized document keeps changing it, and the tokenizer would learn two
encodings of the same grapheme.

### 2.2 Quality filtering

Filters are applied in a fixed order and every rejection is attributed to the
**first** filter that fired, which is what makes the breakdown in §5
interpretable. Filters cover length, script ratio, mojibake, line- and
n-gram-level repetition, digit and symbol ratio, long character runs, and
language identity.

One design note: the script-ratio filter measures Bengali share among **letters
only, excluding digits**. Counting digits in the denominator conflates two
different defects — a table of figures with Bengali labels would be reported as
"wrong script" when the real problem is digit spam. Separating them means each
filter names the defect it actually detects.

### 2.3 Deduplication

Exact duplicates are caught by SHA-256 over whitespace-normalized text. Near
duplicates use **MinHash with LSH banding** over character 5-gram shingles, which
is O(1) per document in the number of documents already indexed — necessary
because pairwise comparison across a 500M-token corpus is intractable.

The similarity threshold is **0.70**, not the more common 0.80. This was measured,
not assumed: on real Bengali prose, replacing one sentence of eight in an article
yields a true Jaccard of ≈0.79, so a 0.80 cutoff lets ordinary syndicated-news
near-duplicates through. At 0.70 those are caught while merely topically related
documents (measured well below 0.4) are kept.

### 2.4 Splitting

Splits are assigned by **salted hash of document content**, not by shuffling.
This is reproducible without holding the corpus in memory, and it is stable under
corpus growth — adding documents never moves an existing document from train to
test, which keeps Phase 2/3 metrics comparable across corpus revisions. Because
assignment is content-based, any exact duplicate that survives cleaning lands in
the *same* split, so duplicates cannot straddle train and test.

Split *ratios* are profile-dependent, because a ratio that is right at full scale
is wrong on a sample. At the full ~500M-token target, Bengali holds out 1% per
split — still roughly 5M tokens, ample for stable perplexity. Assamese holds out
2% at full scale, since its corpus is much smaller. On the `sampled` profile both
languages hold out 6% per split: at 1% of a sampled corpus the validation set came
to only a dozen documents, far too few to measure fertility or perplexity stably.
Because assignment is by content hash, changing the ratio changes only *which*
documents are held out, not the deterministic, reproducible nature of the split.

---

## 3. Preventing Cross-Language Contamination

Because Bengali and Assamese share a script, a script check cannot separate them,
and general-purpose language identifiers routinely label Assamese as Bengali. This
matters directly: the project requires the two corpora to share no documents, and
Bengali text is far more abundant online, so leakage into the Assamese corpus is
the likely failure direction.

`lma/data/language_id.py` implements a transparent scorer over two kinds of
orthographic evidence, chosen so its decisions can be explained rather than
merely trusted:

1. **Exclusive letters** — `ৰ`/`ৱ` appear in Assamese and essentially never in
   correct Bengali; the ratio of `ৰ` to `র` is strongly diagnostic.
2. **Function words** — high-frequency grammatical forms differ between the
   languages (Assamese `আৰু`, `হৈছে`, `কৰি` versus Bengali `এবং`, `হয়েছে`, `করে`).

Scores are per-token rates, so document length does not bias the verdict, and a
language is assigned only when it wins by a margin — ambiguous or short documents
are returned as `unknown` and **dropped**, rather than guessed into a corpus.

This filter is not hypothetical. Spot-checking the Sangraha `verified/asm` split
during development turned up Latin-script documents from an unrelated language;
those are removed by this stage.

### 3.1 Verified independence

The constraint is checked as a property of the artifacts on disk, not assumed from
the pipeline's intent, by `scripts/validate_independence.py`:

{section_independence(validation)}

---

## 4. Dataset Statistics

{section_dataset_table(artifacts)}

### 4.1 On the token counts in this run

The table above reports the **`sampled` profile**, which caps each source so the
whole pipeline can be executed and verified end to end on a laptop. Every number
in this report is a real measurement from that run — nothing is extrapolated — but
the token totals are therefore ~3.5M and ~3.7M rather than the ~500M target.

The 500M-token corpus is produced by the same code with `--profile full`, which
removes the per-source caps, widens the Sangraha `data_files` patterns from one
shard to all of them, and stops the ingest on a token budget rather than a document
count. What the sampled run establishes is that every stage works and every
statistic is real; what remains for the full run is compute time, not code.

Two honest caveats for the full run:

1. **Assamese will fall short of 500M.** Public Assamese text is roughly two orders
   of magnitude scarcer than Bengali. The assignment anticipates this and permits a
   documented shortfall; `scripts/tokenize_corpus.py` reports the exact achieved
   count and the gap, so the shortfall is quantified rather than hidden.
2. **The manual-collection share must be re-established at scale.** It is 25.5%
   (Bengali) and 36.2% (Assamese) here, but those shares come from a fixed number of
   scraped pages against capped downloads. Lifting the download caps without also
   scraping/OCR-ing more material would push the manual share below 20%; the
   scrape caches make that incremental rather than a restart.

### 4.2 Per-source contributions

{section_sources(artifacts)}

---

## 5. Cleaning Effect

{section_cleaning(artifacts)}

---

## 6. Tokenizer Construction

Both tokenizers are trained **from scratch** with SentencePiece on their own
language's training split only. No pretrained tokenizer is used, and no vocabulary
or piece is shared between the two languages.

Key configuration choices and why:

- **`normalization_rule_name: identity`** — the corpus is already normalized by
  the project's own script-aware normalizer. SentencePiece's built-in normalizers
  are not Assamese-aware, and letting one run would risk folding `ৰ`/`ৱ` after all
  the care taken to preserve them.
- **`byte_fallback: true`** — any character outside the vocabulary is encoded as
  UTF-8 byte pieces, so no input is unrepresentable. This drives the unknown rate
  to ≈0, which is why the honest coverage metric is the **byte-fallback rate**
  rather than the unknown rate.

  Two byte-fallback rates are reported, because the raw figure is misleading.
  SentencePiece trains on one-line-per-sentence input and so never learns a piece
  for the newline character; every line break in a multi-line document then costs
  one `<0x0A>` byte token. Measured on the Assamese corpus, that artifact was
  **85% of all byte-fallback tokens** (3.58% of tokens, against 0.65% genuine
  coverage failures). The script-only rate excludes whitespace bytes and is the
  figure that actually reflects vocabulary coverage; the raw rate would otherwise
  grow with document length rather than with coverage.
- **`character_coverage: 0.9995`** — at 1.0 every stray Latin/CJK/symbol
  character in the long tail claims its own vocabulary slot. The remainder falls
  back to bytes.
- **`split_digits: true`** — digits become individual pieces, which generalizes
  to unseen numbers.

### 6.1 Vocabulary selection

Vocabulary size is **chosen empirically**, per the project requirement, by
training candidates across sizes and both algorithms and measuring each on the
same held-out validation text.

Selecting on fertility alone would decide nothing. Fertility falls monotonically
with vocabulary size, so "lowest fertility" always names the largest candidate
swept. The real trade-off is against parameter cost: with tied input/output
embeddings the embedding matrix holds `vocab_size × d_model` parameters, and Phase
2's budget is a fixed ~25M. At d_model=384 a 32k vocabulary spends 12.3M
parameters — roughly half the entire budget — on embeddings alone.

The rule therefore is: discard candidates exceeding the unknown-rate ceiling or
spending more than 40% of the parameter budget on embeddings; among the rest take
the best fertility, then step down to any smaller vocabulary within 3% of it. On
the Assamese sweep this rejects 32k (12.3M embedding parameters) in favour of 24k
(9.2M, 36.9% of budget) at a cost of 3% fertility.

{section_sweep_table(artifacts)}

### 6.2 Held-out tokenizer metrics

{section_tokenizer_table(artifacts)}

### 6.3 Most frequent tokens (held-out text)

{section_top_tokens(artifacts)}

### 6.4 Tokenization examples

{section_examples(artifacts)}

---

## 7. Figures

{figure_block}

---

## 8. Reproduction

```bash
# 0. Environment
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Component self-tests
.venv/bin/python scripts/test_pipeline.py

# 2. Optional: OCR scanned books for the manual-collection share
.venv/bin/python scripts/run_ocr.py --language bn
.venv/bin/python scripts/run_ocr.py --language as

# 3. Build both corpora (independently)
.venv/bin/python scripts/build_corpus.py --config bengali/configs/pipeline.yaml \\
    --sources bengali/configs/sources.yaml --profile {profile}
.venv/bin/python scripts/build_corpus.py --config assamese/configs/pipeline.yaml \\
    --sources assamese/configs/sources.yaml --profile {profile}

# 4. Train tokenizers, with the vocabulary sweep
.venv/bin/python scripts/train_tokenizer.py --config bengali/configs/pipeline.yaml --sweep
.venv/bin/python scripts/train_tokenizer.py --config assamese/configs/pipeline.yaml --sweep

# 5. Exact token counts (and optional Phase 2 binary output)
.venv/bin/python scripts/tokenize_corpus.py --config bengali/configs/pipeline.yaml
.venv/bin/python scripts/tokenize_corpus.py --config assamese/configs/pipeline.yaml

# 6. Verify corpus/tokenizer independence (exits non-zero on violation)
.venv/bin/python scripts/validate_independence.py

# 7. Regenerate this report and all figures
.venv/bin/python scripts/make_report.py
```

Every stage writes its resolved configuration into its output JSON, so any run can
be reconstructed from the artifacts in `bengali/reports/` and `assamese/reports/`.

---

## 9. Deliverables Checklist

| Phase 1 deliverable | Location |
|---|---|
| Dataset collection scripts (both languages) | [`lma/data/sources.py`](../lma/data/sources.py), [`scripts/build_corpus.py`](../scripts/build_corpus.py), [`scripts/run_ocr.py`](../scripts/run_ocr.py) |
| Dataset preprocessing pipelines | [`lma/data/normalize.py`](../lma/data/normalize.py), [`lma/data/clean.py`](../lma/data/clean.py), [`lma/data/language_id.py`](../lma/data/language_id.py) |
| Per-language dataset statistics reports | `bengali/reports/`, `assamese/reports/`, and §4 above |
| Per-language train/validation/test splits | `bengali/data/processed/`, `assamese/data/processed/` |
| Tokenizer training code (both languages) | [`lma/tokenizer/train_tokenizer.py`](../lma/tokenizer/train_tokenizer.py), [`scripts/train_tokenizer.py`](../scripts/train_tokenizer.py) |
| Vocabulary files (one per language) | `bengali/tokenizer/*.vocab`, `assamese/tokenizer/*.vocab` |
| Tokenizer model files (one per language) | `bengali/tokenizer/*.model`, `assamese/tokenizer/*.model` |
"""


def main() -> int:
    """Generate figures and write the Phase 1 report."""
    args = parse_args()
    logger = setup_logging(None, name="make_report")

    report_dir = Path(args.report_dir)
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_artifacts()
    for language, entry in artifacts.items():
        logger.info("%s artifacts present: %s", language, sorted(entry) or "none")

    figures = build_figures(artifacts, figures_dir, logger)
    logger.info("Generated %d figure(s)", len(figures))

    validation = load_independence()
    if validation is not None:
        logger.info("Independence validation: %s", "passed" if validation.get("passed") else "FAILED")

    markdown = build_markdown(artifacts, figures, report_dir, validation)
    output = report_dir / "phase1_report.md"
    output.write_text(markdown, encoding="utf-8")
    logger.info("Wrote %s (%d bytes)", output, len(markdown.encode("utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
