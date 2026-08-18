# Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

**Language Models and Agents — Monsoon 2026 — Individual Project**

Two **completely independent** decoder-only Transformer language models, built from
scratch in PyTorch, with separate datasets, tokenizers, vocabularies, and weights.

| | Model H (higher-resource) | Model L (lower-resource) |
|---|---|---|
| Language | **Hindi** (`hi`) | **Nepali** (`ne`) |
| Directory | [`hindi/`](hindi/) | [`nepali/`](nepali/) |
| Script | Devanagari | Devanagari |
| Corpus | own, independent | own, independent |
| Tokenizer / vocabulary | own, not shared | own, not shared |

> Assamese is on the project's permitted lower-resource list. Bengali and Assamese
> share a script, which is deliberate: holding the writing system constant isolates
> **data scale and quality** as the variable under study. It also creates the main
> data-integrity hazard — see [Cross-language contamination](#cross-language-contamination).

---

## Status

| Phase | Branch | Status |
|---|---|---|
| **Phase 1** — Data collection and tokenizer construction | `phase-1` | **Complete** (verified on the `sampled` profile) |
| Phase 2 — Model implementation, pretraining, evaluation | `phase-2` | Not started |
| Phase 3 — Reasoning finetuning, attention analysis, final report | `phase-3` | Not started |

**Phase 1 report:** [`report/phase1_report.md`](report/phase1_report.md) — all
dataset statistics, tokenizer metrics, tables, and figures.

The full pipeline is implemented and has been executed end to end on the `sampled`
profile, which produced every statistic, tokenizer, and figure in the report. The
`full` profile runs the same code against the ~500M-token target; see
[Run profiles](#run-profiles).

Verify the state of any checkout with:

```bash
.venv/bin/python scripts/test_pipeline.py          # 84 component assertions
.venv/bin/python scripts/validate_independence.py  # corpus/tokenizer independence
.venv/bin/python scripts/check_phase1.py           # deliverables and constraints
```

---

## Repository layout

```
├── lma/                          # Shared, language-agnostic implementation
│   ├── data/
│   │   ├── normalize.py          # Script-aware Unicode normalization (bn vs as)
│   │   ├── language_id.py        # Bengali-vs-Assamese discrimination
│   │   ├── clean.py              # Quality filters + MinHash/LSH deduplication
│   │   ├── sources.py            # HF streaming, web scraping, local/OCR sources
│   │   ├── ocr.py                # PDF/book OCR ingestion
│   │   ├── checkpoint.py         # Resume state for long ingests
│   │   └── splits.py             # Deterministic hash-based train/val/test splits
│   ├── tokenizer/
│   │   └── train_tokenizer.py    # SentencePiece training from scratch
│   ├── stats/
│   │   ├── corpus_stats.py       # Streaming corpus statistics
│   │   ├── tokenizer_stats.py    # Fertility, unk/byte-fallback, vocab sweep
│   │   └── plots.py              # All figures (computation-free)
│   └── utils/logging_utils.py
│
├── bengali/                      # Model H — self-contained
│   ├── configs/pipeline.yaml     # Cleaning, splits, tokenizer, run profiles
│   ├── configs/sources.yaml      # Corpus source manifest
│   ├── data/processed/           # bn_{train,validation,test}.jsonl
│   ├── tokenizer/                # bn_*.model / bn_*.vocab
│   ├── reports/                  # Per-stage JSON statistics
│   └── logs/                     # Timestamped run logs
│
├── assamese/                     # Model L — self-contained, same structure
│
├── scripts/                      # Executable pipeline stages
│   ├── test_pipeline.py          # Component self-tests (84 assertions)
│   ├── build_corpus.py           # Collect -> normalize -> clean -> dedup -> split
│   ├── train_tokenizer.py        # Vocabulary sweep + final tokenizer + evaluation
│   ├── tokenize_corpus.py        # Exact token counts (+ optional Phase 2 .bin)
│   ├── run_ocr.py                # OCR scanned PDFs into the corpus
│   ├── validate_independence.py  # Proves the two models share no data/vocabulary
│   ├── check_phase1.py           # Deliverable and constraint checklist
│   └── make_report.py            # Figures + Markdown report
│
├── notebooks/
│   └── phase1_full_scale_colab.ipynb   # 500M-token run on Colab, checkpointed to Drive
│
├── report/
│   ├── phase1_report.md
│   └── figures/
└── requirements.txt
```

The two language directories are **self-contained and never cross-reference each
other**. Shared code in `lma/` is language-agnostic and is configured per language
through `configs/`, so no data, vocabulary, or statistic is shared between models.

---

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Verify the pipeline components (no network needed)
.venv/bin/python scripts/test_pipeline.py

# Build both corpora, train both tokenizers, and regenerate the report
.venv/bin/python scripts/build_corpus.py --config bengali/configs/pipeline.yaml \
    --sources bengali/configs/sources.yaml --profile sampled
.venv/bin/python scripts/build_corpus.py --config assamese/configs/pipeline.yaml \
    --sources assamese/configs/sources.yaml --profile sampled
.venv/bin/python scripts/train_tokenizer.py --config bengali/configs/pipeline.yaml --sweep
.venv/bin/python scripts/train_tokenizer.py --config assamese/configs/pipeline.yaml --sweep
.venv/bin/python scripts/tokenize_corpus.py --config bengali/configs/pipeline.yaml
.venv/bin/python scripts/tokenize_corpus.py --config assamese/configs/pipeline.yaml
.venv/bin/python scripts/make_report.py
```

### Run profiles

Each `pipeline.yaml` defines two profiles, selected with `--profile`:

- **`sampled`** — bounded run (per-source document caps). Verifies every stage end
  to end and produces real statistics and figures quickly.
- **`full`** — production run targeting ~500M training tokens per language. Streaming
  means the corpus is filtered as it downloads, and the ingest stops once the token
  budget is met, so runtime scales with the target rather than the source size.

To scale up, also widen the Sangraha `data_files` patterns in `configs/sources.yaml`
from a single shard (`verified/ben/data-0.parquet`) to all of them
(`verified/ben/*.parquet`).

---

## Running at full scale (~500M tokens)

**Measured cost for the Bengali 500M ingest:** ~13.5 hours of CPU at 38k chars/sec,
streaming ~30 GB of Parquet across 77 shards. On a laptop with 7 GB RAM the
deduplication index — one MinHash signature per kept document — will exhaust memory
long before the run finishes. **Use Colab**, not a laptop.

[`notebooks/phase1_full_scale_colab.ipynb`](notebooks/phase1_full_scale_colab.ipynb)
runs exactly this pipeline with the caps removed. Open it in Colab, set a **CPU**
runtime (a GPU does nothing for this stage and spends quota you need for Phase 2),
and work through it top to bottom.

### Checkpoint and resume

The ingest checkpoints, because a multi-hour run *will* be interrupted:

```bash
python scripts/build_corpus.py --config bengali/configs/pipeline.yaml \
    --sources bengali/configs/sources.yaml --profile full \
    --checkpoint-dir /content/drive/MyDrive/lma_project/bengali/checkpoints \
    --checkpoint-every 25000 --resume
```

Re-running the identical command after a disconnect continues from the checkpoint.
`--resume` is safe on a first run — no checkpoint simply means start fresh. Use
`--restart` to discard a checkpoint and begin the profile again.

Four pieces of state are persisted, and each is load-bearing:

| State | Why it must be saved |
|---|---|
| Per-source position | Resume skips documents each source already yielded |
| Deduplication index | Without it, the resumed half accumulates repeats of the first — silently |
| Split counters and provenance | So final statistics describe the whole corpus |
| Written-document counts | Output files are reconciled against the checkpoint before appending |

**Verified property:** a run interrupted with `kill -9` and resumed produces a
**byte-identical corpus** to an uninterrupted one — same documents, same order, zero
duplicates, identical per-source counts. Checkpoint writes are atomic (temp file +
rename), so a process killed mid-save leaves the previous checkpoint intact.

### Two things that change at full scale

1. **Assamese will not reach 500M.** Public Assamese text is roughly two orders of
   magnitude scarcer than Bengali. The assignment permits a documented shortfall;
   `scripts/tokenize_corpus.py` reports the exact achieved count and gap.
2. **The manual-collection share must be re-established.** It clears 20% on the
   sampled profile only because downloaded sources are capped. Removing those caps
   multiplies the downloaded side ~100× while scraped volume stays fixed, pushing the
   share below the requirement. Either scrape/OCR substantially more material, or keep
   a cap on downloads — a smaller corpus that meets the requirement beats a larger one
   that does not.

---

## Adding your own sources (manual collection)

The project requires **≥20% of final training tokens to come from manual
collection**. Both manifests ship with working scrape and OCR sources plus
templates. Manual collection has two paths:

**1. Web scraping** — add an entry to `<language>/configs/sources.yaml`:

```yaml
- source_id: manual_scrape_my_site
  type: scrape
  collection: manual          # counted toward the 20% requirement
  enabled: true
  start_urls: [https://example.com/archive]
  allow_patterns: ['example\.com/article/']
  content_selector: null      # null = densest-text-block heuristic
  max_pages: 1000
  delay_seconds: 1.5
  cache_dir: bengali/data/raw/scrape_cache
```

The crawler honours `robots.txt`, rate-limits per host, stays on the seed hosts,
caps total pages, and caches raw HTML so re-runs do not re-fetch. **Verifying that
you are permitted to collect from a given site is your responsibility.**

**2. OCR of books/PDFs** — drop PDFs into `<language>/data/raw/pdf/` and run:

```bash
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-ben tesseract-ocr-asm
.venv/bin/python scripts/run_ocr.py --language bn
```

PDFs with an embedded text layer are extracted directly (faster and more accurate);
image-only pages are rasterized at 300 DPI and passed to Tesseract. Output JSONL is
picked up automatically by the `manual_ocr_*` source on the next corpus build.

`build_corpus.py` reports the achieved manual share and **warns if it is below 20%**.

---

## Design decisions worth knowing

### Cross-language contamination

Bengali and Assamese share the Bengali Unicode block, so a script check cannot
separate them and general-purpose language identifiers routinely mislabel Assamese
as Bengali. Since the project forbids sharing documents between corpora — and
Bengali text is far more abundant online — leakage into the Assamese corpus is the
likely failure direction.

[`lma/data/language_id.py`](lma/data/language_id.py) scores documents on
orthographic evidence: the Assamese-exclusive letters `ৰ` (U+09F0) and `ৱ` (U+09F1),
the `ৰ`:`র` ratio, and divergent high-frequency function words. A document is
accepted only when its language wins by a margin; ambiguous documents are dropped
rather than guessed. Spot-checking Sangraha's `verified/asm` split during
development did turn up unrelated Latin-script text, which this stage removes.

### Normalization must not "fix" Assamese into Bengali

A generic Indic normalizer folds U+09F0 → U+09B0, silently rewriting Assamese as
Bengali. [`lma/data/normalize.py`](lma/data/normalize.py) is therefore
language-profiled: Assamese preserves `ৰ`/`ৱ`, Bengali folds them (where they
indicate OCR noise). ZWNJ/ZWJ are preserved because they control Indic conjunct
formation. The normalizer is idempotent — which required a second NFC pass, since
NFC *decomposes* the nukta letters that the unification rules compose.

### Measured, not assumed, thresholds

The near-duplicate threshold is **0.70**, not the customary 0.80: on real Bengali
prose, replacing one sentence of eight in an article measures at true Jaccard ≈0.79,
so a 0.80 cutoff misses ordinary syndicated-news duplicates. The script-ratio filter
measures Bengali share among **letters only**, excluding digits, so digit-heavy
pages are attributed to the digit filter rather than misreported as "wrong script".

### Reproducibility

Splits are assigned by salted **content hash**, not shuffling — reproducible without
holding the corpus in memory, and stable under corpus growth, so adding documents
never moves an existing document from train to test. Every stage writes its resolved
configuration into its output JSON alongside the results.

---

## Large artifacts (Google Drive)

Per the submission instructions, large artifacts are hosted on Drive rather than
committed. Links are added as each phase produces them.

| Artifact | Phase | Link |
|---|---|---|
| Bengali corpus (processed splits) | 1 | _to be added_ |
| Assamese corpus (processed splits) | 1 | _to be added_ |
| Bengali tokenizer (model + vocab) | 1 | _to be added_ |
| Assamese tokenizer (model + vocab) | 1 | _to be added_ |
| Model H pretrained checkpoint | 2 | _pending_ |
| Model L pretrained checkpoint | 2 | _pending_ |
| Model H / L finetuned checkpoints | 3 | _pending_ |

Tokenizer `.model`/`.vocab` files are small enough to remain in the repository;
processed corpora and checkpoints go to Drive.

---

## Constraints observed

- No pretrained language models, and no pretrained tokenizers — both vocabularies
  are fitted from scratch on this project's own corpora.
- No HuggingFace Transformer model classes. `datasets` is used only as a corpus
  download client; `sentencepiece` only to train vocabularies.
- Not a multilingual model: no shared vocabulary, no shared data, no shared weights.
- Computation and visualization logic are kept in separate functions; every figure
  carries a title, axis labels, and a legend where applicable.
