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

> Nepali is on the project's permitted lower-resource list. Hindi and Nepali share
> the Devanagari script, which is deliberate: holding the writing system constant
> isolates **data scale and quality** as the variable under study. It also creates the
> main data-integrity hazard — the two languages share not only a script but much
> vocabulary, so they cannot be separated by a script check. See
> [Cross-language contamination](#cross-language-contamination).

---

## Status

| Phase | Branch | Status |
|---|---|---|
| **Phase 1** — Data collection and tokenizer construction | `phase-1` | **Complete** — 20/20 checks, built on the `full` profile |
| Phase 2 — Model implementation, pretraining, evaluation | `phase-2` | Not started |
| Phase 3 — Reasoning finetuning, attention analysis, final report | `phase-3` | Not started |

**Phase 1 report:** [`report/phase1_report.md`](report/phase1_report.md) — all
dataset statistics, tokenizer metrics, tables, and figures.

Both corpora were built end to end on the `full` profile and both exceed the
~500M-token target:

| | Training tokens | Manual tokens | Manual share |
|---|---:|---:|---:|
| **Hindi** (Model H) | 578,319,564 | 124,910,381 | **21.6%** |
| **Nepali** (Model L) | 610,989,716 | 364,830,857 | **59.7%** |

The manual share is measured **in training tokens**, as the assignment specifies.
Manually collected prose tokenizes at a different rate to downloaded web text, so
the token share and the word share differ — Hindi is 21.6% manual by token and
19.1% by word.

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
│   │   ├── normalize.py          # Script-aware Unicode normalization (hi vs ne)
│   │   ├── language_id.py        # Hindi-vs-Nepali discrimination (grammar-based)
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
├── hindi/                        # Model H — self-contained
│   ├── configs/pipeline.yaml     # Cleaning, splits, tokenizer, run profiles
│   ├── configs/sources.yaml      # Corpus source manifest
│   ├── data/processed/           # bn_{train,validation,test}.jsonl
│   ├── tokenizer/                # bn_*.model / bn_*.vocab
│   ├── reports/                  # Per-stage JSON statistics
│   └── logs/                     # Timestamped run logs
│
├── nepali/                       # Model L — self-contained, same structure
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
.venv/bin/python scripts/build_corpus.py --config hindi/configs/pipeline.yaml \
    --sources hindi/configs/sources.yaml --profile full
.venv/bin/python scripts/build_corpus.py --config nepali/configs/pipeline.yaml \
    --sources nepali/configs/sources.yaml --profile full
.venv/bin/python scripts/train_tokenizer.py --config hindi/configs/pipeline.yaml --sweep
.venv/bin/python scripts/train_tokenizer.py --config nepali/configs/pipeline.yaml --sweep
.venv/bin/python scripts/tokenize_corpus.py --config hindi/configs/pipeline.yaml
.venv/bin/python scripts/tokenize_corpus.py --config nepali/configs/pipeline.yaml
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

**Measured cost for the Hindi 500M ingest:** ~58 minutes of CPU on an 8-core laptop,
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
python scripts/build_corpus.py --config hindi/configs/pipeline.yaml \
    --sources hindi/configs/sources.yaml --profile full \
    --checkpoint-dir /content/drive/MyDrive/lma_project/hindi/checkpoints \
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

1. **Both languages reached 500M.** The assignment permits a documented shortfall
   for the lower-resource language, but it was not needed here;
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
  cache_dir: hindi/data/raw/scrape_cache
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

Hindi and Nepali share the Devanagari Unicode block, so a script check cannot
separate them, and they share a great deal of vocabulary besides. Since the project
forbids sharing documents between corpora — and Hindi text is far more abundant
online — leakage into the Nepali corpus is the
likely failure direction.

[`lma/data/language_id.py`](lma/data/language_id.py) scores documents on
**grammatical** evidence rather than orthographic, because the two languages share
the script and much of the lexicon. The discriminating forms are the ones present in
almost every sentence: copulas (Hindi `है`/`हैं`/`था` versus Nepali `छ`/`छन्`/`हो`/`थियो`)
and case markers (Hindi `ने`/`को`/`से` versus Nepali `ले`/`लाई`/`बाट`). A document is
accepted only when its language wins by a margin; ambiguous documents are dropped
rather than guessed. On validation samples the separation is clean: Hindi text scores
0.326 on Hindi markers and 0.000 on Nepali ones, and Nepali text the reverse.

This is not hypothetical. A late manual source supplied as "mixed" content measured
99.6% Nepali and 0.1% Hindi; the Hindi documents were dropped here rather than being
allowed into Model L.

### Normalization must not strip the Devanagari nukta

A generic Indic normalizer folds the nukta letters U+0958–U+095F onto their unmarked
bases, which merges distinct Hindi words: `ज़रा` ("a little") becomes `जरा` ("aged").
[`lma/data/normalize.py`](lma/data/normalize.py) is therefore language-profiled and
preserves the nukta for both languages. ZWNJ/ZWJ are preserved because they control
Devanagari conjunct
formation. The normalizer is idempotent — which required a second NFC pass, since
NFC *decomposes* the nukta letters that the unification rules compose.

### Measured, not assumed, thresholds

The near-duplicate threshold is **0.70**, not the customary 0.80: on real Hindi
prose, replacing one sentence of eight in an article measures at true Jaccard ≈0.79,
so a 0.80 cutoff misses ordinary syndicated-news duplicates. The script-ratio filter
measures Devanagari share among **letters only**, excluding digits, so digit-heavy
pages are attributed to the digit filter rather than misreported as "wrong script".

Near-duplicate detection is nevertheless **disabled** in both `pipeline.yaml` files.
On the Hindi run it rejected 0.19% of documents seen while consuming roughly 90% of
the runtime and enough memory to trigger the OOM killer; exact-hash deduplication,
which is retained, removed far more (1.5%) for negligible cost.

### Reproducibility

Splits are assigned by salted **content hash**, not shuffling — reproducible without
holding the corpus in memory, and stable under corpus growth, so adding documents
never moves an existing document from train to test. Every stage writes its resolved
configuration into its output JSON alongside the results.

---

## Large artifacts (Google Drive)

Per the submission instructions, large artifacts are hosted on Google Drive rather
than committed to git. The tokenizer models and vocabularies are small and **are**
in this repository, under `hindi/tokenizer/` and `nepali/tokenizer/`.

**Drive folder:** <https://drive.google.com/drive/folders/1ggiw-RzYTUgkXLhky2u73tObkw7MKbJT?usp=drive_link>

Permissions are set to **"Anyone with the link → Viewer"**, so the folder opens
without an access request.

### Folder structure

```
LMA-Individual-Project/
├── phase1-processed/     11.8 GB   the corpora behind every figure in the report
│   ├── hindi/            hi_train.jsonl, hi_validation.jsonl, hi_test.jsonl
│   └── nepali/           ne_train.jsonl, ne_validation.jsonl, ne_test.jsonl
├── phase1-manual/         5.7 GB   the manually collected sources (>=20% requirement)
│   ├── hindi/            bulk JSONL, wikipedia.txt, scraped/, ocr/
│   └── nepali/           bulk JSONL, scraped/
└── phase1-sources/       11.6 GB   mirrors of the public corpora, for convenience only
    ├── indiccorp.txt                 (AI4Bharat IndicCorp, Hindi)
    └── nepali_corpus_full.parquet    (public Nepali corpus)
```

`phase1-sources/` holds third-party public datasets under their own licences; they
are mirrored only so a reviewer can reproduce a build without re-downloading. Their
provenance is recorded in `hindi/configs/sources.yaml` and
`nepali/configs/sources.yaml`.

### Individual links

| Artifact | Phase | Size | Link |
|---|---|---|---|
| Hindi processed splits (Model H) | 1 | 5.23 GB | [`phase1-processed/hindi/`](https://drive.google.com/drive/folders/1ggiw-RzYTUgkXLhky2u73tObkw7MKbJT?usp=drive_link) |
| Nepali processed splits (Model L) | 1 | 6.61 GB | [`phase1-processed/nepali/`](https://drive.google.com/drive/folders/1ggiw-RzYTUgkXLhky2u73tObkw7MKbJT?usp=drive_link) |
| Hindi manual-collection sources | 1 | 1.31 GB | [`phase1-manual/hindi/`](https://drive.google.com/drive/folders/1ggiw-RzYTUgkXLhky2u73tObkw7MKbJT?usp=drive_link) |
| Nepali manual-collection sources | 1 | 4.36 GB | [`phase1-manual/nepali/`](https://drive.google.com/drive/folders/1ggiw-RzYTUgkXLhky2u73tObkw7MKbJT?usp=drive_link) |
| Public source corpora (optional) | 1 | 11.56 GB | [`phase1-sources/`](https://drive.google.com/drive/folders/1ggiw-RzYTUgkXLhky2u73tObkw7MKbJT?usp=drive_link) |
| Model H pretrained checkpoint | 2 | — | _pending_ |
| Model L pretrained checkpoint | 2 | — | _pending_ |
| Model H / L finetuned checkpoints | 3 | — | _pending_ |

Tokenizer models and vocabularies are **in the repository**, not on Drive:
[`hindi/tokenizer/`](hindi/tokenizer/) and [`nepali/tokenizer/`](nepali/tokenizer/).

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
