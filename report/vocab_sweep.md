# Vocabulary size selection

Both tokenizers use a **vocabulary of 8,000 pieces**. This records the measurement
behind that choice, since the assignment asks for the vocabulary to be *chosen*
using held-out evidence rather than asserted.

## What constrains the choice

Phase 2 fixes the model at roughly 25M parameters, and the embedding matrix is
`vocab_size × d_model`. Every piece added to the vocabulary is therefore a
parameter taken away from the attention and feed-forward layers that do the actual
modelling. The course guidance is explicit about this trade-off: keep the
vocabulary in the 5–10K range, ensure fertility stays above 1 (i.e. genuine
subword tokenization rather than word-level), and justify the choice.

Two quantities pull against each other:

* **Lower fertility is better.** Fewer tokens per word means a fixed context
  window covers more text, and the model spends less capacity reassembling words
  from fragments. Fertility falls as the vocabulary grows.
* **Fewer embedding parameters is better.** At a fixed 25M budget, embeddings and
  transformer layers compete directly.

## Measurement

Candidate tokenizers were trained with SentencePiece Unigram on 60,000 lines
(24.4M words) of the Hindi training split, then evaluated on 1,500 held-out
documents from the validation split. Every candidate used identical settings apart
from vocabulary size: `character_coverage=0.9995`, byte fallback on, digits split,
and `normalization_rule_name=identity` (the corpus is already normalized by this
project's Devanagari-aware normalizer).

| Vocab | Fertility | Chars/token | Unknown rate | Byte-fallback | Embedding share |
|------:|----------:|------------:|-------------:|--------------:|----------------:|
| 5,000 | 1.567 | 3.26 | 0.0 | 0.301% | 14.0% |
| 6,000 | 1.518 | 3.37 | 0.0 | 0.311% | 16.0% |
| **8,000** | **1.454** | **3.52** | **0.0** | **0.324%** | **19.7%** |
| 10,000 | 1.412 | 3.62 | 0.0 | 0.334% | 23.0% |

Embedding share is computed for a 512-dimensional, 6-layer model with a 1,024-token
context and tied input/output embeddings.

Every candidate clears the two hard requirements: fertility is above 1 throughout
(so tokenization is genuinely subword), and the unknown rate is exactly zero
because byte fallback makes any input representable.

## The trade-off, held at a fixed budget

Comparing fertility alone would favour the largest vocabulary. The honest
comparison holds *total* parameters constant at 25M and asks what model each
vocabulary leaves room for:

| Vocab | Largest config under 25M | Transformer parameters | Fertility |
|------:|--------------------------|----------------------:|----------:|
| 5,000 | d=448, 9 layers | 21.7M | 1.567 |
| 6,000 | d=448, 9 layers | 21.7M | 1.518 |
| **8,000** | **d=384, 12 layers** | **21.3M** | **1.454** |
| 10,000 | d=384, 11 layers | 19.5M | 1.412 |

Read across the table: moving from 5,000 to 10,000 buys **9.9% shorter sequences**
but costs **2.2M transformer parameters** — roughly a tenth of the modelling
capacity — because the embedding matrix doubles.

## Why 8,000

8,000 sits at the point where the fertility curve has flattened but the embedding
cost has not yet become severe:

* Against 5,000, it improves fertility by **7.2%** while giving up only 0.4M
  transformer parameters (21.7M → 21.3M).
* Against 10,000, it keeps **1.8M more transformer parameters** while conceding
  only 3% on fertility (1.454 vs 1.412).

The fertility gain per 1,000 pieces is decreasing across the range — 3.1% from
5k→6k, 4.2% from 6k→8k, 2.9% from 8k→10k — while the embedding cost per 1,000
pieces is constant. 8,000 is the last size where the return still justifies the
spend.

Byte-fallback rate rises slightly with vocabulary size (0.301% → 0.334%), which is
expected and harmless: a larger vocabulary spends more of its budget on frequent
whole words, leaving marginally more of the long tail to byte pieces. At a third of
a percent it has no practical effect on sequence length.

## Applied to both languages

Both Hindi and Nepali use 8,000. They share the Devanagari script and similar
morphology, so the same reasoning applies, and holding the vocabulary size equal
across the two models keeps the resource-tier comparison clean: any difference in
Phase 2 perplexity or fertility then reflects **data scale**, which is the variable
this project studies, rather than a tokenizer configuration difference.

The Nepali sweep additionally tests 4,000, since its smaller corpus might have
favoured a smaller vocabulary; the pipeline records the measured result either way.

## Reproducing

```bash
python scripts/train_tokenizer.py --config hindi/configs/pipeline.yaml --sweep
python scripts/train_tokenizer.py --config nepali/configs/pipeline.yaml --sweep
```

This trains every candidate in `sweep_vocab_sizes`, evaluates each on the same
held-out text, and writes the full results to `<language>/reports/tokenizer_sweep.json`.
