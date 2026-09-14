# Phase 3 - Reasoning Finetuning, Attention Analysis, and Final Report

**Language Models and Agents, Monsoon 2026 - Individual Project**

Two independently pretrained decoder-only Transformers, each finetuned on a
synthetic comparative-reasoning corpus generated in its own language. No
data, vocabulary, or weights are shared between them at any stage.

---

## 1. The synthetic reasoning dataset

Generated programmatically rather than downloaded, so every ground-truth
label is known by construction and the corpus is reproducible from a seed.
The task states a chain of pairwise comparisons and asks for the extreme
element, which requires chaining the inequalities rather than matching a
single sentence.

| Property | Model H | Model L |
|---|---|---|
| Total examples | 20,000 | 20,000 |
| Train | 16,000 | 16,000 |
| Validation | 2,000 | 2,000 |
| Test | 2,000 | 2,000 |
| Unique prompts | 20,000 | 20,000 |
| Entity pool | 42 | 33 |
| Comparable dimensions | 4 | 4 |
| Chain lengths | 2, 3, 4 | 2, 3, 4 |

### Leakage control

Three mechanisms, each covering what the others do not. **Chain signatures**
- the unordered entity set plus the dimension - are partitioned across
splits, so no evaluation example re-tests a comparison any training example
covered. A minority of names appear **only in evaluation premises**, testing
that an unfamiliar participant does not derail the comparison. Within those
constraints, assignment is by salted content hash, as in Phase 1.

| Check | Model H | Model L |
|---|---|---|
| Chain-signature overlap (train vs eval) | 0 | 0 |
| Prompt overlap (train vs eval) | 0 | 0 |
| Duplicate prompts | 0 | 0 |
| Evaluation answers that are distractors | 0 | 0 |

**A rejected earlier design, recorded because it is the obvious one.**
Reserving names for the *answer* position - so every test answer was a token
absent from all training targets - makes the task unanswerable rather than
hard. Measured on a finetuned model: 0.0000 exact match on held-out-answer
examples against 0.2600 on examples whose prompts were equally unseen but
whose answers came from the training vocabulary. The evaluation set, not the
model, was the failure.

---

## 2. Finetuning

_Not run yet._

---

## 3. Reasoning results

_Not run yet._

---

## 4. Attention, before and after finetuning

_Not run yet._

---

## 5. The two resource tiers compared

### How data scale and quality differed

| | Model H | Model L |
|---|---|---|
| Training tokens | 580,162,728 | 612,476,098 |
| Manual share of tokens | 21.6% | 59.8% |
| Fertility (tokens/word) | - | - |
| Perplexity | 24.91 | 36.35 |
| Bits-per-byte | 0.5173 | 0.4856 |

### How language-modeling and reasoning results compare

Phase 2 found the two models closer than perplexity alone suggests: Model H
has the better perplexity but the worse bits-per-byte, because each Nepali
token carries more bytes of text. The reasoning task is a different kind of
measurement - it has a correct answer, so exact match is not distorted by
tokenizer granularity the way perplexity is, which makes it the cleaner
cross-tier comparison of the two.

### What tokenizer and corpus factors most affected the lower-resource model

Nepali's tokenizer is the less efficient of the two at equal vocabulary size
- higher fertility, fewer single-token words - so a fixed context holds less
language, and a fixed token budget covers fewer words. Its corpus is also the
narrower one: a single source supplies about half of it, against Hindi's
spread across seven. Volume was never the constraint; Nepali finished with
*more* training tokens than Hindi.

### What evidence explains the observed differences

The strongest available evidence is per-chain-length accuracy against chance,
together with the error profile: a model that fails by naming the wrong
entity has learned the task format and failed the inference, while one that
fails off-chain has not learned the format at all. These distinguish
explanations that aggregate accuracy cannot.

**One caveat, stated plainly.** There is one training run per language. A
difference between the two models could reflect ordinary between-run
variation rather than the resource tier. Establishing the causal claim would
need several seeds per language, which the compute budget did not allow.

---

## 6. Reproduction

```bash
# Component tests - templates, leakage, masking, resume, scoring, ablation
python scripts/test_reasoning.py

# Generate both reasoning datasets (seeded, so this reproduces them exactly)
python scripts/generate_reasoning_data.py --language hi
python scripts/generate_reasoning_data.py --language ne

# Finetune (notebooks/phase3_finetune.ipynb runs this on a GPU host)
python scripts/finetune_reasoning.py --language hi \
    --checkpoint hindi/checkpoints/checkpoint_best.pt \
    --tokenizer hindi/tokenizer/hi_unigram_8000.model

# Score pretrained against finetuned, and compare attention
python scripts/evaluate_reasoning.py --language hi \
    --pretrained hindi/checkpoints/checkpoint_best.pt \
    --finetuned hindi/checkpoints_finetuned/finetuned_best.pt \
    --tokenizer hindi/tokenizer/hi_unigram_8000.model

# Check deliverables, then regenerate this report
python scripts/check_phase3.py
python scripts/make_phase3_report.py
```
