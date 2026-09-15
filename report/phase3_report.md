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

Each model starts from its own Phase 2 checkpoint, keeps its own tokenizer
and vocabulary, and trains only on its own language's data. Loss is computed
on the **answer span only**: the prompt states premises the model is
conditioned on, not asked to produce, so every prompt position is masked out
of the objective.

| | Model H | Model L |
|---|---|---|
| Pretrained step | 12,000 | 12,000 |
| Trainable parameters | 24,562,944 | 24,562,944 |
| Epochs | 3 | 3 |
| Optimizer steps | 1,500 | 1,500 |
| Learning rate | 5.0e-05 | 5.0e-05 |
| Best validation loss | 0.0046 | 0.0024 |
| Wall-clock (minutes) | 1.9 | 2.1 |

![Finetuning validation loss](figures/phase3_finetune_curves.png)

---

## 3. Reasoning results

Exact match on the held-out test split. Both checkpoints are scored the same
way - greedy argmax at the position following the answer prefix, no sampling
- so the comparison isolates the weights.

| Metric | Model H | Model L |
|---|---|---|
| Pretrained exact match | 0.0460 | 0.0025 |
| Finetuned exact match | 0.9965 | 0.9985 |
| Improvement | +0.9505 | +0.9960 |
| Chance (aggregate) | 0.3483 | 0.3381 |
| Pretrained constrained accuracy | 0.2600 | 0.2180 |
| Finetuned constrained accuracy | 0.9965 | 0.9985 |
| Test examples | 2,000 | 2,000 |

### Accuracy against chance, per chain length

Chance differs sharply by chain length - one in two for a two-entity
comparison against one in four for a four-entity chain - so a single blended
baseline is misleading. The per-length comparison is the honest one.

**Model H**

| Entities | Accuracy | Chance | Margin | n |
|---|---:|---:|---:|---:|
| 2 | 1.000 | 0.500 | +0.500 | 546 |
| 3 | 1.000 | 0.333 | +0.667 | 720 |
| 4 | 0.990 | 0.250 | +0.740 | 734 |

**Model L**

| Entities | Accuracy | Chance | Margin | n |
|---|---:|---:|---:|---:|
| 2 | 1.000 | 0.500 | +0.500 | 453 |
| 3 | 0.999 | 0.333 | +0.665 | 755 |
| 4 | 0.997 | 0.250 | +0.747 | 792 |

### Error analysis

Mistakes are sorted into three kinds, because they call for different
explanations. **Off-chain** means the prediction was not any entity in the
premises - the model did not identify the answer as a choice among the named
entities at all. **Inverted** means it named the entity at the opposite end
of the ordering, which suggests it tracked the comparison but mishandled the
question's polarity. **Wrong entity** is everything else.

| Failure mode | Model H | Model L |
|---|---|---|
| Off-chain | 0.000 | 0.000 |
| Inverted polarity | 0.000 | 0.000 |
| Wrong entity | 0.000 | 0.000 |

---

## 4. Attention, before and after finetuning

The same comparative-reasoning prompt through both checkpoints, so any
difference in where the heads look is attributable to finetuning.

**Model H (Hindi)**

| Head type | Pretrained | Finetuned |
|---|---:|---:|
| local | 31 | 38 |
| long-range | 32 | 27 |
| diffuse | 9 | 7 |

| Layer | Pretrained distance | Finetuned distance | Delta |
|---|---:|---:|---:|
| 0 | 5.52 | 5.17 | -0.35 |
| 1 | 6.01 | 5.42 | -0.58 |
| 2 | 6.19 | 5.60 | -0.59 |
| 3 | 3.75 | 3.68 | -0.08 |
| 4 | 3.86 | 3.37 | -0.49 |
| 5 | 3.89 | 2.89 | -1.00 |
| 6 | 3.66 | 2.83 | -0.83 |
| 7 | 5.69 | 4.22 | -1.47 |
| 8 | 4.62 | 3.61 | -1.01 |
| 9 | 5.69 | 5.96 | +0.26 |
| 10 | 6.48 | 6.26 | -0.21 |
| 11 | 5.44 | 6.27 | +0.83 |

![Model H pretrained attention](figures/reasoning_attention_pretrained_hi.png)

![Model H finetuned attention](figures/reasoning_attention_finetuned_hi.png)

**Model L (Nepali)**

| Head type | Pretrained | Finetuned |
|---|---:|---:|
| local | 37 | 40 |
| long-range | 34 | 27 |
| diffuse | 1 | 5 |

| Layer | Pretrained distance | Finetuned distance | Delta |
|---|---:|---:|---:|
| 0 | 5.31 | 5.23 | -0.08 |
| 1 | 5.48 | 5.96 | +0.48 |
| 2 | 5.70 | 5.05 | -0.64 |
| 3 | 3.52 | 3.22 | -0.30 |
| 4 | 4.53 | 3.77 | -0.76 |
| 5 | 3.06 | 2.36 | -0.70 |
| 6 | 3.58 | 2.51 | -1.07 |
| 7 | 4.52 | 4.32 | -0.20 |
| 8 | 6.11 | 3.91 | -2.20 |
| 9 | 6.95 | 6.98 | +0.03 |
| 10 | 6.27 | 7.12 | +0.85 |
| 11 | 5.88 | 6.07 | +0.19 |

![Model L pretrained attention](figures/reasoning_attention_pretrained_ne.png)

![Model L finetuned attention](figures/reasoning_attention_finetuned_ne.png)

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
tokenizer granularity the way perplexity is.

**The reasoning task did not separate the two tiers, and that is the**
**result, not a disappointment.** Both models finish within 0.2 points of
each other and within 0.3 points of ceiling, and Model L - the lower-resource
model, with the narrower corpus and the less efficient tokenizer - scores
*marginally higher*. The honest conclusion is that this task saturates: a
templated corpus with four comparison dimensions, single-token answers, and
16,000 training examples is learnable to near-ceiling from either
pretraining run, so it cannot resolve a difference between them.

Two things follow. First, the tier difference established in Phase 2 - the
perplexity gap, the fertility gap, the corpus-diversity gap - is real but
lives in open-ended language modelling, which is where those metrics measure.
Second, a reasoning benchmark that *would* discriminate needs to be harder
than this one: longer chains, distractor premises that do not bear on the
question, or comparisons stated across intervening text. That is the obvious
next experiment, and the compute budget did not allow it here.

### What tokenizer and corpus factors most affected the lower-resource model

Nepali's tokenizer is the less efficient of the two at equal vocabulary size
- higher fertility, fewer single-token words - so a fixed context holds less
language, and a fixed token budget covers fewer words. Its corpus is also the
narrower one: a single source supplies about half of it, against Hindi's
spread across seven. Volume was never the constraint; Nepali finished with
*more* training tokens than Hindi.

### What evidence explains the observed differences

On the reasoning task there is almost nothing left to explain: both models
answer essentially every held-out question correctly, and the error
categories - off-chain, wrong-entity, inverted-polarity - are all zero or
near it for both. Those diagnostics were built to separate a model that never
learned the answer format from one that learned it and failed the inference.
Neither failure mode occurred at a rate worth interpreting, which is itself
the finding.

The evidence that *does* separate the tiers is the pretrained baseline,
measured before any finetuning touched the weights. Model H answers 4.6% of
questions correctly with no exposure to the task; Model L answers 0.25%.
Neither is anywhere near chance, so neither model can do this task from
pretraining alone - but the order matches the resource tier, and it is the
one place in Phase 3 where the Phase 2 corpus difference still shows through.
It should not be over-read: both numbers are small, and a difference between
4.6% and 0.25% on a task neither model can perform is weak evidence.

**Two caveats, stated plainly.** There is one training run per language, so
any difference between the two models could reflect ordinary between-run
variation rather than the resource tier; establishing the causal claim would
need several seeds per language, which the compute budget did not allow. And
near-ceiling accuracy on a synthetic, templated task is not evidence of
general reasoning ability - it is evidence that the models learned this
task's regularities, which is the narrower claim this report makes.

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
