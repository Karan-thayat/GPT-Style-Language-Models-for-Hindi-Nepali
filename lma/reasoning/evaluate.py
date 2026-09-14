"""Scoring a model on the reasoning task, before and after finetuning.

The headline number the assignment asks for is "pretrained vs. finetuned accuracy
(or exact match) on the synthetic test set". That comparison is only meaningful
if both models are scored the *same* way, so everything here runs identically
against either checkpoint - the pretrained model is not given an easier decoding
path or a more forgiving match rule.

How an answer is scored
-----------------------
Every answer in the generated corpus is a single token by construction (the
entity pools in :mod:`lma.reasoning.templates` are vetted against each language's
own tokenizer), so the model's answer is the token it puts highest probability on
immediately after the answer prefix. That makes exact match unambiguous and
removes decoding temperature as a confound: greedy argmax at one position, no
sampling, no length normalization.

Two accuracies are reported, and the gap between them is informative:

* **Free exact match** - argmax over the whole vocabulary. This is the honest
  measure of whether the model solved the task, and it is what the headline
  number reports.
* **Constrained accuracy** - argmax restricted to the entities named in that
  example's own premises. A pretrained model that has never seen the task will
  score near zero free but near chance constrained, which separates "cannot
  produce a name at all" from "produces a name but the wrong one".

Reporting only the first would understate what a pretrained model knows;
reporting only the second would overstate it.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from ..tokenizer.train_tokenizer import BOS_ID

#: Chance accuracy is 1/len(entities), which differs by chain length, so the
#: report breaks accuracy down per chain length rather than quoting one baseline.
#: See :func:`above_chance`, which is the comparison that should be quoted.


@torch.no_grad()
def score_examples(
    model,
    examples: list[dict],
    tokenizer,
    device: str | None = None,
    batch_size: int = 32,
) -> dict:
    """Evaluate one model on reasoning examples.

    Args:
        model: A :class:`~lma.model.TransformerLM`, pretrained or finetuned.
        examples: Rows carrying ``prompt``, ``answer``, ``entities``, and
            ``chain_length``.
        tokenizer: SentencePiece processor for this language.
        device: Where to run. Defaults to the model's device.
        batch_size: Prompts per forward pass.

    Returns:
        Overall and per-slice accuracy, mean answer log-probability, and the
        predictions themselves for qualitative inspection.
    """
    device = device or str(next(model.parameters()).device)
    model.eval()

    correct_free = 0
    correct_constrained = 0
    total = 0
    answer_logprob_sum = 0.0

    by_chain: dict[int, list[int]] = defaultdict(list)
    by_hops: dict[int, list[int]] = defaultdict(list)
    by_dimension: dict[str, list[int]] = defaultdict(list)
    by_polarity: dict[str, list[int]] = defaultdict(list)
    predictions: list[dict] = []

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]

        # Left-padding would shift the answer position per row; instead each
        # prompt is run to its own final position, so the logits read are always
        # the ones immediately following that prompt's last token.
        encoded = [[BOS_ID, *tokenizer.encode(row["prompt"], out_type=int)] for row in chunk]
        width = max(len(ids) for ids in encoded)
        padded = [ids + [0] * (width - len(ids)) for ids in encoded]
        batch = torch.tensor(padded, dtype=torch.long, device=device)

        logits = model(batch)["logits"]

        for index, row in enumerate(chunk):
            position = len(encoded[index]) - 1
            row_logits = logits[index, position, :].float()
            log_probs = torch.log_softmax(row_logits, dim=-1)

            answer_ids = tokenizer.encode(row["answer"], out_type=int)
            answer_id = answer_ids[0]

            predicted_id = int(row_logits.argmax().item())
            free_hit = int(predicted_id == answer_id)

            # Constrained: restrict to the entities this example actually names.
            candidate_ids = []
            for entity in row["entities"]:
                ids = tokenizer.encode(entity, out_type=int)
                candidate_ids.append(ids[0])
            candidate_tensor = torch.tensor(candidate_ids, device=row_logits.device)
            best_candidate = candidate_ids[
                int(row_logits[candidate_tensor].argmax().item())
            ]
            constrained_hit = int(best_candidate == answer_id)

            correct_free += free_hit
            correct_constrained += constrained_hit
            total += 1
            answer_logprob_sum += float(log_probs[answer_id].item())

            by_chain[row["chain_length"]].append(free_hit)
            by_hops[row.get("hops", row["chain_length"] - 1)].append(free_hit)
            by_dimension[row.get("dimension", "unknown")].append(free_hit)
            by_polarity["most" if row.get("ask_most") else "least"].append(free_hit)

            if len(predictions) < 40:
                predictions.append(
                    {
                        "prompt": row["prompt"],
                        "expected": row["answer"],
                        "predicted": tokenizer.id_to_piece(predicted_id).lstrip("▁"),
                        "correct": bool(free_hit),
                        "chain_length": row["chain_length"],
                    }
                )

    def rates(groups: dict) -> dict:
        """Convert per-slice hit lists into accuracy and support."""
        return {
            str(key): {
                "accuracy": sum(hits) / len(hits) if hits else 0.0,
                "n": len(hits),
            }
            for key, hits in sorted(groups.items(), key=lambda item: str(item[0]))
        }

    mean_logprob = answer_logprob_sum / max(total, 1)
    return {
        "examples": total,
        "exact_match": correct_free / max(total, 1),
        "constrained_accuracy": correct_constrained / max(total, 1),
        "answer_mean_logprob": mean_logprob,
        "answer_perplexity": math.exp(min(-mean_logprob, 20)),
        "by_chain_length": rates(by_chain),
        "by_hops": rates(by_hops),
        "by_dimension": rates(by_dimension),
        "by_polarity": rates(by_polarity),
        "predictions": predictions,
    }


def chance_baseline(examples: list[dict]) -> dict:
    """Accuracy a model would reach by guessing among each example's entities.

    Reported alongside the measured numbers because "above chance" is the only
    meaningful claim for a constrained-choice task, and chance differs by chain
    length: one in two for a two-entity comparison, one in four for a four-entity
    chain.
    """
    total = 0.0
    by_chain: dict[int, list[float]] = defaultdict(list)

    for row in examples:
        entities = len(row["entities"])
        probability = 1.0 / entities if entities else 0.0
        total += probability
        by_chain[row["chain_length"]].append(probability)

    return {
        "overall": total / max(len(examples), 1),
        "by_chain_length": {
            str(key): sum(values) / len(values)
            for key, values in sorted(by_chain.items())
        },
        "support_by_chain_length": {
            str(key): len(values) for key, values in sorted(by_chain.items())
        },
    }


def above_chance(results: dict, baseline: dict) -> dict:
    """Compare accuracy against chance *per chain length*, not in aggregate.

    Quoting one blended baseline is misleading here, because chance differs
    sharply by chain length - one in two for a two-entity comparison against one
    in four for a four-entity chain - so a corpus weighted toward short chains
    carries a high blended baseline that a model can beat on the hard examples
    and still miss overall. The per-length comparison is the honest one, and the
    aggregate is reported only alongside it.

    Args:
        results: :func:`score_examples` output.
        baseline: :func:`chance_baseline` output for the same examples.

    Returns:
        Per-length accuracy, chance, and margin, plus how many lengths clear it.
    """
    per_length = {}
    for key, payload in results.get("by_chain_length", {}).items():
        chance = baseline.get("by_chain_length", {}).get(key, 0.0)
        per_length[key] = {
            "accuracy": payload["accuracy"],
            "chance": chance,
            "margin": payload["accuracy"] - chance,
            "above_chance": payload["accuracy"] > chance,
            "n": payload["n"],
        }

    clearing = [key for key, row in per_length.items() if row["above_chance"]]
    return {
        "by_chain_length": per_length,
        "lengths_above_chance": sorted(clearing),
        "lengths_evaluated": sorted(per_length),
        "aggregate_accuracy": results.get("exact_match", 0.0),
        "aggregate_chance": baseline.get("overall", 0.0),
    }


def compare_pretrained_finetuned(before: dict, after: dict) -> dict:
    """Summarize what finetuning changed.

    Args:
        before: :func:`score_examples` output for the pretrained checkpoint.
        after: The same for the finetuned checkpoint.

    Returns:
        Absolute and relative deltas on the headline metrics, plus per-slice
        exact-match deltas.
    """
    delta_by_chain = {}
    for key, payload in after.get("by_chain_length", {}).items():
        prior = before.get("by_chain_length", {}).get(key, {}).get("accuracy", 0.0)
        delta_by_chain[key] = {
            "pretrained": prior,
            "finetuned": payload["accuracy"],
            "delta": payload["accuracy"] - prior,
            "n": payload["n"],
        }

    return {
        "exact_match": {
            "pretrained": before["exact_match"],
            "finetuned": after["exact_match"],
            "delta": after["exact_match"] - before["exact_match"],
        },
        "constrained_accuracy": {
            "pretrained": before["constrained_accuracy"],
            "finetuned": after["constrained_accuracy"],
            "delta": after["constrained_accuracy"] - before["constrained_accuracy"],
        },
        "answer_mean_logprob": {
            "pretrained": before["answer_mean_logprob"],
            "finetuned": after["answer_mean_logprob"],
            "delta": after["answer_mean_logprob"] - before["answer_mean_logprob"],
        },
        "by_chain_length": delta_by_chain,
    }


def error_analysis(results: dict, examples: list[dict]) -> dict:
    """Characterize the mistakes, for the report's qualitative section.

    Three failure modes are distinguished, because they call for different
    explanations:

    * **off_chain** - the predicted token is not any entity in the premises, so
      the model did not even identify the answer as a choice among the named
      entities.
    * **wrong_entity** - a named entity, but the wrong one: the model understood
      the format and failed the inference.
    * **inverted** - specifically the entity at the opposite end of the ordering,
      which suggests the model tracked the comparison but mishandled the
      question's polarity.
    """
    counts = {"off_chain": 0, "wrong_entity": 0, "inverted": 0}
    index = {row["prompt"]: row for row in examples}

    for prediction in results.get("predictions", []):
        if prediction["correct"]:
            continue
        row = index.get(prediction["prompt"])
        if row is None:
            continue
        predicted = prediction["predicted"]
        entities = row["entities"]

        if predicted not in entities:
            counts["off_chain"] += 1
            continue

        opposite = entities[-1] if row.get("ask_most") else entities[0]
        if predicted == opposite:
            counts["inverted"] += 1
        else:
            counts["wrong_entity"] += 1

    total = sum(counts.values())
    return {
        "counts": counts,
        "fractions": {
            key: (value / total if total else 0.0) for key, value in counts.items()
        },
        "errors_examined": total,
    }
