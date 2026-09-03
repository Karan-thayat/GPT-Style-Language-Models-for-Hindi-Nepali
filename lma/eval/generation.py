"""Generation quality: BLEU, chrF, ROUGE-L, and diversity diagnostics.

The three overlap metrics are implemented here rather than pulled from a library
so that their behaviour on Devanagari is explicit and inspectable. The usual
implementations tokenize on whitespace and Latin punctuation, which is adequate
for Hindi and Nepali - both space-separated - but chrF in particular is defined
over characters, and getting that right for a script with combining marks matters
more than the convenience of an import.

A caveat worth stating up front, and worth repeating in the report: these are
similarity metrics built for tasks with a correct answer, such as translation or
summarization. Open-ended continuation has no single right answer, so a fluent,
sensible continuation that simply differs from the held-out text scores near
zero. They are reported because the assignment asks for them, and they are
informative *comparatively* - between temperatures, and between Model H and Model
L - but their absolute values should not be read as quality.
"""

from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Count n-grams in a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: list[str], references: list[str], max_n: int = 4
) -> dict:
    """Corpus-level BLEU-4 with the standard brevity penalty.

    Corpus-level rather than sentence-level: precision numerators and
    denominators are pooled across the whole corpus before dividing, which is the
    standard definition and far more stable than averaging per-sentence scores
    when individual hypotheses may share no 4-gram with their reference.

    Args:
        hypotheses: Generated continuations.
        references: Reference continuations, aligned with `hypotheses`.
        max_n: Maximum n-gram order. 4 is conventional; the report states it.

    Returns:
        Dict with ``bleu`` (0-100), the per-order precisions, and the brevity
        penalty.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be aligned")

    matches = [0] * max_n
    totals = [0] * max_n
    hyp_length = 0
    ref_length = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()
        hyp_length += len(hyp_tokens)
        ref_length += len(ref_tokens)

        for order in range(1, max_n + 1):
            hyp_ngrams = _ngrams(hyp_tokens, order)
            ref_ngrams = _ngrams(ref_tokens, order)
            # Clipped counts: an n-gram repeated in the hypothesis can only match
            # as often as it appears in the reference, which is what stops a
            # degenerate repetitive output from scoring highly.
            matches[order - 1] += sum(
                min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items()
            )
            totals[order - 1] += max(sum(hyp_ngrams.values()), 0)

    precisions = []
    for matched, total in zip(matches, totals):
        precisions.append(matched / total if total else 0.0)

    # Geometric mean of the precisions. Any zero-precision order sends BLEU to
    # zero, which is common for open-ended generation at order 4.
    if all(p > 0 for p in precisions):
        log_mean = sum(math.log(p) for p in precisions) / max_n
        geometric = math.exp(log_mean)
    else:
        geometric = 0.0

    # Brevity penalty: without it, emitting a single high-precision word would
    # score perfectly.
    if hyp_length == 0:
        brevity = 0.0
    elif hyp_length > ref_length:
        brevity = 1.0
    else:
        brevity = math.exp(1 - ref_length / hyp_length)

    return {
        "bleu": 100 * brevity * geometric,
        "precisions": [100 * p for p in precisions],
        "brevity_penalty": brevity,
        "hypothesis_length": hyp_length,
        "reference_length": ref_length,
    }


def chrf_score(
    hypotheses: list[str], references: list[str], max_n: int = 6, beta: float = 2.0
) -> dict:
    """Corpus-level chrF: F-score over character n-grams.

    Character n-grams rather than word n-grams, which suits morphologically rich
    languages: Hindi and Nepali inflect heavily, so two forms of the same lemma
    share most of their characters but count as entirely different words under
    BLEU. chrF therefore gives partial credit where BLEU gives none, and is the
    more informative of the two here.

    `beta` weights recall over precision; 2.0 is the standard chrF setting.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be aligned")

    total_precision = 0.0
    total_recall = 0.0
    orders_counted = 0

    for order in range(1, max_n + 1):
        matches = 0
        hyp_total = 0
        ref_total = 0

        for hyp, ref in zip(hypotheses, references):
            # Whitespace is removed so chrF measures the characters of the text
            # rather than its spacing, per the standard definition.
            hyp_chars = list(hyp.replace(" ", ""))
            ref_chars = list(ref.replace(" ", ""))

            hyp_ngrams = _ngrams(hyp_chars, order)
            ref_ngrams = _ngrams(ref_chars, order)

            matches += sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
            hyp_total += sum(hyp_ngrams.values())
            ref_total += sum(ref_ngrams.values())

        if hyp_total and ref_total:
            total_precision += matches / hyp_total
            total_recall += matches / ref_total
            orders_counted += 1

    if not orders_counted:
        return {"chrf": 0.0, "precision": 0.0, "recall": 0.0}

    precision = total_precision / orders_counted
    recall = total_recall / orders_counted

    if precision + recall == 0:
        f_score = 0.0
    else:
        beta_sq = beta**2
        f_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

    return {
        "chrf": 100 * f_score,
        "precision": 100 * precision,
        "recall": 100 * recall,
        "beta": beta,
    }


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, computed in O(len(a) * len(b)) time.

    Only the previous row is kept, so memory is linear rather than quadratic -
    which matters when comparing continuations hundreds of tokens long.
    """
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(hypotheses: list[str], references: list[str], beta: float = 1.2) -> dict:
    """Corpus-level ROUGE-L, the F-measure over longest common subsequences.

    Unlike BLEU's contiguous n-grams, the LCS rewards words appearing in the
    right *order* without requiring adjacency, so it tolerates the insertions and
    reorderings that a fluent but different continuation naturally contains.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be aligned")

    f_scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()
        if not hyp_tokens or not ref_tokens:
            f_scores.append(0.0)
            continue

        lcs = _lcs_length(hyp_tokens, ref_tokens)
        if lcs == 0:
            f_scores.append(0.0)
            continue

        precision = lcs / len(hyp_tokens)
        recall = lcs / len(ref_tokens)
        beta_sq = beta**2
        f_scores.append(
            (1 + beta_sq) * precision * recall / (recall + beta_sq * precision)
        )

    mean = sum(f_scores) / max(len(f_scores), 1)
    return {"rouge_l": 100 * mean, "samples": len(f_scores)}


def diversity_metrics(texts: list[str], max_n: int = 4) -> dict:
    """Repetition rate and Distinct-1/2 over generated text.

    These need no reference, which makes them the more honest diagnostics for
    open-ended generation:

    * **Repetition rate** - the fraction of n-grams that are not unique. A model
      stuck in a loop scores near 1; varied text scores low.
    * **Distinct-1 / Distinct-2** - unique unigrams and bigrams as a fraction of
      all produced. Falls as a model collapses onto a few frequent tokens, which
      is the characteristic failure of greedy decoding and low temperature.
    """
    all_tokens: list[str] = []
    per_text_repetition = []

    for text in texts:
        tokens = text.split()
        all_tokens.extend(tokens)

        if len(tokens) >= max_n:
            grams = _ngrams(tokens, max_n)
            total = sum(grams.values())
            unique = len(grams)
            per_text_repetition.append(1 - unique / total if total else 0.0)

    result = {
        "repetition_rate": (
            sum(per_text_repetition) / len(per_text_repetition)
            if per_text_repetition
            else 0.0
        ),
        "repetition_ngram_order": max_n,
        "total_tokens": len(all_tokens),
    }

    for order, name in ((1, "distinct_1"), (2, "distinct_2")):
        grams = _ngrams(all_tokens, order) if len(all_tokens) >= order else Counter()
        total = sum(grams.values())
        result[name] = len(grams) / total if total else 0.0

    return result


def evaluate_generations(
    hypotheses: list[str], references: list[str]
) -> dict:
    """Run every generation metric over one set of continuations."""
    return {
        **corpus_bleu(hypotheses, references),
        **chrf_score(hypotheses, references),
        **rouge_l(hypotheses, references),
        **diversity_metrics(hypotheses),
    }
