"""Intrinsic language-modeling metrics: cross-entropy, perplexity, bits-per-byte.

Perplexity is the natural measure of a language model's fit, but it is not
comparable across models whose tokenizers differ: a model whose tokenizer splits
text into fewer, longer pieces faces fewer prediction decisions per sentence and
scores better without being better. Bits-per-byte removes that advantage by
re-expressing the same total loss per byte of the original UTF-8 text, which is a
property of the text rather than of the segmentation. Both are reported for both
models, as the assignment requires.
"""

from __future__ import annotations

import math

import torch

from ..train.data import TokenBinaryDataset


@torch.no_grad()
def evaluate_language_model(
    model,
    dataset: TokenBinaryDataset,
    batch_size: int = 16,
    max_batches: int | None = None,
    device: str | None = None,
    tokenizer=None,
) -> dict:
    """Cross-entropy, perplexity, and bits-per-byte on a held-out split.

    The split is walked sequentially rather than sampled, so the result is a
    property of the data and not of a random draw.

    Args:
        model: A trained :class:`~lma.model.TransformerLM`.
        dataset: Sampler over the held-out token binary.
        batch_size: Sequences per forward pass.
        max_batches: Stop after this many batches. ``None`` walks the whole split.
        device: Where to run. Defaults to the model's device.
        tokenizer: SentencePiece processor. Required for bits-per-byte, which
            needs the UTF-8 length of the text the tokens decode to.

    Returns:
        Dict with ``cross_entropy`` (nats/token), ``perplexity``, ``tokens``,
        and - when a tokenizer is supplied - ``bits_per_byte`` and ``bytes``.
    """
    device = device or next(model.parameters()).device
    model.eval()

    total_nats = 0.0
    total_tokens = 0
    total_bytes = 0

    for x, y in dataset.sequential_batches(batch_size, limit=max_batches, device=str(device)):
        logits = model(x)["logits"]

        # Sum rather than mean: batches can differ in size, and a mean of means
        # would silently weight a short final batch as heavily as a full one.
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_nats += loss.float().item()
        total_tokens += y.numel()

        if tokenizer is not None:
            # Bits-per-byte needs the byte length of the text these targets
            # represent, decoded with the model's own tokenizer.
            for row in y.tolist():
                total_bytes += len(tokenizer.decode(row).encode("utf-8"))

    cross_entropy = total_nats / max(total_tokens, 1)
    result = {
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(min(cross_entropy, 20)),
        "tokens": total_tokens,
    }

    if tokenizer is not None and total_bytes:
        # Total loss in nats -> bits, spread over the bytes of the source text.
        # log(2) converts nats to bits; dividing by bytes rather than tokens is
        # what makes the figure comparable between different tokenizers.
        result["bits_per_byte"] = total_nats / (total_bytes * math.log(2))
        result["bytes"] = total_bytes
        result["bytes_per_token"] = total_bytes / max(total_tokens, 1)

    return result


def compare_models(results: dict[str, dict]) -> str:
    """Format a side-by-side table of per-model metrics for the report."""
    languages = list(results)
    header = f"| Metric | {' | '.join(languages)} |"
    rule = "|---|" + "---|" * len(languages)

    rows = [header, rule]
    for label, key, fmt in (
        ("Cross-entropy (nats/token)", "cross_entropy", "{:.4f}"),
        ("Perplexity", "perplexity", "{:.2f}"),
        ("Bits-per-byte", "bits_per_byte", "{:.4f}"),
        ("Held-out tokens", "tokens", "{:,}"),
    ):
        cells = []
        for lang in languages:
            value = results[lang].get(key)
            cells.append(fmt.format(value) if value is not None else "-")
        rows.append(f"| {label} | {' | '.join(cells)} |")

    return "\n".join(rows)
