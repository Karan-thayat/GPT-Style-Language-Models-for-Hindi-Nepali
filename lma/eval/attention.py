"""What the attention heads learned: entropy, distance, and heatmaps.

Two scalar summaries per head, both computed only over positions a causal model
may actually attend to:

* **Entropy** measures how spread a head's attention is. A head attending to one
  position has entropy 0; one spreading uniformly over its context has the
  maximum for that width. Because a causal head at position ``t`` can only see
  ``t + 1`` positions, raw entropy grows along the sequence for purely structural
  reasons, so the normalized form - entropy divided by ``log(t + 1)`` - is
  reported alongside it and is the figure to compare across heads.

* **Mean attention distance** is how far back a head looks on average, weighted
  by attention mass. Small values mark local heads (previous-token heads, which
  most models learn early); large values mark heads gathering long-range context.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def attention_statistics(attentions: list[torch.Tensor]) -> dict:
    """Per-head entropy and mean attention distance.

    Args:
        attentions: One ``(B, h, T, T)`` tensor per layer, as returned by
            ``TransformerLM(..., return_attention=True)``.

    Returns:
        Dict with ``entropy``, ``normalized_entropy``, and ``mean_distance``,
        each a ``(n_layer, n_head)`` nested list, plus per-layer means.
    """
    entropy_per_layer = []
    normalized_per_layer = []
    distance_per_layer = []

    for layer_attention in attentions:
        B, H, T, _ = layer_attention.shape
        weights = layer_attention.float()

        # Entropy per query position: -sum p log p. The clamp keeps log(0)
        # finite; masked positions carry exactly zero probability and so
        # contribute nothing to the sum either way.
        logs = torch.log(weights.clamp_min(1e-12))
        entropy = -(weights * logs).sum(dim=-1)          # (B, H, T)

        # Position t attends over t+1 positions, so the maximum possible entropy
        # there is log(t+1). Dividing by it makes heads comparable across
        # positions and sequence lengths.
        positions = torch.arange(T, device=weights.device, dtype=torch.float32)
        max_entropy = torch.log(positions + 1).clamp_min(1e-12)
        normalized = entropy / max_entropy

        # Distance from each query to each key, in positions. Only the causal
        # (lower-triangular) part carries mass, so the weighted sum is the mean
        # look-back distance.
        query_index = positions.view(1, 1, T, 1)
        key_index = positions.view(1, 1, 1, T)
        distance = (query_index - key_index).clamp_min(0)
        mean_distance = (weights * distance).sum(dim=-1)  # (B, H, T)

        # Position 0 can only attend to itself, so its entropy and distance are
        # trivially zero and would drag every head's mean down equally. Excluded.
        entropy_per_layer.append(entropy[:, :, 1:].mean(dim=(0, 2)).tolist())
        normalized_per_layer.append(normalized[:, :, 1:].mean(dim=(0, 2)).tolist())
        distance_per_layer.append(mean_distance[:, :, 1:].mean(dim=(0, 2)).tolist())

    return {
        "entropy": entropy_per_layer,
        "normalized_entropy": normalized_per_layer,
        "mean_distance": distance_per_layer,
        "layer_mean_entropy": [sum(e) / len(e) for e in entropy_per_layer],
        "layer_mean_distance": [sum(d) / len(d) for d in distance_per_layer],
        "n_layer": len(attentions),
        "n_head": len(entropy_per_layer[0]) if entropy_per_layer else 0,
    }


def classify_heads(stats: dict, local_threshold: float = 5.0) -> dict:
    """Sort heads into local, diffuse, and long-range groups.

    A crude but useful summary for the report's discussion: heads whose mean
    attention distance is small are doing positional work (typically attending to
    the previous token or two), while heads with high normalized entropy are
    spreading mass broadly rather than selecting particular positions.

    Args:
        stats: Output of :func:`attention_statistics`.
        local_threshold: Mean distance below which a head counts as local.
    """
    local, long_range, diffuse = [], [], []

    for layer, (distances, normalized) in enumerate(
        zip(stats["mean_distance"], stats["normalized_entropy"])
    ):
        for head, (distance, entropy) in enumerate(zip(distances, normalized)):
            record = {
                "layer": layer,
                "head": head,
                "mean_distance": distance,
                "normalized_entropy": entropy,
            }
            if distance < local_threshold:
                local.append(record)
            elif entropy > 0.8:
                diffuse.append(record)
            else:
                long_range.append(record)

    return {
        "local": local,
        "long_range": long_range,
        "diffuse": diffuse,
        "counts": {
            "local": len(local),
            "long_range": len(long_range),
            "diffuse": len(diffuse),
        },
    }


@torch.no_grad()
def collect_attention(model, token_ids: torch.Tensor, device: str | None = None):
    """Run one sequence through the model and return its attention tensors."""
    device = device or next(model.parameters()).device
    model.eval()
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    output = model(token_ids.to(device), return_attention=True)
    return output["attentions"]


def plot_attention_heatmaps(
    attentions: list[torch.Tensor],
    tokens: list[str],
    layers: list[int],
    heads: list[int],
    out_path,
    title: str = "",
    max_positions: int = 32,
):
    """Save a grid of query-vs-key attention heatmaps.

    The assignment asks for at least one early and one late layer with several
    heads each, labelled with the example sentence's tokens.

    Args:
        attentions: Per-layer ``(B, h, T, T)`` tensors.
        tokens: Token strings for the axis labels.
        layers: Which layers to draw.
        heads: Which heads to draw.
        out_path: Destination PNG.
        title: Figure title, typically naming the model and the sentence.
        max_positions: Truncate to this many positions so labels stay legible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Axis labels are Devanagari tokens, and the default font has no glyphs for
    # them - they would render as empty boxes. Phase 1 already solved this with a
    # fallback chain that puts a Latin font first and Indic faces after it.
    from ..stats.plots import configure_fonts

    configure_fonts()

    # Tick labels are SentencePiece pieces, which mix three character sets:
    # Devanagari, Latin (in loanwords and the piece marker), and the U+2581 block
    # character SentencePiece uses to mark word starts. No single installed face
    # covers all three, so the labels take an ordered *list* of families and each
    # character falls through to whichever face has its glyph. Naming a single
    # Devanagari font here instead turns every Latin character into a box.
    import matplotlib.font_manager as font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}

    # Hosted notebook images often ship no Devanagari face at all, in which case
    # every label renders as an empty box and the figure is useless. Fetch one
    # and register it rather than producing an illegible plot.
    if not any("Devanagari" in name for name in available):
        try:
            import urllib.request

            url = (
                "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
                "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
            )
            font_path = Path("/tmp/NotoSansDevanagari-Regular.ttf")
            if not font_path.exists():
                urllib.request.urlretrieve(url, font_path)
            font_manager.fontManager.addfont(str(font_path))
            available = {font.name for font in font_manager.fontManager.ttflist}
            print(f"  installed Devanagari font for plot labels: {font_path.name}")
        except Exception as error:
            # Not fatal: the numbers are unaffected, only the tick labels.
            print(f"  note: no Devanagari font available, labels may not render ({error})")

    indic_font = [
        name
        for name in (
            "DejaVu Sans",              # Latin, punctuation, and U+2581
            "Noto Sans Devanagari",     # Devanagari
            "Noto Serif Devanagari",
        )
        if name in available
    ] or ["DejaVu Sans"]

    limit = min(max_positions, len(tokens), attentions[0].shape[-1])
    labels = tokens[:limit]

    figure, axes = plt.subplots(
        len(layers), len(heads), figsize=(4 * len(heads), 3.6 * len(layers)), squeeze=False
    )

    for row, layer in enumerate(layers):
        for column, head in enumerate(heads):
            axis = axes[row][column]
            weights = attentions[layer][0, head, :limit, :limit].float().cpu().numpy()

            image = axis.imshow(weights, cmap="viridis", aspect="auto", vmin=0.0)
            axis.set_title(f"Layer {layer}, head {head}", fontsize=10)
            axis.set_xlabel("key position")
            axis.set_ylabel("query position")

            # Label every other position: a full set overlaps illegibly, and the
            # point of the labels is to show *which* tokens a head favours.
            #
            # The Devanagari font is named explicitly here rather than relying on
            # the rcParams family chain: matplotlib resolves tick labels through a
            # separate path that does not fall through the chain, so the glyphs
            # would render as empty boxes despite the font being installed.
            step = max(1, limit // 16)
            axis.set_xticks(range(0, limit, step))
            axis.set_xticklabels(
                labels[::step], rotation=90, fontsize=6, fontfamily=indic_font
            )
            axis.set_yticks(range(0, limit, step))
            axis.set_yticklabels(labels[::step], fontsize=6, fontfamily=indic_font)

            figure.colorbar(image, ax=axis, fraction=0.046, label="attention weight")

    if title:
        figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_head_summary(stats: dict, out_path, title: str = ""):
    """Save the per-head entropy and mean-distance maps as one figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from ..stats.plots import configure_fonts

    configure_fonts()

    entropy = np.array(stats["normalized_entropy"])
    distance = np.array(stats["mean_distance"])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for axis, data, name, cmap in (
        (axes[0], entropy, "Normalized attention entropy", "magma"),
        (axes[1], distance, "Mean attention distance (positions)", "viridis"),
    ):
        image = axis.imshow(data, cmap=cmap, aspect="auto")
        axis.set_title(name)
        axis.set_xlabel("head")
        axis.set_ylabel("layer")
        axis.set_xticks(range(data.shape[1]))
        axis.set_yticks(range(data.shape[0]))
        figure.colorbar(image, ax=axis, fraction=0.046)

    if title:
        figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path
