"""Figures for the Phase 3 reasoning analysis.

Visualization only: every function here takes numbers already computed by
:mod:`lma.reasoning.evaluate` and draws them, so computation and plotting stay in
separate functions as the assignment requires.
"""

from __future__ import annotations


def plot_answer_attention(pretrained: dict, finetuned: dict, out_path, title: str = ""):
    """Draw attention on the correct answer by layer, before and after finetuning.

    Args:
        pretrained: :func:`~lma.reasoning.evaluate.answer_attention_by_layer`
            output for the pretrained checkpoint.
        finetuned: The same for the finetuned checkpoint.
        out_path: Destination PNG.
        title: Figure title, typically naming the model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..stats.plots import configure_fonts

    configure_fonts()

    layers = list(range(len(pretrained["answer_max_head"])))
    figure, axis = plt.subplots(figsize=(8, 4.5))

    axis.plot(layers, pretrained["answer_max_head"], marker="o", color="tab:blue",
              label="pretrained - strongest head, correct answer")
    axis.plot(layers, finetuned["answer_max_head"], marker="o", color="tab:orange",
              label="finetuned - strongest head, correct answer")
    axis.plot(layers, pretrained["other_entity_max_head"], marker=".", linestyle="--",
              color="tab:blue", alpha=0.6,
              label="pretrained - strongest head, per other entity")
    axis.plot(layers, finetuned["other_entity_max_head"], marker=".", linestyle="--",
              color="tab:orange", alpha=0.6,
              label="finetuned - strongest head, per other entity")

    axis.set_title(title or "Attention from the answer position onto the named entities")
    axis.set_xlabel("layer")
    axis.set_ylabel("attention mass from the final prompt position")
    axis.set_xticks(layers)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path
