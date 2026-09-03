"""Phase 2 evaluation: language-modeling metrics, generation quality, attention."""

from .attention import (
    attention_statistics,
    classify_heads,
    collect_attention,
    plot_attention_heatmaps,
    plot_head_summary,
)
from .generation import (
    chrf_score,
    corpus_bleu,
    diversity_metrics,
    evaluate_generations,
    rouge_l,
)
from .lm_metrics import compare_models, evaluate_language_model

__all__ = [
    "evaluate_language_model",
    "compare_models",
    "corpus_bleu",
    "chrf_score",
    "rouge_l",
    "diversity_metrics",
    "evaluate_generations",
    "attention_statistics",
    "classify_heads",
    "collect_attention",
    "plot_attention_heatmaps",
    "plot_head_summary",
]
