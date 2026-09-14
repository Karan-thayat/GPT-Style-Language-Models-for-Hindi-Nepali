"""Phase 3: synthetic comparative-reasoning data, finetuning, and evaluation.

The dataset is generated programmatically rather than downloaded, as the
assignment requires, so every ground-truth label is known by construction and the
whole corpus is reproducible from a seed.
"""

from .templates import TEMPLATES, Dimension, LanguageTemplates, get_templates

__all__ = ["TEMPLATES", "Dimension", "LanguageTemplates", "get_templates"]
