"""Decoder-only Transformer implemented from primitive PyTorch layers.

Phase 2 of the project forbids ``nn.Transformer*``, HuggingFace model classes,
and any pre-built attention block, so every tensor operation in the forward pass
is written out explicitly here.
"""

from .config import ModelConfig
from .transformer import (
    CausalSelfAttention,
    DecoderBlock,
    FeedForward,
    TransformerLM,
)

__all__ = [
    "ModelConfig",
    "CausalSelfAttention",
    "FeedForward",
    "DecoderBlock",
    "TransformerLM",
]
