"""Model configuration, with an exact parameter count derived from the shapes.

The assignment fixes a target of roughly 25M trainable parameters per model and
asks for the exact count to be reported, so the arithmetic lives here next to the
hyperparameters rather than being measured only after a model is built. That also
makes it possible to search for a configuration that hits the budget without
allocating anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    """Hyperparameters for one decoder-only language model.

    The two models in this project share an architecture but not a vocabulary:
    each uses the tokenizer trained on its own corpus in Phase 1, so
    `vocab_size` differs and the embedding and output layers differ in size.

    Attributes:
        vocab_size: Number of tokenizer pieces. From the Phase 1 tokenizer.
        n_layer: Number of stacked decoder blocks.
        n_head: Number of attention heads. Must divide `n_embd`.
        n_embd: Model dimension `d`.
        n_inner: Inner width of the feed-forward network. Defaults to 4x `n_embd`,
            the usual ratio, and the assignment requires it to exceed `n_embd`.
        block_size: Maximum context length in tokens. Learned positional
            embeddings are allocated for exactly this many positions, which is
            the hard limit on sequence length at inference (see `TransformerLM`).
        dropout: Applied to embeddings, attention probabilities, and sublayer
            outputs.
        tie_weights: Share the input embedding matrix with the output projection.
            Saves `vocab_size * n_embd` parameters, which at these sizes is a
            fifth of the whole budget.
        bias: Whether linear layers carry bias terms.
        use_positional_embeddings: Whether position is injected at all. Setting
            this to ``False`` is the assignment's optional ablation. Without it, a
            single attention layer sees the tokens before each position as an
            unordered set and cannot tell their orderings apart. The causal mask
            still leaks order into deeper layers - each position's state
            summarizes a prefix of a different length - so how much of the lost
            signal a deep stack recovers is what the ablation measures. Kept as a
            flag so the ablated model is the same code path, and so a checkpoint
            records which variant produced it.
    """

    vocab_size: int
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    n_inner: int | None = None
    block_size: int = 512
    dropout: float = 0.1
    tie_weights: bool = True
    bias: bool = True
    use_positional_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
        if self.n_inner <= self.n_embd:
            raise ValueError(
                f"n_inner ({self.n_inner}) must exceed n_embd ({self.n_embd}); the "
                "feed-forward network is required to project up before projecting back"
            )

    @property
    def head_dim(self) -> int:
        """Per-head subspace width `d_k = d / h`."""
        return self.n_embd // self.n_head

    def parameter_count(self) -> dict[str, int]:
        """Exact trainable-parameter count, broken down by component.

        Computed from shapes rather than by instantiating the model, so a
        configuration can be checked against the 25M budget cheaply.

        Returns:
            Counts per component plus a ``total`` key.
        """
        d, v, ff = self.n_embd, self.vocab_size, self.n_inner
        b = 1 if self.bias else 0

        token_embedding = v * d
        # The ablation removes the positional table entirely rather than zeroing
        # it, so the parameter count reflects what is actually trained.
        position_embedding = self.block_size * d if self.use_positional_embeddings else 0

        # Per block: four d x d projections (Q, K, V, O) and two feed-forward
        # matrices, plus two LayerNorms (weight and bias each).
        attn = 4 * (d * d + b * d)
        mlp = (d * ff + b * ff) + (ff * d + b * d)
        norms = 2 * (2 * d)
        per_block = attn + mlp + norms
        blocks = self.n_layer * per_block

        final_norm = 2 * d
        # The output projection is free when tied to the input embedding.
        head = 0 if self.tie_weights else v * d

        total = token_embedding + position_embedding + blocks + final_norm + head
        return {
            "token_embedding": token_embedding,
            "position_embedding": position_embedding,
            "blocks": blocks,
            "per_block": per_block,
            "final_norm": final_norm,
            "lm_head": head,
            "total": total,
        }

    @property
    def n_params(self) -> int:
        """Total trainable parameters."""
        return self.parameter_count()["total"]

    def embedding_share(self) -> float:
        """Fraction of the budget spent on embeddings rather than on layers.

        The quantity the Phase 1 vocabulary sweep was chosen against: every piece
        added to the vocabulary is a parameter taken away from the blocks that do
        the modelling.
        """
        counts = self.parameter_count()
        embeddings = counts["token_embedding"] + counts["position_embedding"]
        return embeddings / counts["total"]

    def to_dict(self) -> dict:
        """Serialize for checkpointing and for the run's config record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Rebuild from a checkpoint payload, ignoring unknown keys."""
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})
