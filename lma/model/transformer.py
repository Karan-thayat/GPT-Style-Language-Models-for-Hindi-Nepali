"""A decoder-only (GPT-style) Transformer written from primitive layers.

Only ``nn.Linear``, ``nn.Embedding``, ``nn.LayerNorm``, and ``nn.Dropout`` are
used. Attention is computed explicitly - the projection, the head reshape, the
scaled dot product, the causal mask, the softmax, and the output projection are
all written out - because the point of this phase is to be able to explain every
tensor operation in the forward pass.

Shape convention throughout: ``B`` batch, ``T`` time (sequence position), ``d``
model dimension, ``h`` heads, ``dk = d / h`` per-head width.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Each position attends to itself and to earlier positions only. The mask is
    additive and applied to the scores *before* the softmax, so masked positions
    receive zero probability rather than being renormalized afterwards.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        # One matrix producing Q, K and V together. Mathematically identical to
        # three separate d x d projections, but a single matmul is faster.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Upper-triangular mask, held as a buffer so it moves with .to(device)
        # and is saved in no gradient. Bool: True where attention is forbidden.
        mask = torch.triu(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply causal self-attention.

        Args:
            x: Input activations, shape ``(B, T, d)``.
            return_attention: Also return the post-softmax attention
                probabilities, shape ``(B, h, T, T)``. Used by the Phase 2
                attention analysis (heatmaps, entropy, mean distance); left off
                during training because materializing them costs memory.

        Returns:
            ``(output, attention)`` where output has shape ``(B, T, d)`` and
            attention is ``None`` unless requested.
        """
        B, T, d = x.shape

        # Project to queries, keys and values: (B, T, d) -> (B, T, 3d) -> 3 x (B, T, d)
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        # Split each into heads and move the head axis in front of time, so that
        # every head is an independent (T, dk) sequence for the matmul below:
        #   (B, T, d) -> (B, T, h, dk) -> (B, h, T, dk)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product scores: (B, h, T, dk) @ (B, h, dk, T) -> (B, h, T, T).
        #
        # The 1/sqrt(dk) scaling matters. q and k have unit-ish variance per
        # component, so their dot product over dk dimensions has variance
        # proportional to dk. Without the scaling, scores grow with model width,
        # the softmax saturates, and its gradient vanishes - training stalls for
        # exactly the wider models the scaling was meant to enable.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: -inf on future positions, so softmax assigns them exactly
        # zero. Sliced to T because the final batch of a sequence may be shorter.
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values: (B, h, T, T) @ (B, h, T, dk) -> (B, h, T, dk)
        y = attn @ v

        # Concatenate heads back together: (B, h, T, dk) -> (B, T, h, dk) -> (B, T, d).
        # contiguous() because transpose leaves a non-contiguous view that view()
        # cannot reinterpret.
        y = y.transpose(1, 2).contiguous().view(B, T, d)

        y = self.resid_dropout(self.proj(y))
        return y, (attn if return_attention else None)


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Two linear layers with a GELU between them, applied identically at every
    position. The inner dimension is wider than the model dimension (4x by
    default): this is where most of the block's capacity lives, and where the
    model does per-token computation that attention - which only mixes
    information between positions - cannot do.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_inner, bias=config.bias)
        self.proj = nn.Linear(config.n_inner, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, d)`` to ``(B, T, d)`` through the wider inner layer."""
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class DecoderBlock(nn.Module):
    """One pre-norm decoder block: attention, then feed-forward, both residual.

    Pre-norm (normalize *entering* each sublayer, add the raw residual) rather
    than post-norm. With pre-norm the residual path from input to loss carries no
    normalization, so gradients reach early layers undiminished and deep stacks
    train without a warmup-dependent divergence. Post-norm places a LayerNorm on
    that path and needs careful warmup to remain stable at this depth.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run both sublayers, each wrapped in a residual connection."""
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attention)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    Learned absolute positional embeddings are used rather than sinusoidal or
    rotary. The reason is that this model is trained and evaluated at a single
    fixed context length, where learned positions are simplest and train fastest;
    the cost is that the model cannot be run on sequences longer than
    `block_size`, because no embedding exists for those positions. Sinusoidal or
    RoPE encodings would extrapolate, at the price of extra machinery this
    project does not need.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # The ablation omits the table entirely rather than zeroing it, so the
        # parameters are genuinely absent from the model and from its count.
        self.position_embedding = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.use_positional_embeddings
            else None
        )
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            # One matrix serving as both the input embedding and the output
            # projection. Saves vocab_size x n_embd parameters - about a fifth of
            # this budget - and ties the representation of a token to the vector
            # that predicts it.
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Scaled init for the residual projections. Each block adds into the
        # residual stream, so with n_layer blocks the variance accumulated at the
        # top grows with depth; scaling these by 1/sqrt(2 * n_layer) keeps
        # activations stable at initialization regardless of depth.
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Normal(0, 0.02) for weights, zeros for biases - the GPT-2 scheme."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters, counting a tied embedding matrix only once."""
        seen: set[int] = set()
        total = 0
        for param in self.parameters():
            if trainable_only and not param.requires_grad:
                continue
            if id(param) in seen:
                continue
            seen.add(id(param))
            total += param.numel()
        return total

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict:
        """Predict next-token logits, and optionally the language-modeling loss.

        Args:
            idx: Token ids, shape ``(B, T)``.
            targets: Next-token ids, shape ``(B, T)``. When given, the causal
                language-modeling loss is returned alongside the logits.
            return_attention: Collect per-layer attention probabilities.

        Returns:
            Dict with ``logits`` ``(B, T, vocab_size)``, ``loss`` (or None), and
            ``attentions`` (list of ``(B, h, T, T)`` tensors, or None).
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}; "
                "learned positional embeddings exist only for block_size positions"
            )

        # Token identity plus position. Addition, not concatenation: the two
        # signals share the same d dimensions, and the model learns to separate
        # them. Without the positional term, one attention layer sees the tokens
        # before each position as an unordered set, so on its own it cannot tell
        # "dog bites man" from "man bites dog". The causal mask leaks some order
        # into deeper layers, and the positional-embedding ablation measures how
        # much of that lost signal a deep stack recovers.
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.position_embedding(pos)
        x = self.embed_dropout(x)

        attentions = [] if return_attention else None
        for block in self.blocks:
            x, attn = block(x, return_attention)
            if return_attention:
                attentions.append(attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy between the logits at position t and the token at
            # t+1, averaged over every position in the batch. The caller supplies
            # targets already shifted by one.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "attentions": attentions}

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend a prompt.

        Args:
            idx: Prompt token ids, shape ``(B, T)``.
            max_new_tokens: How many tokens to append.
            temperature: Divides the logits before softmax. Below 1 sharpens the
                distribution, above 1 flattens it. Ignored when `greedy`.
            top_k: Restrict sampling to the k most likely tokens.
            greedy: Take the argmax instead of sampling.
            eos_id: Stop early once every sequence has produced this token.

        Returns:
            The prompt with the generated continuation appended.
        """
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            # Crop to the context window: positions beyond block_size have no
            # learned embedding, so the model can only condition on the most
            # recent block_size tokens.
            idx_cond = idx[:, -self.config.block_size :]
            logits = self(idx_cond)["logits"][:, -1, :]

            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            if eos_id is not None:
                # Once a sequence has emitted EOS, keep emitting it so the tensor
                # stays rectangular; the caller truncates at the first EOS.
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_id),
                    next_token,
                )
                finished |= next_token.squeeze(1) == eos_id

            idx = torch.cat([idx, next_token], dim=1)
            if eos_id is not None and bool(finished.all()):
                break

        return idx
