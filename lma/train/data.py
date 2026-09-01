"""Batching for causal language-model pretraining.

The corpus is one flat ``uint16`` array of token ids on disk, produced by the
tokenization step. Batches are random windows into that array, memory-mapped so
the file is never loaded whole - a 1.2 GB corpus streams from page cache while
resident memory stays near zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenBinaryDataset:
    """Random fixed-length windows over a flat array of token ids.

    Two details that matter for correctness:

    * **Targets are inputs shifted by one.** A window of ``block_size + 1``
      tokens yields ``x = w[:-1]`` and ``y = w[1:]``, so position ``t`` predicts
      the token at ``t + 1``. Sampling ``block_size + 1`` rather than
      ``block_size`` avoids the off-by-one that silently trains the model to
      predict its own input.
    * **Windows may cross document boundaries.** Documents were concatenated with
      an EOS token between them, so the model sees EOS as an ordinary token and
      learns that text ends there. This is standard for pretraining and means no
      document structure has to be tracked at batch time.
    """

    def __init__(self, path: str | Path, block_size: int, seed: int = 0) -> None:
        """Open a token binary for sampling.

        Args:
            path: ``.bin`` file of ``uint16`` token ids.
            block_size: Context length. Windows are ``block_size + 1`` long.
            seed: Seeds this dataset's own generator, so batch order is
                reproducible and independent of global torch/numpy state.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"token binary not found: {self.path}")

        self.block_size = block_size
        # mode="r" keeps the file read-only and lets several workers - or both
        # languages' loaders - share the same page cache.
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.rng = np.random.default_rng(seed)

        if len(self.tokens) <= block_size + 1:
            raise ValueError(
                f"{self.path.name} holds {len(self.tokens):,} tokens, too few for "
                f"block_size {block_size}"
            )

    def __len__(self) -> int:
        """Number of distinct starting positions."""
        return len(self.tokens) - self.block_size - 1

    def batch(self, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one batch of random windows.

        Returns:
            ``(x, y)``, both ``(batch_size, block_size)`` int64 tensors, with
            ``y`` shifted one position ahead of ``x``.
        """
        starts = self.rng.integers(0, len(self), size=batch_size)

        # Build in numpy, convert once: per-sample torch indexing is markedly
        # slower than one contiguous copy.
        window = np.stack(
            [self.tokens[s : s + self.block_size + 1].astype(np.int64) for s in starts]
        )
        batch = torch.from_numpy(window)

        x = batch[:, :-1].contiguous()
        y = batch[:, 1:].contiguous()

        if device != "cpu":
            # non_blocking with pinned memory overlaps the host-to-device copy
            # with compute on the previous batch.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y

    def sequential_batches(
        self, batch_size: int, limit: int | None = None, device: str = "cpu"
    ):
        """Iterate the corpus in order, for deterministic evaluation.

        Validation loss should not depend on a random draw, so held-out
        evaluation walks contiguous non-overlapping windows from the start of the
        file instead of sampling.

        Args:
            batch_size: Windows per batch.
            limit: Stop after this many batches. ``None`` walks the whole split.
            device: Where to place the tensors.

        Yields:
            ``(x, y)`` pairs shaped as in :meth:`batch`.
        """
        stride = self.block_size
        position = 0
        produced = 0

        while position + (batch_size * stride) + 1 <= len(self.tokens):
            if limit is not None and produced >= limit:
                return

            starts = [position + i * stride for i in range(batch_size)]
            window = np.stack(
                [self.tokens[s : s + self.block_size + 1].astype(np.int64) for s in starts]
            )
            batch = torch.from_numpy(window)

            x = batch[:, :-1].contiguous()
            y = batch[:, 1:].contiguous()
            if device != "cpu":
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)

            yield x, y
            position += batch_size * stride
            produced += 1

    def get_rng_state(self) -> dict:
        """Sampler RNG state, for checkpointing.

        The dataset draws window offsets from its own generator rather than the
        global one, so resuming a run has to restore this too - otherwise a
        resumed run sees a different batch sequence than an uninterrupted one and
        the two diverge.
        """
        return self.rng.bit_generator.state

    def set_rng_state(self, state: dict) -> None:
        """Restore a sampler RNG state saved by :meth:`get_rng_state`."""
        self.rng.bit_generator.state = state

    def stats(self) -> dict:
        """Summary for the run log."""
        return {
            "path": str(self.path),
            "tokens": int(len(self.tokens)),
            "bytes": self.path.stat().st_size,
            "windows": len(self),
            "block_size": self.block_size,
        }
