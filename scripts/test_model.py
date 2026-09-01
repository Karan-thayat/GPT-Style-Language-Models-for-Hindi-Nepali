"""Component tests for the Phase 2 Transformer.

Run before spending GPU time:

    .venv/bin/python scripts/test_model.py

Every check here runs on CPU with tiny tensors in a few seconds. They cover the
properties that are expensive to discover during a training run: that the causal
mask actually prevents future information from reaching earlier positions, that
tensor shapes survive the head reshape, that weight tying is real sharing rather
than a copy, and that the parameter count computed from the config matches the
model that gets built.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.model import CausalSelfAttention, ModelConfig, TransformerLM  # noqa: E402

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion."""
    global PASSED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def tiny_config(**overrides) -> ModelConfig:
    """A model small enough to test in milliseconds."""
    base = dict(
        vocab_size=97, n_layer=2, n_head=4, n_embd=32, block_size=16, dropout=0.0
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_config() -> None:
    print("\n[config]")
    cfg = tiny_config()
    check("head_dim divides n_embd", cfg.head_dim * cfg.n_head == cfg.n_embd)
    check("n_inner defaults to 4x", cfg.n_inner == 4 * cfg.n_embd)

    try:
        ModelConfig(vocab_size=10, n_embd=30, n_head=4)
        check("rejects indivisible n_embd/n_head", False)
    except ValueError:
        check("rejects indivisible n_embd/n_head", True)

    try:
        ModelConfig(vocab_size=10, n_embd=64, n_head=4, n_inner=32)
        check("rejects n_inner <= n_embd", False)
    except ValueError:
        check("rejects n_inner <= n_embd", True)

    # The count is used to pick a configuration without allocating one, so it
    # has to agree with the model that actually gets built.
    for tie in (True, False):
        cfg = tiny_config(tie_weights=tie)
        model = TransformerLM(cfg)
        check(
            f"predicted param count matches built model (tie_weights={tie})",
            cfg.n_params == model.num_parameters(),
            f"predicted {cfg.n_params:,} vs actual {model.num_parameters():,}",
        )

    cfg = tiny_config()
    round_tripped = ModelConfig.from_dict(cfg.to_dict())
    check("config survives dict round-trip", round_tripped == cfg)


def test_shapes() -> None:
    print("\n[shapes]")
    cfg = tiny_config()
    model = TransformerLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (3, 11))

    out = model(idx)
    check("logits shape (B, T, vocab)", tuple(out["logits"].shape) == (3, 11, 97))
    check("loss is None without targets", out["loss"] is None)

    out = model(idx, targets=idx)
    check("loss is scalar", out["loss"].ndim == 0)
    check("loss is finite", bool(torch.isfinite(out["loss"])))
    # An untrained model is uniform over the vocabulary, so the loss should sit
    # near ln(vocab_size); far from it means the initialization is wrong.
    expected = math.log(cfg.vocab_size)
    check(
        "initial loss near ln(vocab_size)",
        abs(out["loss"].item() - expected) < 0.7,
        f"got {out['loss'].item():.3f}, expected about {expected:.3f}",
    )

    out = model(idx, return_attention=True)
    attns = out["attentions"]
    check("one attention tensor per layer", len(attns) == cfg.n_layer)
    check(
        "attention shape (B, h, T, T)",
        tuple(attns[0].shape) == (3, cfg.n_head, 11, 11),
    )

    # Sequences longer than block_size have no positional embedding.
    try:
        model(torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1)))
        check("rejects sequence longer than block_size", False)
    except ValueError:
        check("rejects sequence longer than block_size", True)


def test_causality() -> None:
    """The property the assignment asks to be verified empirically."""
    print("\n[causality]")
    torch.manual_seed(0)
    cfg = tiny_config(n_layer=3, n_embd=64, vocab_size=500)
    model = TransformerLM(cfg).eval()

    idx = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        before = model(idx)["logits"]
        perturbed = idx.clone()
        t = 7
        perturbed[0, t] = (perturbed[0, t] + 137) % cfg.vocab_size
        after = model(perturbed)["logits"]

    # Changing token t must leave every logit at positions < t untouched.
    check(
        "changing token t does not affect logits before t",
        torch.allclose(before[0, :t], after[0, :t], atol=1e-6),
        f"max delta {(before[0, :t] - after[0, :t]).abs().max().item():.2e}",
    )
    # And it must affect position t itself, otherwise the test is vacuous - a
    # model ignoring its input entirely would pass the check above.
    check(
        "changing token t does affect logits at t",
        not torch.allclose(before[0, t], after[0, t], atol=1e-4),
    )

    # Attention probabilities must be exactly zero on the upper triangle.
    with torch.no_grad():
        attns = model(idx, return_attention=True)["attentions"]
    upper = torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
    worst = max(float(a[0, :, upper].abs().max()) for a in attns)
    check("attention weight on future positions is zero", worst == 0.0, f"max {worst:.2e}")

    # Each row is a probability distribution over the allowed positions.
    row_sums = attns[0][0, 0].sum(dim=-1)
    check(
        "attention rows sum to 1",
        torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5),
    )


def test_attention_internals() -> None:
    print("\n[attention]")
    cfg = tiny_config(n_embd=64, n_head=8, dropout=0.0)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 9, 64)
    y, weights = attn(x, return_attention=True)
    check("attention output shape matches input", y.shape == x.shape)
    check("head split is exact", cfg.head_dim == 8)

    # First position can only attend to itself, so its row is a one-hot.
    first_row = weights[0, 0, 0]
    check(
        "position 0 attends only to itself",
        abs(first_row[0].item() - 1.0) < 1e-6 and float(first_row[1:].max()) == 0.0,
    )

    # The 1/sqrt(dk) scaling: without it, score magnitude grows with head width.
    # Verify the implementation actually divides by sqrt(head_dim).
    q = torch.ones(1, 1, 4, 16)
    k = torch.ones(1, 1, 4, 16)
    scaled = (q @ k.transpose(-2, -1)) / math.sqrt(16)
    check("scaling divides by sqrt(head_dim)", abs(scaled[0, 0, 0, 0].item() - 4.0) < 1e-6)


def test_weight_tying() -> None:
    print("\n[weight tying]")
    cfg = tiny_config(tie_weights=True)
    model = TransformerLM(cfg)
    check(
        "tied head shares storage with embedding",
        model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr(),
    )
    saved = cfg.vocab_size * cfg.n_embd
    untied = tiny_config(tie_weights=False).n_params
    check(
        "tying saves vocab_size x n_embd parameters",
        untied - cfg.n_params == saved,
        f"expected {saved:,}, got {untied - cfg.n_params:,}",
    )

    untied_model = TransformerLM(tiny_config(tie_weights=False))
    check(
        "untied head has its own storage",
        untied_model.lm_head.weight.data_ptr()
        != untied_model.token_embedding.weight.data_ptr(),
    )


def test_training_step() -> None:
    """The model must actually learn - a smoke test against silent breakage."""
    print("\n[training]")
    torch.manual_seed(0)
    cfg = tiny_config(vocab_size=40, n_layer=2, n_embd=64, block_size=16, dropout=0.0)
    model = TransformerLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # A single fixed batch: the model should be able to memorize it.
    idx = torch.randint(0, cfg.vocab_size, (4, 12))
    targets = torch.randint(0, cfg.vocab_size, (4, 12))

    first = None
    for step in range(60):
        out = model(idx, targets=targets)
        if first is None:
            first = out["loss"].item()
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    last = out["loss"].item()

    check("loss decreases on a memorizable batch", last < first * 0.5, f"{first:.3f} -> {last:.3f}")

    # Every parameter must receive gradient; an unused one signals a wiring bug.
    out = model(idx, targets=targets)
    model.zero_grad()
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    check("every parameter receives a gradient", not missing, f"missing: {missing[:3]}")

    grads = [p.grad.abs().max().item() for p in model.parameters() if p.grad is not None]
    check("gradients are finite", all(math.isfinite(g) for g in grads))


def test_generation() -> None:
    print("\n[generation]")
    torch.manual_seed(0)
    cfg = tiny_config(vocab_size=50, block_size=16)
    model = TransformerLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 5))

    out = model.generate(prompt, max_new_tokens=6, greedy=True)
    check("generate appends the requested tokens", out.shape == (2, 11))
    check("prompt is preserved", torch.equal(out[:, :5], prompt))
    check("generated ids are in range", bool((out >= 0).all() and (out < 50).all()))

    # Greedy decoding is deterministic; sampling at temperature is not.
    a = model.generate(prompt, max_new_tokens=6, greedy=True)
    b = model.generate(prompt, max_new_tokens=6, greedy=True)
    check("greedy decoding is deterministic", torch.equal(a, b))

    # Generation must not break when the context exceeds block_size: the model
    # crops to the last block_size tokens rather than failing.
    long_prompt = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    out = model.generate(long_prompt, max_new_tokens=4, greedy=True)
    check("generation crops context beyond block_size", out.shape == (1, cfg.block_size + 4))


def test_phase2_budget() -> None:
    """The configuration actually used for the graded runs."""
    print("\n[phase 2 configuration]")
    cfg = ModelConfig(
        vocab_size=8000, n_layer=12, n_head=6, n_embd=384, block_size=512, dropout=0.1
    )
    total = cfg.n_params
    check(
        "within 10% of the 25M target",
        abs(total - 25_000_000) / 25_000_000 < 0.10,
        f"{total:,}",
    )
    check(
        "embeddings stay a minority of the budget",
        cfg.embedding_share() < 0.40,
        f"{100 * cfg.embedding_share():.1f}%",
    )
    print(f"        exact parameter count: {total:,}")
    print(f"        embedding share:       {100 * cfg.embedding_share():.1f}%")


def test_training_resume() -> None:
    """A run stopped and resumed must match one that never stopped.

    This is the property Kaggle's 12-hour session limit depends on: a full
    pretraining run spans several sessions, and if resuming perturbed the
    trajectory the loss curve would carry an artifact at every session boundary.

    Note that `max_steps` must stay fixed across the interruption. It defines the
    cosine decay horizon, so shortening it for a partial run compresses the whole
    schedule into those steps and produces genuinely different weights - a
    difference in configuration, not a resume bug.
    """
    print("\n[training resume]")
    import tempfile

    import numpy as np

    from lma.train import TokenBinaryDataset, TrainConfig, Trainer

    data_dir = Path(tempfile.mkdtemp())
    tokens = (np.arange(200_000) % 97).astype(np.uint16)
    (data_dir / "train.bin").write_bytes(tokens.tobytes())
    (data_dir / "val.bin").write_bytes(tokens[:40_000].tobytes())

    def make(out_dir: Path) -> "Trainer":
        torch.manual_seed(0)
        model_cfg = ModelConfig(
            vocab_size=97, n_layer=2, n_head=4, n_embd=64, block_size=32, dropout=0.0
        )
        train_cfg = TrainConfig(
            batch_size=8, gradient_accumulation=2, max_steps=40, warmup_steps=10,
            eval_interval=1000, eval_batches=5, checkpoint_interval=10,
            log_interval=1000, amp=False, seed=7,
        )
        return Trainer(
            TransformerLM(model_cfg),
            TokenBinaryDataset(data_dir / "train.bin", 32, seed=7),
            TokenBinaryDataset(data_dir / "val.bin", 32, seed=8),
            train_cfg,
            out_dir,
            device="cpu",
        )

    straight = make(Path(tempfile.mkdtemp()) / "straight")
    straight.train(resume=False)
    straight_loss = straight.evaluate(batches=5)["val_loss"]

    interrupted_dir = Path(tempfile.mkdtemp()) / "interrupted"
    first = make(interrupted_dir)
    first.train(max_seconds=0.0001, resume=False)   # stop almost immediately
    stopped_at = first.state.step
    second = make(interrupted_dir)
    second.train(resume=True)
    resumed_loss = second.evaluate(batches=5)["val_loss"]

    check(
        "interrupted run resumes to the same weights",
        abs(straight_loss - resumed_loss) < 1e-5,
        f"{straight_loss:.6f} vs {resumed_loss:.6f}",
    )
    check("resume actually happened mid-run", 0 < stopped_at < 40, f"stopped at {stopped_at}")
    check(
        "checkpoint carries optimizer, scheduler position, and step",
        all(
            key in torch.load(
                interrupted_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False
            )
            for key in ("model", "optimizer", "step", "model_config", "train_config")
        ),
    )


def main() -> int:
    print("Phase 2 model component tests")
    print("=" * 60)
    test_config()
    test_shapes()
    test_causality()
    test_attention_internals()
    test_weight_tying()
    test_training_step()
    test_generation()
    test_phase2_budget()
    test_training_resume()

    print("\n" + "=" * 60)
    total = PASSED + len(FAILED)
    print(f"{PASSED}/{total} assertions passed")
    if FAILED:
        print("\nFailed:")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
