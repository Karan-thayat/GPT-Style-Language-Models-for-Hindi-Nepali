"""Pretraining loop for the Phase 2 language models.

Built around one constraint that shapes every design choice here: Kaggle kills a
session after 12 hours, and a full run needs more than that. So training is
resumable at any step - checkpoints carry model weights, optimizer state,
scheduler position, step counter, RNG state, and the config - and a fresh session
picks up exactly where the last one stopped rather than restarting.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from ..model import ModelConfig, TransformerLM
from .data import TokenBinaryDataset


@dataclass
class TrainConfig:
    """Optimization hyperparameters for one pretraining run."""

    batch_size: int = 24
    gradient_accumulation: int = 4
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 500
    max_steps: int = 12_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_batches: int = 100
    checkpoint_interval: int = 1_000
    log_interval: int = 20
    seed: int = 20260812
    amp: bool = True
    compile: bool = False

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed per optimizer step, ignoring block_size."""
        return self.batch_size * self.gradient_accumulation

    def to_dict(self) -> dict:
        """Serialize for checkpointing and for the run's config record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        """Rebuild from a checkpoint payload, ignoring unknown keys.

        Unknown keys are dropped rather than raising, so a checkpoint written by
        an earlier version - before a field existed, or after one was removed -
        still loads instead of stranding a trained model.
        """
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Learning rate at `step`: linear warmup, then cosine decay.

    Warmup exists because Adam's second-moment estimate is unreliable in the
    first few hundred steps; starting at full learning rate there produces the
    large early updates that destabilize deep pre-norm stacks. Cosine decay
    afterwards spends most of the budget at a high rate and anneals smoothly,
    which reliably beats a constant rate at a fixed step count.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(cfg.warmup_steps, 1)

    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + coeff * (cfg.learning_rate - cfg.min_learning_rate)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay applied only where it belongs.

    Decaying biases, LayerNorm gains, and embeddings hurts: those parameters set
    scale and offset rather than mixing features, and shrinking them toward zero
    is a bias with no regularizing benefit. Only matrices that participate in
    matrix multiplications - the 2D weights - are decayed.
    """
    decay, no_decay = [], []
    seen: set[int] = set()

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if param.dim() >= 2 and not isinstance(module, nn.Embedding):
                decay.append(param)
            else:
                no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused AdamW is markedly faster on CUDA and a no-op elsewhere.
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=fused
    )


def _as_byte_tensor(state) -> torch.Tensor:
    """Coerce a saved RNG state into the CPU uint8 tensor torch requires.

    ``torch.set_rng_state`` and ``torch.cuda.set_rng_state_all`` accept only a
    ByteTensor on CPU, and raise ``TypeError: RNG state must be a
    torch.ByteTensor`` otherwise. A state that has been through
    ``torch.save``/``torch.load`` - especially across torch versions or between
    hosts - does not always come back in that exact form.
    """
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.uint8)
    return state.cpu().to(torch.uint8)


@dataclass
class TrainState:
    """Everything needed to resume a run mid-flight."""

    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    history: list[dict] = field(default_factory=list)
    wall_seconds: float = 0.0


class Trainer:
    """Pretrain one language model, resumably.

    Args:
        model: The model to train.
        train_data: Sampler over the training binary.
        val_data: Sampler over the validation binary.
        cfg: Optimization hyperparameters.
        out_dir: Where checkpoints and the run log are written.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model: TransformerLM,
        train_data: TokenBinaryDataset,
        val_data: TokenBinaryDataset,
        cfg: TrainConfig,
        out_dir: str | Path,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = build_optimizer(self.model, cfg)
        self.state = TrainState()

        # Mixed precision: bfloat16 where supported (no loss scaling needed,
        # same exponent range as fp32), otherwise fp16 with a gradient scaler to
        # keep small gradients from flushing to zero.
        self.amp_enabled = cfg.amp and self.device == "cuda"
        self.amp_dtype = torch.float32
        if self.amp_enabled:
            supported = torch.cuda.is_bf16_supported()
            self.amp_dtype = torch.bfloat16 if supported else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )

        if cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        torch.manual_seed(cfg.seed)

    # -- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, batches: int | None = None) -> dict:
        """Mean cross-entropy over held-out data, and the perplexity implied.

        Walks the validation split in order rather than sampling, so the number
        is comparable across steps and across runs.
        """
        self.model.eval()
        n_batches = batches if batches is not None else self.cfg.eval_batches

        losses = []
        for x, y in self.val_data.sequential_batches(
            self.cfg.batch_size, limit=n_batches, device=self.device
        ):
            with torch.autocast(
                device_type=self.device, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                loss = self.model(x, targets=y)["loss"]
            losses.append(loss.float().item())

        self.model.train()
        mean_loss = sum(losses) / max(len(losses), 1)
        return {
            "val_loss": mean_loss,
            # Perplexity is exp of the mean cross-entropy, per the assignment.
            "val_perplexity": math.exp(min(mean_loss, 20)),
            "batches": len(losses),
        }

    # -- checkpointing ------------------------------------------------------

    def save_checkpoint(self, name: str = "latest") -> Path:
        """Write a checkpoint that a fresh session can resume from.

        Carries model weights, optimizer state, the step counter, and the config,
        as the assignment requires - plus RNG state, so a resumed run draws the
        same batch sequence it would have drawn had it never stopped. The
        learning-rate schedule is a pure function of the step, so storing the step
        stores the scheduler position.
        """
        model = getattr(self.model, "_orig_mod", self.model)  # unwrap torch.compile
        payload = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "best_val_loss": self.state.best_val_loss,
            "history": self.state.history,
            "wall_seconds": self.state.wall_seconds,
            "model_config": model.config.to_dict(),
            "train_config": self.cfg.to_dict(),
            # The sampler's own generator, without which a resumed run draws a
            # different batch sequence and diverges from an uninterrupted one.
            "train_sampler_rng": self.train_data.get_rng_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        path = self.out_dir / f"checkpoint_{name}.pt"
        # Write to a temporary file first: a session killed mid-write would
        # otherwise leave a truncated checkpoint and lose the whole run.
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore a run from a checkpoint written by :meth:`save_checkpoint`."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        model = getattr(self.model, "_orig_mod", self.model)
        model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])

        self.state = TrainState(
            step=payload["step"],
            tokens_seen=payload["tokens_seen"],
            best_val_loss=payload["best_val_loss"],
            history=payload.get("history", []),
            wall_seconds=payload.get("wall_seconds", 0.0),
        )

        # Restoring RNG state makes a resumed run follow the same trajectory as an
        # uninterrupted one. It is a fidelity improvement, not a correctness
        # requirement: the model, optimizer, and step above are what actually
        # continue the run. So every restore below is best-effort - a checkpoint
        # written under a different torch version or GPU count can carry states
        # this build will not accept, and refusing to resume over that would
        # throw away a valid checkpoint for no benefit.
        if payload.get("train_sampler_rng") is not None:
            try:
                self.train_data.set_rng_state(payload["train_sampler_rng"])
            except (ValueError, KeyError, TypeError) as error:
                print(f"  note: sampler RNG state not restored ({error})")

        if payload.get("torch_rng") is not None:
            try:
                torch.set_rng_state(_as_byte_tensor(payload["torch_rng"]))
            except (RuntimeError, TypeError) as error:
                print(f"  note: CPU RNG state not restored ({error})")

        if payload.get("cuda_rng") and torch.cuda.is_available():
            try:
                states = [_as_byte_tensor(state) for state in payload["cuda_rng"]]
                # set_rng_state_all expects exactly one state per visible device;
                # a checkpoint from a host with a different GPU count would
                # otherwise raise deep inside the CUDA lazy-init callback.
                if len(states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(states)
                else:
                    print(
                        f"  note: CUDA RNG state not restored "
                        f"({len(states)} saved, {torch.cuda.device_count()} device(s) here)"
                    )
            except (RuntimeError, TypeError) as error:
                print(f"  note: CUDA RNG state not restored ({error})")

    def find_resume_point(self) -> Path | None:
        """Latest checkpoint in the output directory, if any."""
        latest = self.out_dir / "checkpoint_latest.pt"
        return latest if latest.exists() else None

    # -- training -----------------------------------------------------------

    def train(self, max_seconds: float | None = None, resume: bool = True) -> TrainState:
        """Run the training loop.

        Args:
            max_seconds: Stop cleanly after this much wall time, saving a
                checkpoint first. Set it below the platform's session limit so
                the run ends on its own terms rather than being killed mid-step.
            resume: Continue from ``checkpoint_latest.pt`` when one exists.

        Returns:
            The final training state.
        """
        if resume:
            found = self.find_resume_point()
            if found:
                self.load_checkpoint(found)
                print(f"resumed from {found.name} at step {self.state.step:,}")

        cfg = self.cfg
        block = self.train_data.block_size
        self.model.train()
        started = time.time()
        window_start = time.time()
        window_loss = 0.0
        window_steps = 0

        while self.state.step < cfg.max_steps:
            lr = cosine_lr(self.state.step, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # Gradient accumulation: several micro-batches summed into one
            # update, so the effective batch can exceed what fits in memory.
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(cfg.gradient_accumulation):
                x, y = self.train_data.batch(cfg.batch_size, device=self.device)
                with torch.autocast(
                    device_type=self.device, dtype=self.amp_dtype, enabled=self.amp_enabled
                ):
                    loss = self.model(x, targets=y)["loss"]
                    # Divide so the accumulated gradient is the mean over
                    # micro-batches, not their sum.
                    loss = loss / cfg.gradient_accumulation

                self.scaler.scale(loss).backward()
                step_loss += loss.float().item()

            if cfg.grad_clip > 0:
                # Unscale before clipping, or the threshold would apply to
                # scaled gradients and mean nothing.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.state.step += 1
            self.state.tokens_seen += cfg.tokens_per_step * block
            window_loss += step_loss
            window_steps += 1

            if self.state.step % cfg.log_interval == 0:
                elapsed = time.time() - window_start
                tps = (cfg.tokens_per_step * block * window_steps) / max(elapsed, 1e-9)
                print(
                    f"step {self.state.step:>6,}/{cfg.max_steps:,}  "
                    f"loss {window_loss / window_steps:.4f}  "
                    f"lr {lr:.2e}  {tps / 1e3:.1f}k tok/s  "
                    f"seen {self.state.tokens_seen / 1e6:.0f}M",
                    flush=True,
                )
                window_loss = 0.0
                window_steps = 0
                window_start = time.time()

            if self.state.step % cfg.eval_interval == 0:
                metrics = self.evaluate()
                metrics.update(step=self.state.step, lr=lr, tokens_seen=self.state.tokens_seen)
                self.state.history.append(metrics)
                print(
                    f"  eval step {self.state.step:,}: "
                    f"val_loss {metrics['val_loss']:.4f}  "
                    f"ppl {metrics['val_perplexity']:.2f}",
                    flush=True,
                )
                if metrics["val_loss"] < self.state.best_val_loss:
                    self.state.best_val_loss = metrics["val_loss"]
                    self.save_checkpoint("best")
                self._write_history()

            if self.state.step % cfg.checkpoint_interval == 0:
                self.save_checkpoint("latest")

            if max_seconds is not None and (time.time() - started) > max_seconds:
                self.state.wall_seconds += time.time() - started
                self.save_checkpoint("latest")
                self._write_history()
                print(
                    f"\nstopped at step {self.state.step:,} after "
                    f"{(time.time() - started) / 60:.0f} min (time budget reached); "
                    "rerun to continue from this checkpoint"
                )
                return self.state

        self.state.wall_seconds += time.time() - started
        self.save_checkpoint("latest")
        final = self.evaluate()
        final.update(step=self.state.step, tokens_seen=self.state.tokens_seen, final=True)
        self.state.history.append(final)
        self._write_history()
        print(
            f"\ntraining complete at step {self.state.step:,}: "
            f"val_loss {final['val_loss']:.4f}  ppl {final['val_perplexity']:.2f}"
        )
        return self.state

    def _write_history(self) -> None:
        """Persist the metric history for the report's loss curves."""
        path = self.out_dir / "training_history.json"
        path.write_text(
            json.dumps(
                {
                    "step": self.state.step,
                    "tokens_seen": self.state.tokens_seen,
                    "best_val_loss": self.state.best_val_loss,
                    "wall_seconds": round(self.state.wall_seconds, 1),
                    "train_config": self.cfg.to_dict(),
                    "history": self.state.history,
                },
                indent=1,
            )
        )
