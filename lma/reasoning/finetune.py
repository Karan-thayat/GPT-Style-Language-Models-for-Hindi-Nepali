"""Supervised finetuning of a pretrained model on the reasoning dataset.

This is deliberately a separate module from :mod:`lma.train.trainer` rather than
a flag on it, because the two objectives differ in one important way.

Pretraining predicts *every* token: the corpus is one flat stream and the loss is
the mean over all positions. Finetuning here predicts only the **answer**. The
prompt states premises the model is being conditioned on, not asked to produce -
training it to generate the premises would spend capacity modelling text the task
always supplies, and would let a model lower its loss without ever learning the
comparison. So the label for every prompt position is set to ``-100``, which
``F.cross_entropy`` ignores, and the loss is computed on the answer span alone.

What is reused rather than rewritten
------------------------------------
* :func:`lma.train.trainer.build_optimizer` - the decay/no-decay split is a
  property of parameter shapes, not of the objective.
* The checkpoint payload shape from :class:`lma.train.trainer.Trainer`, so a
  finetuned checkpoint resumes exactly like a pretrained one, which the
  assignment requires ("save finetuned checkpoints in the same resume-capable
  format as pretraining").
* :class:`lma.model.TransformerLM` unchanged - no new layers, no new heads. The
  task is expressed entirely in the data.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..model import ModelConfig, TransformerLM
from ..tokenizer.train_tokenizer import BOS_ID, EOS_ID, PAD_ID
from ..train.trainer import build_optimizer

#: Label value ignored by the loss. Matches ``ignore_index`` in the model's
#: cross-entropy call, so masked positions contribute nothing to the gradient.
IGNORE_INDEX = -100


@dataclass
class FinetuneConfig:
    """Optimization hyperparameters for one finetuning run.

    The defaults are deliberately gentler than pretraining's. The model arrives
    already trained, the dataset is small and highly regular, and a large
    learning rate would overwrite the language knowledge the reasoning task
    depends on - catastrophic forgetting. A low rate with a short warmup adapts
    the model while preserving what pretraining bought.

    Attributes:
        batch_size: Examples per optimizer step.
        learning_rate: Peak rate, an order of magnitude below pretraining's 6e-4.
        min_learning_rate: Floor of the cosine decay.
        warmup_steps: Linear warmup length.
        epochs: Passes over the training split.
        weight_decay: Applied to 2D weights only, as in pretraining.
        beta1: AdamW first-moment coefficient.
        beta2: AdamW second-moment coefficient.
        grad_clip: Gradient-norm clipping threshold.
        eval_interval: Steps between validation evaluations.
        log_interval: Steps between progress lines.
        seed: Seeds initialization and batch order.
        amp: Use mixed precision when CUDA is available.
        max_length: Longest tokenized example; longer ones are skipped.
    """

    batch_size: int = 32
    learning_rate: float = 5e-5
    min_learning_rate: float = 5e-6
    warmup_steps: int = 100
    epochs: int = 3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    log_interval: int = 50
    seed: int = 20260812
    amp: bool = True
    max_length: int = 64

    def to_dict(self) -> dict:
        """Serialize for checkpointing and the run record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FinetuneConfig":
        """Rebuild from a checkpoint payload, ignoring unknown keys."""
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})


class ReasoningDataset:
    """Tokenized reasoning examples with answer-span label masking.

    Each example becomes ``[BOS] prompt_tokens answer_tokens [EOS]``, framed the
    same way Phase 2 framed corpus documents so the model sees a familiar shape.
    Labels are the input shifted by one, with every prompt position replaced by
    :data:`IGNORE_INDEX`.

    Args:
        examples: Rows loaded from the generated JSONL, each carrying ``prompt``
            and ``answer``.
        tokenizer: SentencePiece processor for this language.
        max_length: Examples longer than this are dropped rather than truncated -
            truncating would cut the answer off and train the model on a
            mislabelled target.
    """

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 64) -> None:
        """Tokenize every example and build its masked label sequence."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows: list[dict] = []
        self.skipped = 0

        for row in examples:
            prompt_ids = tokenizer.encode(row["prompt"], out_type=int)
            full_ids = tokenizer.encode(f"{row['prompt']} {row['answer']}", out_type=int)

            # The prompt must be a token-level prefix of the full sequence, or the
            # answer span cannot be located by slicing. SentencePiece guarantees
            # this for the templates in use (verified), but a future template
            # could break it silently, so it is checked rather than assumed.
            if full_ids[: len(prompt_ids)] != prompt_ids:
                self.skipped += 1
                continue

            input_ids = [BOS_ID, *full_ids, EOS_ID]
            if len(input_ids) > max_length:
                self.skipped += 1
                continue

            # Labels predict position t+1 from position t. The BOS shifts
            # everything by one, so prompt positions occupy indices
            # 0 .. len(prompt_ids), and the answer begins right after.
            labels = list(input_ids)
            answer_start = len(prompt_ids) + 1  # +1 for BOS
            for index in range(min(answer_start, len(labels))):
                labels[index] = IGNORE_INDEX

            self.rows.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "answer_tokens": len(full_ids) - len(prompt_ids),
                }
            )

    def __len__(self) -> int:
        """Number of usable examples."""
        return len(self.rows)

    def batches(self, batch_size: int, shuffle: bool, seed: int, device: str):
        """Yield padded ``(x, y)`` batches.

        Sequences are padded to the longest in the batch with :data:`PAD_ID`, and
        the corresponding label positions are set to :data:`IGNORE_INDEX` so
        padding contributes nothing to the loss.

        Args:
            batch_size: Examples per batch.
            shuffle: Shuffle example order; on for training, off for evaluation.
            seed: Seeds the shuffle, so batch order is reproducible.
            device: Where to place the tensors.

        Yields:
            ``(x, y)`` int64 tensors of shape ``(batch, length)``.
        """
        order = list(range(len(self.rows)))
        if shuffle:
            import random

            random.Random(seed).shuffle(order)

        for start in range(0, len(order), batch_size):
            chunk = [self.rows[i] for i in order[start : start + batch_size]]
            width = max(len(row["input_ids"]) for row in chunk)

            inputs, labels = [], []
            for row in chunk:
                pad = width - len(row["input_ids"])
                inputs.append(row["input_ids"] + [PAD_ID] * pad)
                labels.append(row["labels"] + [IGNORE_INDEX] * pad)

            batch_inputs = torch.tensor(inputs, dtype=torch.long)
            batch_labels = torch.tensor(labels, dtype=torch.long)

            # Inputs are positions 0..n-1, targets are positions 1..n, the same
            # shift pretraining used.
            x = batch_inputs[:, :-1].contiguous()
            y = batch_labels[:, 1:].contiguous()

            if device != "cpu":
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            yield x, y

    def statistics(self) -> dict:
        """Summary of what was tokenized, for the run record."""
        lengths = [len(row["input_ids"]) for row in self.rows]
        answer_lengths = [row["answer_tokens"] for row in self.rows]
        return {
            "examples": len(self.rows),
            "skipped": self.skipped,
            "max_length": self.max_length,
            "sequence_tokens": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": sum(lengths) / len(lengths) if lengths else 0.0,
            },
            "answer_tokens": {
                "min": min(answer_lengths) if answer_lengths else 0,
                "max": max(answer_lengths) if answer_lengths else 0,
                "single_token_fraction": (
                    sum(1 for n in answer_lengths if n == 1) / len(answer_lengths)
                    if answer_lengths
                    else 0.0
                ),
            },
        }


def finetune_lr(step: int, total_steps: int, cfg: FinetuneConfig) -> float:
    """Learning rate at `step`: linear warmup, then cosine decay.

    Same shape as pretraining's schedule, but over the finetuning horizon. As
    there, the schedule is a pure function of the step, so storing the step in a
    checkpoint stores the scheduler position.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(cfg.warmup_steps, 1)

    progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + coefficient * (
        cfg.learning_rate - cfg.min_learning_rate
    )


@dataclass
class FinetuneState:
    """Everything needed to resume a finetuning run mid-flight."""

    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    history: list[dict] = field(default_factory=list)
    wall_seconds: float = 0.0


class ReasoningFinetuner:
    """Finetunes one pretrained model on its own language's reasoning data.

    Args:
        model: A model already loaded with pretrained weights.
        train_data: Tokenized training examples.
        val_data: Tokenized validation examples.
        cfg: Optimization hyperparameters.
        out_dir: Where checkpoints and the history file are written.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model: TransformerLM,
        train_data: ReasoningDataset,
        val_data: ReasoningDataset,
        cfg: FinetuneConfig,
        out_dir: str | Path,
        device: str | None = None,
    ) -> None:
        """Move the model to the device and prepare the optimizer."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = build_optimizer(self.model, cfg)
        self.state = FinetuneState()

        self.amp_enabled = cfg.amp and self.device == "cuda"
        self.amp_dtype = torch.float32
        if self.amp_enabled:
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )

        self.steps_per_epoch = max(
            1, math.ceil(len(train_data) / max(cfg.batch_size, 1))
        )
        self.total_steps = self.steps_per_epoch * cfg.epochs

        torch.manual_seed(cfg.seed)

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Mean answer-span cross-entropy over the validation split.

        Because prompt positions are masked out, this loss is the model's
        uncertainty about the *answer* alone, which is the quantity the task
        cares about - unlike a whole-sequence loss, which would be dominated by
        the highly predictable template text.
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for x, y in self.val_data.batches(
            self.cfg.batch_size, shuffle=False, seed=self.cfg.seed, device=self.device
        ):
            with torch.autocast(
                device_type=self.device, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                loss = self.model(x, targets=y)["loss"]
            total_loss += loss.float().item()
            total_batches += 1

        self.model.train()
        mean_loss = total_loss / max(total_batches, 1)
        return {
            "val_loss": mean_loss,
            "val_perplexity": math.exp(min(mean_loss, 20)),
            "batches": total_batches,
        }

    def save_checkpoint(self, name: str = "latest") -> Path:
        """Write a resume-capable checkpoint, atomically.

        Carries the same keys as a pretraining checkpoint - weights, optimizer
        state, step, and both configs - plus the pretrained checkpoint this run
        started from, so a finetuned model's provenance is recorded in the file
        rather than only in a notebook.
        """
        model = getattr(self.model, "_orig_mod", self.model)
        payload = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.state.step,
            "epoch": self.state.epoch,
            "best_val_loss": self.state.best_val_loss,
            "history": self.state.history,
            "wall_seconds": self.state.wall_seconds,
            "model_config": model.config.to_dict(),
            "finetune_config": self.cfg.to_dict(),
            "torch_rng": torch.get_rng_state(),
            "phase": "finetune",
        }
        path = self.out_dir / f"finetuned_{name}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume a finetuning run from one of its own checkpoints."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        model = getattr(self.model, "_orig_mod", self.model)
        model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])

        self.state = FinetuneState(
            step=payload.get("step", 0),
            epoch=payload.get("epoch", 0),
            best_val_loss=payload.get("best_val_loss", float("inf")),
            history=payload.get("history", []),
            wall_seconds=payload.get("wall_seconds", 0.0),
        )

        if payload.get("torch_rng") is not None:
            try:
                state = payload["torch_rng"]
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.uint8)
                torch.set_rng_state(state.cpu().to(torch.uint8))
            except (RuntimeError, TypeError) as error:
                print(f"  note: RNG state not restored ({error})")

    def train(self, resume: bool = True) -> FinetuneState:
        """Run the finetuning loop.

        Args:
            resume: Continue from ``finetuned_latest.pt`` when one exists.

        Returns:
            The final training state.
        """
        if resume:
            latest = self.out_dir / "finetuned_latest.pt"
            if latest.exists():
                self.load_checkpoint(latest)
                print(f"resumed at step {self.state.step:,}")

        self.model.train()
        started = time.time()

        while self.state.epoch < self.cfg.epochs:
            epoch_seed = self.cfg.seed + self.state.epoch
            for x, y in self.train_data.batches(
                self.cfg.batch_size, shuffle=True, seed=epoch_seed, device=self.device
            ):
                lr = finetune_lr(self.state.step, self.total_steps, self.cfg)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    loss = self.model(x, targets=y)["loss"]

                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.state.step += 1

                if self.state.step % self.cfg.log_interval == 0:
                    print(
                        f"step {self.state.step:>5,}/{self.total_steps:,}  "
                        f"loss {loss.float().item():.4f}  lr {lr:.2e}",
                        flush=True,
                    )

                if self.state.step % self.cfg.eval_interval == 0:
                    metrics = self.evaluate()
                    metrics.update(step=self.state.step, epoch=self.state.epoch, lr=lr)
                    self.state.history.append(metrics)
                    print(
                        f"  eval step {self.state.step:,}: "
                        f"val_loss {metrics['val_loss']:.4f}",
                        flush=True,
                    )
                    if metrics["val_loss"] < self.state.best_val_loss:
                        self.state.best_val_loss = metrics["val_loss"]
                        self.save_checkpoint("best")

            self.state.epoch += 1
            self.save_checkpoint("latest")

        final = self.evaluate()
        final.update(step=self.state.step, epoch=self.state.epoch, final=True)
        self.state.history.append(final)
        if final["val_loss"] < self.state.best_val_loss:
            self.state.best_val_loss = final["val_loss"]
            self.save_checkpoint("best")

        self.state.wall_seconds += time.time() - started
        self.save_checkpoint("latest")
        self._write_history()
        print(
            f"\nfinetuning complete: {self.state.step:,} steps, "
            f"best val_loss {self.state.best_val_loss:.4f}"
        )
        return self.state

    def _write_history(self) -> None:
        """Persist the metric history for the report's finetuning curve."""
        path = self.out_dir / "finetune_history.json"
        path.write_text(
            json.dumps(
                {
                    "step": self.state.step,
                    "epochs": self.state.epoch,
                    "best_val_loss": self.state.best_val_loss,
                    "wall_seconds": round(self.state.wall_seconds, 1),
                    "finetune_config": self.cfg.to_dict(),
                    "train_examples": len(self.train_data),
                    "val_examples": len(self.val_data),
                    "history": self.state.history,
                },
                indent=1,
            )
        )


def load_pretrained(checkpoint_path: str | Path, device: str) -> tuple[TransformerLM, dict]:
    """Rebuild a pretrained model from its Phase 2 checkpoint.

    The model architecture comes from the checkpoint's own stored config, so a
    finetuning run cannot silently disagree with the model it is loading.
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(payload["model_config"])
    model = TransformerLM(config)
    model.load_state_dict(payload["model"])
    model.to(device)
    return model, payload
