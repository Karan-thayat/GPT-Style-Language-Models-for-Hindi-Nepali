"""Generate the Kaggle notebook that pretrains one Phase 2 model.

Generated rather than hand-written so the code stays reviewable as ordinary
Python here, and so both languages share one source of truth. Cell sources are
stored as nbformat expects - a list of lines each keeping its newline - because
joining them into a single string fuses statements and produces cells that raise
SyntaxError on the first run.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "notebooks" / "phase2_pretrain.ipynb"

#: Source files shipped inside the notebook. The GitHub Classroom repository is
#: private, so Kaggle cannot clone it; embedding the package keeps the notebook
#: self-contained and guarantees it runs the same code that is committed here.
EMBEDDED_MODULES = [
    "lma/__init__.py",
    "lma/model/__init__.py",
    "lma/model/config.py",
    "lma/model/transformer.py",
    "lma/train/__init__.py",
    "lma/train/data.py",
    "lma/train/trainer.py",
]


def _lines(text: str) -> list[str]:
    """Split a cell body into nbformat's line list, preserving newlines."""
    body = text.strip("\n")
    parts = body.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _embedded_source() -> str:
    """Pack the lma package into one base64 blob for the notebook."""
    payload = {}
    for rel in EMBEDDED_MODULES:
        path = REPO_ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"cannot embed missing module: {path}")
        payload[rel] = path.read_text(encoding="utf-8")

    raw = json.dumps(payload).encode("utf-8")
    blob = base64.b64encode(gzip.compress(raw, 9)).decode()
    return "\n".join(
        '    "' + blob[i : i + 96] + '"' for i in range(0, len(blob), 96)
    )


def build() -> dict:
    cells: list[dict] = []

    cells.append(markdown("""
# Phase 2: pretrain one language model

Trains a 24.6M-parameter decoder-only Transformer on one language's corpus.
Run it **once per language** by changing `LANGUAGE` below.

### Before running

* Accelerator: **GPU** (P100 or T4). Settings -> Accelerator.
* Add the tokenized data: **+ Add Data** -> search your datasets ->
  `lma-phase2-tokenized`.
* On the second and later sessions, also add your own checkpoint dataset so the
  run continues instead of starting over. See the resume section at the bottom.

### Why this is split across sessions

Kaggle stops a session after 12 hours. A full run needs more than that, so the
trainer checkpoints every 1,000 steps and stops cleanly before the limit. A fresh
session resumes from the last checkpoint and continues on exactly the same
trajectory - this is verified bit-exact by `scripts/test_model.py`.
"""))

    cells.append(markdown("""
## 1. Choose the language

`hi` is Model H (Hindi, higher-resource); `ne` is Model L (Nepali).
"""))

    cells.append(code("""
LANGUAGE = "hi"        # "hi" or "ne"
TIME_BUDGET_HOURS = 10.5   # stop and checkpoint before Kaggle's 12-hour cutoff

assert LANGUAGE in ("hi", "ne")
print(f"training Model {'H (Hindi)' if LANGUAGE == 'hi' else 'L (Nepali)'}")
"""))

    cells.append(markdown("""
## 2. Unpack the project code

The `lma` package is embedded in this notebook rather than cloned: the GitHub
Classroom repository is private, so an unauthenticated clone from Kaggle fails.
The SHA-256 of each module is printed so it can be matched against the committed
source.
"""))

    cells.append(code(f"""
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

EMBEDDED_SOURCE = (
{_embedded_source()}
)

SRC = Path("/kaggle/working/src")
modules = json.loads(gzip.decompress(base64.b64decode(EMBEDDED_SOURCE)))

for rel, text in modules.items():
    path = SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    print(f"  {{rel:32s}} {{len(text):>6,}} bytes  sha256:{{digest}}")

sys.path.insert(0, str(SRC))
print(f"\\nunpacked {{len(modules)}} modules")
"""))

    cells.append(markdown("""
## 3. Locate the data and confirm the GPU
"""))

    cells.append(code("""
import torch

INPUT = Path("/kaggle/input")

# rglob rather than a fixed depth: Kaggle nests a dataset under its own name, and
# the tokenization notebook wrote its output inside a "tokenized/" folder, so the
# binaries can sit two or three levels down depending on how the dataset was
# created. Searching recursively covers every arrangement.
candidates = sorted(INPUT.rglob("*_train.bin"))
if not candidates:
    available = sorted(p.name for p in INPUT.iterdir()) if INPUT.exists() else []
    raise FileNotFoundError(
        "No *_train.bin found under /kaggle/input. "
        f"Attached datasets: {available or 'none'}. "
        "Use '+ Add Data' to attach lma-phase2-tokenized."
    )

DATA_DIR = candidates[0].parent
print("data:", DATA_DIR)
for path in sorted(DATA_DIR.glob("*.bin")):
    print(f"   {path.stat().st_size / 1e9:6.2f} GB  {path.name}")

# Fail early and clearly if the split this run needs is missing, rather than
# deep inside the training loop.
for split in ("train", "validation"):
    expected = DATA_DIR / f"{LANGUAGE}_{split}.bin"
    if not expected.exists():
        raise FileNotFoundError(f"{expected} is missing from the attached dataset")

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\\nGPU: {name}  {total:.1f} GB  bf16={torch.cuda.is_bf16_supported()}")
else:
    print("\\nWARNING: no GPU. Set Accelerator to GPU in Settings, or this will take days.")
"""))

    cells.append(markdown("""
## 4. Build the model

Identical architecture for both languages, so any difference measured in Phase 2
reflects data scale rather than capacity. Only the vocabulary differs, and here
it happens to be 8,000 for both.
"""))

    cells.append(code("""
from lma.model import ModelConfig, TransformerLM
from lma.train import TokenBinaryDataset, TrainConfig, Trainer

model_config = ModelConfig(
    vocab_size=8000,
    n_layer=12,
    n_head=6,
    n_embd=384,
    n_inner=1536,
    block_size=512,
    dropout=0.1,
    tie_weights=True,
)

model = TransformerLM(model_config)
counts = model_config.parameter_count()

print(f"parameters: {model.num_parameters():,}")
for key in ("token_embedding", "position_embedding", "blocks", "final_norm"):
    print(f"   {key:20s} {counts[key]:>12,}")
print(f"   embedding share:     {100 * model_config.embedding_share():.1f}%")
"""))

    cells.append(markdown("""
## 5. Train

The trainer resumes automatically when a checkpoint is present in the output
directory, and stops cleanly once the time budget is reached so the session ends
on its own terms rather than being killed mid-step.
"""))

    cells.append(code("""
OUT = Path("/kaggle/working") / f"{LANGUAGE}_model"
OUT.mkdir(parents=True, exist_ok=True)

# Seed a checkpoint from a previous session's dataset, if one is attached.
for previous in INPUT.glob("*/*_model/checkpoint_latest.pt"):
    if previous.parent.name == OUT.name and not (OUT / "checkpoint_latest.pt").exists():
        import shutil

        for item in previous.parent.glob("*"):
            shutil.copy2(item, OUT / item.name)
        print(f"seeded from {previous.parent}")
        break

train_config = TrainConfig(
    batch_size=24,
    gradient_accumulation=4,
    learning_rate=6e-4,
    min_learning_rate=6e-5,
    warmup_steps=500,
    max_steps=12_000,        # keep FIXED across sessions: it sets the decay horizon
    weight_decay=0.1,
    grad_clip=1.0,
    eval_interval=500,
    eval_batches=100,
    checkpoint_interval=1_000,
    log_interval=50,
    seed=20260812,
    amp=True,
)

train_data = TokenBinaryDataset(
    DATA_DIR / f"{LANGUAGE}_train.bin", model_config.block_size, seed=train_config.seed
)
val_data = TokenBinaryDataset(
    DATA_DIR / f"{LANGUAGE}_validation.bin", model_config.block_size, seed=train_config.seed + 1
)

print("train:", f"{train_data.stats()['tokens']:,} tokens")
print("val:  ", f"{val_data.stats()['tokens']:,} tokens")

tokens_per_step = train_config.tokens_per_step * model_config.block_size
print(f"\\n{tokens_per_step:,} tokens/step x {train_config.max_steps:,} steps "
      f"= {tokens_per_step * train_config.max_steps / 1e9:.2f}B tokens "
      f"({tokens_per_step * train_config.max_steps / train_data.stats()['tokens']:.2f} epochs)")

trainer = Trainer(model, train_data, val_data, train_config, OUT)
state = trainer.train(max_seconds=TIME_BUDGET_HOURS * 3600)
"""))

    cells.append(markdown("""
## 6. Where the run got to
"""))

    cells.append(code("""
import math

print(f"step {state.step:,} / {train_config.max_steps:,}")
print(f"tokens seen: {state.tokens_seen / 1e9:.2f}B")
print(f"best val loss: {state.best_val_loss:.4f}  (ppl {math.exp(state.best_val_loss):.2f})")
print(f"wall time: {state.wall_seconds / 3600:.2f} h")

if state.history:
    print("\\n   step      val_loss      ppl")
    for row in state.history[-10:]:
        print(f"  {row['step']:>7,}   {row['val_loss']:>8.4f}   {row['val_perplexity']:>8.2f}")

if state.step < train_config.max_steps:
    print("\\nNot finished. Save this output as a dataset, attach it to a new")
    print("session alongside lma-phase2-tokenized, and run again to continue.")
else:
    print("\\nTraining complete.")
"""))

    cells.append(markdown("""
## 7. Loss curve
"""))

    cells.append(code("""
import matplotlib.pyplot as plt

if state.history:
    steps = [row["step"] for row in state.history]
    losses = [row["val_loss"] for row in state.history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(steps, losses, marker="o", markersize=3)
    axes[0].set_title(f"Validation loss - Model {'H' if LANGUAGE == 'hi' else 'L'}")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross-entropy")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, [math.exp(min(v, 20)) for v in losses], marker="o", markersize=3, color="tab:orange")
    axes[1].set_title("Validation perplexity")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("perplexity")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(OUT / "loss_curve.png", dpi=150)
    plt.show()
else:
    print("no evaluations recorded yet")
"""))

    cells.append(markdown("""
## 8. A sample of what it writes

Not a graded metric - just a quick check that the model produces text in the
right script rather than noise.
"""))

    cells.append(code("""
prompt = torch.zeros((1, 1), dtype=torch.long, device=trainer.device)
generated = trainer.model.generate(prompt, max_new_tokens=60, temperature=0.8, top_k=50)
print("token ids:", generated[0, :20].tolist(), "...")
print("\\n(decode with the Phase 1 tokenizer during evaluation)")
"""))

    cells.append(markdown("""
## 9. Save the checkpoint

**Do not skip this.** `/kaggle/working` is wiped when the session ends.

Save the notebook version, then from the **Output** tab create a dataset named
`lma-phase2-<language>-checkpoint`. Attach it to the next session together with
`lma-phase2-tokenized`, and the run continues from where it stopped.
"""))

    cells.append(code("""
for path in sorted(OUT.iterdir()):
    print(f"{path.stat().st_size / 1e6:9.1f} MB  {path.name}")

print(f"\\nSave as a Kaggle Dataset named lma-phase2-{LANGUAGE}-checkpoint")
"""))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1))

    import ast

    failures = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        stripped = "\n".join(
            "" if line.lstrip().startswith("!") else line for line in source.split("\n")
        )
        try:
            ast.parse(stripped)
        except SyntaxError as exc:
            failures += 1
            print(f"  cell {index}: {exc}")

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(notebook['cells'])} cells, {size_kb:.0f} KB)")
    print("all code cells parse" if not failures else f"{failures} cell(s) failed to parse")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
