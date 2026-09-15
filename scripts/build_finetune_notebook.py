#!/usr/bin/env python3
"""Generate the Kaggle notebook that finetunes one Phase 3 model.

Generated rather than hand-written, for the same reasons as the Phase 2
notebooks: the code stays reviewable as ordinary Python here, both languages
share one source of truth, and the notebook is guaranteed to run the same code
that is committed in ``lma/``.

    python scripts/build_finetune_notebook.py

Writes ``notebooks/phase3_finetune.ipynb``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "notebooks" / "phase3_finetune.ipynb"

#: Where the modules are written inside the Kaggle session.
SRC_DIR = "/kaggle/working/src"

#: Source files shipped inside the notebook. The GitHub Classroom repository is
#: private, so Kaggle cannot clone it; embedding the package keeps the notebook
#: self-contained. Phase 3 needs the model, the optimizer helper from training,
#: and the whole reasoning package.
EMBEDDED_MODULES = [
    "lma/__init__.py",
    "lma/model/__init__.py",
    "lma/model/config.py",
    "lma/model/transformer.py",
    "lma/train/__init__.py",
    "lma/train/data.py",
    "lma/train/trainer.py",
    "lma/tokenizer/__init__.py",
    "lma/tokenizer/train_tokenizer.py",
    "lma/eval/__init__.py",
    "lma/eval/attention.py",
    "lma/eval/generation.py",
    "lma/eval/lm_metrics.py",
    "lma/stats/__init__.py",
    "lma/stats/plots.py",
    "lma/reasoning/__init__.py",
    "lma/reasoning/templates.py",
    "lma/reasoning/generate.py",
    "lma/reasoning/finetune.py",
    "lma/reasoning/evaluate.py",
    "lma/reasoning/plots.py",
    "lma/data/__init__.py",
    "lma/data/splits.py",
]

#: Packages whose ``__init__.py`` carries no code. These cannot be written with
#: ``%%writefile``, which errors on an empty cell body, so they are created with
#: ``touch()`` instead. Every empty marker must be listed: a package missing its
#: ``__init__.py`` fails to import inside the Kaggle session, and the failure
#: surfaces only once the notebook is already running there.
EMPTY_PACKAGES = [
    "lma/__init__.py",
    "lma/tokenizer/__init__.py",
    "lma/stats/__init__.py",
    "lma/data/__init__.py",
]


def _lines(text: str) -> list[str]:
    """Split a cell body into nbformat's line list, preserving newlines."""
    body = text.strip("\n")
    parts = body.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


_CELL_COUNTER = itertools.count()


def _cell_id() -> str:
    """Return a unique cell id, which nbformat 4.5+ expects."""
    return f"cell{next(_CELL_COUNTER):03d}"


def markdown(text: str) -> dict:
    """Build a markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": _lines(text),
    }


def code(text: str) -> dict:
    """Build a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(),
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _module_cells() -> list[dict]:
    """Emit one ``%%writefile`` cell per embedded module."""
    cells: list[dict] = []

    directories = sorted({str(Path(m).parent) for m in EMBEDDED_MODULES if Path(m).parent != Path(".")})
    cells.append(
        code(
            "import sys\n"
            "from pathlib import Path\n\n"
            f'SRC = Path("{SRC_DIR}")\n'
            f"for package in {directories!r}:\n"
            "    (SRC / package).mkdir(parents=True, exist_ok=True)\n\n"
            "if str(SRC) not in sys.path:\n"
            "    sys.path.insert(0, str(SRC))\n\n"
            'print(f"module directory ready: {SRC}")'
        )
    )

    empty = [m for m in EMPTY_PACKAGES if (REPO_ROOT / m).exists() and not (REPO_ROOT / m).read_text().strip()]
    if empty:
        body = "\n".join(f'(SRC / "{m}").touch()' for m in empty)
        cells.append(
            code(
                "# Empty package markers. These cannot use %%writefile, which errors\n"
                "# on an empty cell body, so they are created directly.\n"
                f"{body}\n\n"
                f'print("created {len(empty)} empty package marker(s)")'
            )
        )

    for relative in EMBEDDED_MODULES:
        path = REPO_ROOT / relative
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        if not source.strip():
            continue
        cells.append(code(f"%%writefile {SRC_DIR}/{relative}\n{source}"))

    return cells


def build() -> dict:
    """Assemble the notebook."""
    cells: list[dict] = []

    cells.append(markdown("""
# Phase 3: finetune one model on comparative reasoning

Finetunes a Phase 2 pretrained model on its own language's synthetic
comparative-reasoning dataset, then scores pretrained against finetuned on the
held-out test split and compares their attention.

Run it **once per language** by changing `LANGUAGE` below.

### Before running

* Accelerator: **GPU** (T4 or P100). Settings -> Accelerator.
* **+ Add Data**: the pretrained checkpoint dataset for this language, e.g.
  `lma-phase2-hi-final`.
* The reasoning dataset is generated inside this notebook from the committed
  generator, so nothing else needs attaching - and because generation is seeded,
  it reproduces the corpus in the repository exactly.

### Why this is much cheaper than pretraining

Pretraining consumed ~590M tokens over 12,000 steps. Finetuning sees 16,000 short
examples of about 20 tokens each, so a few epochs is a matter of minutes rather
than hours. The 12-hour session limit is not a constraint here, but the run is
still checkpointed and resumable, because the assignment requires finetuned
checkpoints in the same resume-capable format as pretraining.
"""))

    cells.append(markdown("## 1. Choose the language"))
    cells.append(code("""
LANGUAGE = "hi"        # "hi" or "ne" - must match the attached checkpoint

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
TOTAL_EXAMPLES = 20000   # same default as scripts/generate_reasoning_data.py

assert LANGUAGE in ("hi", "ne")
print(f"finetuning Model {'H (Hindi)' if LANGUAGE == 'hi' else 'L (Nepali)'}")
"""))

    cells.append(markdown("""
## 2. The project code

The `lma` package travels inside this notebook rather than being cloned: the
GitHub Classroom repository is private, so an unauthenticated `git clone` from
Kaggle fails. These are the committed sources, written out verbatim.
"""))
    cells.extend(_module_cells())

    cells.append(markdown("""
## 3. Find the pretrained checkpoint and the tokenizer

Both are attached datasets, so this searches `/kaggle/input` recursively rather
than assuming a fixed depth - Kaggle nests a dataset under its own name, and the
producing notebook may have written into a subfolder.
"""))
    cells.append(code("""
import torch
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / f"{LANGUAGE}_finetuned"
OUT.mkdir(parents=True, exist_ok=True)

# Prefer the best checkpoint (lowest validation loss) over the latest one, and
# only one belonging to THIS language - attaching both languages' datasets must
# not silently finetune the wrong model.
candidates = [
    p for p in list(INPUT.rglob("checkpoint_best.pt")) + list(INPUT.rglob("checkpoint_latest.pt"))
    if LANGUAGE in str(p).lower()
]
if not candidates:
    attached = sorted(p.name for p in INPUT.iterdir()) if INPUT.exists() else []
    raise FileNotFoundError(
        f"No checkpoint for {LANGUAGE} under /kaggle/input. Attached: {attached or 'none'}"
    )
CHECKPOINT = sorted(candidates)[0]
print("checkpoint:", CHECKPOINT)

tokenizers = sorted(INPUT.rglob(f"{LANGUAGE}_unigram_*.model"))
if tokenizers:
    TOKENIZER = tokenizers[0]
else:
    # The tokenizers are small and committed to the repository, so fall back to
    # a copy written by the cell below if no dataset supplied one.
    TOKENIZER = WORK / f"{LANGUAGE}_unigram_8000.model"
print("tokenizer:", TOKENIZER)
"""))

    cells.append(markdown("""
### If the tokenizer was not attached

The `.model` files live in the repository under `hindi/tokenizer/` and
`nepali/tokenizer/`. Upload the one for this language as a dataset, or paste it
here. The cell below only checks; it does not invent a tokenizer, because a
different vocabulary would silently invalidate the pretrained weights.
"""))
    cells.append(code("""
if not Path(TOKENIZER).exists():
    raise FileNotFoundError(
        f"{TOKENIZER} not found. Attach the Phase 1 tokenizer for {LANGUAGE} as a "
        "dataset (hindi/tokenizer/ or nepali/tokenizer/ in the repository)."
    )

import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
print("vocabulary:", tokenizer.get_piece_size())
"""))

    cells.append(markdown("""
## 4. Generate the reasoning dataset

Generated here from the committed generator rather than uploaded, so the notebook
is self-contained. Generation is seeded, so this reproduces the corpus in the
repository byte for byte - and the leakage checks below are the same ones
`scripts/generate_reasoning_data.py` enforces.
"""))
    cells.append(code("""
from lma.reasoning.generate import (
    GeneratorConfig, ReasoningDatasetBuilder, dataset_statistics,
)

builder = ReasoningDatasetBuilder(
    GeneratorConfig(language=LANGUAGE, total_examples=TOTAL_EXAMPLES)
)
examples = builder.build()
stats = dataset_statistics(examples, builder)

print("split sizes:", stats["by_split"])
print("chain lengths:", stats["by_chain_length"])
print()
print("leakage checks:")
for name, value in stats["leakage_checks"].items():
    print(f"  {name}: {value if not isinstance(value, list) else len(value)}")

violations = [
    name for name in (
        "chain_signature_overlap", "prompt_overlap_train_eval",
        "duplicate_prompts", "distractor_answers_in_eval",
    )
    if stats["leakage_checks"][name]
]
assert not violations, f"leakage checks failed: {violations}"
print("\\nall leakage checks passed")
"""))

    cells.append(code("""
print("sample training examples:\\n")
for example in examples[:4]:
    print(" ", example.text)
    print(f"    answer={example.answer}  chain={example.chain_length}  hops={example.hops}")
"""))

    cells.append(markdown("""
## 5. Load the pretrained model

The architecture comes from the checkpoint's own stored config, so a finetuning
run cannot silently disagree with the model it is loading.
"""))
    cells.append(code("""
from lma.reasoning.finetune import (
    FinetuneConfig, ReasoningDataset, ReasoningFinetuner, load_pretrained,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, payload = load_pretrained(CHECKPOINT, device)

print(f"parameters      : {model.num_parameters():,}")
print(f"pretrained step : {payload.get('step', 0):,}")
print(f"tokens seen     : {payload.get('tokens_seen', 0) / 1e6:.0f}M")
print(f"device          : {device}")
"""))

    cells.append(markdown("""
## 6. Score the pretrained model first

This is the "before" half of the comparison the assignment asks for, and it has
to be measured **before** finetuning touches the weights.
"""))
    cells.append(code("""
from lma.reasoning.evaluate import (
    above_chance, chance_baseline, compare_pretrained_finetuned, error_analysis,
    score_examples,
)

test_rows = [e.to_dict() for e in examples if e.split == "test"]
train_rows = [e.to_dict() for e in examples if e.split == "train"]
val_rows = [e.to_dict() for e in examples if e.split == "validation"]

before = score_examples(model, test_rows, tokenizer, device=device)
baseline = chance_baseline(test_rows)

print(f"pretrained exact match : {before['exact_match']:.4f}")
print(f"constrained accuracy   : {before['constrained_accuracy']:.4f}")
print(f"chance (aggregate)     : {baseline['overall']:.4f}")
"""))

    cells.append(markdown("""
## 7. Finetune

Loss is computed on the **answer span only**: the prompt states premises the
model is conditioned on, not asked to produce. Training it to generate the
premises would spend capacity on text the task always supplies.
"""))
    cells.append(code("""
config = FinetuneConfig(
    batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE,
)

train_data = ReasoningDataset(train_rows, tokenizer, config.max_length)
val_data = ReasoningDataset(val_rows, tokenizer, config.max_length)
print(f"train {len(train_data)}, validation {len(val_data)}")
print("tokenization:", train_data.statistics()["sequence_tokens"])

finetuner = ReasoningFinetuner(model, train_data, val_data, config, OUT, device=device)
state = finetuner.train(resume=True)
"""))

    cells.append(markdown("""
## 8. Score the finetuned model

Same examples, same decoding, same match rule - the only thing that changed is
the weights. The model scored is the checkpoint with the **lowest validation
loss**, reloaded from disk, not whatever happens to be in memory after the final
step: `finetuned_best.pt` is the file that gets published, so the reported
numbers must describe exactly that file.
"""))
    cells.append(code("""
del model
model, best_payload = load_pretrained(OUT / "finetuned_best.pt", device)
print(f"scoring finetuned_best.pt (step {best_payload['step']:,}, "
      f"val loss {best_payload['best_val_loss']:.4f})")

after = score_examples(model, test_rows, tokenizer, device=device)
comparison = compare_pretrained_finetuned(before, after)
margins = above_chance(after, baseline)

print(f"exact match : {before['exact_match']:.4f} -> {after['exact_match']:.4f}"
      f"  (delta {comparison['exact_match']['delta']:+.4f})")
print(f"constrained : {before['constrained_accuracy']:.4f} -> {after['constrained_accuracy']:.4f}")
print()
print("per chain length (finetuned):")
for key, row in sorted(margins["by_chain_length"].items()):
    verdict = "above" if row["above_chance"] else "below"
    print(f"  {key} entities: acc {row['accuracy']:.3f}  chance {row['chance']:.3f}"
          f"  margin {row['margin']:+.3f}  n={row['n']}  ({verdict} chance)")
"""))

    cells.append(code("""
print("error profile (finetuned):")
errors = error_analysis(after, test_rows)
for name, fraction in errors["fractions"].items():
    print(f"  {name:14s} {fraction:.3f}")

print("\\nsample predictions:")
for prediction in after["predictions"][:8]:
    mark = "OK " if prediction["correct"] else "BAD"
    print(f"  [{mark}] expected={prediction['expected']:10s} "
          f"predicted={prediction['predicted']:10s} chain={prediction['chain_length']}")
"""))

    cells.append(markdown("""
## 9. Attention: pretrained against finetuned

The same comparative-reasoning prompts through both models, so any difference in
where the heads look is attributable to finetuning. Head statistics are averaged
over 200 test prompts before heads are classified - one prompt is a sample of
one - and the answer-attention measurement asks directly how much the position
that predicts the answer attends to the correct entity versus the others. The
heatmaps show one prompt, as an illustration of those averages.
"""))
    cells.append(code("""
from lma.eval import plot_attention_heatmaps, plot_head_summary
from lma.reasoning.evaluate import answer_attention_by_layer, averaged_attention_statistics
from lma.reasoning.plots import plot_answer_attention
from lma.tokenizer.train_tokenizer import BOS_ID

FIGURES = WORK / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
LANGUAGE_NAME = "Hindi" if LANGUAGE == "hi" else "Nepali"
ATTENTION_PROMPTS = 200

prompt = test_rows[0]["prompt"]
print("illustration prompt:", prompt, "\\n")

# The pretrained weights were overwritten in place by finetuning, so reload them
# for the comparison rather than trusting the in-memory model.
pretrained_model, _ = load_pretrained(CHECKPOINT, device)

attention_summary = {}
for name, this_model in (("pretrained", pretrained_model), ("finetuned", model)):
    averaged = averaged_attention_statistics(
        this_model, test_rows, tokenizer, device=device, limit=ATTENTION_PROMPTS
    )
    answer = answer_attention_by_layer(
        this_model, test_rows, tokenizer, device=device, limit=ATTENTION_PROMPTS
    )
    attention_summary[name] = {
        "layer_mean_entropy": averaged["layer_mean_entropy"],
        "layer_mean_distance": averaged["layer_mean_distance"],
        "normalized_entropy": averaged["normalized_entropy"],
        "mean_distance": averaged["mean_distance"],
        "head_groups": averaged["head_groups"],
        "prompts": averaged["prompts"],
        "answer_attention": answer,
    }
    print(f"{name:11s} head types: {averaged['head_groups']}  "
          f"(averaged over {averaged['prompts']} prompts)")

    ids = torch.tensor([[BOS_ID, *tokenizer.encode(prompt, out_type=int)]], dtype=torch.long)
    with torch.no_grad():
        attentions = this_model(ids.to(device), return_attention=True)["attentions"]
    pieces = [tokenizer.id_to_piece(int(t)) for t in ids[0].tolist()]
    last = this_model.config.n_layer - 1
    plot_attention_heatmaps(
        attentions, pieces,
        layers=[0, last], heads=list(range(min(3, this_model.config.n_head))),
        out_path=FIGURES / f"reasoning_attention_{name}_{LANGUAGE}.png",
        title=f"{LANGUAGE_NAME} reasoning prompt - {name} model (layers 0 and {last})",
    )
    plot_head_summary(
        averaged, FIGURES / f"reasoning_head_summary_{name}_{LANGUAGE}.png",
        title=f"{LANGUAGE_NAME} per-head attention - {name} model, "
              f"averaged over {averaged['prompts']} test prompts",
    )

plot_answer_attention(
    attention_summary["pretrained"]["answer_attention"],
    attention_summary["finetuned"]["answer_attention"],
    FIGURES / f"reasoning_answer_attention_{LANGUAGE}.png",
    title=f"{LANGUAGE_NAME} - attention from the answer position, averaged over "
          f"{attention_summary['finetuned']['answer_attention']['prompts']} test prompts",
)

print()
print("layer  pretrained  finetuned   delta")
shifts = []
for layer, (pre, post) in enumerate(zip(
    attention_summary["pretrained"]["layer_mean_distance"],
    attention_summary["finetuned"]["layer_mean_distance"],
)):
    shifts.append({"layer": layer, "pretrained": pre, "finetuned": post, "delta": post - pre})
    print(f"{layer:5d}  {pre:10.2f}  {post:9.2f}  {post - pre:+6.2f}")
attention_summary["distance_shift_by_layer"] = shifts
"""))

    cells.append(code("""
from IPython.display import Image, display

for name in ("pretrained", "finetuned"):
    print(name)
    display(Image(filename=str(FIGURES / f"reasoning_attention_{name}_{LANGUAGE}.png")))
"""))

    cells.append(markdown("""
## 10. Save the results

Everything the Phase 3 report needs. Download `phase3_evaluation.json` and place
it at `<language>/reports/phase3_evaluation.json` in the repository, copy the
figures into `report/figures/`, then run `scripts/make_phase3_report.py`.
"""))
    cells.append(code("""
import json

results = {
    "language": LANGUAGE,
    "model": "Model H" if LANGUAGE == "hi" else "Model L",
    "test_examples": len(test_rows),
    "pretrained_checkpoint": str(CHECKPOINT),
    "finetuned_checkpoint": str(OUT / "finetuned_best.pt"),
    "pretrained_step": payload.get("step"),
    "finetuned_step": best_payload.get("step"),
    "finetuned_best_val_loss": best_payload.get("best_val_loss"),
    "checkpoint_selection": "lowest answer-span validation loss (finetuned_best.pt)",
    "attention_prompts": attention_summary["finetuned"]["prompts"],
    "parameters": model.num_parameters(),
    "model_config": model.config.to_dict(),
    "finetune_config": config.to_dict(),
    "chance_baseline": baseline,
    "pretrained": before,
    "finetuned": after,
    "comparison": comparison,
    "above_chance": margins,
    "error_analysis": {
        "pretrained": error_analysis(before, test_rows),
        "finetuned": errors,
    },
    "attention": attention_summary,
    "attention_prompt": prompt,
    "dataset": stats,
}

path = WORK / "phase3_evaluation.json"
path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
print("wrote", path)

record = {
    "language": LANGUAGE,
    "pretrained_checkpoint": str(CHECKPOINT),
    "pretrained_step": payload.get("step"),
    "parameters": model.num_parameters(),
    "model_config": model.config.to_dict(),
    "finetune_config": config.to_dict(),
    "steps": state.step,
    "epochs": state.epoch,
    "best_val_loss": state.best_val_loss,
    "wall_seconds": round(state.wall_seconds, 1),
    "train_data": train_data.statistics(),
    "val_data": val_data.statistics(),
    "history": state.history,
}
record_path = WORK / "finetune_record.json"
record_path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
print("wrote", record_path)
"""))

    cells.append(markdown("""
## 11. Save the checkpoint

**Do not skip this.** `/kaggle/working` is wiped when the session ends.

Save the notebook version, then from the **Output** tab create a dataset named
`lma-phase3-<language>-finetuned`. Put its Drive link in the README, which the
submission instructions require for the finetuned checkpoints.
"""))
    cells.append(code("""
for path in sorted(OUT.iterdir()):
    print(f"{path.stat().st_size / 1e6:9.1f} MB  {path.name}")

print(f"\\nSave as a Kaggle Dataset named lma-phase3-{LANGUAGE}-finetuned")
print("Then upload finetuned_best.pt to Google Drive and put the link in README.md")
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
    """Write the notebook and check that every code cell parses."""
    notebook = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1))

    import ast

    failures = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Strip IPython magics and shell escapes before parsing: they are valid
        # in a notebook but not in plain Python.
        stripped = "\n".join(
            "" if line.lstrip().startswith(("!", "%%", "%")) else line
            for line in source.split("\n")
        )
        try:
            ast.parse(stripped)
        except SyntaxError as error:
            failures += 1
            print(f"  cell {index}: {error}")

    size_kb = OUTPUT.stat().st_size / 1024
    print(
        f"wrote {OUTPUT.relative_to(REPO_ROOT)} "
        f"({len(notebook['cells'])} cells, {size_kb:.0f} KB)"
    )
    print("all code cells parse" if not failures else f"{failures} cell(s) failed to parse")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
