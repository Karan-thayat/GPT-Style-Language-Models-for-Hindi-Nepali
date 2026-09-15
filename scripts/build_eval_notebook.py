"""Generate the Kaggle notebook that evaluates a trained Phase 2 model.

Generated rather than hand-written so the code stays reviewable as ordinary
Python in this repository. Cell sources are stored as nbformat expects - a list
of lines each keeping its newline - because joining them into one string fuses
statements and produces cells that raise SyntaxError on the first run.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "notebooks" / "phase2_evaluate.ipynb"

#: Where the bonus ablation evaluation notebook is written by ``--ablation``.
ABLATION_OUTPUT = REPO_ROOT / "notebooks" / "bonus_nopos_evaluate.ipynb"

#: The evaluation code, shipped inside the notebook. The GitHub Classroom
#: repository is private, so Kaggle cannot clone it.
EMBEDDED_MODULES = [
    "lma/__init__.py",
    "lma/model/__init__.py",
    "lma/model/config.py",
    "lma/model/transformer.py",
    "lma/train/__init__.py",
    "lma/train/data.py",
    "lma/train/trainer.py",
    "lma/eval/__init__.py",
    "lma/eval/lm_metrics.py",
    "lma/eval/generation.py",
    "lma/eval/attention.py",
    "lma/stats/__init__.py",
    "lma/stats/plots.py",
]


def _lines(text: str) -> list[str]:
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
    payload = {}
    for rel in EMBEDDED_MODULES:
        path = REPO_ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"cannot embed missing module: {path}")
        payload[rel] = path.read_text(encoding="utf-8")

    raw = json.dumps(payload).encode("utf-8")
    blob = base64.b64encode(gzip.compress(raw, 9)).decode()
    return "\n".join('    "' + blob[i : i + 96] + '"' for i in range(0, len(blob), 96))


def _embedded_tokenizers() -> str:
    """Base64+gzip the Phase 1 tokenizers; bits-per-byte needs them."""
    parts = []
    for lang, rel in (
        ("hi", "hindi/tokenizer/hi_unigram_8000.model"),
        ("ne", "nepali/tokenizer/ne_unigram_8000.model"),
    ):
        path = REPO_ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"missing tokenizer: {path}")
        blob = base64.b64encode(gzip.compress(path.read_bytes(), 9)).decode()
        wrapped = "\n".join(
            '    "' + blob[i : i + 96] + '"' for i in range(0, len(blob), 96)
        )
        parts.append(f'"{lang}": (\n{wrapped}\n),')
    return "\n".join(parts)


def build() -> dict:
    cells: list[dict] = []

    cells.append(markdown("""
# Phase 2: evaluate a trained model

Runs the whole of section 2.3 for one language and writes a JSON result file:

* validation cross-entropy, perplexity, and bits-per-byte on the held-out split
* BLEU-4, chrF, and ROUGE-L for greedy decoding and temperatures 0.5, 1.0, 1.5
* repetition rate and Distinct-1 / Distinct-2
* attention entropy and mean attention distance per head, plus heatmaps from an
  early and a late layer

### Before running

* Accelerator: **GPU** (T4 x2 works; a P100 may lack kernels for the installed
  torch build - the setup cell checks and says so).
* **+ Add Data**: `lma-phase2-tokenized` **and** the checkpoint dataset for this
  language, e.g. `lma-phase2-hi-checkpoint`.
* Set `LANGUAGE` below to match the checkpoint you attached.
"""))

    cells.append(code("""
LANGUAGE = "hi"        # "hi" or "ne" - must match the attached checkpoint

SAMPLES = 100          # held-out prompts to generate from
PROMPT_TOKENS = 32     # prefix length
CONTINUATION_TOKENS = 64
EVAL_BATCHES = 200     # batches of held-out text for perplexity and BPB

assert LANGUAGE in ("hi", "ne")
print(f"evaluating Model {'H (Hindi)' if LANGUAGE == 'hi' else 'L (Nepali)'}")
"""))

    cells.append(markdown("""
## 1. Unpack the project code and tokenizers
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

EMBEDDED_TOKENIZERS = {{
{_embedded_tokenizers()}
}}

WORK = Path("/kaggle/working")
SRC = WORK / "src"

modules = json.loads(gzip.decompress(base64.b64decode(EMBEDDED_SOURCE)))
for rel, text in modules.items():
    path = SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
sys.path.insert(0, str(SRC))
print(f"unpacked {{len(modules)}} modules")

TOK_DIR = WORK / "tokenizers"
TOK_DIR.mkdir(parents=True, exist_ok=True)
for lang, blob in EMBEDDED_TOKENIZERS.items():
    raw = gzip.decompress(base64.b64decode(blob))
    (TOK_DIR / f"{{lang}}_unigram_8000.model").write_bytes(raw)
    print(f"  {{lang}} tokenizer  sha256:{{hashlib.sha256(raw).hexdigest()[:12]}}")
"""))

    cells.append(markdown("""
## 2. Find the data and the checkpoint

Both are attached datasets, so this searches `/kaggle/input` recursively rather
than assuming a fixed depth - Kaggle nests a dataset under its own name, and the
producing notebook may have written into a subfolder.
"""))

    cells.append(code("""
import torch

INPUT = Path("/kaggle/input")

train_bins = sorted(INPUT.rglob(f"{LANGUAGE}_test.bin"))
if not train_bins:
    attached = sorted(p.name for p in INPUT.iterdir()) if INPUT.exists() else []
    raise FileNotFoundError(
        f"No {LANGUAGE}_test.bin under /kaggle/input. Attached: {attached or 'none'}"
    )
DATA_DIR = train_bins[0].parent
print("data:", DATA_DIR)

# Prefer the best checkpoint (lowest validation loss) over the latest one.
checkpoints = sorted(INPUT.rglob("checkpoint_best.pt")) or sorted(
    INPUT.rglob("checkpoint_latest.pt")
)
if not checkpoints:
    attached = sorted(p.name for p in INPUT.iterdir()) if INPUT.exists() else []
    raise FileNotFoundError(
        f"No checkpoint_*.pt under /kaggle/input. Attached: {attached or 'none'}. "
        f"Add the lma-phase2-{LANGUAGE}-checkpoint dataset."
    )
CHECKPOINT = checkpoints[0]
print("checkpoint:", CHECKPOINT)
print(f"   {CHECKPOINT.stat().st_size / 1e6:.0f} MB")

if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    print(f"\\nGPU: {torch.cuda.get_device_name(0)}  sm_{capability[0]}{capability[1]}")
    try:
        probe = torch.zeros(8, device="cuda") + 1
        _ = (probe @ probe).item()
        print("   GPU compute check: OK")
    except RuntimeError as error:
        raise RuntimeError(
            f"This torch build has no kernels for sm_{capability[0]}{capability[1]}. "
            f"Switch the Accelerator to another GPU type. Original error: {error}"
        ) from error
else:
    print("\\nNo GPU - evaluation will work but will be slow.")
"""))

    cells.append(markdown("""
## 3. Load the model
"""))

    cells.append(code("""
import sentencepiece as spm

from lma.model import ModelConfig, TransformerLM
from lma.train import TokenBinaryDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model_config = ModelConfig.from_dict(payload["model_config"])
model = TransformerLM(model_config)
model.load_state_dict(payload["model"])
model.to(device).eval()

tokenizer = spm.SentencePieceProcessor(
    model_file=str(TOK_DIR / f"{LANGUAGE}_unigram_8000.model")
)

print(f"parameters   : {model.num_parameters():,}")
print(f"trained steps: {payload.get('step', 0):,}")
print(f"tokens seen  : {payload.get('tokens_seen', 0) / 1e9:.2f}B")
print(f"best val loss: {payload.get('best_val_loss', float('nan')):.4f}")
"""))

    cells.append(markdown("""
## 4. Language-modeling metrics

Perplexity is `exp` of the mean cross-entropy. It is not comparable across models
with different tokenizers, so bits-per-byte is reported alongside it: the same
total loss expressed per byte of the underlying UTF-8 text, which is a property
of the text rather than of the segmentation.
"""))

    cells.append(code("""
from lma.eval import evaluate_language_model

test_data = TokenBinaryDataset(DATA_DIR / f"{LANGUAGE}_test.bin", model_config.block_size)

lm_metrics = evaluate_language_model(
    model, test_data, batch_size=16, max_batches=EVAL_BATCHES,
    device=device, tokenizer=tokenizer,
)

print(f"cross-entropy : {lm_metrics['cross_entropy']:.4f} nats/token")
print(f"perplexity    : {lm_metrics['perplexity']:.2f}")
print(f"bits-per-byte : {lm_metrics['bits_per_byte']:.4f}")
print(f"tokens        : {lm_metrics['tokens']:,}")
print(f"bytes/token   : {lm_metrics['bytes_per_token']:.2f}")
"""))

    cells.append(markdown("""
## 5. Generation quality

Continuations from held-out prefixes, scored against the text that actually
followed them, under greedy decoding and three temperatures.
"""))

    cells.append(code("""
from lma.eval import evaluate_generations

DECODE_SETTINGS = [
    {"name": "greedy", "greedy": True},
    {"name": "temperature_0.5", "temperature": 0.5},
    {"name": "temperature_1.0", "temperature": 1.0},
    {"name": "temperature_1.5", "temperature": 1.5},
]

window = PROMPT_TOKENS + CONTINUATION_TOKENS
stride = window * 4          # spread prompts across the split

prompts, references = [], []
for index in range(SAMPLES):
    start = index * stride
    if start + window >= len(test_data.tokens):
        break
    chunk = [int(t) for t in test_data.tokens[start : start + window]]
    prompts.append(torch.tensor(chunk[:PROMPT_TOKENS], dtype=torch.long))
    references.append(tokenizer.decode(chunk[PROMPT_TOKENS:]))

print(f"{len(prompts)} prompts of {PROMPT_TOKENS} tokens\\n")

generation_results = {}
samples = {}

for setting in DECODE_SETTINGS:
    name = setting["name"]
    kwargs = {k: v for k, v in setting.items() if k != "name"}

    outputs = []
    for start in range(0, len(prompts), 8):
        batch = torch.stack(prompts[start : start + 8]).to(device)
        generated = model.generate(batch, max_new_tokens=CONTINUATION_TOKENS, **kwargs)
        for row in generated[:, batch.shape[1]:].tolist():
            outputs.append(tokenizer.decode(row))

    metrics = evaluate_generations(outputs, references[: len(outputs)])
    generation_results[name] = metrics
    samples[name] = outputs[:3]

    print(
        f"{name:16s} BLEU {metrics['bleu']:5.2f}  chrF {metrics['chrf']:5.2f}  "
        f"ROUGE-L {metrics['rouge_l']:5.2f}  rep {metrics['repetition_rate']:.3f}  "
        f"D-1 {metrics['distinct_1']:.3f}  D-2 {metrics['distinct_2']:.3f}"
    )
"""))

    cells.append(markdown("""
### What the model actually writes
"""))

    cells.append(code("""
for name in ("greedy", "temperature_1.0"):
    print(f"--- {name} ---")
    for text in samples[name][:2]:
        print(" ", text.strip()[:200])
    print()
"""))

    cells.append(markdown("""
## 6. Attention analysis
"""))

    cells.append(code("""
from lma.eval import attention_statistics, classify_heads, plot_attention_heatmaps, plot_head_summary

FIGURES = WORK / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sample_ids = torch.tensor(
    [int(t) for t in test_data.tokens[:64]], dtype=torch.long
).unsqueeze(0)

with torch.no_grad():
    attentions = model(sample_ids.to(device), return_attention=True)["attentions"]

stats = attention_statistics(attentions)
groups = classify_heads(stats)

print(f"{stats['n_layer']} layers x {stats['n_head']} heads")
print(f"head types: {groups['counts']}")
print()
print("layer  mean entropy  mean distance")
for index, (entropy, distance) in enumerate(
    zip(stats["layer_mean_entropy"], stats["layer_mean_distance"])
):
    print(f"{index:5d}  {entropy:12.3f}  {distance:13.2f}")

pieces = [tokenizer.id_to_piece(int(t)) for t in sample_ids[0].tolist()]
label = "Hindi (Model H)" if LANGUAGE == "hi" else "Nepali (Model L)"

plot_attention_heatmaps(
    attentions, pieces,
    layers=[0, model_config.n_layer - 1], heads=[0, 1, 2],
    out_path=FIGURES / f"attention_heatmaps_{LANGUAGE}.png",
    title=f"Attention patterns - {label}",
)
plot_head_summary(
    stats, FIGURES / f"attention_summary_{LANGUAGE}.png",
    title=f"Per-head statistics - {label}",
)
print(f"\\nwrote figures to {FIGURES}")
"""))

    cells.append(code("""
from IPython.display import Image, display

display(Image(filename=str(FIGURES / f"attention_heatmaps_{LANGUAGE}.png")))
display(Image(filename=str(FIGURES / f"attention_summary_{LANGUAGE}.png")))
"""))

    cells.append(markdown("""
## 7. Save the results

Everything the Phase 2 report needs, in one JSON file plus the figures. Save this
notebook's output as a dataset (or download the JSON) and place it at
`<language>/reports/phase2_evaluation.json` in the repository, then run
`scripts/make_phase2_report.py`.
"""))

    cells.append(code("""
results = {
    "language": LANGUAGE,
    "checkpoint": str(CHECKPOINT),
    "parameters": model.num_parameters(),
    "training_step": payload.get("step"),
    "tokens_seen": payload.get("tokens_seen"),
    "model_config": payload.get("model_config"),
    "language_modeling": lm_metrics,
    "generation": generation_results,
    "samples": samples,
    "attention": {
        "normalized_entropy": stats["normalized_entropy"],
        "mean_distance": stats["mean_distance"],
        "layer_mean_entropy": stats["layer_mean_entropy"],
        "layer_mean_distance": stats["layer_mean_distance"],
        "head_groups": groups["counts"],
    },
}

out_path = WORK / f"phase2_evaluation_{LANGUAGE}.json"
out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

# Also copy the training history out of the checkpoint dataset, since the report
# draws its loss curves from it.
for history in INPUT.rglob("training_history.json"):
    target = WORK / f"training_history_{LANGUAGE}.json"
    target.write_text(history.read_text())
    print(f"copied {target.name}")
    break
"""))

    cells.append(markdown("""
### Then

Download `phase2_evaluation_<lang>.json` and `training_history_<lang>.json` from
the Output tab, put them in `<language>/reports/` (renaming the evaluation file
to `phase2_evaluation.json`), copy the two figures into `report/figures/`, and
run:

```bash
python scripts/make_phase2_report.py
```
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


#: The ablation evaluation is the Phase 2 evaluation with exactly these edits, so
#: the ablated and standard models are measured identically. Each edit must apply
#: the stated number of times: one that silently failed to apply could evaluate
#: the standard checkpoint, or overwrite its results, under the ablation's name.
ABLATION_EDITS = (
    ("# Phase 2: evaluate a trained model\n\n"
     "Runs the whole of section 2.3 for one language and writes a JSON result file:",
     "# Bonus ablation: evaluate the model trained without positional embeddings\n\n"
     "Runs exactly the Phase 2 evaluation (section 2.3) on the ablated checkpoint - the\n"
     "same held-out text, prompts, decoding settings and 64-token attention sample as\n"
     "the standard model's evaluation, so the two sets of numbers compare directly -\n"
     "and writes a JSON result file:", 1),
    ("the checkpoint dataset for this\n  language, e.g. `lma-phase2-hi-checkpoint`.",
     "the ablation checkpoint dataset,\n  e.g. `lma-bonus-hi-nopos-checkpoint`. Only the `<language>_nopos` folder is\n"
     "  accepted, so the standard model's checkpoint cannot be evaluated by mistake.", 1),
    ("# Prefer the best checkpoint (lowest validation loss) over the latest one.\n"
     "checkpoints = sorted(INPUT.rglob(\"checkpoint_best.pt\")) or sorted(\n"
     "    INPUT.rglob(\"checkpoint_latest.pt\")\n)",
     "# Prefer the best checkpoint (lowest validation loss) over the latest one, and\n"
     "# accept only the ablation run's own folder.\n"
     "RUN_DIR = f\"{LANGUAGE}_nopos\"\n"
     "checkpoints = sorted(\n"
     "    p for p in INPUT.rglob(\"checkpoint_best.pt\") if p.parent.name == RUN_DIR\n"
     ") or sorted(p for p in INPUT.rglob(\"checkpoint_latest.pt\") if p.parent.name == RUN_DIR)", 1),
    ("lma-phase2-{LANGUAGE}-checkpoint", "lma-bonus-{LANGUAGE}-nopos-checkpoint", 1),
    ("model.load_state_dict(payload[\"model\"])\nmodel.to(device).eval()",
     "model.load_state_dict(payload[\"model\"])\nmodel.to(device).eval()\n"
     "assert model.position_embedding is None, "
     "\"this checkpoint has positional embeddings - it is not the ablation\"", 1),
    ("f\"attention_heatmaps_{LANGUAGE}.png\"", "f\"attention_heatmaps_{LANGUAGE}_nopos.png\"", 2),
    ("f\"attention_summary_{LANGUAGE}.png\"", "f\"attention_summary_{LANGUAGE}_nopos.png\"", 2),
    ("title=f\"Attention patterns - {label}\",",
     "title=f\"Attention patterns - {label}, no positional embeddings\",", 1),
    ("title=f\"Per-head statistics - {label}\",",
     "title=f\"Per-head statistics - {label}, no positional embeddings\",", 1),
    ("    \"language\": LANGUAGE,\n    \"checkpoint\": str(CHECKPOINT),",
     "    \"language\": LANGUAGE,\n    \"variant\": \"no_positional_embeddings\",\n"
     "    \"checkpoint\": str(CHECKPOINT),", 1),
    ("out_path = WORK / f\"phase2_evaluation_{LANGUAGE}.json\"",
     "out_path = WORK / f\"bonus_nopos_evaluation_{LANGUAGE}.json\"", 1),
    ("for history in INPUT.rglob(\"training_history.json\"):\n"
     "    target = WORK / f\"training_history_{LANGUAGE}.json\"",
     "for history in (p for p in INPUT.rglob(\"training_history.json\") if p.parent.name == RUN_DIR):\n"
     "    target = WORK / f\"training_history_{LANGUAGE}_nopos.json\"", 1),
    ("Everything the Phase 2 report needs, in one JSON file plus the figures. Save this\n"
     "notebook's output as a dataset (or download the JSON) and place it at\n"
     "`<language>/reports/phase2_evaluation.json` in the repository, then run\n"
     "`scripts/make_phase2_report.py`.",
     "Everything the ablation comparison needs. Download these from the Output tab and\n"
     "drop them anywhere in the project folder:\n\n"
     "* `bonus_nopos_evaluation_<language>.json`\n"
     "* `training_history_<language>_nopos.json`\n"
     "* `figures/attention_heatmaps_<language>_nopos.png`\n"
     "* `figures/attention_summary_<language>_nopos.png`", 1),
    ("Download `phase2_evaluation_<lang>.json` and `training_history_<lang>.json` from\n"
     "the Output tab, put them in `<language>/reports/` (renaming the evaluation file\n"
     "to `phase2_evaluation.json`), copy the two figures into `report/figures/`, and\n"
     "run:\n\n```bash\npython scripts/make_phase2_report.py\n```",
     "The comparison against the standard model is generated from these files by\n"
     "`scripts/make_ablation_report.py`.", 1),
)


def ablate(notebook: dict) -> dict:
    """Turn the Phase 2 evaluation notebook into the ablation evaluation."""
    applied = [0] * len(ABLATION_EDITS)
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "EMBEDDED_SOURCE = (" in text:
            continue  # the embedded modules are the committed code, never edited here
        for index, (old, new, _) in enumerate(ABLATION_EDITS):
            if old in text:
                applied[index] += text.count(old)
                text = text.replace(old, new)
        cell["source"] = _lines(text)
    for (old, _, expected), count in zip(ABLATION_EDITS, applied):
        if count != expected:
            raise RuntimeError(f"ablation edit applied {count} time(s), expected {expected}: {old[:60]!r}")
    return notebook


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="store_true",
                        help="write the no-positional-embeddings bonus notebook instead")
    args = parser.parse_args()

    notebook = ablate(build()) if args.ablation else build()
    output = ABLATION_OUTPUT if args.ablation else OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1))

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

    size_kb = output.stat().st_size / 1024
    print(f"wrote {output.relative_to(REPO_ROOT)} ({len(notebook['cells'])} cells, {size_kb:.0f} KB)")
    print("all code cells parse" if not failures else f"{failures} cell(s) failed to parse")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
