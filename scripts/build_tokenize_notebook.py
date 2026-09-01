"""Generate the Kaggle notebook that turns the Phase 1 corpora into training binaries.

The notebook is generated rather than hand-edited so its code stays reviewable as
ordinary Python in this repository. Cell sources are stored the way nbformat
expects - a list of lines, each keeping its trailing newline - because writing
them as one joined string silently fuses statements together and produces a
notebook whose cells raise SyntaxError on the first run.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "notebooks" / "phase2_tokenize_corpora.ipynb"

DRIVE_FOLDER = "https://drive.google.com/drive/folders/16iUGbzj89eBr4R8P77irYxDB4lT_EAC2"

#: Tokenizers are embedded in the notebook rather than cloned. The GitHub
#: Classroom repository is private, so an unauthenticated `git clone` from Kaggle
#: prompts for credentials and fails; making the repository public to work around
#: that would be the wrong trade for a graded submission. The two models are about
#: 200 KB each and compress to under 100 KB, so carrying them inline costs little
#: and removes the dependency entirely.
TOKENIZERS = {
    "hi": REPO_ROOT / "hindi" / "tokenizer" / "hi_unigram_8000.model",
    "ne": REPO_ROOT / "nepali" / "tokenizer" / "ne_unigram_8000.model",
}


def _embedded_tokenizers() -> str:
    """Base64+gzip the tokenizer models for inlining into the notebook."""
    import base64
    import gzip

    parts = []
    for lang, path in TOKENIZERS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; the notebook cannot embed the {lang} tokenizer"
            )
        blob = base64.b64encode(gzip.compress(path.read_bytes(), 9)).decode()
        # Wrap so the notebook JSON stays readable rather than one enormous line.
        wrapped = "\n".join(
            '    "' + blob[i : i + 96] + '"' for i in range(0, len(blob), 96)
        )
        parts.append(f'"{lang}": (\n{wrapped}\n),')
    return "\n".join(parts)


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


def build() -> dict:
    cells: list[dict] = []

    cells.append(markdown(f"""
# Phase 1 -> Phase 2: tokenize the corpora into training binaries

Run this **once** on Kaggle with **CPU** (this step needs no GPU, so do not spend
GPU quota on it). It downloads the Phase 1 processed splits from Google Drive,
encodes them with the Phase 1 tokenizers, and writes compact `uint16` binaries
that training memory-maps.

Why binaries rather than training from JSONL: the vocabulary is 8,000, so a token
id fits in 16 bits. 11.8 GB of JSONL becomes about 2.4 GB, training does no JSON
parsing or tokenization in the data path, and the result can be saved as a Kaggle
Dataset that later sessions mount in seconds.

### Disk budget

`/kaggle/working` holds roughly **20 GB**. The full Drive folder is about 29 GB,
so this notebook downloads **only the processed splits** (11.8 GB) and deletes
each JSONL as soon as it is encoded, keeping peak usage near 12 GB.
`Phase1-Manual` and `Phase1-Sources` are Phase 1 provenance and are not needed
for training.
"""))

    cells.append(code("""
!pip install -q gdown sentencepiece
"""))

    cells.append(code("""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")


def disk() -> None:
    "Print how much of the Kaggle working disk is left."
    st = os.statvfs(WORK)
    free = st.f_bavail * st.f_frsize / 1e9
    used = (st.f_blocks - st.f_bavail) * st.f_frsize / 1e9
    print(f"disk: {used:.1f} GB used, {free:.1f} GB free")


disk()
"""))

    cells.append(markdown("""
## 0. Clear anything left by a previous attempt

`/kaggle/working` persists for the life of the session, so a run that failed
partway leaves its downloads behind and the next attempt starts with less room.
"""))

    cells.append(code("""
for name in ("corpora", "repo", "tokenized"):
    shutil.rmtree(WORK / name, ignore_errors=True)

disk()
"""))

    cells.append(markdown("""
## 1. Tokenizers

The two Phase 1 SentencePiece models are **embedded in this notebook**, so there
is nothing to download and no credentials to supply. The GitHub Classroom
repository is private, so cloning it from Kaggle would prompt for a username and
fail; the models are small enough (about 200 KB each, under 100 KB compressed)
that carrying them inline is simpler than any of the alternatives.

The SHA-256 of each decoded model is printed so it can be checked against the
Phase 1 artifacts if there is ever any doubt about which tokenizer was used.
"""))

    cells.append(code(f"""
import base64
import gzip
import hashlib

# Phase 1 SentencePiece models, gzipped and base64-encoded.
EMBEDDED_TOKENIZERS = {{
{_embedded_tokenizers()}
}}

TOK = WORK / "tokenizers"
TOK.mkdir(parents=True, exist_ok=True)

for lang, blob in EMBEDDED_TOKENIZERS.items():
    raw = gzip.decompress(base64.b64decode(blob))
    path = TOK / f"{{lang}}_unigram_8000.model"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()[:16]
    print(f"  {{path.name}}  {{len(raw) / 1024:.0f}} KB  sha256:{{digest}}")

models = sorted(TOK.glob("*_unigram_8000.model"))
assert len(models) == 2, f"expected 2 tokenizer models, found {{len(models)}}"
"""))

    cells.append(markdown(f"""
## 2. Download only the processed splits

Point `PROCESSED_FOLDER` at the **`Phase1-Processed` subfolder**, not the parent:
open that subfolder in Drive, then Share -> Copy link. Using the parent folder
pulls `Phase1-Manual` and `Phase1-Sources` as well, which overruns the disk.

Expected: `hi_train.jsonl` 5.13 GB, `ne_train.jsonl` 6.34 GB, plus four small
validation and test files.
"""))

    cells.append(code(f"""
PROCESSED_FOLDER = "{DRIVE_FOLDER}"

RAW = WORK / "corpora"
RAW.mkdir(parents=True, exist_ok=True)

subprocess.run(
    ["gdown", "--folder", PROCESSED_FOLDER, "-O", str(RAW), "--remaining-ok"],
    check=False,
)

for path in sorted(p for p in RAW.rglob("*.jsonl") if p.is_file()):
    print(f"{{path.stat().st_size / 1e9:7.2f}} GB  {{path.relative_to(RAW)}}")

disk()
"""))

    cells.append(markdown("""
If a large train file came down materially smaller than expected, Drive's
virus-scan interstitial truncated it. Fetch that one by its own file id: in
Drive, right-click the file -> Share -> Copy link, and take the id between `/d/`
and `/view`.
"""))

    cells.append(code("""
# FILE_ID = "paste the id here"
# subprocess.run(["gdown", FILE_ID, "-O", str(RAW / "hi_train.jsonl")], check=True)
"""))

    cells.append(markdown("""
## 3. Encode, freeing each split as it is consumed

Documents are concatenated into one stream with an EOS token between them, the
standard arrangement for causal language-model pretraining: document boundaries
become an ordinary token the model can learn, and training windows may start at
any offset without tracking document structure.

Each JSONL is deleted immediately after it is encoded, which is what keeps peak
disk usage near 12 GB rather than 14 GB.
"""))

    cells.append(code("""
import sentencepiece as spm

OUT = WORK / "tokenized"
OUT.mkdir(exist_ok=True)

CHUNK_DOCS = 20_000  # documents buffered between flushes


def locate(pattern: str, root: Path) -> Path:
    "Find the single file matching a glob, or fail loudly."
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return hits[0]


def encode_split(jsonl_path: Path, sp, out_path: Path) -> dict:
    "Encode one split into a flat uint16 array; return a manifest entry."
    eos = sp.eos_id()
    assert sp.get_piece_size() <= 65535, "uint16 requires a vocabulary under 65535"

    n_docs = 0
    n_tokens = 0
    buffer: list[int] = []
    started = time.time()

    with open(out_path, "wb") as out:
        with open(jsonl_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    text = json.loads(line).get("text", "")
                except Exception:
                    continue
                if not text:
                    continue

                ids = sp.encode(text)
                buffer.extend(ids)
                buffer.append(eos)
                n_docs += 1
                n_tokens += len(ids) + 1

                if n_docs % CHUNK_DOCS == 0:
                    np.asarray(buffer, dtype=np.uint16).tofile(out)
                    buffer.clear()
                    rate = n_tokens / max(time.time() - started, 1e-9)
                    print(
                        f"   {n_docs:>9,} docs  {n_tokens / 1e6:>8.1f}M tokens"
                        f"  {rate / 1e6:.2f}M tok/s",
                        flush=True,
                    )

        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(out)

    return {
        "documents": n_docs,
        "tokens": n_tokens,
        "bytes": out_path.stat().st_size,
        "seconds": round(time.time() - started, 1),
    }


manifest = {}

for lang in ("hi", "ne"):
    tokenizer_path = locate(f"{lang}_unigram_8000.model", TOK)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    manifest[lang] = {"vocab_size": sp.get_piece_size(), "splits": {}}

    for split in ("train", "validation", "test"):
        source = locate(f"{lang}_{split}.jsonl", RAW)
        print(f"\\n[{lang}/{split}]  {source.stat().st_size / 1e9:.2f} GB")

        entry = encode_split(source, sp, OUT / f"{lang}_{split}.bin")
        manifest[lang]["splits"][split] = entry
        print(f"   {entry['tokens']:,} tokens -> {entry['bytes'] / 1e9:.2f} GB")

        source.unlink()  # free the JSONL right away
        disk()

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("\\nwrote manifest.json")
"""))

    cells.append(markdown("""
## 4. Verify before trusting it

Two checks worth making now rather than three hours into a training run: that the
token counts line up with Phase 1, and that decoding a window from the middle of
each file returns readable text in the right language. A truncated download or a
mismatched tokenizer shows up immediately here.
"""))

    cells.append(code("""
EXPECTED = {"hi": 578_319_564, "ne": 610_989_716}  # Phase 1 reported train tokens

for lang, spec in manifest.items():
    got = spec["splits"]["train"]["tokens"]
    want = EXPECTED[lang]
    # The binary adds one EOS per document, so it should sit slightly above the
    # Phase 1 figure - a large gap in either direction means something is wrong.
    print(f"{lang}: {got:,} vs Phase 1 {want:,}  ({100 * (got - want) / want:+.2f}%)")

print()

for lang in ("hi", "ne"):
    sp = spm.SentencePieceProcessor(model_file=str(locate(f"{lang}_unigram_8000.model", TOK)))
    array = np.memmap(OUT / f"{lang}_train.bin", dtype=np.uint16, mode="r")
    print(f"[{lang}] {len(array):,} tokens on disk")
    print("   ", sp.decode([int(t) for t in array[10_000:10_060]])[:200])
    print()
"""))

    cells.append(markdown("""
## 5. Save as a Kaggle Dataset

**Do not skip this.** `/kaggle/working` is wiped when the session ends, so
without saving, the whole run is lost.

Save the notebook version, then open the **Output** tab and choose *Create
Dataset*. Name it `lma-phase2-tokenized`. Training sessions then mount it
read-only at `/kaggle/input/lma-phase2-tokenized/` in seconds.
"""))

    cells.append(code("""
shutil.rmtree(RAW, ignore_errors=True)
shutil.rmtree(TOK, ignore_errors=True)

total = 0
for path in sorted(OUT.iterdir()):
    total += path.stat().st_size
    print(f"{path.stat().st_size / 1e9:7.2f} GB  {path.name}")

print(f"\\n{total / 1e9:.2f} GB total")
print("Save this as a Kaggle Dataset named lma-phase2-tokenized")
disk()
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
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1))

    # Every code cell must be parseable Python, minus IPython shell escapes.
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

    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(notebook['cells'])} cells)")
    print("all code cells parse" if not failures else f"{failures} cell(s) failed to parse")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
