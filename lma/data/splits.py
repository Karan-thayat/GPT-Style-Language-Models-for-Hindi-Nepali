"""Deterministic train/validation/test splitting.

Design decisions
----------------
**Hash-based assignment, not shuffling.** Each document's split is a function of
a hash of its content and a seed. This makes the split reproducible without
holding the corpus in memory, and stable under corpus growth: adding documents
never moves an existing document from train to test, which is what keeps the
Phase 2/3 evaluation numbers comparable across corpus revisions.

**Source-aware assignment.** The hash is salted per source so a source that
happens to dominate the corpus cannot skew one split. Optionally, whole sources
can be pinned to a split.

**Leakage control.** Documents are assigned by content hash, so exact duplicates
that survive cleaning land in the *same* split rather than straddling train and
test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, TextIO

SPLIT_NAMES = ("train", "validation", "test")


@dataclass
class SplitConfig:
    """Split ratios and assignment policy.

    Attributes:
        train: Fraction assigned to training.
        validation: Fraction assigned to validation.
        test: Fraction assigned to test.
        seed: Hash salt; changing it produces a different but equally
            reproducible partition.
        pinned_sources: Optional ``{source_id: split}`` overrides, for holding a
            whole source out (e.g. keeping one OCR'd book entirely in test).
    """

    train: float = 0.98
    validation: float = 0.01
    test: float = 0.01
    seed: int = 20260812
    pinned_sources: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that ratios are positive and sum to 1."""
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if min(self.train, self.validation, self.test) < 0:
            raise ValueError("Split ratios must be non-negative")
        for source, split in self.pinned_sources.items():
            if split not in SPLIT_NAMES:
                raise ValueError(f"Pinned source {source!r} has invalid split {split!r}")


class SplitAssigner:
    """Assigns documents to splits by salted content hash."""

    def __init__(self, config: SplitConfig) -> None:
        """Precompute the cumulative split boundaries."""
        self.config = config
        self._train_edge = config.train
        self._val_edge = config.train + config.validation

    def assign(self, text: str, source_id: str = "") -> str:
        """Return the split name for one document.

        Args:
            text: Document text; only its hash is used.
            source_id: Source identifier, used for salting and for pinning.

        Returns:
            One of ``"train"``, ``"validation"``, ``"test"``.
        """
        pinned = self.config.pinned_sources.get(source_id)
        if pinned is not None:
            return pinned

        payload = f"{self.config.seed}:{source_id}:{' '.join(text.split())}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        position = int.from_bytes(digest, "big") / float(1 << 64)

        if position < self._train_edge:
            return "train"
        if position < self._val_edge:
            return "validation"
        return "test"


class SplitWriter:
    """Writes documents into one JSONL file per split, tracking counts.

    Intended for use as a context manager so file handles always close::

        with SplitWriter(Path("data/processed"), "bn") as writer:
            writer.write("train", text, "wikipedia", "downloaded")
    """

    def __init__(self, output_dir: Path, language: str, append: bool = False) -> None:
        """Prepare the output directory and per-split counters.

        Args:
            output_dir: Directory to write the per-split JSONL files into.
            language: ISO code, used in the output filenames.
            append: Open the split files for appending instead of truncating, and
                start the counters from `restore_counts`. Required when resuming a
                checkpointed run - opening with ``"w"`` would silently discard
                every document written before the interruption.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.language = language
        self.append = append
        self._handles: dict[str, TextIO] = {}
        self.counts: dict[str, int] = {name: 0 for name in SPLIT_NAMES}
        self.chars: dict[str, int] = {name: 0 for name in SPLIT_NAMES}
        self.by_collection: dict[str, dict[str, int]] = {name: {} for name in SPLIT_NAMES}
        self.by_source: dict[str, dict[str, int]] = {name: {} for name in SPLIT_NAMES}

    def restore_counts(self, summary: dict) -> None:
        """Seed the counters from a previous run's :meth:`summary` output.

        Called when resuming so the final statistics cover the whole corpus rather
        than only the documents written since the last checkpoint.
        """
        self.counts.update({k: v for k, v in summary.get("documents", {}).items() if k in self.counts})
        self.chars.update({k: v for k, v in summary.get("characters", {}).items() if k in self.chars})
        for split, payload in summary.get("characters_by_collection", {}).items():
            if split in self.by_collection:
                self.by_collection[split] = dict(payload)
        for split, payload in summary.get("characters_by_source", {}).items():
            if split in self.by_source:
                self.by_source[split] = dict(payload)

    def __enter__(self) -> "SplitWriter":
        """Open one output file per split, truncating unless appending."""
        mode = "a" if self.append else "w"
        for name in SPLIT_NAMES:
            path = self.output_dir / f"{self.language}_{name}.jsonl"
            self._handles[name] = path.open(mode, encoding="utf-8")
        return self

    def __exit__(self, *exc_info) -> None:
        """Flush and close all output files."""
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self._handles.clear()

    def write(self, split: str, text: str, source_id: str, collection: str) -> None:
        """Append one document to a split and update the counters."""
        handle = self._handles[split]
        handle.write(json.dumps({"text": text, "source_id": source_id, "collection": collection}, ensure_ascii=False))
        handle.write("\n")
        self.counts[split] += 1
        self.chars[split] += len(text)
        self.by_collection[split][collection] = self.by_collection[split].get(collection, 0) + len(text)
        self.by_source[split][source_id] = self.by_source[split].get(source_id, 0) + len(text)

    def flush(self) -> None:
        """Force buffered documents to disk, so a checkpoint matches the files.

        A checkpoint records how many documents have been written. If the OS still
        holds some in a buffer when the process dies, the files would be shorter
        than the checkpoint claims and the resumed run would double-count.
        """
        for handle in self._handles.values():
            handle.flush()

    def summary(self) -> dict:
        """Return per-split document counts, characters, and provenance breakdowns."""
        return {
            "language": self.language,
            "documents": dict(self.counts),
            "characters": dict(self.chars),
            "characters_by_collection": {k: dict(v) for k, v in self.by_collection.items()},
            "characters_by_source": {k: dict(v) for k, v in self.by_source.items()},
            "files": {name: str(self.output_dir / f"{self.language}_{name}.jsonl") for name in SPLIT_NAMES},
        }


def iter_jsonl(path: Path, text_field: str = "text") -> Iterator[dict]:
    """Yield parsed rows from a JSONL file, skipping malformed lines."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if text_field in row:
                yield row


def iter_texts(paths: Iterable[Path], text_field: str = "text") -> Iterator[str]:
    """Yield just the text field from a sequence of JSONL files."""
    for path in paths:
        for row in iter_jsonl(Path(path), text_field):
            yield row[text_field]
