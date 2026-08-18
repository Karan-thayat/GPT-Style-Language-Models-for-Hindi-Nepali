"""Checkpoint and resume support for long corpus ingests.

Why this exists
---------------
A ~500M-token ingest runs for many hours. Colab sessions disconnect, laptops
sleep, and processes get killed. Without resume support, every interruption
restarts from zero, which in practice means the run never finishes.

What has to be saved for a resume to be *correct*
-------------------------------------------------
Continuing is easy; continuing without corrupting the corpus is the hard part.
Four pieces of state matter, and omitting any one produces a subtly wrong corpus
rather than an obvious failure:

1. **Per-source position** - how many documents each source already yielded. On
   resume the source is re-streamed and this many documents are skipped. Sources
   must therefore be deterministically ordered, which Hugging Face streaming and
   the sorted local/scrape sources both are.
2. **Deduplication state** - the exact-hash set and the MinHash/LSH index. Drop
   these and documents already in the corpus stop being recognized as duplicates,
   so the resumed portion silently accumulates repeats of earlier content.
3. **Split counters and provenance** - so the final statistics describe the whole
   corpus rather than only the last segment.
4. **Written-document counts** - to verify the output files agree with the
   checkpoint. If a crash lost buffered writes, the files are shorter than the
   checkpoint claims; resuming blindly would leave a gap.

Format
------
The dedup state dominates the checkpoint size (one 96-slot signature per kept
document), so it is stored as a compressed pickle, while the small bookkeeping
state is JSON for inspectability. Writes go to a temporary file and are then
atomically renamed, because a checkpoint half-written during a kill is worse than
no checkpoint at all.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Bumped when the checkpoint layout changes incompatibly.
CHECKPOINT_VERSION = 1


@dataclass
class IngestState:
    """Everything needed to resume a corpus build exactly where it stopped.

    Attributes:
        language: ISO code, checked on resume to prevent cross-language mix-ups.
        profile: Profile name, checked for the same reason.
        source_positions: ``{source_id: documents_already_yielded}``.
        completed_sources: Sources fully exhausted, skipped entirely on resume.
        seen: Documents examined, for the cleaning statistics.
        kept: Documents that passed every filter.
        rejected: Rejection counts by reason.
        kept_chars: Characters in kept documents.
        kept_words: Words in kept documents.
        train_chars: Characters written to the train split, for the token budget.
        split_summary: Last :meth:`SplitWriter.summary` output.
        corpus_stats_state: Pickled :class:`CorpusStatistics` internals.
        elapsed_seconds: Cumulative wall time across all segments.
        segments: How many times this run has been started or resumed.
    """

    language: str
    profile: str
    source_positions: dict[str, int] = field(default_factory=dict)
    completed_sources: list[str] = field(default_factory=list)
    seen: int = 0
    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    kept_chars: int = 0
    kept_words: int = 0
    train_chars: int = 0
    #: Characters kept per collection ("manual" / "downloaded"), so a resumed run
    #: restores the budget split rather than restarting the ratio from zero.
    chars_by_collection: dict = field(default_factory=dict)
    split_summary: dict = field(default_factory=dict)
    corpus_stats_state: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    segments: int = 0

    def documents_written(self) -> int:
        """Total documents recorded as written across all splits."""
        return sum(self.split_summary.get("documents", {}).values())


class CheckpointManager:
    """Saves and loads ingest checkpoints atomically.

    Args:
        checkpoint_dir: Directory to hold the checkpoint files. On Colab this
            should live on mounted Drive so it survives the VM, not in /content.
        language: ISO code for the run being checkpointed.
        keep_backup: Retain the previous checkpoint as ``*.bak``, so a corrupted
            save still leaves one recoverable generation.
    """

    def __init__(self, checkpoint_dir: Path, language: str, keep_backup: bool = True) -> None:
        """Resolve the checkpoint paths and ensure the directory exists."""
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.language = language
        self.keep_backup = keep_backup
        self.state_path = self.dir / f"{language}_ingest_state.json"
        self.dedup_path = self.dir / f"{language}_dedup_state.pkl.gz"

    def exists(self) -> bool:
        """True when a resumable checkpoint is present."""
        return self.state_path.exists()

    # --- saving ---------------------------------------------------------------

    def save(self, state: IngestState, dedup_payload: dict | None = None) -> None:
        """Write a checkpoint atomically.

        The dedup state is written first: if the process dies between the two
        writes, the bookkeeping state is the one that decides whether a checkpoint
        is usable, so it must never point at a dedup file that was not finished.

        Args:
            state: Bookkeeping state to persist.
            dedup_payload: Serialized deduplicator state from
                :meth:`lma.data.clean.DocumentCleaner.dedup_state`.
        """
        if dedup_payload is not None:
            self._atomic_write_pickle(self.dedup_path, dedup_payload)

        payload = {
            "version": CHECKPOINT_VERSION,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "language": state.language,
            "profile": state.profile,
            "source_positions": state.source_positions,
            "completed_sources": state.completed_sources,
            "seen": state.seen,
            "kept": state.kept,
            "rejected": state.rejected,
            "kept_chars": state.kept_chars,
            "kept_words": state.kept_words,
            "train_chars": state.train_chars,
            "split_summary": state.split_summary,
            "corpus_stats_state": state.corpus_stats_state,
            "elapsed_seconds": state.elapsed_seconds,
            "segments": state.segments,
            "has_dedup_state": self.dedup_path.exists(),
        }
        self._atomic_write_json(self.state_path, payload)

    def _atomic_write_json(self, path: Path, payload: dict) -> None:
        """Write JSON to a temp file, fsync, then rename over the target."""
        if self.keep_backup and path.exists():
            path.replace(path.with_suffix(path.suffix + ".bak"))
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)

    def _atomic_write_pickle(self, path: Path, payload: dict) -> None:
        """Write a gzipped pickle to a temp file, then rename over the target."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    # --- loading --------------------------------------------------------------

    def load(self, expected_profile: str | None = None) -> tuple[IngestState, dict | None] | None:
        """Load a checkpoint, or return None when there is nothing usable.

        Args:
            expected_profile: When given, refuse a checkpoint written under a
                different profile. Resuming a `sampled` checkpoint into a `full`
                run would apply the wrong caps and split ratios to half the corpus.

        Returns:
            ``(state, dedup_payload)``, or None if no valid checkpoint exists.

        Raises:
            ValueError: On a version or language mismatch, which indicates the
                checkpoint belongs to a different run and must not be used.
        """
        if not self.state_path.exists():
            return None

        with self.state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {payload.get('version')} does not match "
                f"{CHECKPOINT_VERSION}; delete {self.state_path} to start fresh."
            )
        if payload.get("language") != self.language:
            raise ValueError(
                f"Checkpoint is for language {payload.get('language')!r}, not {self.language!r}."
            )
        if expected_profile is not None and payload.get("profile") != expected_profile:
            raise ValueError(
                f"Checkpoint was written under profile {payload.get('profile')!r} but this run "
                f"uses {expected_profile!r}. Resuming across profiles would apply different caps "
                f"and split ratios to different parts of one corpus. Delete {self.state_path} "
                f"to start this profile from scratch."
            )

        state = IngestState(
            language=payload["language"],
            profile=payload["profile"],
            source_positions=payload.get("source_positions", {}),
            completed_sources=payload.get("completed_sources", []),
            seen=payload.get("seen", 0),
            kept=payload.get("kept", 0),
            rejected=payload.get("rejected", {}),
            kept_chars=payload.get("kept_chars", 0),
            kept_words=payload.get("kept_words", 0),
            train_chars=payload.get("train_chars", 0),
            split_summary=payload.get("split_summary", {}),
            corpus_stats_state=payload.get("corpus_stats_state", {}),
            elapsed_seconds=payload.get("elapsed_seconds", 0.0),
            segments=payload.get("segments", 0),
        )

        dedup_payload = None
        if payload.get("has_dedup_state") and self.dedup_path.exists():
            with gzip.open(self.dedup_path, "rb") as handle:
                dedup_payload = pickle.load(handle)

        return state, dedup_payload

    def clear(self) -> None:
        """Delete the checkpoint, marking the run complete.

        Called on successful completion so a later re-run starts fresh instead of
        resuming from the end of a finished corpus.
        """
        for path in (self.state_path, self.dedup_path,
                     self.state_path.with_suffix(self.state_path.suffix + ".bak")):
            path.unlink(missing_ok=True)


def truncate_jsonl_to(path: Path, keep_lines: int) -> int:
    """Trim a JSONL file to its first `keep_lines` complete lines.

    This reconciles the output files with the checkpoint. A process killed
    mid-write can leave more lines than the last checkpoint recorded, or a
    truncated final line; either would corrupt the resumed corpus. Rewriting to
    exactly the checkpointed line count makes the resume idempotent.

    Args:
        path: JSONL file to trim.
        keep_lines: Number of leading lines to retain.

    Returns:
        The number of lines the file has after trimming.
    """
    path = Path(path)
    if not path.exists():
        return 0

    tmp = path.with_suffix(path.suffix + ".trim")
    kept = 0
    with path.open("r", encoding="utf-8") as source, tmp.open("w", encoding="utf-8") as target:
        for line in source:
            if kept >= keep_lines:
                break
            # A final line without a newline was interrupted mid-write; drop it.
            if not line.endswith("\n"):
                break
            target.write(line)
            kept += 1
    tmp.replace(path)
    return kept
