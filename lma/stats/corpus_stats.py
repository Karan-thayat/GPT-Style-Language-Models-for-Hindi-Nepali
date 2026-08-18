"""Corpus statistics for the Phase 1 dataset report.

Computes everything the deliverables ask for - size, sources, cleaning effect,
manual fraction - plus the distributional detail that makes the two resource
tiers comparable: document length quantiles, type/token ratio, and script
composition.

Counters are streaming and bounded: full length distributions are kept as
histograms and quantiles are estimated from a reservoir sample, so a 500M-token
corpus is summarized in constant memory.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

_RE_WORD = re.compile(r"[^\s]+")
#: Per-script classes; the statistics object picks one from its language.
_RE_SCRIPT = {
    "devanagari": re.compile(r"[ऀ-ॿ]"),
    "bengali": re.compile(r"[ঀ-৿]"),
}
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_SENTENCE = re.compile(r"[।!?\n]+")


@dataclass
class CorpusStatistics:
    """Accumulates corpus statistics over a stream of documents.

    Attributes:
        language: ISO code for reporting.
        reservoir_size: Number of document lengths retained for quantiles.
        max_vocab_tracked: Cap on distinct word types held in memory; beyond this
            the type/token ratio is reported as a lower bound.
    """

    language: str
    #: Writing system; defaults from the language when left as None.
    script: str | None = None
    reservoir_size: int = 200_000
    max_vocab_tracked: int = 3_000_000

    documents: int = 0
    characters: int = 0
    words: int = 0
    sentences: int = 0
    script_chars: int = 0
    latin_chars: int = 0
    digit_chars: int = 0
    whitespace_chars: int = 0
    punctuation_chars: int = 0

    documents_by_source: Counter = field(default_factory=Counter)
    characters_by_source: Counter = field(default_factory=Counter)
    words_by_source: Counter = field(default_factory=Counter)
    documents_by_collection: Counter = field(default_factory=Counter)
    characters_by_collection: Counter = field(default_factory=Counter)
    words_by_collection: Counter = field(default_factory=Counter)

    _word_types: set = field(default_factory=set, repr=False)
    _word_types_saturated: bool = False
    #: Word-type count carried over from a checkpoint; see load_state_dict for
    #: why the types themselves are not stored.
    _restored_word_types: int = 0
    _length_reservoir: list = field(default_factory=list, repr=False)
    _reservoir_seen: int = 0
    _char_counter: Counter = field(default_factory=Counter, repr=False)
    _rng_state: int = 0x243F6A8885A308D3

    def __post_init__(self) -> None:
        """Resolve the script for this language and cache its regex."""
        from ..data.normalize import LANGUAGE_SCRIPTS_DEFAULT

        self._script = self.script or LANGUAGE_SCRIPTS_DEFAULT.get(self.language, "devanagari")
        self._script_re = _RE_SCRIPT[self._script]

    def _next_random(self) -> int:
        """Advance a small deterministic PRNG for reservoir sampling."""
        x = self._rng_state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self._rng_state = x & 0xFFFFFFFFFFFFFFFF
        return self._rng_state

    def add(self, text: str, source_id: str = "unknown", collection: str = "unknown") -> None:
        """Fold one document into the running statistics."""
        words = _RE_WORD.findall(text)
        n_chars = len(text)
        n_words = len(words)

        self.documents += 1
        self.characters += n_chars
        self.words += n_words
        self.sentences += sum(1 for part in _RE_SENTENCE.split(text) if part.strip())

        self.script_chars += len(self._script_re.findall(text))
        self.latin_chars += len(_RE_LATIN.findall(text))
        for ch in text:
            if ch.isdigit():
                self.digit_chars += 1
            elif ch.isspace():
                self.whitespace_chars += 1
            elif unicodedata.category(ch).startswith("P"):
                self.punctuation_chars += 1

        self.documents_by_source[source_id] += 1
        self.characters_by_source[source_id] += n_chars
        self.words_by_source[source_id] += n_words
        self.documents_by_collection[collection] += 1
        self.characters_by_collection[collection] += n_chars
        self.words_by_collection[collection] += n_words

        if not self._word_types_saturated:
            self._word_types.update(words)
            if len(self._word_types) > self.max_vocab_tracked:
                self._word_types_saturated = True

        self._char_counter.update(ch for ch in text if not ch.isspace())

        # Reservoir sampling of document word-lengths for quantiles.
        self._reservoir_seen += 1
        if len(self._length_reservoir) < self.reservoir_size:
            self._length_reservoir.append(n_words)
        else:
            index = self._next_random() % self._reservoir_seen
            if index < self.reservoir_size:
                self._length_reservoir[index] = n_words

    def state_dict(self) -> dict:
        """Return every accumulator for checkpointing.

        The reservoir and its seen-count are included together: restoring one
        without the other would bias the sampling probability for the remainder of
        the run, skewing the document-length quantiles.
        """
        return {
            "documents": self.documents,
            "characters": self.characters,
            "words": self.words,
            "sentences": self.sentences,
            "script_chars": self.script_chars,
            "latin_chars": self.latin_chars,
            "digit_chars": self.digit_chars,
            "whitespace_chars": self.whitespace_chars,
            "punctuation_chars": self.punctuation_chars,
            "documents_by_source": dict(self.documents_by_source),
            "characters_by_source": dict(self.characters_by_source),
            "words_by_source": dict(self.words_by_source),
            "documents_by_collection": dict(self.documents_by_collection),
            "characters_by_collection": dict(self.characters_by_collection),
            "words_by_collection": dict(self.words_by_collection),
            "word_types_count": len(self._word_types),
            "word_types_saturated": self._word_types_saturated,
            "length_reservoir": list(self._length_reservoir),
            "reservoir_seen": self._reservoir_seen,
            "char_counter": dict(self._char_counter),
            "rng_state": self._rng_state,
        }

    def load_state_dict(self, payload: dict) -> None:
        """Restore accumulators from a checkpoint.

        Note the one lossy field: the set of distinct word *types* is not stored,
        because at corpus scale it holds millions of strings and would dominate the
        checkpoint. Only its cardinality is restored, so type/token ratio after a
        resume is a lower bound - which :meth:`as_dict` already reports through the
        ``is_lower_bound`` flag.
        """
        self.documents = payload.get("documents", 0)
        self.characters = payload.get("characters", 0)
        self.words = payload.get("words", 0)
        self.sentences = payload.get("sentences", 0)
        # Accept the old key too, so a checkpoint written before the field was
        # renamed for Devanagari support still resumes rather than restarting.
        self.script_chars = payload.get("script_chars", payload.get("bengali_chars", 0))
        self.latin_chars = payload.get("latin_chars", 0)
        self.digit_chars = payload.get("digit_chars", 0)
        self.whitespace_chars = payload.get("whitespace_chars", 0)
        self.punctuation_chars = payload.get("punctuation_chars", 0)

        self.documents_by_source = Counter(payload.get("documents_by_source", {}))
        self.characters_by_source = Counter(payload.get("characters_by_source", {}))
        self.words_by_source = Counter(payload.get("words_by_source", {}))
        self.documents_by_collection = Counter(payload.get("documents_by_collection", {}))
        self.characters_by_collection = Counter(payload.get("characters_by_collection", {}))
        self.words_by_collection = Counter(payload.get("words_by_collection", {}))

        self._restored_word_types = payload.get("word_types_count", 0)
        self._word_types_saturated = payload.get("word_types_saturated", False) or self._restored_word_types > 0
        self._length_reservoir = list(payload.get("length_reservoir", []))
        self._reservoir_seen = payload.get("reservoir_seen", 0)
        self._char_counter = Counter(payload.get("char_counter", {}))
        self._rng_state = payload.get("rng_state", self._rng_state)

    @staticmethod
    def _quantile(sorted_values: list[int], q: float) -> float:
        """Linear-interpolated quantile of a pre-sorted list."""
        if not sorted_values:
            return 0.0
        position = q * (len(sorted_values) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return float(sorted_values[low])
        return sorted_values[low] + (position - low) * (sorted_values[high] - sorted_values[low])

    def as_dict(self, top_chars: int = 40) -> dict:
        """Return a JSON-serializable summary of every accumulated statistic."""
        lengths = sorted(self._length_reservoir)
        non_space_chars = max(1, self.characters - self.whitespace_chars)
        manual_words = self.words_by_collection.get("manual", 0)

        return {
            "language": self.language,
            "totals": {
                "documents": self.documents,
                "characters": self.characters,
                "words": self.words,
                "sentences": self.sentences,
                "bytes_utf8_estimate": self.characters * 3,
            },
            "averages": {
                "characters_per_document": self.characters / self.documents if self.documents else 0.0,
                "words_per_document": self.words / self.documents if self.documents else 0.0,
                "characters_per_word": self.characters / self.words if self.words else 0.0,
                "words_per_sentence": self.words / self.sentences if self.sentences else 0.0,
            },
            "document_length_words": {
                "min": lengths[0] if lengths else 0,
                "p10": self._quantile(lengths, 0.10),
                "p25": self._quantile(lengths, 0.25),
                "median": self._quantile(lengths, 0.50),
                "p75": self._quantile(lengths, 0.75),
                "p90": self._quantile(lengths, 0.90),
                "p99": self._quantile(lengths, 0.99),
                "max": lengths[-1] if lengths else 0,
                "sampled_documents": len(lengths),
            },
            "script_composition": {
                "script_char_ratio": self.script_chars / non_space_chars,
                "script": self._script,
                "latin_char_ratio": self.latin_chars / non_space_chars,
                "digit_char_ratio": self.digit_chars / non_space_chars,
                "punctuation_char_ratio": self.punctuation_chars / non_space_chars,
            },
            "lexical_diversity": {
                # Types seen in this segment plus the count carried over from a
                # checkpoint. Because the two sets may overlap, this is an upper
                # bound on a resumed run and is flagged as approximate below.
                "word_types": len(self._word_types) + self._restored_word_types,
                "type_token_ratio": (
                    (len(self._word_types) + self._restored_word_types) / self.words if self.words else 0.0
                ),
                "is_lower_bound": self._word_types_saturated,
                "resumed_from_checkpoint": self._restored_word_types > 0,
            },
            "collection_split": {
                "documents": dict(self.documents_by_collection),
                "words": dict(self.words_by_collection),
                "characters": dict(self.characters_by_collection),
                "manual_word_fraction": (manual_words / self.words) if self.words else 0.0,
            },
            "by_source": {
                "documents": dict(self.documents_by_source.most_common()),
                "words": dict(self.words_by_source.most_common()),
                "characters": dict(self.characters_by_source.most_common()),
            },
            "top_characters": [
                {"char": ch, "codepoint": f"U+{ord(ch):04X}", "name": _safe_char_name(ch), "count": count}
                for ch, count in self._char_counter.most_common(top_chars)
            ],
            "distinct_characters": len(self._char_counter),
        }


def _safe_char_name(ch: str) -> str:
    """Unicode name of a character, or a placeholder when it has none."""
    try:
        return unicodedata.name(ch)
    except ValueError:
        return "<unnamed>"


def word_frequency(texts: Iterable[str], top_k: int = 100) -> list[tuple[str, int]]:
    """Return the `top_k` most frequent whitespace-delimited words."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(_RE_WORD.findall(text))
    return counter.most_common(top_k)


def compute_split_statistics(
    split_files: dict[str, Path],
    language: str,
    text_field: str = "text",
) -> dict:
    """Compute statistics for every split of one language.

    Args:
        split_files: Mapping of split name to its JSONL file.
        language: ISO code.
        text_field: JSON field holding text.

    Returns:
        ``{split_name: statistics_dict}`` plus a combined ``"all"`` entry.
    """
    from ..data.splits import iter_jsonl

    results: dict[str, dict] = {}
    combined = CorpusStatistics(language=language)

    for split_name, path in split_files.items():
        path = Path(path)
        if not path.exists():
            continue
        stats = CorpusStatistics(language=language)
        for row in iter_jsonl(path, text_field):
            text = row[text_field]
            source = row.get("source_id", "unknown")
            collection = row.get("collection", "unknown")
            stats.add(text, source, collection)
            combined.add(text, source, collection)
        results[split_name] = stats.as_dict()

    results["all"] = combined.as_dict()
    return results


def iter_split_texts(path: Path, text_field: str = "text", limit: int | None = None) -> Iterator[str]:
    """Yield up to `limit` document texts from a split JSONL file."""
    from ..data.splits import iter_jsonl

    for index, row in enumerate(iter_jsonl(Path(path), text_field)):
        if limit is not None and index >= limit:
            return
        yield row[text_field]
