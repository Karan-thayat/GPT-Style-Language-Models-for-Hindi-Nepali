"""Document-level quality filtering and deduplication.

The cleaner is a pipeline of independent, individually-countable filters. Every
rejection is attributed to the first filter that fired, which is what makes the
dataset statistics report meaningful: it shows *why* documents were lost, not
just how many.

Filters, in order of application
-------------------------------
1. Length - too short to contain usable language modelling signal.
2. Script ratio - the document must be predominantly Bengali-script, which
   removes off-script pages, navigation chrome, and code blocks. Measured over
   letters only, so digit-heavy pages are attributed to the digit filter instead.
   The script is chosen per language: Devanagari for Hindi, Bengali for Assamese.
3. Replacement/mojibake - a high rate of U+FFFD indicates broken decoding.
4. Repetition - boilerplate and generated spam repeat lines or n-grams.
5. Symbol and digit ratio - tables, price lists, and link farms.
6. Language identity - see :mod:`lma.data.language_id`. Hindi and Assamese use
   different scripts so the check is decisive; Bengali vs Assamese needs
   orthographic evidence.
7. Exact duplicates - SHA-256 over normalized text.
8. Near duplicates - MinHash + LSH over character 5-gram shingles.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict

from .language_id import IndicLanguageIdentifier
from .normalize import LANGUAGE_SCRIPTS_DEFAULT

#: Per-script character classes, selected by the cleaner from its config.
_RE_SCRIPT = {
    "devanagari": re.compile(r"[ऀ-ॿ]"),
    "bengali": re.compile(r"[ঀ-৿]"),
}
_RE_WORD = re.compile(r"\S+")
#: Letters only (no digits) - see the script-ratio filter for why that matters.
_RE_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass
class CleanerConfig:
    """Thresholds for :class:`DocumentCleaner`.

    Defaults are tuned for web/Wikipedia-style Indic text: permissive enough to
    keep ordinary prose, strict enough to drop chrome and machine-generated
    pages. Every value is written to the run manifest so results stay reproducible.
    """

    language: str = "hi"
    #: Writing system, which selects the script-ratio character class. Defaults
    #: from the language when left as None.
    script: str | None = None
    min_chars: int = 200
    min_words: int = 25
    max_chars: int = 500_000
    #: Minimum fraction of *letters* (digits excluded) that are Bengali-script.
    min_script_ratio: float = 0.70
    max_replacement_ratio: float = 0.002
    #: Maximum fraction of lines that are exact duplicates of another line.
    max_duplicate_line_ratio: float = 0.30
    #: Maximum share of the document taken up by its most frequent 10-gram.
    max_top_ngram_ratio: float = 0.20
    max_digit_ratio: float = 0.20
    max_symbol_ratio: float = 0.10
    #: Longest run of one character allowed (catches "aaaaaa..." padding).
    max_char_run: int = 20
    check_language_identity: bool = True
    dedup_exact: bool = True
    dedup_near: bool = True
    #: Jaccard similarity at which documents are treated as duplicates. Measured
    #: on real Bengali prose, replacing one sentence of eight in an article gives
    #: a true Jaccard of ~0.79 over character 5-gram shingles - so a 0.80 cutoff
    #: lets ordinary syndicated-news near-duplicates through. 0.70 catches those
    #: while still keeping documents that merely share a topic (measured well
    #: below 0.4).
    near_dup_threshold: float = 0.70
    #: Signature length. The standard error of a MinHash estimate is
    #: ~sqrt(J(1-J)/permutations), about 0.04 at 96 permutations and J=0.8, so
    #: 96 is adequate given the threshold has 0.09 of headroom above the case
    #: it must catch. Raising this narrows the grey zone at a linear cost.
    minhash_permutations: int = 96
    shingle_size: int = 5


@dataclass
class CleanerStats:
    """Counters describing one cleaning run."""

    seen: int = 0
    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    kept_chars: int = 0
    kept_words: int = 0

    def reject(self, reason: str) -> None:
        """Record a rejection attributed to `reason`."""
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def as_dict(self) -> dict:
        """Return a JSON-serializable view including the derived keep rate."""
        payload = asdict(self)
        payload["keep_rate"] = (self.kept / self.seen) if self.seen else 0.0
        payload["rejected"] = dict(sorted(self.rejected.items(), key=lambda kv: -kv[1]))
        return payload


class MinHashDeduplicator:
    """Near-duplicate detector using MinHash signatures with LSH banding.

    Character n-gram shingles are hashed into `permutations` independent minima;
    two documents whose signatures agree in any full band become candidates, and
    the estimated Jaccard similarity of their signatures decides the verdict.
    This is O(1) per document in the number of documents already seen, which
    matters because the corpus will not fit in memory as a pairwise comparison.

    Args:
        permutations: Signature length. More is more accurate and slower.
        threshold: Estimated Jaccard similarity at or above which a document is
            considered a duplicate.
        shingle_size: Character n-gram width.
    """

    _MERSENNE = (1 << 61) - 1

    def __init__(self, permutations: int = 96, threshold: float = 0.80, shingle_size: int = 5) -> None:
        """Build the hash coefficients and choose an LSH banding for `threshold`."""
        self.permutations = permutations
        self.threshold = threshold
        self.shingle_size = shingle_size
        rng = _DeterministicRandom(seed=0xC0FFEE)
        self._coeff_a = [rng.randint(1, self._MERSENNE - 1) for _ in range(permutations)]
        self._coeff_b = [rng.randint(0, self._MERSENNE - 1) for _ in range(permutations)]
        self._bands, self._rows = self._choose_banding(permutations, threshold)
        self._buckets: list[dict[int, list[int]]] = [{} for _ in range(self._bands)]
        self._signatures: list[tuple[int, ...]] = []

    @staticmethod
    def _choose_banding(permutations: int, threshold: float) -> tuple[int, int]:
        """Pick (bands, rows) whose LSH S-curve inflects closest to `threshold`.

        The probability that a pair with similarity s shares a band is
        ``1 - (1 - s**rows)**bands``; we search integer factorizations of
        `permutations` for the one whose 50% point sits nearest `threshold`.
        """
        best = (1, permutations)
        best_error = float("inf")
        for rows in range(1, permutations + 1):
            if permutations % rows:
                continue
            bands = permutations // rows
            inflection = (1.0 / bands) ** (1.0 / rows)
            error = abs(inflection - threshold)
            if error < best_error:
                best_error, best = error, (bands, rows)
        return best

    def _shingles(self, text: str) -> set[int]:
        """Hash the document's character n-grams into a set of 64-bit integers."""
        compact = " ".join(text.split())
        if len(compact) <= self.shingle_size:
            return {hash(compact) & 0xFFFFFFFFFFFFFFFF} if compact else set()
        size = self.shingle_size
        return {
            int.from_bytes(hashlib.blake2b(compact[i : i + size].encode("utf-8"), digest_size=8).digest(), "big")
            for i in range(len(compact) - size + 1)
        }

    def _signature(self, shingles: set[int]) -> tuple[int, ...]:
        """Compute the MinHash signature of a shingle set."""
        if not shingles:
            return tuple([0] * self.permutations)
        prime = self._MERSENNE
        signature = []
        for a, b in zip(self._coeff_a, self._coeff_b):
            signature.append(min(((a * s + b) % prime) for s in shingles))
        return tuple(signature)

    @staticmethod
    def _similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        """Estimated Jaccard similarity: the fraction of agreeing signature slots."""
        matches = sum(1 for x, y in zip(left, right) if x == y)
        return matches / len(left)

    def is_duplicate(self, text: str, add: bool = True) -> bool:
        """Test `text` against everything seen so far.

        Args:
            text: Document text.
            add: When True and the document is unique, index it so later
                documents are compared against it.

        Returns:
            True when a near-duplicate has already been seen.
        """
        signature = self._signature(self._shingles(text))
        candidates: set[int] = set()
        band_keys = []
        for band in range(self._bands):
            chunk = signature[band * self._rows : (band + 1) * self._rows]
            key = int.from_bytes(hashlib.blake2b(repr(chunk).encode(), digest_size=8).digest(), "big")
            band_keys.append(key)
            candidates.update(self._buckets[band].get(key, ()))

        for index in candidates:
            if self._similarity(signature, self._signatures[index]) >= self.threshold:
                return True

        if add:
            doc_id = len(self._signatures)
            self._signatures.append(signature)
            for band, key in enumerate(band_keys):
                self._buckets[band].setdefault(key, []).append(doc_id)
        return False

    def state_dict(self) -> dict:
        """Return the index contents for checkpointing.

        Only the signatures are stored; the LSH buckets are rebuilt from them on
        load. The buckets are pure derived data - roughly as large as the
        signatures themselves - so recomputing costs a few seconds and halves the
        checkpoint size.
        """
        return {
            "permutations": self.permutations,
            "threshold": self.threshold,
            "shingle_size": self.shingle_size,
            "signatures": self._signatures,
        }

    def load_state_dict(self, payload: dict) -> None:
        """Restore a checkpointed index and rebuild the LSH buckets.

        Raises:
            ValueError: If the saved parameters differ from this instance's. A
                signature computed under different hash coefficients or a different
                shingle size is meaningless here, so silently accepting it would
                make deduplication stop working while appearing to succeed.
        """
        for field_name in ("permutations", "shingle_size"):
            saved = payload.get(field_name)
            current = getattr(self, field_name)
            if saved != current:
                raise ValueError(
                    f"Checkpointed deduplicator has {field_name}={saved} but this run uses "
                    f"{current}; the saved signatures are not comparable."
                )

        self._signatures = list(payload["signatures"])
        self._buckets = [{} for _ in range(self._bands)]
        for doc_id, signature in enumerate(self._signatures):
            for band in range(self._bands):
                chunk = signature[band * self._rows : (band + 1) * self._rows]
                key = int.from_bytes(hashlib.blake2b(repr(chunk).encode(), digest_size=8).digest(), "big")
                self._buckets[band].setdefault(key, []).append(doc_id)


class _DeterministicRandom:
    """Tiny xorshift PRNG so dedup hashing is reproducible without global seeding."""

    def __init__(self, seed: int) -> None:
        """Initialize the generator state."""
        self._state = seed & 0xFFFFFFFFFFFFFFFF or 0x9E3779B97F4A7C15

    def _next(self) -> int:
        """Advance the state and return a 64-bit value."""
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self._state = x & 0xFFFFFFFFFFFFFFFF
        return self._state

    def randint(self, low: int, high: int) -> int:
        """Return an integer in [low, high]."""
        return low + self._next() % (high - low + 1)


class DocumentCleaner:
    """Applies quality filters and deduplication to a stream of documents.

    Usage::

        cleaner = DocumentCleaner(CleanerConfig(language="as"))
        for doc in raw_documents:
            kept = cleaner.process(doc)
            if kept is not None:
                write(kept)
        print(cleaner.stats.as_dict())
    """

    def __init__(self, config: CleanerConfig) -> None:
        """Set up filters, the exact-hash set, and the near-duplicate index."""
        self.config = config
        self.stats = CleanerStats()
        self._seen_hashes: set[bytes] = set()
        self._script = config.script or LANGUAGE_SCRIPTS_DEFAULT.get(config.language, "devanagari")
        self._script_re = _RE_SCRIPT[self._script]
        self._identifier = IndicLanguageIdentifier() if config.check_language_identity else None
        self._near_dup = (
            MinHashDeduplicator(
                permutations=config.minhash_permutations,
                threshold=config.near_dup_threshold,
                shingle_size=config.shingle_size,
            )
            if config.dedup_near
            else None
        )

    # --- individual filters ----------------------------------------------------

    def _reject_reason(self, text: str) -> str | None:
        """Return the name of the first filter that rejects `text`, else None."""
        cfg = self.config

        n_chars = len(text)
        if n_chars < cfg.min_chars:
            return "too_short_chars"
        if n_chars > cfg.max_chars:
            return "too_long"

        words = _RE_WORD.findall(text)
        if len(words) < cfg.min_words:
            return "too_short_words"

        # Script ratio is measured over *letters*, not all alphanumerics. Counting
        # digits in the denominator conflates two different defects: a page of
        # numbers with Bengali labels would be rejected as "wrong script" when the
        # real problem is digit spam, which the digit-ratio filter below reports
        # correctly. Keeping the denominator to letters means each filter names the
        # defect it actually detects, which is what makes the rejection breakdown
        # in the dataset report interpretable.
        letters = _RE_LETTER.findall(text)
        if not letters:
            return "no_letters"
        script_hits = len(self._script_re.findall(text))
        if script_hits / len(letters) < cfg.min_script_ratio:
            return "low_script_ratio"

        if text.count("�") / n_chars > cfg.max_replacement_ratio:
            return "mojibake"

        lines = [line for line in (ln.strip() for ln in text.split("\n")) if line]
        if len(lines) >= 5:
            counts = Counter(lines)
            duplicated = sum(count for count in counts.values() if count > 1)
            if duplicated / len(lines) > cfg.max_duplicate_line_ratio:
                return "repeated_lines"

        if len(words) >= 50:
            grams = Counter(tuple(words[i : i + 10]) for i in range(len(words) - 9))
            top_gram, top_count = grams.most_common(1)[0]
            if (top_count * 10) / len(words) > cfg.max_top_ngram_ratio:
                return "repeated_ngram"

        digits = sum(1 for ch in text if ch.isdigit())
        if digits / n_chars > cfg.max_digit_ratio:
            return "high_digit_ratio"

        symbols = sum(1 for ch in text if unicodedata.category(ch).startswith(("S", "P")))
        if symbols / n_chars > cfg.max_symbol_ratio:
            return "high_symbol_ratio"

        if self._has_long_run(text, cfg.max_char_run):
            return "long_char_run"

        if self._identifier is not None and not self._identifier.accepts(text, cfg.language):
            return "wrong_language"

        return None

    @staticmethod
    def _has_long_run(text: str, limit: int) -> bool:
        """True when any character repeats more than `limit` times consecutively."""
        run_char = ""
        run_len = 0
        for ch in text:
            if ch == run_char:
                run_len += 1
                if run_len > limit and not ch.isspace():
                    return True
            else:
                run_char, run_len = ch, 1
        return False

    # --- public API ------------------------------------------------------------

    def process(self, text: str) -> str | None:
        """Filter one document.

        Args:
            text: Already-normalized document text.

        Returns:
            The document if it passed every filter and is not a duplicate,
            otherwise None. Statistics are updated either way.
        """
        self.stats.seen += 1

        reason = self._reject_reason(text)
        if reason is not None:
            self.stats.reject(reason)
            return None

        if self.config.dedup_exact:
            digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).digest()
            if digest in self._seen_hashes:
                self.stats.reject("exact_duplicate")
                return None
            self._seen_hashes.add(digest)

        if self._near_dup is not None and self._near_dup.is_duplicate(text):
            self.stats.reject("near_duplicate")
            return None

        self.stats.kept += 1
        self.stats.kept_chars += len(text)
        self.stats.kept_words += len(_RE_WORD.findall(text))
        return text

    # --- checkpointing --------------------------------------------------------

    def dedup_state(self) -> dict:
        """Return the deduplication state for checkpointing.

        Both dedup structures must be saved. Without them, a resumed run no longer
        recognizes documents already in the corpus as duplicates, so the second
        half of the corpus silently accumulates repeats of the first - a corruption
        that produces no error and is invisible until training.
        """
        return {
            "exact_hashes": self._seen_hashes,
            "near_dup": self._near_dup.state_dict() if self._near_dup is not None else None,
        }

    def load_dedup_state(self, payload: dict) -> None:
        """Restore checkpointed deduplication state."""
        self._seen_hashes = set(payload.get("exact_hashes", set()))
        near_payload = payload.get("near_dup")
        if near_payload is not None and self._near_dup is not None:
            self._near_dup.load_state_dict(near_payload)

    def restore_stats(self, payload: dict) -> None:
        """Seed the cleaning counters from a checkpoint, so totals cover the run."""
        self.stats.seen = payload.get("seen", 0)
        self.stats.kept = payload.get("kept", 0)
        self.stats.rejected = dict(payload.get("rejected", {}))
        self.stats.kept_chars = payload.get("kept_chars", 0)
        self.stats.kept_words = payload.get("kept_words", 0)
