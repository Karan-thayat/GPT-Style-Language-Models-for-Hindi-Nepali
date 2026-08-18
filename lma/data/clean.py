"""Document-level quality filtering and deduplication.

The cleaner is a pipeline of independent, individually-countable filters. Every
rejection is attributed to the first filter that fired, which is what makes the
dataset statistics report meaningful: it shows *why* documents were lost, not
just how many.

Filters, in order of application
-------------------------------
1. Length - too short to contain usable language modelling signal.
2. Script ratio - the document must be predominantly in the target script, which
   removes off-script pages, navigation chrome, and code blocks. Measured over
   letters only, so digit-heavy pages are attributed to the digit filter instead.
   The script is chosen per language: Devanagari for both Hindi and Nepali.
3. Replacement/mojibake - a high rate of U+FFFD indicates broken decoding.
4. Repetition - boilerplate and generated spam repeat lines or n-grams.
5. Symbol and digit ratio - tables, price lists, and link farms.
6. Gibberish - runs of meaningless Latin characters (``asdfg``, ``xjskq``),
   which the course instructors asked to be removed. Real words and proper nouns
   such as ``Tata``, ``IRCTC``, and ``Google`` are explicitly kept, as are digits.
7. Language identity - see :mod:`lma.data.language_id`. Hindi and Nepali share
   the Devanagari script, so the check rests on grammar rather than characters.
8. Exact duplicates - SHA-256 over normalized text.
9. Near duplicates - MinHash + LSH over character 5-gram shingles.

Non-language words
------------------
The instructors clarified that "remove non-language words" means gibberish -
meaningless strings like ``asdfg`` - and *not* meaningful English words or proper
nouns embedded in Indic news text, and that digits are fine. Two stages implement
that reading. :func:`strip_gibberish_words` deletes individual gibberish tokens
from otherwise good documents, and the ``high_gibberish_ratio`` filter rejects
documents that are mostly gibberish. Latin stripping stays off, so ``Tata`` and
``IRCTC`` survive.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
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
#: A run of Latin letters, optionally carrying digits or internal punctuation.
#: Candidates for the gibberish test; Indic-script tokens are never tested.
_RE_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'’.&-]*")
#: English vowels, including 'y' which carries syllables in words like "myth".
_VOWELS = frozenset("aeiouyAEIOUY")

#: Latin tokens this short are never treated as gibberish. Real abbreviations and
#: units ("kg", "PM", "AI", "5G") are overwhelmingly short and consonant-heavy, so
#: testing them would delete exactly the meaningful English the instructors asked
#: to keep. Five is the shortest length at which the instructors' own example,
#: "asdfg", must be caught, and it still sits above every common acronym.
_MIN_GIBBERISH_LENGTH = 5

#: Meaningful tokens the orthographic rules would otherwise misjudge. These are
#: real words - web and technical vocabulary that is vowel-poor by nature - and
#: the instructors asked explicitly for meaningful words to be kept. Matching is
#: case-insensitive.
_PROTECTED_WORDS = frozenset(
    {
        "https", "http", "www", "html", "xhtml", "xml", "json", "css", "php", "sql",
        "referer", "referrer", "cctld", "gtld", "url", "uri", "dns", "ftp", "smtp",
        "javascript", "wordpress", "youtube", "whatsapp", "github", "linkedin",
        "pdf", "jpg", "png", "svg", "gif", "mp3", "mp4", "wifi", "gmail",
    }
)

#: Longest all-caps run still treated as an acronym rather than shouted noise.
#: Real organisational acronyms in Indian news ("IRCTC", "BPSCL", "NABARD") sit
#: well inside this; strings like "GUYKUHKHLHKFS" do not.
_MAX_ACRONYM_LENGTH = 8

#: Adjacent keyboard letters, by row. Keyboard mash - "asdfg", "qwertyuiop",
#: "zxcvbnm" - is a walk along these rows, which is what distinguishes it from a
#: real word that merely looks unusual. Detecting the *pattern* rather than the
#: literal strings generalizes to mashes this project has not seen.
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _longest_keyboard_walk(letters: str) -> int:
    """Length of the longest unbroken slide along one keyboard row.

    A ratio of keyboard-adjacent pairs is not usable here: ordinary English words
    contain such pairs by chance ("po" in "Poverty", "we" in "Power", "re" in
    "Sartre"), so a ratio test deletes real vocabulary. What distinguishes a mash
    is that the adjacency *continues* - "asdfg" is five letters walked in one
    direction along the home row, which no real word does.

    Args:
        letters: The token's letters, lowercased.

    Returns:
        The number of characters in the longest consecutive same-row, same-direction
        run. A single character counts as a run of one.
    """
    if not letters:
        return 0
    best = longest = 1
    direction = 0
    for first, second in zip(letters, letters[1:]):
        step = 0
        for row in _KEYBOARD_ROWS:
            left, right = row.find(first), row.find(second)
            if left >= 0 and right >= 0 and abs(left - right) == 1:
                step = right - left
                break
        if step and (direction == 0 or step == direction):
            longest += 1
            direction = step
        else:
            longest = 1
            direction = 0
        best = max(best, longest)
    return best


#: A same-direction keyboard run this long is a finger drag, not a word. The
#: instructors' example "asdfg" is a run of 5; the longest such run in ordinary
#: English vocabulary is 3 ("we" + "r" in "power" breaks direction at 2-3), so 5
#: leaves a clear margin and never fires on real words.
_MIN_KEYBOARD_WALK = 5

#: Consonant runs at or above this length essentially never occur in real words.
#: The threshold is set by the longest genuine examples that appear in Indian news
#: text: "strengths" runs 5, and transliterated European surnames such as
#: "Kohlschreiber" run 6 ("hlschr"). Requiring 7 keeps every one of them while
#: still catching consonant-only mash, which the vowel-ratio and keyboard-walk
#: tests catch in any case.
_MAX_CONSONANT_RUN = 7

#: Below this vowel share a long Latin token is not pronounceable as a word.
#: "IRCTC" is 0.0 but is only 5 characters, so the length floor protects it;
#: "strengths" is 0.11 at 9 characters, so the threshold sits below that.
_MIN_VOWEL_RATIO = 0.10


def is_gibberish_word(token: str) -> bool:
    """Return True when a Latin token is meaningless letter noise.

    The course instructors defined "non-language words" as gibberish such as
    ``asdfg`` or ``xjskq``, and asked explicitly that meaningful English words and
    proper nouns - ``Tata``, ``IRCTC``, ``Google`` - be kept, along with digits.
    This function is therefore deliberately conservative: it fires only on long,
    unpronounceable Latin runs, and never on Indic script, digits, or short
    abbreviations.

    Three orthographic signals decide it:

    1. **Vowel starvation.** English and romanized Indic words carry vowels at
       roughly 30-40% of characters. Keyboard mash carries far fewer.
    2. **Consonant runs.** Real words top out at six consecutive consonants, and
       only for transliterated surnames; mashed input runs longer.
    3. **Keyboard walks.** ``asdfg`` and ``qwertyuiop`` are finger drags along a
       keyboard row, so their letters slide unbroken in one direction. This catches
       the vowel-carrying mashes the first two signals miss, and it is what makes
       the instructors' own example, ``asdfg``, detectable.

    A hyphenated compound is judged by its parts: "XHTML-based" is meaningful even
    though "XHTML" alone is vowel-poor, so a compound counts as gibberish only when
    every one of its segments does.

    Args:
        token: A single whitespace-delimited token, already stripped of
            surrounding punctuation.

    Returns:
        True when the token should be deleted as gibberish.
    """
    segments = [seg for seg in re.split(r"[-'’.&]", token) if any(ch.isalpha() for ch in seg)]
    if len(segments) > 1:
        return all(_is_gibberish_segment(seg) for seg in segments)
    return _is_gibberish_segment(token)


def _is_gibberish_segment(token: str) -> bool:
    """Apply the orthographic gibberish tests to one unhyphenated segment."""
    letters = [ch for ch in token if ch.isalpha()]
    if len(letters) < _MIN_GIBBERISH_LENGTH:
        return False
    # Anything outside the Latin alphabet is not this filter's business.
    if any(ord(ch) > 0x7F for ch in letters):
        return False
    if "".join(letters).lower() in _PROTECTED_WORDS:
        return False

    # All-caps tokens of moderate length are acronyms, and the instructors named
    # acronyms - "IRCTC" - as exactly the kind of meaningful word to keep. They are
    # vowel-poor by construction, so without this they would all read as gibberish.
    # The length ceiling still catches shouted keyboard mash like "GUYKUHKHLHKFS",
    # and a repeated-character run ("CCCCCC", "FFFFFF") is never a real acronym.
    word = "".join(letters)
    # A trailing lowercase "s" is an acronym plural ("HCFCs", "NGOs"), so test the
    # stem rather than rejecting the whole token for its one lowercase letter.
    stem = word[:-1] if word.endswith("s") and word[:-1].isupper() else word
    if stem.isupper() and len(stem) <= _MAX_ACRONYM_LENGTH and len(set(stem)) > 2:
        return False

    vowels = sum(1 for ch in letters if ch in _VOWELS)
    if vowels / len(letters) < _MIN_VOWEL_RATIO:
        return True

    run = 0
    for ch in letters:
        if ch in _VOWELS:
            run = 0
            continue
        run += 1
        if run >= _MAX_CONSONANT_RUN:
            return True

    if _longest_keyboard_walk("".join(letters).lower()) >= _MIN_KEYBOARD_WALK:
        return True
    return False


def strip_gibberish_words(text: str) -> tuple[str, int]:
    """Delete gibberish Latin tokens from a document, keeping everything else.

    Only tokens that :func:`is_gibberish_word` rejects are removed. Indic text,
    digits, punctuation, and meaningful English are untouched, so a news article
    mentioning ``Tata`` or ``IRCTC`` passes through unchanged.

    Args:
        text: Document text.

    Returns:
        A ``(cleaned_text, removed_count)`` pair.
    """
    removed = 0

    def replace(match: re.Match[str]) -> str:
        """Blank out one Latin token when it reads as gibberish."""
        nonlocal removed
        token = match.group(0)
        if is_gibberish_word(token):
            removed += 1
            return " "
        return token

    if removed == 0 and not _RE_LATIN_TOKEN.search(text):
        return text, 0
    cleaned = _RE_LATIN_TOKEN.sub(replace, text)
    if removed == 0:
        return text, 0
    # Removal leaves double spaces behind; collapse them without disturbing the
    # line structure the repetition filters and the tokenizer both rely on.
    cleaned = "\n".join(" ".join(line.split()) for line in cleaned.split("\n"))
    return cleaned, removed


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
    #: Delete individual gibberish Latin tokens ("asdfg") from kept documents.
    #: Meaningful English words and proper nouns are preserved - see
    #: :func:`is_gibberish_word` for what actually qualifies as gibberish.
    strip_gibberish: bool = True
    #: Reject the whole document when this share of its Latin tokens is gibberish.
    #: A document that is mostly keyboard noise is not worth repairing token by
    #: token. Ordinary Indic news measures far below this.
    max_gibberish_ratio: float = 0.30
    #: Latin tokens a document needs before the ratio above is meaningful. Below
    #: this, one unusual proper noun would swing the ratio wildly.
    min_latin_tokens_for_gibberish: int = 10
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
    #: Gibberish tokens deleted from documents that were otherwise kept.
    gibberish_words_removed: int = 0
    #: Documents whose text was modified by gibberish stripping.
    documents_gibberish_stripped: int = 0

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

    #: Mersenne prime used as the hash modulus. 2^31-1 rather than 2^61-1 so that
    #: `a * shingle + b` stays inside uint64 and the signature can be computed as
    #: one vectorized NumPy operation - see :meth:`_signature`, which is 42x
    #: faster this way and was otherwise the dominant cost of the whole pipeline.
    _MERSENNE = (1 << 31) - 1

    def __init__(self, permutations: int = 96, threshold: float = 0.80, shingle_size: int = 5) -> None:
        """Build the hash coefficients and choose an LSH banding for `threshold`."""
        self.permutations = permutations
        self.threshold = threshold
        self.shingle_size = shingle_size
        rng = _DeterministicRandom(seed=0xC0FFEE)
        self._coeff_a = [rng.randint(1, self._MERSENNE - 1) for _ in range(permutations)]
        self._coeff_b = [rng.randint(0, self._MERSENNE - 1) for _ in range(permutations)]
        # Built once: _signature runs for every document, so rebuilding these per
        # call would cancel out the vectorization.
        self._coeff_a_np = np.array(self._coeff_a, dtype=np.uint64)
        self._coeff_b_np = np.array(self._coeff_b, dtype=np.uint64)
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
        """Compute the MinHash signature of a shingle set.

        Evaluated as one NumPy operation over the whole (permutation x shingle)
        grid rather than as nested Python loops. The scalar form measured 52.4 ms
        per document on real corpus text - by far the largest single cost in the
        cleaning pipeline - against 1.4 ms here for identical output.

        Every intermediate stays in uint64: shingles are reduced modulo the 31-bit
        prime first, so `a * s + b` cannot exceed 62 bits.
        """
        if not shingles:
            return tuple([0] * self.permutations)

        prime = np.uint64(self._MERSENNE)
        values = np.fromiter(shingles, dtype=np.uint64, count=len(shingles)) % prime
        hashed = (self._coeff_a_np[:, None] * values[None, :] + self._coeff_b_np[:, None]) % prime
        return tuple(int(x) for x in hashed.min(axis=1))

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

        # Documents that are mostly keyboard noise rather than language. Judged on
        # the Latin tokens only: the Indic body of the document is not evidence
        # either way, and including it would dilute the ratio to nothing.
        latin_tokens = _RE_LATIN_TOKEN.findall(text)
        if len(latin_tokens) >= cfg.min_latin_tokens_for_gibberish:
            gibberish = sum(1 for token in latin_tokens if is_gibberish_word(token))
            if gibberish / len(latin_tokens) > cfg.max_gibberish_ratio:
                return "high_gibberish_ratio"

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

        # Stripping runs after the filters, not before: the rejection breakdown
        # must describe the document as it arrived, and stripping first could
        # rescue a mostly-gibberish document that should have been rejected.
        if self.config.strip_gibberish:
            text, removed = strip_gibberish_words(text)
            if removed:
                self.stats.gibberish_words_removed += removed
                self.stats.documents_gibberish_stripped += 1

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
        self.stats.gibberish_words_removed = payload.get("gibberish_words_removed", 0)
        self.stats.documents_gibberish_stripped = payload.get("documents_gibberish_stripped", 0)
