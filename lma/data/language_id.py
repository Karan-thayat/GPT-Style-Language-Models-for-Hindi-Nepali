"""Discriminating Assamese from Bengali text.

The problem
-----------
Both languages are written in the same Unicode block, so a script detector
cannot separate them, and off-the-shelf language identifiers routinely label
Assamese as Bengali. Because Phase 1 forbids sharing documents between the two
corpora, we need a positive test that a document belongs to the language we
think it does - otherwise Bengali text leaks into the Assamese corpus (the
likelier direction, since Bengali text is far more abundant online).

The approach
------------
A lightweight, transparent scorer over orthographic evidence rather than a
trained classifier, so its decisions can be explained during evaluation:

1. **Exclusive letters.** ``ৰ`` (U+09F0) and ``ৱ`` (U+09F1) occur in Assamese
   and essentially never in correct Bengali. Their Bengali counterparts ``র``
   and ``ব`` occur in both, but Assamese uses ``ৰ`` where Bengali uses ``র``,
   so the *ratio* between the two is strongly diagnostic.
2. **Function-word evidence.** Short grammatical words differ between the
   languages and are frequent enough to show up in any real document
   (e.g. Assamese ``আৰু``/``হৈছে`` vs Bengali ``এবং``/``হয়েছে``).

Scores are per-token-rate based, so document length does not bias the verdict.
A document is accepted for a language only when that language wins by a margin,
which means ambiguous short documents are rejected rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import ASSAMESE_RA, ASSAMESE_WA, BENGALI_RA

_RE_TOKEN = re.compile(r"[ঀ-৿]+")

#: Assamese function words / high-frequency forms with Bengali counterparts.
_ASSAMESE_MARKERS: frozenset[str] = frozenset(
    {
        "আৰু",      # and
        "হৈছে",     # has become
        "কৰি",      # doing
        "কৰা",      # to do
        "কৰে",      # does
        "নাই",      # is not
        "হ'ব",      # will be
        "গৈ",       # having gone
        "মোৰ",      # my
        "তেওঁ",     # he/she (honorific)
        "সিহঁত",    # they
        "ইয়াত",     # here
        "বুলি",     # saying / that
        "লগত",      # with
        "অসম",      # Assam
        "অসমীয়া",   # Assamese
        "পৰা",      # from
        "হৈ",       # being
        "নহয়",      # is not
        "যেন",      # as if
    }
)

#: Bengali function words that are rare or absent in Assamese.
_BENGALI_MARKERS: frozenset[str] = frozenset(
    {
        "এবং",      # and
        "হয়েছে",    # has become
        "করে",      # does
        "করা",      # to do
        "করেন",     # does (honorific)
        "থেকে",     # from
        "তিনি",     # he/she (honorific)
        "তাদের",    # their
        "আমাদের",   # our
        "এই",       # this
        "সেই",      # that
        "নয়",       # is not
        "হবে",      # will be
        "খুব",      # very
        "সঙ্গে",     # with
        "বাংলা",    # Bengali
        "পশ্চিমবঙ্গ",  # West Bengal
        "কিন্তু",    # but
        "জন্য",     # for
        "সাথে",     # with
    }
)


@dataclass(frozen=True)
class LanguageVerdict:
    """Outcome of a Bengali-vs-Assamese decision for one document.

    Attributes:
        language: ``"as"``, ``"bn"``, or ``"unknown"`` when evidence is too thin.
        assamese_score: Evidence rate favouring Assamese.
        bengali_score: Evidence rate favouring Bengali.
        evidence_tokens: Number of script tokens the decision was based on.
    """

    language: str
    assamese_score: float
    bengali_score: float
    evidence_tokens: int

    @property
    def margin(self) -> float:
        """Absolute difference between the two scores."""
        return abs(self.assamese_score - self.bengali_score)


class BengaliAssameseIdentifier:
    """Scores a document as Assamese or Bengali using orthographic evidence.

    Args:
        min_tokens: Documents with fewer script tokens than this are reported as
            ``unknown`` - too little evidence to be worth a guess.
        min_margin: Minimum score gap required to commit to a language.
    """

    def __init__(self, min_tokens: int = 12, min_margin: float = 0.004) -> None:
        """Store decision thresholds."""
        self.min_tokens = min_tokens
        self.min_margin = min_margin

    def identify(self, text: str) -> LanguageVerdict:
        """Classify one document.

        Args:
            text: Normalized document text.

        Returns:
            A :class:`LanguageVerdict`. Callers should treat ``unknown`` as
            "do not use", not as "either is fine".
        """
        tokens = _RE_TOKEN.findall(text)
        n_tokens = len(tokens)
        if n_tokens < self.min_tokens:
            return LanguageVerdict("unknown", 0.0, 0.0, n_tokens)

        as_marker_hits = sum(1 for token in tokens if token in _ASSAMESE_MARKERS)
        bn_marker_hits = sum(1 for token in tokens if token in _BENGALI_MARKERS)

        # Exclusive-letter evidence, weighted up because it is near-decisive.
        assamese_letters = text.count(ASSAMESE_RA) + text.count(ASSAMESE_WA)
        bengali_ra = text.count(BENGALI_RA)

        as_score = (as_marker_hits + 3.0 * assamese_letters) / n_tokens
        bn_score = (bn_marker_hits + 0.35 * bengali_ra) / n_tokens

        if as_score - bn_score >= self.min_margin:
            language = "as"
        elif bn_score - as_score >= self.min_margin:
            language = "bn"
        else:
            language = "unknown"
        return LanguageVerdict(language, as_score, bn_score, n_tokens)

    def accepts(self, text: str, language: str) -> bool:
        """Return True when `text` is confidently the requested language.

        An ``unknown`` verdict returns False, so thin or ambiguous documents are
        dropped instead of being assigned to whichever corpus is being built.
        """
        return self.identify(text).language == language
