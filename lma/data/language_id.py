"""Verifying that a document is written in the language we think it is.

Two very different problems live here, because the project's two languages sit in
different Unicode blocks while the harder historical case does not.

**Hindi vs Assamese - easy.** Devanagari (U+0900-U+097F) and Bengali-Assamese
(U+0980-U+09FF) share no characters, so a script-share test separates them
completely. Cross-contamination between Model H and Model L is therefore
detectable with certainty rather than inference.

**Bengali vs Assamese - hard.** These share one script, so a script test cannot
tell them apart and general-purpose language identifiers routinely label Assamese
as Bengali. That case is still handled here, because Assamese corpora are
frequently polluted with Bengali text and the filter is what keeps Model L's
corpus honest. It relies on orthographic evidence:

1. **Exclusive letters** - ``ৰ`` (U+09F0) and ``ৱ`` (U+09F1) occur in Assamese
   and essentially never in correct Bengali. The ratio of ``ৰ`` to ``র`` is
   strongly diagnostic even in short documents.
2. **Function words** - frequent grammatical forms differ between the languages
   (Assamese ``আৰু``/``হৈছে`` against Bengali ``এবং``/``হয়েছে``).

Scores are per-token rates so document length does not bias the verdict, and a
language is assigned only when it wins by a margin: ambiguous documents come back
as ``unknown`` and are dropped rather than guessed into a corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import ASSAMESE_RA, ASSAMESE_WA, BENGALI_RA

_RE_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_RE_BENGALI = re.compile(r"[ঀ-৿]")
_RE_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_RE_BENGALI_TOKEN = re.compile(r"[ঀ-৿]+")

#: Which script each supported language is written in.
LANGUAGE_SCRIPTS = {"hi": "devanagari", "as": "bengali", "bn": "bengali"}

#: Assamese function words and high-frequency forms with Bengali counterparts.
_ASSAMESE_MARKERS: frozenset[str] = frozenset(
    {
        "আৰু", "হৈছে", "কৰি", "কৰা", "কৰে", "নাই", "হ'ব", "গৈ", "মোৰ", "তেওঁ",
        "সিহঁত", "ইয়াত", "বুলি", "লগত", "অসম", "অসমীয়া", "পৰা", "হৈ", "নহয়", "যেন",
    }
)

#: Bengali function words that are rare or absent in Assamese.
_BENGALI_MARKERS: frozenset[str] = frozenset(
    {
        "এবং", "হয়েছে", "করে", "করা", "করেন", "থেকে", "তিনি", "তাদের", "আমাদের",
        "এই", "সেই", "নয়", "হবে", "খুব", "সঙ্গে", "বাংলা", "পশ্চিমবঙ্গ", "কিন্তু",
        "জন্য", "সাথে",
    }
)


@dataclass(frozen=True)
class LanguageVerdict:
    """Outcome of a language check for one document.

    Attributes:
        language: ``"hi"``, ``"as"``, ``"bn"``, or ``"unknown"``.
        assamese_score: Evidence rate favouring Assamese (Bengali script only).
        bengali_score: Evidence rate favouring Bengali (Bengali script only).
        evidence_tokens: Tokens the decision was based on.
        script: Dominant script detected, or ``"mixed"`` when none dominates.
    """

    language: str
    assamese_score: float = 0.0
    bengali_score: float = 0.0
    evidence_tokens: int = 0
    script: str = "unknown"

    @property
    def margin(self) -> float:
        """Absolute gap between the Assamese and Bengali scores."""
        return abs(self.assamese_score - self.bengali_score)


class IndicLanguageIdentifier:
    """Checks whether a document is in the expected language.

    Args:
        min_tokens: Documents with fewer script tokens are reported ``unknown`` -
            too little evidence to justify a verdict.
        min_margin: Minimum score gap required to commit to Bengali or Assamese.
        min_script_share: Fraction of letters that must belong to a script before
            that script is considered dominant.
    """

    def __init__(
        self,
        min_tokens: int = 12,
        min_margin: float = 0.004,
        min_script_share: float = 0.60,
    ) -> None:
        """Store the decision thresholds."""
        self.min_tokens = min_tokens
        self.min_margin = min_margin
        self.min_script_share = min_script_share

    def detect_script(self, text: str) -> str:
        """Return the dominant script: ``devanagari``, ``bengali``, or ``mixed``."""
        letters = _RE_LETTER.findall(text)
        if not letters:
            return "unknown"
        total = len(letters)
        devanagari = len(_RE_DEVANAGARI.findall(text)) / total
        bengali = len(_RE_BENGALI.findall(text)) / total

        if devanagari >= self.min_script_share:
            return "devanagari"
        if bengali >= self.min_script_share:
            return "bengali"
        return "mixed"

    def identify(self, text: str) -> LanguageVerdict:
        """Classify one document.

        Devanagari text is reported as Hindi directly - the script is decisive.
        Bengali-script text goes on to the Bengali-vs-Assamese scorer.
        """
        script = self.detect_script(text)

        if script == "devanagari":
            tokens = len(text.split())
            if tokens < self.min_tokens:
                return LanguageVerdict("unknown", evidence_tokens=tokens, script=script)
            return LanguageVerdict("hi", evidence_tokens=tokens, script=script)

        if script != "bengali":
            return LanguageVerdict("unknown", script=script)

        tokens = _RE_BENGALI_TOKEN.findall(text)
        n_tokens = len(tokens)
        if n_tokens < self.min_tokens:
            return LanguageVerdict("unknown", evidence_tokens=n_tokens, script=script)

        as_markers = sum(1 for token in tokens if token in _ASSAMESE_MARKERS)
        bn_markers = sum(1 for token in tokens if token in _BENGALI_MARKERS)

        # Exclusive-letter evidence, weighted up because it is near-decisive.
        assamese_letters = text.count(ASSAMESE_RA) + text.count(ASSAMESE_WA)
        bengali_ra = text.count(BENGALI_RA)

        as_score = (as_markers + 3.0 * assamese_letters) / n_tokens
        bn_score = (bn_markers + 0.35 * bengali_ra) / n_tokens

        if as_score - bn_score >= self.min_margin:
            language = "as"
        elif bn_score - as_score >= self.min_margin:
            language = "bn"
        else:
            language = "unknown"
        return LanguageVerdict(language, as_score, bn_score, n_tokens, script)

    def accepts(self, text: str, language: str) -> bool:
        """Return True when `text` is confidently the requested language.

        An ``unknown`` verdict returns False, so thin or ambiguous documents are
        dropped instead of being assigned to whichever corpus is being built.
        """
        return self.identify(text).language == language


#: Backwards-compatible alias from when the project used Bengali as Model H.
BengaliAssameseIdentifier = IndicLanguageIdentifier
