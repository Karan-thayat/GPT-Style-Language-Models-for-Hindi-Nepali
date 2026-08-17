"""Unicode normalization for Bengali-script text (Bengali and Assamese).

Why this module is script-aware
-------------------------------
Bengali and Assamese share the Bengali Unicode block (U+0980-U+09FF), but they
are *not* the same orthography. Two code points are decisive:

* U+09F0 BENGALI LETTER RA WITH MIDDLE DIAGONAL (``ৰ``) - the Assamese ``ra``.
* U+09F1 BENGALI LETTER RA WITH LOWER DIAGONAL (``ৱ``) - the Assamese ``wa``.

Bengali writes ``ra`` as U+09B0 (``র``) and has no ``ৱ``. A naive "Indic
normalizer" folds U+09F0 into U+09B0, which silently converts Assamese into
Bengali and would destroy exactly the distinction this project is built on. So
this normalizer keeps U+09F0/U+09F1 intact and is configured per language: the
Bengali profile maps the Assamese letters *in* to their Bengali equivalents
(they only appear in Bengali text as OCR noise or mis-tagged Assamese), while
the Assamese profile preserves them.

What is always normalized
-------------------------
* NFC composition, so that canonically equivalent sequences share one encoding.
* Legacy pre-composed sequences that NFC does not touch, notably the
  nukta-based letters and the two-part vowel signs U+09CB/U+09CC, which appear
  in the wild both pre-composed and as ``U+09C7 + U+09BE / U+09D7``.
* Zero-width and invisible characters, except ZWNJ/ZWJ which are orthographically
  meaningful in Indic conjunct control and are only collapsed when repeated.
* Punctuation and digit variants, whitespace, and control characters.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --- Character-class constants -------------------------------------------------

#: Bengali/Assamese script block.
BENGALI_BLOCK = (0x0980, 0x09FF)

#: Assamese-specific letters that must survive normalization in Assamese text.
ASSAMESE_RA = "ৰ"
ASSAMESE_WA = "ৱ"
BENGALI_RA = "র"
BENGALI_BA = "ব"

#: Bengali digits, mapped to ASCII so that numeric text tokenizes consistently.
BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"

#: Invisible characters that carry no orthographic meaning and are stripped.
#: ZWNJ (U+200C) and ZWJ (U+200D) are deliberately excluded - they control
#: conjunct formation in Indic scripts.
_INVISIBLES = (
    "­"  # soft hyphen
    "​"  # zero width space
    "‎‏"  # LTR/RTL marks
    "  "  # line/paragraph separators
    "‪‫‬‭‮"  # bidi embedding
    "⁠"  # word joiner
    "﻿"  # BOM / zero width no-break space
)

#: Sequences unified so the tokenizer does not learn two encodings of one grapheme.
#:
#: Note on the nukta letters: U+09DC/U+09DD/U+09DF have canonical *decompositions*,
#: so NFC actively splits them into base letter + U+09BC nukta. Composing them here
#: therefore leaves the string outside NFC, which is why `normalize` runs a second
#: NFC pass afterwards. The two-part vowel signs are the opposite case - NFC does
#: not touch them, so only the explicit rule unifies them.
_CANONICAL_SEQUENCES: tuple[tuple[str, str], ...] = (
    ("ো", "ো"),  # e + aa  -> o
    ("ৌ", "ৌ"),  # e + au length mark -> au
    ("ড়", "ড়"),  # dda + nukta -> rra
    ("ঢ়", "ঢ়"),  # ddha + nukta -> rha
    ("য়", "য়"),  # ya + nukta -> yya
)

#: Punctuation and symbol variants folded to a single canonical form. The
#: Bengali danda U+0964 is kept as-is because it is real sentence punctuation.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    "৤": "।",  # deprecated Bengali danda -> danda
    "৥": "॥",
}

_PUNCT_TABLE = str.maketrans(_PUNCT_MAP)
_INVISIBLE_TABLE = str.maketrans({ch: None for ch in _INVISIBLES})
_DIGIT_TABLE = str.maketrans({src: str(i) for i, src in enumerate(BENGALI_DIGITS)})

_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_REPEATED_ZW = re.compile(r"[‌‍]{2,}")
_RE_ZW_AT_EDGE = re.compile(r"(?:^|(?<=\s))[‌‍]+|[‌‍]+(?=\s|$)")
_RE_SPACES = re.compile(r"[ \t\r\f\v]+")
_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,;:!?।॥\.])")
_RE_REPEATED_PUNCT = re.compile(r"([।!?])\1{2,}")


@dataclass(frozen=True)
class NormalizerConfig:
    """Per-language switches for :class:`BengaliScriptNormalizer`.

    Attributes:
        language: ISO code, used only for provenance in reports.
        preserve_assamese_letters: When True, U+09F0/U+09F1 are left untouched
            (Assamese). When False they are folded to their Bengali counterparts,
            which is the right behaviour for a Bengali corpus where their
            presence indicates OCR noise or mis-tagged text.
        map_digits_to_ascii: Fold Bengali digits to ASCII digits.
        strip_latin: Remove Latin-script runs. Off by default; the cleaner
            handles script-ratio filtering instead, which is less destructive.
    """

    language: str
    preserve_assamese_letters: bool
    map_digits_to_ascii: bool = True
    strip_latin: bool = False
    extra_replacements: dict[str, str] = field(default_factory=dict)


#: Ready-made profiles for the two project languages.
BENGALI_PROFILE = NormalizerConfig(language="bn", preserve_assamese_letters=False)
ASSAMESE_PROFILE = NormalizerConfig(language="as", preserve_assamese_letters=True)

_PROFILES = {"bn": BENGALI_PROFILE, "as": ASSAMESE_PROFILE}


def get_profile(language: str) -> NormalizerConfig:
    """Return the normalizer profile for an ISO language code (``bn`` or ``as``)."""
    try:
        return _PROFILES[language]
    except KeyError as exc:
        raise ValueError(f"No normalizer profile for language {language!r}; expected one of {sorted(_PROFILES)}") from exc


class BengaliScriptNormalizer:
    """Deterministic, idempotent text normalizer for Bengali-script languages.

    Normalization is applied in a fixed order: control/invisible stripping, NFC
    composition, canonical sequence unification, a second NFC pass, per-language
    letter policy, punctuation and digit folding, then whitespace tidying.

    The double NFC pass is deliberate. The first pass puts arbitrary input into a
    single canonical encoding so the sequence rules can match reliably. Those
    rules then compose the nukta letters (U+09DC/U+09DD/U+09DF), which NFC itself
    decomposes - so without a second pass the output would not be NFC, and
    normalizing an already-normalized document would keep changing it. Idempotence
    matters because documents can be normalized more than once across pipeline
    stages, and a non-idempotent normalizer would silently produce two encodings
    of the same text for the tokenizer to learn separately.
    """

    def __init__(self, config: NormalizerConfig) -> None:
        """Store the profile and pre-build its language-specific translation table."""
        self.config = config
        letter_map: dict[str, str | None] = {}
        if not config.preserve_assamese_letters:
            letter_map[ASSAMESE_RA] = BENGALI_RA
            letter_map[ASSAMESE_WA] = BENGALI_BA
        letter_map.update(config.extra_replacements)
        self._letter_table = str.maketrans(letter_map) if letter_map else None
        self._latin_re = re.compile(r"[A-Za-z]+") if config.strip_latin else None

    def normalize(self, text: str) -> str:
        """Normalize one document and return the cleaned string.

        Args:
            text: Raw document text in Bengali or Assamese script.

        Returns:
            The normalized text. Empty input yields an empty string.
        """
        if not text:
            return ""

        text = text.translate(_INVISIBLE_TABLE)
        text = _RE_CONTROL.sub("", text)
        text = unicodedata.normalize("NFC", text)

        for source, target in _CANONICAL_SEQUENCES:
            if source in text:
                text = text.replace(source, target)
        # Re-compose: the nukta rules above produce characters that NFC would have
        # decomposed, so a second pass is what makes the output NFC-stable and the
        # whole normalizer idempotent.
        text = unicodedata.normalize("NFC", text)

        if self._letter_table is not None:
            text = text.translate(self._letter_table)

        text = text.translate(_PUNCT_TABLE)
        if self.config.map_digits_to_ascii:
            text = text.translate(_DIGIT_TABLE)
        if self._latin_re is not None:
            text = self._latin_re.sub(" ", text)

        # Collapse meaningless zero-width runs but keep single interior ZWNJ/ZWJ.
        text = _RE_REPEATED_ZW.sub(lambda m: m.group(0)[0], text)
        text = _RE_ZW_AT_EDGE.sub("", text)

        text = _RE_REPEATED_PUNCT.sub(r"\1", text)
        text = _RE_SPACES.sub(" ", text)
        text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _RE_BLANK_LINES.sub("\n\n", text)
        return text.strip()

    def __call__(self, text: str) -> str:
        """Alias for :meth:`normalize` so the object can be used as a function."""
        return self.normalize(text)
