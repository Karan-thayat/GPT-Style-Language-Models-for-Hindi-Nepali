"""Script-aware Unicode normalization for Indic text.

The project's two models use different writing systems - Hindi in Devanagari
(U+0900-U+097F) and Assamese in the Bengali-Assamese block (U+0980-U+09FF) - so
normalization is parameterized by script rather than hard-coded to one. The
shared machinery (NFC handling, invisible-character policy, punctuation and digit
folding, whitespace) is identical; only the per-script tables differ.

Why a generic "Indic normalizer" is not good enough
---------------------------------------------------
Off-the-shelf normalizers routinely destroy orthographic distinctions that this
project depends on:

* **Devanagari nukta letters.** क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़ (U+0958-U+095F) have canonical
  decompositions, so NFC splits them into base + U+093C nukta. Both forms occur
  in real Hindi text, and a tokenizer that sees both learns two encodings of one
  grapheme. Worse, some normalizers *drop* the nukta entirely, which merges
  distinct words - ज़रा (zaraa, "a little") becomes जरा (jaraa, "aged").
* **Assamese ৰ/ৱ.** U+09F0/U+09F1 are the Assamese ``ra`` and ``wa``. Folding
  them to the Bengali र/ব silently converts Assamese into Bengali.
* **ZWNJ/ZWJ.** These control conjunct formation in every Indic script. Stripping
  them changes how text renders and how words are segmented.

What is always normalized
-------------------------
* NFC composition, applied twice - before and after the explicit sequence rules,
  because those rules compose characters that NFC itself decomposes.
* Legacy and variant encodings of the same grapheme, per script.
* Invisible characters, except ZWNJ/ZWJ which are only collapsed when repeated.
* Punctuation and digit variants, whitespace, and control characters.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --- Script blocks -------------------------------------------------------------

#: Devanagari, used for Hindi.
DEVANAGARI_BLOCK = (0x0900, 0x097F)

#: Bengali-Assamese, used for Assamese (and Bengali).
BENGALI_BLOCK = (0x0980, 0x09FF)

#: Regex character classes for each script, used by the cleaner and statistics.
DEVANAGARI_CLASS = r"ऀ-ॿ"
BENGALI_CLASS = r"ঀ-৿"

# --- Assamese-specific letters -------------------------------------------------

#: These must survive normalization in Assamese text; see the module docstring.
ASSAMESE_RA = "ৰ"
ASSAMESE_WA = "ৱ"
BENGALI_RA = "র"
BENGALI_BA = "ব"

# --- Digits --------------------------------------------------------------------

#: Devanagari digits ०-९, folded to ASCII so numbers tokenize consistently.
DEVANAGARI_DIGITS = "०१२३४५६७८९"

#: Bengali digits ০-৯.
BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"

# --- Invisible characters ------------------------------------------------------

#: Stripped everywhere. ZWNJ (U+200C) and ZWJ (U+200D) are deliberately absent -
#: they control conjunct formation and are orthographically meaningful.
#: Invisible characters stripped everywhere, given as explicit code points.
#:
#: Written as escapes rather than as literal characters on purpose. An earlier
#: version listed them literally, and the entry intended to be U+2028/U+2029
#: (LINE/PARAGRAPH SEPARATOR) was silently replaced by two ordinary U+0020
#: spaces during an edit - which made the normalizer delete every space in every
#: document, fusing whole paragraphs into a single token. Escapes cannot be
#: corrupted that way, and the check below fails loudly if one ever is.
#:
#: ZWNJ (U+200C) and ZWJ (U+200D) are deliberately absent: they control conjunct
#: formation in Indic scripts and are orthographically meaningful.
_INVISIBLE_CODEPOINTS = (
    0x00AD,  # soft hyphen
    0x200B,  # zero width space
    0x200E,  # left-to-right mark
    0x200F,  # right-to-left mark
    0x2028,  # line separator
    0x2029,  # paragraph separator
    0x202A,  # left-to-right embedding
    0x202B,  # right-to-left embedding
    0x202C,  # pop directional formatting
    0x202D,  # left-to-right override
    0x202E,  # right-to-left override
    0x2060,  # word joiner
    0xFEFF,  # BOM / zero width no-break space
)

#: Guard: whitespace that separates words must never be stripped. This is the
#: assertion that would have caught the bug described above at import time.
assert 0x0020 not in _INVISIBLE_CODEPOINTS, "SPACE must never be treated as invisible"
assert 0x0009 not in _INVISIBLE_CODEPOINTS, "TAB must never be treated as invisible"
assert 0x000A not in _INVISIBLE_CODEPOINTS, "NEWLINE must never be treated as invisible"

_INVISIBLES = "".join(chr(cp) for cp in _INVISIBLE_CODEPOINTS)

# --- Per-script canonical sequences --------------------------------------------

#: Devanagari. The nukta letters U+0958-U+095F have canonical decompositions, so
#: NFC splits them; these rules compose them back and a second NFC pass then
#: re-decomposes, which is what makes the result NFC-stable and idempotent.
#: Preserving the nukta is essential: dropping it merges distinct Hindi words.
_DEVANAGARI_SEQUENCES: tuple[tuple[str, str], ...] = (
    ("क़", "क़"),  # ka + nukta  -> qa
    ("ख़", "ख़"),  # kha + nukta -> khha
    ("ग़", "ग़"),  # ga + nukta  -> ghha
    ("ज़", "ज़"),  # ja + nukta  -> za
    ("ड़", "ड़"),  # dda + nukta -> dddha
    ("ढ़", "ढ़"),  # ddha + nukta -> rha
    ("फ़", "फ़"),  # pha + nukta -> fa
    ("य़", "य़"),  # ya + nukta  -> yya
)

#: Bengali-Assamese. The two-part vowel signs are the opposite case from the
#: nukta letters - NFC does not touch them, so only these explicit rules unify
#: the pre-composed and decomposed spellings.
_BENGALI_SEQUENCES: tuple[tuple[str, str], ...] = (
    ("ো", "ো"),  # e + aa  -> o
    ("ৌ", "ৌ"),  # e + au length mark -> au
    ("ড়", "ড়"),  # dda + nukta -> rra
    ("ঢ়", "ঢ়"),  # ddha + nukta -> rha
    ("য়", "য়"),  # ya + nukta -> yya
)

# --- Punctuation ---------------------------------------------------------------

#: Variant forms folded to one spelling. The danda U+0964 and double danda
#: U+0965 are shared across Indic scripts and are kept - they are real sentence
#: punctuation, not decoration.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    "৤": "।",  # deprecated Bengali danda -> danda
    "৥": "॥",
    "॰": ".",  # Devanagari abbreviation sign
}

_PUNCT_TABLE = str.maketrans(_PUNCT_MAP)
_INVISIBLE_TABLE = str.maketrans({ch: None for ch in _INVISIBLES})

_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_REPEATED_ZW = re.compile(r"[‌‍]{2,}")
_RE_ZW_AT_EDGE = re.compile(r"(?:^|(?<=\s))[‌‍]+|[‌‍]+(?=\s|$)")
_RE_SPACES = re.compile(r"[ \t\r\f\v]+")
_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,;:!?।॥\.])")
_RE_REPEATED_PUNCT = re.compile(r"([।!?])\1{2,}")


@dataclass(frozen=True)
class NormalizerConfig:
    """Per-language normalization policy.

    Attributes:
        language: ISO code, recorded in reports for provenance.
        script: ``"devanagari"`` or ``"bengali"``; selects the sequence and digit
            tables.
        preserve_assamese_letters: Bengali-script only. When True, U+09F0/U+09F1
            are left intact (Assamese). When False they fold to their Bengali
            counterparts, which is correct for a Bengali corpus where their
            presence indicates OCR noise or mis-tagged text. Irrelevant for
            Devanagari.
        map_digits_to_ascii: Fold the script's digits to ASCII.
        strip_latin: Remove Latin runs. Off by default; the cleaner's script-ratio
            filter handles mixed-script documents less destructively.
        extra_replacements: Additional character substitutions.
    """

    language: str
    script: str = "devanagari"
    preserve_assamese_letters: bool = False
    map_digits_to_ascii: bool = True
    strip_latin: bool = False
    extra_replacements: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the script name."""
        if self.script not in {"devanagari", "bengali"}:
            raise ValueError(f"script must be 'devanagari' or 'bengali', got {self.script!r}")


#: Ready-made profiles. Hindi is Model H (higher-resource); Nepali is Model L.
#: Both are Devanagari, so they share the same normalization rules entirely -
#: the languages differ in grammar and vocabulary, not orthographic convention.
HINDI_PROFILE = NormalizerConfig(language="hi", script="devanagari")
NEPALI_PROFILE = NormalizerConfig(language="ne", script="devanagari")
ASSAMESE_PROFILE = NormalizerConfig(language="as", script="bengali", preserve_assamese_letters=True)
BENGALI_PROFILE = NormalizerConfig(language="bn", script="bengali", preserve_assamese_letters=False)

_PROFILES = {
    "hi": HINDI_PROFILE,
    "ne": NEPALI_PROFILE,
    "as": ASSAMESE_PROFILE,
    "bn": BENGALI_PROFILE,
}

#: Default writing system per language, for callers that need the script but
#: not a whole normalizer profile.
LANGUAGE_SCRIPTS_DEFAULT = {code: profile.script for code, profile in _PROFILES.items()}


def get_profile(language: str) -> NormalizerConfig:
    """Return the normalizer profile for an ISO language code.

    Args:
        language: ``"hi"``, ``"as"``, or ``"bn"``.

    Raises:
        ValueError: For an unknown language code.
    """
    try:
        return _PROFILES[language]
    except KeyError as exc:
        raise ValueError(
            f"No normalizer profile for language {language!r}; expected one of {sorted(_PROFILES)}"
        ) from exc


def script_class(script: str) -> str:
    """Return the regex character class for a script name."""
    return DEVANAGARI_CLASS if script == "devanagari" else BENGALI_CLASS


class IndicNormalizer:
    """Deterministic, idempotent normalizer for Devanagari and Bengali-script text.

    Normalization runs in a fixed order: invisible and control stripping, NFC,
    the script's canonical sequence rules, a second NFC pass, the per-language
    letter policy, punctuation and digit folding, then whitespace tidying.

    The **double NFC pass** is the subtle part. The first puts arbitrary input
    into one canonical encoding so the sequence rules match reliably. Those rules
    then compose nukta letters that NFC itself decomposes, so without a second
    pass the output would not be NFC and re-normalizing an already-normalized
    document would keep changing it. Idempotence matters because documents can be
    normalized more than once across pipeline stages, and a non-idempotent
    normalizer would hand the tokenizer two encodings of the same text.
    """

    def __init__(self, config: NormalizerConfig) -> None:
        """Build the per-script tables this profile needs."""
        self.config = config

        if config.script == "devanagari":
            self._sequences = _DEVANAGARI_SEQUENCES
            digits = DEVANAGARI_DIGITS
        else:
            self._sequences = _BENGALI_SEQUENCES
            digits = BENGALI_DIGITS

        self._digit_table = str.maketrans({src: str(i) for i, src in enumerate(digits)})

        letter_map: dict[str, str | None] = {}
        if config.script == "bengali" and not config.preserve_assamese_letters:
            letter_map[ASSAMESE_RA] = BENGALI_RA
            letter_map[ASSAMESE_WA] = BENGALI_BA
        letter_map.update(config.extra_replacements)
        self._letter_table = str.maketrans(letter_map) if letter_map else None

        self._latin_re = re.compile(r"[A-Za-z]+") if config.strip_latin else None

    def normalize(self, text: str) -> str:
        """Normalize one document.

        Args:
            text: Raw document text.

        Returns:
            The normalized text; empty input yields an empty string.
        """
        if not text:
            return ""

        text = text.translate(_INVISIBLE_TABLE)
        text = _RE_CONTROL.sub("", text)
        text = unicodedata.normalize("NFC", text)

        for source, target in self._sequences:
            if source in text:
                text = text.replace(source, target)
        # See the class docstring: this second pass is what keeps the output
        # NFC-stable, and therefore the normalizer idempotent.
        text = unicodedata.normalize("NFC", text)

        if self._letter_table is not None:
            text = text.translate(self._letter_table)

        text = text.translate(_PUNCT_TABLE)
        if self.config.map_digits_to_ascii:
            text = text.translate(self._digit_table)
        if self._latin_re is not None:
            text = self._latin_re.sub(" ", text)

        text = _RE_REPEATED_ZW.sub(lambda m: m.group(0)[0], text)
        text = _RE_ZW_AT_EDGE.sub("", text)

        text = _RE_REPEATED_PUNCT.sub(r"\1", text)
        text = _RE_SPACES.sub(" ", text)
        text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _RE_BLANK_LINES.sub("\n\n", text)
        return text.strip()

    def __call__(self, text: str) -> str:
        """Alias for :meth:`normalize`."""
        return self.normalize(text)


#: Backwards-compatible alias. The class was named for the Bengali script before
#: Hindi became Model H; the implementation is script-parameterized now.
BengaliScriptNormalizer = IndicNormalizer
