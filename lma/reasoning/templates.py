"""Devanagari surface forms for the synthetic comparative-reasoning task.

Everything language-specific lives here, separated from the generation logic in
:mod:`lma.reasoning.generate`, so the two languages are configured rather than
special-cased and a third language could be added without touching the sampler.

Why the wording was chosen the way it was
-----------------------------------------
The assignment requires the finetuning text to be "in the target language's
script and natural phrasing for that language (not English prompts with foreign
names only)". Three constraints shaped every string below, and all three were
measured against this project's own Phase 1 tokenizers rather than assumed:

1. **No byte fallback.** A piece the vocabulary does not cover is encoded as raw
   UTF-8 bytes, which makes an answer span several unpredictable tokens and
   turns exact-match scoring into guesswork. Measured: ``उत्तर:`` byte-falls-back
   in Nepali because that vocabulary has no ``:`` piece, while ``उत्तर -`` is
   clean in both languages. Hence :data:`ANSWER_PREFIX` uses the dash.

2. **Single-token answers.** Every entity name below encodes to exactly one
   ``▁``-prefixed piece in its language's tokenizer, so the answer is one token.
   That is what lets the finetuning loss be masked to the answer span exactly,
   and what makes exact match unambiguous. Numbers were rejected as answers for
   the opposite reason: ``split_digits: true`` means ``12`` becomes ``▁ 1 2``,
   three tokens whose partial emission would score as a wrong answer.

3. **Grammatical authenticity.** The comparative constructions differ
   structurally between the two languages, and using one language's frame with
   the other's words would produce text the project's own language filter in
   ``lma/data/language_id.py`` would reject. Hindi compares with the postposition
   ``से`` and the copula ``है``; Nepali suffixes ``भन्दा`` directly onto the
   standard of comparison and uses the copula ``छ``. These are the same forms
   that filter relies on to tell the two languages apart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """One comparable attribute, with both poles of its comparison.

    A dimension supplies the adjective used to state a premise and the two
    superlatives used to ask a question, so that "taller" premises can be
    queried for either the tallest or the shortest entity.

    Attributes:
        key: Stable identifier used in the dataset record and in statistics.
        premise_more: Adjective phrase asserting the subject exceeds the
            standard, e.g. Hindi ``लंबा`` (tall) in "A is taller than B".
        question_most: Superlative asking for the maximum, e.g. "who is tallest".
        question_least: Superlative asking for the minimum.
    """

    key: str
    premise_more: str
    question_most: str
    question_least: str


@dataclass(frozen=True)
class LanguageTemplates:
    """Every surface string needed to render one language's examples.

    Attributes:
        language: ISO code, ``"hi"`` or ``"ne"``.
        entities: Single-token personal names used as the compared entities.
        dimensions: Comparable attributes available to the sampler.
        premise: Format string taking ``subject``, ``standard`` and ``adjective``.
        question: Format string taking ``superlative``.
        answer_prefix: Text introducing the answer, and the boundary the
            finetuning loss is masked from.
    """

    language: str
    entities: tuple[str, ...]
    dimensions: tuple[Dimension, ...]
    premise: str
    question: str
    answer_prefix: str

    def render_premise(self, subject: str, standard: str, dimension: Dimension) -> str:
        """Render "``subject`` is more ``dimension`` than ``standard``"."""
        return self.premise.format(
            subject=subject, standard=standard, adjective=dimension.premise_more
        )

    def render_question(self, dimension: Dimension, ask_most: bool) -> str:
        """Render the superlative question for one dimension."""
        superlative = dimension.question_most if ask_most else dimension.question_least
        return self.question.format(superlative=superlative)


#: Hindi (Model H). The comparative frame is "<subject> <standard> से <adjective>
#: है" - the standard of comparison takes the postposition से, and the sentence
#: closes with the copula है, both of which are Hindi markers in language_id.py.
HINDI = LanguageTemplates(
    language="hi",
    entities=(
        "राम", "श्याम", "मोहन", "सीता", "गीता", "अमित", "रवि", "विजय",
        "अनिल", "सुनील", "दीपक", "अर्जुन", "संजय", "राहुल", "कृष्ण",
        "शिव", "गणेश", "सूरज", "मनोज", "अजय", "विनोद", "रमेश", "सुरेश",
        "महेश", "राजेश", "दिनेश", "नरेश", "हरि", "गोपाल", "प्रकाश",
        "संतोष", "अशोक", "रोहित", "सचिन", "विकास", "पवन", "नितिन",
        "किरण", "लक्ष्मी", "रेखा", "माया", "राधा",
    ),
    dimensions=(
        Dimension("height", "लंबा", "सबसे लंबा", "सबसे छोटा"),
        Dimension("age", "बड़ा", "सबसे बड़ा", "सबसे छोटा"),
        Dimension("speed", "तेज़", "सबसे तेज़", "सबसे धीमा"),
        Dimension("wealth", "अमीर", "सबसे अमीर", "सबसे गरीब"),
    ),
    premise="{subject} {standard} से {adjective} है।",
    question="{superlative} कौन है?",
    answer_prefix="उत्तर -",
)

#: Nepali (Model L). The comparative frame suffixes भन्दा directly onto the
#: standard ("श्यामभन्दा अग्लो छ"), and closes with the copula छ. Both are Nepali
#: markers in language_id.py, so generated text is positively identified as
#: Nepali rather than merely "Devanagari".
NEPALI = LanguageTemplates(
    language="ne",
    entities=(
        "राम", "श्याम", "हरि", "सीता", "विनोद", "रमेश", "प्रकाश", "कमला",
        "सुरेश", "गोपाल", "कृष्ण", "शिव", "गणेश", "विष्णु", "मोहन",
        "विजय", "अर्जुन", "महेश", "सन्तोष", "विकास", "आनन्द", "किरण",
        "लक्ष्मी", "रेखा", "माया", "बिकास", "निर्मला", "दुर्गा", "भीम",
        "कुमार", "यादव", "थापा", "शर्मा",
    ),
    dimensions=(
        Dimension("height", "अग्लो", "सबैभन्दा अग्लो", "सबैभन्दा होचो"),
        Dimension("age", "जेठो", "सबैभन्दा जेठो", "सबैभन्दा कान्छो"),
        Dimension("speed", "छिटो", "सबैभन्दा छिटो", "सबैभन्दा ढिलो"),
        Dimension("wealth", "धनी", "सबैभन्दा धनी", "सबैभन्दा गरिब"),
    ),
    premise="{subject} {standard}भन्दा {adjective} छ।",
    question="{superlative} को हो?",
    answer_prefix="उत्तर -",
)

#: Registry, so scripts select by language code exactly as the Phase 1 configs do.
TEMPLATES: dict[str, LanguageTemplates] = {"hi": HINDI, "ne": NEPALI}


def get_templates(language: str) -> LanguageTemplates:
    """Return the template set for a language code.

    Args:
        language: ``"hi"`` or ``"ne"``.

    Raises:
        KeyError: If the language has no templates defined.
    """
    if language not in TEMPLATES:
        raise KeyError(
            f"no reasoning templates for {language!r}; have {sorted(TEMPLATES)}"
        )
    return TEMPLATES[language]
