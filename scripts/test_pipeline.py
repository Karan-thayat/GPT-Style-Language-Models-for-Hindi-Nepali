#!/usr/bin/env python3
"""Self-tests for the Phase 1 data and tokenizer components.

These are correctness checks on the pieces where a silent bug would corrupt the
whole project - especially the Bengali/Assamese distinction, since folding the
Assamese letters or mislabelling documents would break the independence
constraint without any visible error.

Run with::

    python scripts/test_pipeline.py
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lma.data.clean import CleanerConfig, DocumentCleaner, MinHashDeduplicator
from lma.data.language_id import BengaliAssameseIdentifier
from lma.data.normalize import (
    ASSAMESE_RA,
    ASSAMESE_WA,
    BENGALI_RA,
    BengaliScriptNormalizer,
    get_profile,
)
from lma.data.splits import SplitAssigner, SplitConfig

_PASSED = 0
_FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion result."""
    global _PASSED
    if condition:
        _PASSED += 1
        print(f"  PASS  {name}")
    else:
        _FAILED.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


# --- normalization -------------------------------------------------------------

def test_normalization() -> None:
    """Normalization must be idempotent and must respect the language profile."""
    print("\n[normalization]")
    bn = BengaliScriptNormalizer(get_profile("bn"))
    asm = BengaliScriptNormalizer(get_profile("as"))

    # The decisive property: Assamese letters survive in Assamese, fold in Bengali.
    assamese_text = f"{ASSAMESE_RA}াম {ASSAMESE_WA}াৰি আৰু অসমীয়া ভাষা"
    check(
        "Assamese profile preserves ৰ (U+09F0)",
        ASSAMESE_RA in asm.normalize(assamese_text),
        "the Assamese ra was lost",
    )
    check(
        "Assamese profile preserves ৱ (U+09F1)",
        ASSAMESE_WA in asm.normalize(assamese_text),
        "the Assamese wa was lost",
    )
    bn_out = bn.normalize(assamese_text)
    check(
        "Bengali profile folds ৰ -> র",
        ASSAMESE_RA not in bn_out and BENGALI_RA in bn_out,
        f"got {bn_out!r}",
    )

    # Idempotence: normalizing twice must equal normalizing once.
    samples = [
        "এটি   একটি​ পরীক্ষা।।।",
        f"বাংলা {ASSAMESE_RA}‌তথ্য ১২৩৪",
        "কোোন শব্দ",
        "যায়‍‍‍না",
    ]
    for normalizer, tag in ((bn, "bn"), (asm, "as")):
        for sample in samples:
            once = normalizer.normalize(sample)
            twice = normalizer.normalize(once)
            check(f"idempotent [{tag}] {sample[:18]!r}", once == twice, f"{once!r} != {twice!r}")

    # Two-part vowel signs must be unified with their pre-composed form.
    decomposed = "কো"   # ke + aa  -> ko
    precomposed = "কো"        # ko
    check(
        "two-part vowel ো unified",
        bn.normalize(decomposed) == bn.normalize(precomposed),
        f"{bn.normalize(decomposed)!r} != {bn.normalize(precomposed)!r}",
    )

    # Nukta forms must be unified.
    check(
        "nukta ড় unified",
        bn.normalize("ড়") == bn.normalize("ড়"),
        "dda+nukta did not equal rra",
    )

    # Output must be valid NFC.
    out = asm.normalize("অসমীয়া ভাষা আৰু সংস্কৃতি")
    check("output is NFC", out == unicodedata.normalize("NFC", out))

    # Digits fold to ASCII; danda is preserved as sentence punctuation.
    check("Bengali digits -> ASCII", bn.normalize("১২৩") == "123", f"got {bn.normalize('১২৩')!r}")
    check("danda preserved", "।" in bn.normalize("শেষ হল।"))

    # Zero-width joiners are kept inside words (they control conjuncts).
    interior = bn.normalize("ক‍খ")
    check("interior ZWJ preserved", "‍" in interior, f"got {interior!r}")

    check("empty input safe", bn.normalize("") == "")


# --- language identification ---------------------------------------------------

def test_language_id() -> None:
    """The identifier must separate genuine Assamese from Bengali."""
    print("\n[language identification]")
    identifier = BengaliAssameseIdentifier()

    assamese = (
        "অসম ভাৰতৰ উত্তৰ-পূৱ অঞ্চলত অৱস্থিত এখন ৰাজ্য। অসমীয়া ভাষা ইয়াৰ ৰাজ্যিক ভাষা "
        "আৰু ই ইণ্ডো-আৰ্য ভাষা পৰিবাৰৰ অন্তৰ্গত। ব্ৰহ্মপুত্ৰ নদী অসমৰ মাজেদি বৈ গৈছে "
        "আৰু ইয়াৰ উপত্যকাত বহুতো মানুহে বাস কৰে।"
    )
    bengali = (
        "পশ্চিমবঙ্গ ভারতের একটি রাজ্য এবং এর রাজধানী কলকাতা। বাংলা ভাষা এখানে সরকারি "
        "ভাষা হিসেবে ব্যবহৃত হয়েছে এবং এটি ইন্দো-আর্য ভাষা পরিবারের অন্তর্গত। গঙ্গা নদী "
        "এই রাজ্যের মধ্য দিয়ে প্রবাহিত হয়েছে এবং এর অনেক মানুষ কৃষিকাজ করে।"
    )

    as_verdict = identifier.identify(assamese)
    bn_verdict = identifier.identify(bengali)
    check("Assamese detected as 'as'", as_verdict.language == "as", f"got {as_verdict}")
    check("Bengali detected as 'bn'", bn_verdict.language == "bn", f"got {bn_verdict}")
    check("accepts() agrees for Assamese", identifier.accepts(assamese, "as"))
    check("accepts() rejects Bengali as Assamese", not identifier.accepts(bengali, "as"))
    check("accepts() rejects Assamese as Bengali", not identifier.accepts(assamese, "bn"))

    # Too-short input must be 'unknown', not a coin flip.
    check("short text -> unknown", identifier.identify("অসম ভাল").language == "unknown")
    check("empty text -> unknown", identifier.identify("").language == "unknown")
    # Non-Bengali-script text (the Latin-script contamination seen in Sangraha).
    check("Latin text -> unknown", identifier.identify("Uru-o ti ibidaba anduane homogonaga" * 5).language == "unknown")

    # Hindi vs Nepali is the pair that matters now, and it is the hard case: both
    # are Devanagari, so the script cannot separate them and the decision rests on
    # grammar. Getting this wrong would let Hindi - far more abundant online -
    # contaminate Model L's corpus invisibly.
    hindi = (
        "भारत दक्षिण एशिया में स्थित एक विशाल देश है और इसकी राजधानी नई दिल्ली है। यहाँ हिन्दी भाषा सबसे अधिक बोली जाती है और यह देवनागरी लिपि में लिखी जाती है। गंगा नदी इस देश की सबसे पवित्र नदी मानी जाती है और लाखों लोग इसके किनारे रहते हैं।"
    )
    check("Hindi detected as hi", identifier.identify(hindi).language == "hi",
          f"got {identifier.identify(hindi)}")
    check("script detected as devanagari", identifier.detect_script(hindi) == "devanagari")
    check("Hindi rejected from the Assamese corpus", not identifier.accepts(hindi, "as"))
    check("Assamese rejected from the Hindi corpus", not identifier.accepts(assamese, "hi"))

    nepali = (
        "नेपाल दक्षिण एसियामा रहेको एउटा सानो देश हो र यसको राजधानी काठमाडौं हो। यहाँ नेपाली भाषा बोलिन्छ र यो देवनागरी लिपिमा लेखिन्छ। सगरमाथा संसारकै अग्लो हिमाल हो र यो नेपालमा पर्दछ। उनले भनेका छन् कि यो काम गरेको छ। धेरै मानिसहरू कृषिमा निर्भर छन् र गाउँमा बस्छन्।"
    )
    ne_verdict = identifier.identify(nepali)
    hi_verdict = identifier.identify(hindi)
    check("Nepali detected as ne", ne_verdict.language == "ne", f"got {ne_verdict}")
    check("Nepali is Devanagari script", ne_verdict.script == "devanagari")
    check("Hindi still detected as hi", hi_verdict.language == "hi", f"got {hi_verdict}")
    check("Nepali rejected from the Hindi corpus", not identifier.accepts(nepali, "hi"),
          f"ne scores hi={ne_verdict.hindi_score:.4f} ne={ne_verdict.nepali_score:.4f}")
    check("Hindi rejected from the Nepali corpus", not identifier.accepts(hindi, "ne"),
          f"hi scores hi={hi_verdict.hindi_score:.4f} ne={hi_verdict.nepali_score:.4f}")
    # The separation must be decisive, not marginal, or real-world documents with
    # fewer marker words would flip.
    check("Nepali score dominates for Nepali text",
          ne_verdict.nepali_score > 3 * max(ne_verdict.hindi_score, 1e-9),
          f"hi={ne_verdict.hindi_score:.4f} ne={ne_verdict.nepali_score:.4f}")
    check("Hindi score dominates for Hindi text",
          hi_verdict.hindi_score > 3 * max(hi_verdict.nepali_score, 1e-9),
          f"hi={hi_verdict.hindi_score:.4f} ne={hi_verdict.nepali_score:.4f}")


# --- deduplication -------------------------------------------------------------

def test_dedup() -> None:
    """MinHash must catch near-duplicates without collapsing distinct documents."""
    print("\n[deduplication]")
    dedup = MinHashDeduplicator(permutations=96, threshold=0.70)

    base = "বাংলা ভাষা একটি ইন্দো-আর্য ভাষা যা দক্ষিণ এশিয়ায় প্রচলিত এবং এর সমৃদ্ধ সাহিত্য রয়েছে। " * 4
    check("first document is not a duplicate", not dedup.is_duplicate(base))
    check("exact repeat is a duplicate", dedup.is_duplicate(base))

    # Realistic near-duplicate: a long, non-repetitive article with a date and a
    # noun phrase changed - the ordinary syndicated-news variant, measured at true
    # Jaccard 0.94.
    #
    # The fixture must be long AND non-repetitive. Character shingles of repeated
    # or templated text collapse into a small unique set, so a single edit to such
    # a document drops true Jaccard to ~0.68 - correct behaviour by the metric, but
    # a misleading way to test the detector.
    article = (
        "ভারতীয় উপমহাদেশের পূর্বাঞ্চলে অবস্থিত এই রাজ্যটির ইতিহাস অত্যন্ত প্রাচীন। "
        "প্রত্নতাত্ত্বিক খননকার্যে এখানে বহু গুরুত্বপূর্ণ নিদর্শন আবিষ্কৃত হয়েছে। "
        "মধ্যযুগে এই অঞ্চলে শিল্প ও বাণিজ্যের যথেষ্ট প্রসার ঘটেছিল বলে জানা যায়। "
        "নদীমাতৃক এই ভূখণ্ডে কৃষিকাজ চিরকালই প্রধান জীবিকা হিসেবে বিবেচিত হয়েছে। "
        "ঊনবিংশ শতাব্দীতে শিক্ষা ও সংস্কৃতির ক্ষেত্রে এক নবজাগরণ পরিলক্ষিত হয়। "
        "সাহিত্য, সংগীত ও চিত্রকলায় এই সময়ের অবদান আজও স্মরণীয় বলে গণ্য করা হয়। "
        "স্বাধীনতা আন্দোলনে এই অঞ্চলের বহু মানুষ সক্রিয়ভাবে অংশগ্রহণ করেছিলেন। "
        "বর্তমানে শিল্পায়ন ও নগরায়ণের ফলে জনবিন্যাসে উল্লেখযোগ্য পরিবর্তন এসেছে। "
        "জলবায়ু পরিবর্তনের প্রভাব এই অঞ্চলের কৃষি ব্যবস্থার উপরেও পড়তে শুরু করেছে। "
        "শিক্ষাপ্রতিষ্ঠানের সংখ্যা বৃদ্ধি পাওয়ায় সাক্ষরতার হার ক্রমশ উন্নত হয়েছে। "
        "পর্যটন শিল্প এই রাজ্যের অর্থনীতিতে এক নতুন সম্ভাবনার দ্বার উন্মোচন করেছে। "
        "যোগাযোগ ব্যবস্থার উন্নতির ফলে প্রত্যন্ত অঞ্চলগুলিও মূল ভূখণ্ডের সঙ্গে যুক্ত হয়েছে। "
    )
    check("long article is not a duplicate", not dedup.is_duplicate(article))
    near = article.replace("ঊনবিংশ শতাব্দীতে", "বিংশ শতাব্দীর প্রথমার্ধে").replace(
        "পর্যটন শিল্প", "হস্তশিল্প"
    )
    check("near-duplicate detected", dedup.is_duplicate(near))

    different = "ভারতের ইতিহাস অত্যন্ত প্রাচীন এবং বৈচিত্র্যময় একটি বিষয় যা নিয়ে বহু গবেষণা হয়েছে। " * 4
    check("distinct document is not a duplicate", not dedup.is_duplicate(different))

    # The estimator itself: MinHash must track true Jaccard within sampling error.
    # Checked directly, because the accept/reject tests above cannot distinguish a
    # correct estimator near the threshold from a broken one.
    probe = MinHashDeduplicator(permutations=256, threshold=0.70)
    left, right = probe._shingles(article), probe._shingles(near)
    true_jaccard = len(left & right) / len(left | right)
    estimate = probe._similarity(probe._signature(left), probe._signature(right))
    check(
        "MinHash estimate tracks true Jaccard",
        abs(estimate - true_jaccard) < 0.06,
        f"true={true_jaccard:.3f} est={estimate:.3f}",
    )


# --- cleaning ------------------------------------------------------------------

def test_cleaner() -> None:
    """Each filter must fire on its own failure mode and pass good text."""
    print("\n[cleaning]")
    good = (
        "পশ্চিমবঙ্গ ভারতের পূর্বদিকে অবস্থিত একটি রাজ্য এবং এর রাজধানী কলকাতা শহর। "
        "এই রাজ্যে বাংলা ভাষা সরকারি ভাষা হিসেবে ব্যবহৃত হয়েছে এবং এখানকার সংস্কৃতি "
        "অত্যন্ত সমৃদ্ধ বলে বিবেচিত হয়। গঙ্গা নদী এই রাজ্যের মধ্য দিয়ে প্রবাহিত হয়েছে "
        "এবং কৃষিকাজ এখানকার প্রধান জীবিকা হিসেবে পরিচিত। অনেক মানুষ এখানে বসবাস করেন "
        "এবং তাদের জন্য এই শহর খুব গুরুত্বপূর্ণ বলে মনে করা হয়। "
    )
    cleaner = DocumentCleaner(CleanerConfig(language="bn"))
    check("good Bengali document kept", cleaner.process(good) is not None,
          f"rejected: {cleaner.stats.rejected}")

    # Each probe text must trip its own filter. Note that filters are ordered and
    # only the first to fire is recorded, so a probe must not also violate an
    # earlier filter - a repetitive digit string, for instance, trips the
    # repetition filter before the digit-ratio one ever runs.
    cases = {
        "too_short_chars": "ছোট লেখা।",
        "low_script_ratio": "This is entirely English text with no Bengali content at all. " * 8,
        # Must stay above min_script_ratio (0.70 of alphanumerics are Bengali) while
        # exceeding max_digit_ratio (0.20 of characters are digits) - a table of
        # figures with Bengali labels, which is the real-world case.
        "high_digit_ratio": " ".join(
            f"মোট {index * 1379:07d} ও {index * 977:07d} জন" for index in range(40)
        ),
    }
    for expected, text in cases.items():
        probe = DocumentCleaner(CleanerConfig(language="bn"))
        probe.process(text)
        got = list(probe.stats.rejected)
        check(f"filter fires: {expected}", expected in got, f"got {got}")

    # Repetition filter, using otherwise-valid Bengali.
    repeated = ("এটি একটি পুনরাবৃত্ত বাক্য যা বহুবার লেখা হয়েছে এখানে ঠিক একইভাবে বারবার।\n" * 30)
    probe = DocumentCleaner(CleanerConfig(language="bn"))
    probe.process(repeated)
    check("repetition filter fires", bool({"repeated_lines", "repeated_ngram"} & set(probe.stats.rejected)),
          f"got {list(probe.stats.rejected)}")

    # Hindi documents must pass the Devanagari cleaner and be rejected by the
    # Assamese one. This is the Model H / Model L independence boundary, and
    # because the scripts differ it is a decisive check rather than a heuristic.
    hindi_doc = (
        "भारत दक्षिण एशिया में स्थित एक विशाल देश है और इसकी राजधानी नई दिल्ली है। "
        "यहाँ पर हिन्दी भाषा सबसे अधिक बोली जाती है तथा यह देवनागरी लिपि में लिखी जाती है। "
        "गंगा नदी इस देश की सबसे पवित्र नदी मानी जाती है और लाखों लोग इसके किनारे बसे हुए हैं। "
        "कृषि यहाँ की अर्थव्यवस्था का प्रमुख आधार है और अधिकांश जनसंख्या गाँवों में निवास करती है। "
        "देश की संस्कृति अत्यन्त प्राचीन और विविधतापूर्ण मानी जाती है जिसमें अनेक परम्पराएँ सम्मिलित हैं। "
    )
    hi_cleaner = DocumentCleaner(CleanerConfig(language="hi"))
    check("Hindi document kept by the Devanagari cleaner",
          hi_cleaner.process(hindi_doc) is not None,
          f"rejected: {hi_cleaner.stats.rejected}")

    # Nepali must pass its own cleaner and be rejected by the Hindi one. Both are
    # Devanagari, so this exercises the grammar-based identity filter rather than
    # the script-ratio filter.
    # Varied sentences, not one paragraph repeated: repeating a block trips the
    # n-gram repetition filter, which would mask what this test is checking.
    nepali_doc = (
        "नेपाल दक्षिण एसियामा रहेको एउटा सानो देश हो र यसको राजधानी काठमाडौं हो। "
        "यहाँ नेपाली भाषा बोलिन्छ र यो देवनागरी लिपिमा लेखिन्छ। "
        "सगरमाथा संसारकै अग्लो हिमाल हो र यो नेपालको उत्तरी भागमा पर्दछ। "
        "उनले भनेका छन् कि यो काम राम्रोसँग गरेको छ। "
        "धेरै मानिसहरू कृषिमा निर्भर छन् र गाउँमा बस्छन्। "
        "पर्यटन नेपालको अर्थतन्त्रको महत्त्वपूर्ण आधार हो भन्ने कुरा सबैले बुझेका छन्। "
        "काठमाडौं उपत्यकामा धेरै पुराना मन्दिरहरू छन् जुन विश्व सम्पदा सूचीमा परेका छन्। "
        "हिमाली क्षेत्रमा बस्ने मानिसहरूको जीवनशैली फरक छ र त्यहाँ हिउँ धेरै पर्दछ। "
        "नेपाली साहित्यमा लक्ष्मीप्रसाद देवकोटाको योगदान निकै ठूलो रहेको छ। "
        "देशका विभिन्न भागमा फरक फरक भाषा र संस्कृति रहेका छन् भन्ने कुरा सत्य हो। "
    )
    ne_cleaner = DocumentCleaner(CleanerConfig(language="ne"))
    check("Nepali document kept by the Nepali cleaner",
          ne_cleaner.process(nepali_doc) is not None,
          f"rejected: {ne_cleaner.stats.rejected}")
    probe_hi = DocumentCleaner(CleanerConfig(language="hi"))
    probe_hi.process(nepali_doc)
    check("Nepali rejected from the Hindi corpus",
          "wrong_language" in probe_hi.stats.rejected,
          f"got {list(probe_hi.stats.rejected)}")
    probe = DocumentCleaner(CleanerConfig(language="as"))
    probe.process(hindi_doc)
    check("Hindi rejected from the Assamese corpus",
          bool(probe.stats.rejected), f"got {list(probe.stats.rejected)}")

    # Cross-language rejection: Bengali text must not enter the Assamese corpus.
    probe = DocumentCleaner(CleanerConfig(language="as"))
    probe.process(good)
    check("Bengali rejected from Assamese corpus", "wrong_language" in probe.stats.rejected,
          f"got {list(probe.stats.rejected)}")

    # Statistics must be internally consistent.
    check("stats: seen == kept + rejected",
          cleaner.stats.seen == cleaner.stats.kept + sum(cleaner.stats.rejected.values()))


# --- splits --------------------------------------------------------------------

def test_splits() -> None:
    """Split assignment must be deterministic, stable, and correctly proportioned."""
    print("\n[splits]")
    assigner = SplitAssigner(SplitConfig(train=0.8, validation=0.1, test=0.1))

    check("deterministic", assigner.assign("একটি নথি", "src") == assigner.assign("একটি নথি", "src"))

    counts = {"train": 0, "validation": 0, "test": 0}
    for index in range(20_000):
        counts[assigner.assign(f"নথি সংখ্যা {index} এর বিষয়বস্তু", "src")] += 1
    train_share = counts["train"] / 20_000
    check("train share ~80%", 0.78 <= train_share <= 0.82, f"got {train_share:.3f}")

    # Identical content must land in the same split, so duplicates cannot straddle
    # train and test.
    text = "একই বিষয়বস্তু দুইবার"
    check("duplicate content -> same split", assigner.assign(text, "a") == assigner.assign(text, "a"))

    pinned = SplitAssigner(SplitConfig(train=0.8, validation=0.1, test=0.1, pinned_sources={"heldout": "test"}))
    check("pinned source honoured", all(pinned.assign(f"doc {i}", "heldout") == "test" for i in range(200)))

    try:
        SplitConfig(train=0.5, validation=0.1, test=0.1)
        check("bad ratios rejected", False, "no exception raised")
    except ValueError:
        check("bad ratios rejected", True)


# --- source manifests ----------------------------------------------------------

def test_source_manifests() -> None:
    """Every enabled manifest entry must instantiate, for both languages.

    This is a regression test. Manifest bookkeeping keys such as `limit` are
    consumed by the loader, not by the source classes, and passing one through to a
    constructor raises TypeError. Because `iter_documents` deliberately tolerates
    runtime source failures, such an error surfaced only as a warning while the run
    continued and silently produced a corpus with no downloaded documents at all.
    """
    print("\n[source manifests]")
    from lma.data.sources import build_source, load_manifest

    for language, manifest in (("hi", "hindi/configs/sources.yaml"), ("ne", "nepali/configs/sources.yaml")):
        path = REPO_ROOT / manifest
        specs = load_manifest(path)
        check(f"[{language}] manifest loads with enabled sources", bool(specs), "no enabled sources")
        for spec in specs:
            source_id = spec.get("source_id", "<unnamed>")
            try:
                build_source(spec)
                check(f"[{language}] builds source {source_id}", True)
            except Exception as exc:  # noqa: BLE001 - reporting is the point of the test
                check(f"[{language}] builds source {source_id}", False, f"{type(exc).__name__}: {exc}")

        # Manifest limits must be integers; a string would silently break slicing.
        for spec in specs:
            limit = spec.get("limit")
            check(
                f"[{language}] limit is an int for {spec.get('source_id')}",
                limit is None or isinstance(limit, int),
                f"got {type(limit).__name__}",
            )


# --- OCR ingestion -------------------------------------------------------------

def test_ocr() -> None:
    """Exercise the OCR module's dependency reporting and text-layer extraction.

    True OCR needs Tesseract, which may not be installed, so those checks are
    skipped rather than failed when the binary is absent. The embedded-text path
    only needs `pdftotext` and is tested against a real PDF, because that is the
    path most manually collected PDFs actually take - re-recognizing a page that
    already carries a text layer is both slower and less accurate.
    """
    print("\n[OCR]")
    from lma.data.ocr import OcrConfig, PdfTextExtractor, check_dependencies, install_hint

    available = check_dependencies(need_ocr=True)
    check("dependency check reports every tool", set(available) == {"pdftotext", "pdftoppm", "tesseract"},
          f"got {sorted(available)}")
    check("install hint names the language pack", "tesseract-ocr-hin" in install_hint("hi"))
    check("install hint is language-specific", "tesseract-ocr-nep" in install_hint("ne"))

    if not available.get("pdftotext"):
        print("  SKIP  embedded-text extraction (pdftotext not installed)")
        return

    sample_pdf = REPO_ROOT / "LMA_Individual_Project_v1.pdf"
    if not sample_pdf.exists():
        print("  SKIP  embedded-text extraction (no sample PDF present)")
        return

    # The sample PDF is the English assignment brief, so it is extracted under a
    # language with no script profile. Using an Indic language here would - quite
    # correctly - trip the text-layer quality gate, since English contains no
    # Devanagari at all.
    extractor = PdfTextExtractor(OcrConfig(language="en"))
    text, metadata = extractor.extract_document(sample_pdf)
    check("extracts text from a PDF text layer", len(text) > 1000, f"got {len(text)} chars")
    check("reports pages extracted", metadata["pages_extracted"] > 0, f"got {metadata['pages_extracted']}")
    check("prefers the embedded text layer over OCR", metadata["methods"].get("embedded", 0) > 0,
          f"methods={metadata['methods']}")

    # The text-layer quality gate. This is the check that keeps a corrupt PDF out
    # of the corpus: the supplied Hindi books all carry a text layer that is long
    # but wrong, because a legacy font's broken ToUnicode map drops the i-matra
    # entirely - किया extracts as कया. Length alone cannot detect that.
    hi_extractor = PdfTextExtractor(OcrConfig(language="hi"))
    good_hindi = "भारत एक विशाल देश है और यहाँ हिंदी बोली जाती है। उन्होंने कहा कि यह किया गया। " * 40
    usable, metrics = hi_extractor.assess_text_layer(good_hindi)
    check("accepts a clean Hindi text layer", usable, f"metrics={metrics}")

    # Same text with every i-matra stripped, which is exactly the corruption seen.
    broken_hindi = good_hindi.replace("\u093f", "")
    usable, metrics = hi_extractor.assess_text_layer(broken_hindi)
    check("rejects a text layer with no i-matra", not usable,
          f"metrics={metrics}")
    check("names the reason for rejection", metrics.get("reason") == "missing_imatra",
          f"got {metrics.get('reason')}")

    # Latin mojibake, the other failure mode of a broken font mapping.
    usable, metrics = hi_extractor.assess_text_layer("Ȉ Æ ȱ ȳ Ȏ " * 200)
    check("rejects Latin-Extended mojibake", not usable, f"metrics={metrics}")


# --- checkpointing -------------------------------------------------------------

def test_checkpointing() -> None:
    """Checkpoint state must round-trip, and limits must be lifetime-scoped.

    The end-to-end property that matters - a resumed run produces a byte-identical
    corpus to an uninterrupted one - is verified by interrupting a real build; see
    the Phase 1 report. These are the unit-level invariants that make it hold.
    """
    print("\n[checkpointing]")
    import tempfile
    from lma.data.checkpoint import CheckpointManager, IngestState, truncate_jsonl_to
    from lma.data.clean import MinHashDeduplicator

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Round-trip of the bookkeeping state.
        manager = CheckpointManager(tmp_dir, "bn")
        check("no checkpoint initially", not manager.exists())
        state = IngestState(language="bn", profile="full", seen=100, kept=90, train_chars=12345)
        state.source_positions["wikipedia_bn"] = 42
        manager.save(state, {"exact_hashes": {b"abc"}, "near_dup": None})
        check("checkpoint exists after save", manager.exists())

        loaded = manager.load(expected_profile="full")
        check("checkpoint round-trips", loaded is not None)
        restored, dedup = loaded
        check("restores counters", restored.seen == 100 and restored.kept == 90, f"got {restored.seen}/{restored.kept}")
        check("restores source positions", restored.source_positions.get("wikipedia_bn") == 42)
        check("restores dedup payload", dedup is not None and b"abc" in dedup["exact_hashes"])

        # A profile mismatch must be refused: resuming a `sampled` checkpoint into a
        # `full` run would apply different caps and split ratios to one corpus.
        try:
            manager.load(expected_profile="sampled")
            check("refuses profile mismatch", False, "no exception raised")
        except ValueError:
            check("refuses profile mismatch", True)

        manager.clear()
        check("clear removes the checkpoint", not manager.exists())

        # MinHash index survives a save/load and still detects duplicates.
        dedup_a = MinHashDeduplicator(permutations=96, threshold=0.70)
        article = "বাংলা ভাষার সমৃদ্ধ সাহিত্য ঐতিহ্য বহু শতাব্দী ধরে গড়ে উঠেছে এখানে। " * 6
        dedup_a.is_duplicate(article)
        dedup_b = MinHashDeduplicator(permutations=96, threshold=0.70)
        dedup_b.load_state_dict(dedup_a.state_dict())
        check("restored index still detects duplicates", dedup_b.is_duplicate(article))

        # Mismatched parameters must be refused rather than silently accepted -
        # signatures from a different configuration are not comparable.
        try:
            MinHashDeduplicator(permutations=64, threshold=0.70).load_state_dict(dedup_a.state_dict())
            check("refuses mismatched dedup params", False, "no exception raised")
        except ValueError:
            check("refuses mismatched dedup params", True)

        # JSONL reconciliation trims a file to the checkpointed line count and drops
        # a torn final line.
        jsonl = tmp_dir / "sample.jsonl"
        jsonl.write_text('{"text":"a"}\n{"text":"b"}\n{"text":"c"}\n{"text":"tor', encoding="utf-8")
        check("truncates to checkpointed count", truncate_jsonl_to(jsonl, 2) == 2)
        check("dropped the torn trailing line", jsonl.read_text(encoding="utf-8").count("\n") == 2)


# --- byte-level BPE decoding ---------------------------------------------------

def test_detokenize() -> None:
    """Byte-level BPE decoding must round-trip Bengali text exactly.

    This is the property the whole token-dump ingest rests on: if decoding were
    lossy, the recovered corpus would differ from the original and every statistic
    computed from it would be wrong in a way nothing downstream could detect.
    """
    print("\n[byte-level BPE decoding]")
    import json
    import tempfile
    import numpy as np
    from lma.data.detokenize import ByteLevelBpeDecoder, byte_decoder

    encoder = {b: c for c, b in byte_decoder().items()}
    check("byte table covers all 256 values", len(encoder) == 256, f"got {len(encoder)}")

    # Build a minimal byte-level vocabulary: one piece per byte value.
    vocab = {encoder[b]: b for b in range(256)}
    vocab["<s>"] = 256
    vocab["<pad>"] = 257

    with tempfile.TemporaryDirectory() as tmp:
        vocab_file = Path(tmp) / "vocab.json"
        vocab_file.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        decoder = ByteLevelBpeDecoder(vocab_file)

        check("special tokens identified", 256 in decoder.special_ids and 257 in decoder.special_ids,
              f"got {sorted(decoder.special_ids)}")

        original = "বাংলা ভাষার সমৃদ্ধ সাহিত্য ঐতিহ্য রয়েছে। ১২৩ ABC"
        ids = np.array([vocab[encoder[b]] for b in original.encode("utf-8")], dtype=np.uint16)
        check("round-trips Bengali exactly", decoder.decode_ids(ids) == original,
              f"got {decoder.decode_ids(ids)!r}")

        # Padding must be dropped rather than decoded into <s><s><s>...
        padded = np.concatenate([np.full(50, 256, dtype=np.uint16), ids])
        check("drops special-token padding", decoder.decode_ids(padded, decoder.special_ids) == original,
              "padding leaked into the decoded text")

        # Streaming a dump must recover the documents inside it.
        text = ("প্রথম নথির বিষয়বস্তু এখানে লেখা হয়েছে বিস্তারিতভাবে। " * 12
                + "\n\n"
                + "দ্বিতীয় নথিটি সম্পূর্ণ আলাদা একটি বিষয় নিয়ে আলোচনা করে। " * 12)
        stream_ids = np.array([vocab[encoder[b]] for b in text.encode("utf-8")], dtype=np.uint16)
        bin_file = Path(tmp) / "dump.bin"
        stream_ids.tofile(bin_file)

        docs = list(decoder.stream_documents(bin_file, chunk_tokens=10_000))
        check("recovers documents from a dump", len(docs) >= 2, f"got {len(docs)}")
        joined = " ".join(docs)
        check("recovered text preserves content",
              "প্রথম" in joined and "দ্বিতীয়" in joined,
              "content was lost during streaming")


def main() -> int:
    """Run every self-test and report the outcome."""
    print("=" * 68)
    print("Phase 1 component self-tests")
    print("=" * 68)
    test_normalization()
    test_language_id()
    test_dedup()
    test_cleaner()
    test_splits()
    test_source_manifests()
    test_ocr()
    test_checkpointing()
    test_detokenize()

    print("\n" + "=" * 68)
    print(f"{_PASSED} passed, {len(_FAILED)} failed")
    if _FAILED:
        print("\nFailures:")
        for failure in _FAILED:
            print(f"  - {failure}")
        return 1
    print("All component tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
