"""OCR ingestion for manually collected books and PDFs.

This is the second half of the manual-collection requirement: scanned books and
PDFs become JSONL that :class:`lma.data.sources.LocalFileSource` can read.

Two paths are supported and auto-detected:

* **Embedded text** - many Indic PDFs (government reports, Wikisource exports)
  already contain a text layer. ``pdftotext`` extracts it directly, which is
  faster and far more accurate than re-recognizing the page images.
* **True OCR** - for scans with no text layer, pages are rasterized and passed
  to Tesseract with the Bengali or Assamese traineddata.

Both are external tools rather than Python dependencies, so this module checks
for them and reports precisely what to install if they are missing. Nothing here
is a pretrained *language model* - OCR is text acquisition, the same category as
downloading a corpus.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

#: Tesseract language codes for the project's two languages.
TESSERACT_LANGS = {"hi": "hin", "ne": "nep", "bn": "ben", "as": "asm"}

#: Unicode block bounds per language, used to test whether an extracted text
#: layer is in the script it claims to be.
_SCRIPT_RANGES = {
    "hi": ("\u0900", "\u097f"),
    "ne": ("\u0900", "\u097f"),
    "bn": ("\u0980", "\u09ff"),
    "as": ("\u0980", "\u09ff"),
}

_RE_HYPHEN_BREAK = re.compile(r"-\n(?=\S)")
_RE_PAGE_NUMBER_LINE = re.compile(r"^\s*[\d০-৯]{1,4}\s*$", re.MULTILINE)


@dataclass
class OcrConfig:
    """Settings for :class:`PdfTextExtractor`.

    Attributes:
        language: ``"bn"`` or ``"as"``; selects the Tesseract model.
        dpi: Rasterization resolution for true OCR. 300 is the accuracy/speed
            sweet spot for Indic scripts; below 200 conjuncts degrade badly.
        prefer_embedded_text: Try the PDF's text layer before rasterizing.
        min_chars_per_page: A page yielding less than this from the text layer is
            treated as image-only and sent to OCR instead.
        psm: Tesseract page segmentation mode; 1 enables automatic layout with
            orientation detection.
        drop_page_numbers: Strip lines that are only a page number.
        join_hyphenated: Rejoin words broken across a line by a hyphen.
    """

    language: str = "hi"
    dpi: int = 300
    prefer_embedded_text: bool = True
    min_chars_per_page: int = 120
    #: Minimum share of a page's characters that must belong to the target script
    #: before its embedded text layer is trusted. A PDF using a legacy non-Unicode
    #: font extracts as Latin-Extended mojibake, which is long enough to pass a
    #: length check but is not the language at all.
    min_script_ratio: float = 0.50
    #: Minimum share of Devanagari characters that must be the i-matra U+093F.
    #: Real Hindi runs 2.5-4%; a broken font mapping that renders this vowel sign
    #: positionally often drops it entirely, giving a rate of 0. Set to 0 to
    #: disable this check for scripts where it does not apply.
    min_imatra_ratio: float = 0.005
    #: Minimum share that must be the virama U+094D. Real Hindi runs 4-6%; a value
    #: far below that means conjuncts were lost.
    min_virama_ratio: float = 0.005
    psm: int = 1
    drop_page_numbers: bool = True
    join_hyphenated: bool = True
    #: Parallel OCR workers. Tesseract is single-threaded per page and pages are
    #: independent, so throughput scales with cores. Defaults to one per core.
    workers: int = 0
    #: Pages rasterized per batch. Larger batches amortize pdftoppm startup but
    #: hold more PNGs on disk at once; each 300 DPI page is roughly 150-500 KB.
    batch_pages: int = 50


def _ocr_one_image(args: tuple[str, str, int]) -> tuple[str, str]:
    """OCR a single rendered page. Module-level so it can be pickled to a worker.

    Args:
        args: ``(image_path, tesseract_lang, psm)``.

    Returns:
        ``(image_path, text)``; empty text when Tesseract fails, so one bad page
        cannot abort a multi-hour job.
    """
    image_path, lang, psm = args
    # Tesseract parallelizes internally with OpenMP. Left unconstrained, N worker
    # processes each spawn their own threads and oversubscribe the CPU - measured
    # here at 4x SLOWER than the serial baseline. Pinning each worker to one
    # thread and parallelizing at the page level instead gives 0.61s/page against
    # 2.40s/page serial on 8 cores.
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "-l", lang, "--psm", str(psm)],
            capture_output=True, check=True, timeout=300, env=env,
        )
        return image_path, result.stdout.decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return image_path, ""


class MissingDependencyError(RuntimeError):
    """Raised when a required external binary is not installed."""


def check_dependencies(need_ocr: bool = True) -> dict[str, str | None]:
    """Report which external tools are available.

    Args:
        need_ocr: Also check the rasterizer and Tesseract, not just ``pdftotext``.

    Returns:
        Mapping of tool name to resolved path, or None when absent.
    """
    tools = ["pdftotext"]
    if need_ocr:
        tools += ["pdftoppm", "tesseract"]
    return {tool: shutil.which(tool) for tool in tools}


def install_hint(language: str = "hi") -> str:
    """Return the apt command that installs every OCR dependency."""
    lang_pack = TESSERACT_LANGS.get(language, "hin")
    return (
        "sudo apt-get install -y poppler-utils tesseract-ocr "
        f"tesseract-ocr-{lang_pack}"
    )


class PdfTextExtractor:
    """Extracts text from PDFs, using the embedded text layer when present.

    Args:
        config: Extraction settings.

    Raises:
        MissingDependencyError: If no extraction path is available at all.
    """

    def __init__(self, config: OcrConfig) -> None:
        """Validate the toolchain and record which paths are usable."""
        self.config = config
        available = check_dependencies(need_ocr=True)
        self._has_pdftotext = available.get("pdftotext") is not None
        self._has_ocr = available.get("pdftoppm") is not None and available.get("tesseract") is not None
        self.last_text_layer_metrics: dict = {}
        if not self._has_pdftotext and not self._has_ocr:
            raise MissingDependencyError(
                "No PDF text extraction tool found. Install with:\n  " + install_hint(config.language)
            )

    # --- extraction paths ------------------------------------------------------

    def _embedded_text(self, pdf_path: Path) -> list[str]:
        """Extract per-page text from the PDF's text layer via ``pdftotext``."""
        if not self._has_pdftotext:
            return []
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
                capture_output=True,
                check=True,
                timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return []
        text = result.stdout.decode("utf-8", errors="replace")
        return text.split("\f")

    def _ocr_page_range(self, pdf_path: Path, first: int, last: int) -> list[str]:
        """Rasterize pages [first, last] and OCR them, one worker per core.

        Rasterization is a single pdftoppm call (it is fast and I/O bound), while
        the Tesseract calls are distributed across a process pool. Pages are
        independent, so this scales close to linearly with core count - the
        difference between an overnight job and a coffee break at book scale.
        """
        if not self._has_ocr:
            return []
        lang = TESSERACT_LANGS.get(self.config.language, "hin")
        workers = self.config.workers or (os.cpu_count() or 1)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            try:
                subprocess.run(
                    [
                        "pdftoppm", "-r", str(self.config.dpi), "-png",
                        "-f", str(first), "-l", str(last),
                        str(pdf_path), str(tmp_dir / "page"),
                    ],
                    capture_output=True, check=True, timeout=3600,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return []

            images = sorted(tmp_dir.glob("page-*.png"))
            if not images:
                return []

            tasks = [(str(image), lang, self.config.psm) for image in images]
            if workers <= 1 or len(tasks) == 1:
                results = dict(_ocr_one_image(task) for task in tasks)
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    results = dict(pool.map(_ocr_one_image, tasks, chunksize=1))

            # Restore page order: the pool returns completions out of order.
            return [results.get(str(image), "") for image in images]

    def _page_count(self, pdf_path: Path) -> int:
        """Return the PDF's page count, or 0 when it cannot be determined."""
        if not shutil.which("pdfinfo"):
            return 0
        try:
            result = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, check=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0
        match = re.search(r"^Pages:\s+(\d+)", result.stdout.decode("utf-8", "replace"), re.MULTILINE)
        return int(match.group(1)) if match else 0

    # --- post-processing -------------------------------------------------------

    def _postprocess(self, text: str) -> str:
        """Apply OCR-specific tidying (hyphen joins, page-number removal)."""
        if self.config.join_hyphenated:
            text = _RE_HYPHEN_BREAK.sub("", text)
        if self.config.drop_page_numbers:
            text = _RE_PAGE_NUMBER_LINE.sub("", text)
        return "\n".join(line.rstrip() for line in text.split("\n")).strip()

    # --- public API ------------------------------------------------------------

    def assess_text_layer(self, text: str) -> tuple[bool, dict]:
        """Judge whether an extracted text layer is actually usable.

        Length is not evidence of correctness. A PDF whose embedded font lacks a
        valid ToUnicode map extracts plenty of characters - they are simply the
        wrong ones, typically Latin-Extended code points standing in for Indic
        glyphs. Worse, positionally-rendered vowel signs (notably the Devanagari
        i-matra, which draws to the *left* of its consonant) are frequently
        dropped, so the text reads as plausible-looking but systematically
        misspelt words.

        Args:
            text: Text extracted from the PDF's embedded layer.

        Returns:
            ``(usable, metrics)``. `metrics` records the measured ratios so a
            rejection can be explained rather than merely asserted.
        """
        if not text.strip():
            return False, {"reason": "empty", "chars": 0}

        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return False, {"reason": "no_letters", "chars": len(text)}

        script_range = _SCRIPT_RANGES.get(self.config.language)
        if script_range is None:
            return True, {"reason": "no_script_profile"}

        low, high = script_range
        script_chars = [ch for ch in text if low <= ch <= high]
        script_ratio = len(script_chars) / len(letters)

        metrics = {
            "chars": len(text),
            "script_ratio": script_ratio,
            "script_chars": len(script_chars),
        }

        if script_ratio < self.config.min_script_ratio:
            metrics["reason"] = "low_script_ratio"
            return False, metrics

        # Devanagari-specific: the vowel signs a broken mapping tends to lose.
        if self.config.language in {"hi", "ne"} and script_chars:
            n_script = len(script_chars)
            imatra = text.count("\u093f") / n_script
            virama = text.count("\u094d") / n_script
            metrics["imatra_ratio"] = imatra
            metrics["virama_ratio"] = virama
            if imatra < self.config.min_imatra_ratio:
                metrics["reason"] = "missing_imatra"
                return False, metrics
            if virama < self.config.min_virama_ratio:
                metrics["reason"] = "missing_virama"
                return False, metrics

        metrics["reason"] = "ok"
        return True, metrics

    def extract(self, pdf_path: Path):
        """Extract text page by page, choosing the best method per page.

        Args:
            pdf_path: PDF to read.

        Yields:
            ``(page_number, text, method)`` where method is ``"embedded"`` or
            ``"ocr"``. Blank pages are skipped.
        """
        pdf_path = Path(pdf_path)
        embedded = self._embedded_text(pdf_path) if self.config.prefer_embedded_text else []

        # Sample the text layer before trusting it. Judging the whole document at
        # once is both cheaper and more reliable than per-page checks, because a
        # broken font mapping is a property of the file, not of any one page.
        if embedded:
            sample = "\n".join(embedded[: min(40, len(embedded))])
            usable, metrics = self.assess_text_layer(sample)
            self.last_text_layer_metrics = metrics
            if not usable:
                embedded = []

        if embedded:
            for page_no, page_text in enumerate(embedded, start=1):
                cleaned = self._postprocess(page_text)
                if len(cleaned) >= self.config.min_chars_per_page:
                    yield page_no, cleaned, "embedded"
                else:
                    # Single-page fallback for a document whose text layer is
                    # mostly fine but has scanned inserts. Rare by construction:
                    # a wholly broken layer is rejected up front by
                    # `assess_text_layer`, which routes the file to the batched
                    # OCR path below instead.
                    ocr_pages = self._ocr_page_range(pdf_path, page_no, page_no)
                    if ocr_pages:
                        ocr_text = self._postprocess(ocr_pages[0])
                        if ocr_text:
                            yield page_no, ocr_text, "ocr"
            return

        total = self._page_count(pdf_path) or 1
        # Render and OCR in batches rather than materializing the whole book. A
        # single pdftoppm call over thousands of pages delays the first OCR result
        # by minutes and holds every 300-DPI PNG on disk simultaneously; batching
        # bounds both, and lets the caller see progress while the job runs.
        batch = max(1, self.config.batch_pages)
        for start in range(1, total + 1, batch):
            end = min(start + batch - 1, total)
            for offset, page_text in enumerate(self._ocr_page_range(pdf_path, start, end)):
                cleaned = self._postprocess(page_text)
                if cleaned:
                    yield start + offset, cleaned, "ocr"

    def extract_document(self, pdf_path: Path) -> tuple[str, dict]:
        """Extract a whole PDF as one document plus provenance metadata.

        Returns:
            ``(text, metadata)`` where metadata records the per-method page counts.
        """
        pages: list[str] = []
        methods: dict[str, int] = {}
        for _, text, method in self.extract(pdf_path):
            pages.append(text)
            methods[method] = methods.get(method, 0) + 1
        return "\n\n".join(pages), {
            "source_file": str(pdf_path),
            "pages_extracted": len(pages),
            "methods": methods,
        }


def ingest_directory(
    pdf_dir: Path,
    output_jsonl: Path,
    config: OcrConfig,
    one_document_per_page: bool = False,
) -> dict:
    """OCR every PDF in a directory into a JSONL file for the corpus pipeline.

    Args:
        pdf_dir: Directory searched recursively for ``*.pdf``.
        output_jsonl: Destination JSONL, one document per line.
        config: Extraction settings.
        one_document_per_page: Emit each page separately instead of one document
            per book. Useful when pages are independent (newspaper scans).

    Returns:
        A summary dict with per-file page counts and total characters.
    """
    pdf_dir = Path(pdf_dir)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    extractor = PdfTextExtractor(config)

    summary = {"files": [], "documents": 0, "characters": 0, "language": config.language}
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for pdf_path in sorted(pdf_dir.rglob("*.pdf")):
            file_docs = 0
            file_chars = 0
            if one_document_per_page:
                for page_no, text, method in extractor.extract(pdf_path):
                    handle.write(
                        json.dumps(
                            {"text": text, "url": f"{pdf_path}#page={page_no}", "method": method},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    file_docs += 1
                    file_chars += len(text)
            else:
                text, metadata = extractor.extract_document(pdf_path)
                if text.strip():
                    handle.write(json.dumps({"text": text, "url": str(pdf_path), **metadata}, ensure_ascii=False) + "\n")
                    file_docs = 1
                    file_chars = len(text)

            summary["files"].append({"path": str(pdf_path), "documents": file_docs, "characters": file_chars})
            summary["documents"] += file_docs
            summary["characters"] += file_chars

    return summary
