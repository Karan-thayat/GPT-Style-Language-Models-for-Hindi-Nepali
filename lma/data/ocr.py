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
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

#: Tesseract language codes for the project's two languages.
TESSERACT_LANGS = {"hi": "hin", "bn": "ben", "as": "asm"}

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
    psm: int = 1
    drop_page_numbers: bool = True
    join_hyphenated: bool = True


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
        """Rasterize pages [first, last] and OCR each one with Tesseract."""
        if not self._has_ocr:
            return []
        lang = TESSERACT_LANGS.get(self.config.language, "hin")
        pages: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            try:
                subprocess.run(
                    [
                        "pdftoppm", "-r", str(self.config.dpi), "-png",
                        "-f", str(first), "-l", str(last),
                        str(pdf_path), str(tmp_dir / "page"),
                    ],
                    capture_output=True, check=True, timeout=1800,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return []

            for image in sorted(tmp_dir.glob("page-*.png")):
                try:
                    result = subprocess.run(
                        ["tesseract", str(image), "stdout", "-l", lang, "--psm", str(self.config.psm)],
                        capture_output=True, check=True, timeout=600,
                    )
                    pages.append(result.stdout.decode("utf-8", errors="replace"))
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    pages.append("")
        return pages

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

    def extract(self, pdf_path: Path) -> Iterator[tuple[int, str, str]]:
        """Extract text page by page, choosing the best method per page.

        Args:
            pdf_path: PDF to read.

        Yields:
            ``(page_number, text, method)`` where method is ``"embedded"`` or
            ``"ocr"``. Blank pages are skipped.
        """
        pdf_path = Path(pdf_path)
        embedded = self._embedded_text(pdf_path) if self.config.prefer_embedded_text else []

        if embedded:
            for page_no, page_text in enumerate(embedded, start=1):
                cleaned = self._postprocess(page_text)
                if len(cleaned) >= self.config.min_chars_per_page:
                    yield page_no, cleaned, "embedded"
                else:
                    ocr_pages = self._ocr_page_range(pdf_path, page_no, page_no)
                    if ocr_pages:
                        ocr_text = self._postprocess(ocr_pages[0])
                        if ocr_text:
                            yield page_no, ocr_text, "ocr"
            return

        total = self._page_count(pdf_path) or 1
        for page_text in self._ocr_page_range(pdf_path, 1, total):
            cleaned = self._postprocess(page_text)
            if cleaned:
                yield 0, cleaned, "ocr"

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
