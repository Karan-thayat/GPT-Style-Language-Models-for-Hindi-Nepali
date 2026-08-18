"""Corpus sources: public datasets and manually collected text.

Every source yields :class:`Document` objects carrying provenance, so that the
dataset statistics can report the manual-vs-downloaded token split the project
requires. Sources are declared in a YAML/JSON manifest per language rather than
hard-coded, which keeps the "add your own sources" workflow a config change.

Two families of source exist:

``downloaded``
    Public corpora streamed from the Hugging Face hub (used as *data*, not as a
    pretrained model - the project only forbids pretrained models/tokenizers).

``manual``
    Text the student collects: site scrapes and OCR of books/PDFs. These are
    what satisfy the >=20% manual-collection requirement.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

_RE_TAG = re.compile(r"<[^>]+>")


@dataclass
class Document:
    """One raw document with the provenance needed for dataset statistics.

    Attributes:
        text: Raw (un-normalized) document text.
        source_id: Manifest identifier of the source that produced it.
        collection: Either ``"downloaded"`` or ``"manual"``.
        url: Origin URL or file path, when known.
    """

    text: str
    source_id: str
    collection: str
    url: str | None = None

    def to_json(self) -> str:
        """Serialize to a single JSONL line."""
        return json.dumps(
            {"text": self.text, "source_id": self.source_id, "collection": self.collection, "url": self.url},
            ensure_ascii=False,
        )


class DocumentSource(Protocol):
    """Anything that can stream :class:`Document` objects."""

    source_id: str
    collection: str

    def stream(self, limit: int | None = None) -> Iterator[Document]:
        """Yield documents, stopping after `limit` if given."""
        ...


@dataclass
class HuggingFaceSource:
    """Streams a text field out of a Hugging Face dataset.

    Streaming mode is used throughout so that a 500M-token corpus never needs to
    be materialized on disk before filtering.

    Attributes:
        source_id: Manifest identifier.
        dataset: Hub dataset path, e.g. ``"wikimedia/wikipedia"``.
        config: Dataset configuration name, e.g. ``"20231101.bn"``.
        data_files: Explicit file glob(s) within the repo. Needed for datasets
            that organize languages as directories rather than configs - notably
            AI4Bharat Sangraha, whose only configs are ``verified``/``unverified``
            /``synthetic`` with the language as a subdirectory, so selecting a
            language requires a path pattern such as ``verified/ben/*.parquet``.
        split: Split to read.
        text_field: Field holding the document text.
        collection: Always ``"downloaded"``.
        trust_remote_code: Passed through for datasets with loading scripts.
    """

    source_id: str
    dataset: str
    config: str | None = None
    data_files: str | list[str] | None = None
    split: str = "train"
    text_field: str = "text"
    collection: str = "downloaded"
    trust_remote_code: bool = False

    def stream(self, limit: int | None = None) -> Iterator[Document]:
        """Yield documents from the hub dataset in streaming mode."""
        from datasets import load_dataset

        kwargs: dict = {"split": self.split, "streaming": True}
        # `data_files` selects a language directory directly; when it is given the
        # config name must be omitted, since the two are mutually exclusive ways
        # of choosing what to load.
        if self.data_files:
            kwargs["data_files"] = self.data_files
        elif self.config:
            kwargs["name"] = self.config
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True

        dataset = load_dataset(self.dataset, **kwargs)
        for index, row in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            text = row.get(self.text_field)
            if isinstance(text, str) and text.strip():
                yield Document(text=text, source_id=self.source_id, collection=self.collection, url=row.get("url"))


@dataclass
class LocalFileSource:
    """Reads documents from local files - the drop-in point for OCR output.

    Supported layouts:

    * ``.jsonl`` / ``.jsonl.gz`` - one JSON object per line with a text field.
    * ``.txt`` - the whole file is one document by default. Set
      `split_on_blank_lines` for blank-line-separated blocks (typical of OCR page
      dumps), or `one_document_per_line` for large corpus files that put one
      document on each line.

    `one_document_per_line` streams rather than reading the file whole, which is
    what makes multi-gigabyte corpora usable: reading a 5.9 GB file into memory is
    not an option on a machine with 7 GB of RAM.

    This is how OCR'd books enter the pipeline: run OCR with any tool, write the
    text out here, and point the manifest at the directory.

    Attributes:
        source_id: Manifest identifier.
        path: File or directory to read.
        glob: Pattern used when `path` is a directory.
        text_field: JSON field holding text, for JSONL inputs.
        collection: Usually ``"manual"`` for OCR/typed text.
        split_on_blank_lines: Treat blank-line-separated blocks in ``.txt`` files
            as separate documents (typical for OCR page dumps).
    """

    source_id: str
    path: str
    glob: str = "**/*"
    text_field: str = "text"
    collection: str = "manual"
    split_on_blank_lines: bool = False
    #: Treat each line of a ``.txt`` file as one document, streaming rather than
    #: loading the file. Required for large corpus dumps; see the class docstring.
    one_document_per_line: bool = False
    #: Skip lines shorter than this before they reach the cleaner. Cheap guard
    #: against the stray blank or truncated lines common in bulk corpus files.
    min_line_chars: int = 50

    def _iter_paths(self) -> Iterator[Path]:
        """Yield the concrete files this source covers."""
        root = Path(self.path).expanduser()
        if root.is_file():
            yield root
            return
        if not root.exists():
            return
        for candidate in sorted(root.glob(self.glob)):
            if candidate.is_file() and candidate.suffix in {".txt", ".jsonl", ".gz", ".json"}:
                yield candidate

    def stream(self, limit: int | None = None) -> Iterator[Document]:
        """Yield documents from every matching local file."""
        emitted = 0
        for file_path in self._iter_paths():
            opener = gzip.open if file_path.suffix == ".gz" else open
            with opener(file_path, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore[operator]
                if file_path.name.endswith((".jsonl", ".jsonl.gz", ".json.gz")) or file_path.suffix == ".jsonl":
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = row.get(self.text_field)
                        if isinstance(text, str) and text.strip():
                            yield Document(text, self.source_id, self.collection, row.get("url") or str(file_path))
                            emitted += 1
                            if limit is not None and emitted >= limit:
                                return
                elif self.one_document_per_line:
                    # Streaming path: never materialize the file. A 5.9 GB corpus
                    # read whole would exhaust memory long before any document
                    # reached the cleaner.
                    for line in handle:
                        line = line.strip()
                        if len(line) < self.min_line_chars:
                            continue
                        yield Document(line, self.source_id, self.collection, str(file_path))
                        emitted += 1
                        if limit is not None and emitted >= limit:
                            return
                else:
                    content = handle.read()
                    blocks = re.split(r"\n\s*\n", content) if self.split_on_blank_lines else [content]
                    for block in blocks:
                        if block.strip():
                            yield Document(block, self.source_id, self.collection, str(file_path))
                            emitted += 1
                            if limit is not None and emitted >= limit:
                                return


@dataclass
class WebScrapeSource:
    """Politely crawls a site and extracts article text - the manual-collection path.

    Deliberately conservative: it honours ``robots.txt``, rate-limits requests,
    stays on the configured host, and caps how many pages it fetches. The point
    is to collect text the student can account for, not to mirror a site.

    Attributes:
        source_id: Manifest identifier.
        start_urls: Seed pages to crawl from.
        allow_patterns: Regexes an URL must match to be followed.
        deny_patterns: Regexes that exclude an URL.
        content_selector: CSS selector for the article body. When None, a
            readability-style heuristic picks the densest text block.
        max_pages: Hard cap on fetched pages.
        delay_seconds: Minimum delay between requests to one host.
        max_depth: Link-following depth from the seeds.
        user_agent: Sent on every request so the crawl is identifiable.
        cache_dir: When set, raw HTML is cached so re-runs do not re-fetch.
    """

    source_id: str
    start_urls: list[str]
    allow_patterns: list[str] = field(default_factory=list)
    deny_patterns: list[str] = field(default_factory=list)
    content_selector: str | None = None
    max_pages: int = 500
    delay_seconds: float = 1.0
    max_depth: int = 2
    collection: str = "manual"
    user_agent: str = "LMA-Individual-Project/1.0 (student coursework crawler; contact via course HackMD)"
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        """Compile URL patterns and prepare the robots.txt cache."""
        self._allow = [re.compile(p) for p in self.allow_patterns]
        self._deny = [re.compile(p) for p in self.deny_patterns]
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request: dict[str, float] = {}

    def _robots_allows(self, url: str, session) -> bool:
        """Check `url` against its host's robots.txt, fetching it once per host."""
        host = urlparse(url).netloc
        if host not in self._robots:
            parser = RobotFileParser()
            robots_url = urljoin(f"{urlparse(url).scheme}://{host}", "/robots.txt")
            try:
                response = session.get(robots_url, timeout=15)
                parser.parse(response.text.splitlines() if response.ok else [])
            except Exception:
                # Unreachable robots.txt: treat as disallowed rather than assume consent.
                parser.parse(["User-agent: *", "Disallow: /"])
            self._robots[host] = parser
        return self._robots[host].can_fetch(self.user_agent, url)

    def _throttle(self, url: str) -> None:
        """Sleep so that requests to one host stay `delay_seconds` apart."""
        host = urlparse(url).netloc
        elapsed = time.monotonic() - self._last_request.get(host, 0.0)
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request[host] = time.monotonic()

    def _should_follow(self, url: str, seed_hosts: set[str]) -> bool:
        """Decide whether a discovered URL is in scope for this crawl."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.netloc not in seed_hosts:
            return False
        if any(pattern.search(url) for pattern in self._deny):
            return False
        if self._allow and not any(pattern.search(url) for pattern in self._allow):
            return False
        return True

    def _cache_path(self, url: str) -> Path | None:
        """Return the on-disk cache location for `url`, if caching is enabled."""
        if not self.cache_dir:
            return None
        import hashlib

        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        path = Path(self.cache_dir).expanduser() / self.source_id
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{digest}.html.gz"

    @staticmethod
    def _extract_text(html: str, selector: str | None) -> str:
        """Pull readable article text out of an HTML page.

        With a selector, only matching nodes are used. Without one, the densest
        block-level container wins, which handles most news and wiki layouts.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "figure"]):
            tag.decompose()

        if selector:
            nodes = soup.select(selector)
            if nodes:
                return "\n\n".join(node.get_text("\n", strip=True) for node in nodes)

        best_text = ""
        for node in soup.find_all(["article", "main", "div", "section"]):
            paragraphs = node.find_all("p", recursive=True)
            if len(paragraphs) < 3:
                continue
            text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs)
            if len(text) > len(best_text):
                best_text = text
        if best_text:
            return best_text
        return soup.get_text("\n", strip=True)

    def stream(self, limit: int | None = None) -> Iterator[Document]:
        """Crawl the configured site and yield one document per article page."""
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": self.user_agent})

        seed_hosts = {urlparse(u).netloc for u in self.start_urls}
        queue: list[tuple[str, int]] = [(u, 0) for u in self.start_urls]
        visited: set[str] = set()
        fetched = 0
        emitted = 0

        while queue and fetched < self.max_pages:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            cache_path = self._cache_path(url)
            html: str | None = None
            if cache_path is not None and cache_path.exists():
                with gzip.open(cache_path, "rt", encoding="utf-8", errors="replace") as handle:
                    html = handle.read()
            else:
                if not self._robots_allows(url, session):
                    continue
                self._throttle(url)
                try:
                    response = session.get(url, timeout=30)
                except Exception:
                    continue
                fetched += 1
                if not response.ok or "html" not in response.headers.get("Content-Type", ""):
                    continue
                response.encoding = response.encoding or "utf-8"
                html = response.text
                if cache_path is not None:
                    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
                        handle.write(html)

            if not html:
                continue

            text = self._extract_text(html, self.content_selector)
            if text.strip():
                yield Document(text=text, source_id=self.source_id, collection=self.collection, url=url)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

            if depth < self.max_depth:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "lxml")
                for anchor in soup.find_all("a", href=True):
                    candidate = urljoin(url, anchor["href"]).split("#")[0]
                    if candidate not in visited and self._should_follow(candidate, seed_hosts):
                        queue.append((candidate, depth + 1))


def build_source(spec: dict) -> DocumentSource:
    """Instantiate a source from one manifest entry.

    Args:
        spec: Manifest dict with a ``type`` key of ``huggingface``, ``local``, or
            ``scrape``, plus that class's fields.

    Returns:
        The constructed source.

    Raises:
        ValueError: On an unknown source type.
    """
    spec = dict(spec)
    kind = spec.pop("type")
    # Manifest-level bookkeeping keys, consumed by the loader rather than by the
    # source classes. `limit` in particular is applied by `iter_documents` when it
    # calls `stream()`, so it must not reach the constructor.
    for meta_key in ("enabled", "notes", "limit"):
        spec.pop(meta_key, None)
    if kind == "huggingface":
        return HuggingFaceSource(**spec)
    if kind == "local":
        return LocalFileSource(**spec)
    if kind == "scrape":
        return WebScrapeSource(**spec)
    raise ValueError(f"Unknown source type {kind!r}")


def load_manifest(path: Path) -> list[dict]:
    """Load a source manifest, returning only entries that are enabled.

    Args:
        path: Path to a JSON or YAML manifest with a top-level ``sources`` list.

    Returns:
        Enabled source specs in manifest order.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    return [entry for entry in payload.get("sources", []) if entry.get("enabled", True)]


def iter_documents(
    specs: Iterable[dict],
    per_source_limit: int | None = None,
    skip_positions: dict[str, int] | None = None,
    completed_sources: Iterable[str] = (),
    on_progress=None,
) -> Iterator[Document]:
    """Stream documents from every spec in turn, isolating per-source failures.

    A source that fails (network error, dataset moved) is reported and skipped so
    that one broken entry cannot abort a long ingest.

    Args:
        specs: Source manifest entries.
        per_source_limit: Profile-wide cap on documents per source.
        skip_positions: ``{source_id: n}`` - discard the first n documents of that
            source, used to resume a checkpointed run. This relies on sources
            yielding documents in a deterministic order, which holds for Hugging
            Face streaming and for the sorted local and scrape sources.
        completed_sources: Source ids already exhausted, skipped entirely.
        on_progress: Optional ``callable(source_id, documents_yielded)`` invoked per
            emitted document, so the caller can track per-source position for
            checkpointing.

    Yields:
        Documents from every source in manifest order.
    """
    skip_positions = skip_positions or {}
    completed = set(completed_sources)

    for spec in specs:
        source_id = spec.get("source_id", "<unnamed>")
        if source_id in completed:
            print(f"[sources] skipping already-completed source {source_id!r}")
            continue
        # Construction happens outside the try below on purpose. A spec that cannot
        # even be instantiated is a configuration error - a bad field name, a typo -
        # and it would produce an empty corpus for that source on every run. Those
        # must fail loudly and immediately. Only *runtime* failures (network, a
        # dataset that moved) are tolerated and skipped, since those are transient
        # and should not abort a long ingest.
        source = build_source(spec)
        # An explicit per-source `limit` in the manifest is a deliberate statement
        # about corpus composition (for example, capping downloaded sources so
        # manually collected text reaches its required share), so it wins over the
        # profile-wide cap. When both are present the stricter one applies.
        source_limit = spec.get("limit")
        if source_limit is not None and per_source_limit is not None:
            limit = min(source_limit, per_source_limit)
        else:
            limit = source_limit if source_limit is not None else per_source_limit
        skip = skip_positions.get(source_id, 0)
        if skip:
            print(f"[sources] resuming {source_id!r}: skipping first {skip:,} document(s)")

        # The limit is a cap on the source's *lifetime* position, not on documents
        # emitted in this segment. Adding the skipped prefix on top of it would let
        # each resume grant the source a fresh full quota, so an interrupted run
        # would end up with a larger corpus than an uninterrupted one - which is
        # exactly the kind of silent divergence checkpointing must not introduce.
        stream_limit = limit
        if limit is not None and skip >= limit:
            print(f"[sources] {source_id!r} already reached its limit of {limit:,}; skipping")
            if on_progress is not None:
                on_progress(source_id, -1)
            continue

        emitted = 0
        try:
            for index, document in enumerate(source.stream(limit=stream_limit)):
                if index < skip:
                    continue
                emitted += 1
                yield document
                if on_progress is not None:
                    on_progress(source_id, index + 1)
        except Exception as exc:  # noqa: BLE001 - one bad source must not kill the run
            print(f"[sources] WARNING: source {source_id!r} failed after {emitted:,} document(s): "
                  f"{type(exc).__name__}: {exc}")
        else:
            if on_progress is not None:
                # Signal exhaustion so a resume does not re-stream this source. A
                # source that raised is deliberately *not* marked complete, so the
                # next attempt retries it from its last position.
                on_progress(source_id, -1)
