"""Recovering source text from a byte-level BPE token dump.

Why this exists
---------------
A pre-tokenized corpus (``*.bin`` of uint16 token ids plus a HuggingFace-style
``vocab.json``/``merges.txt``) cannot be fed to this project's pipeline, which
needs raw text: normalization, quality filtering, deduplication, and split
assignment all operate on documents, not ids. Byte-level BPE is losslessly
invertible, so the original bytes can be recovered exactly and re-processed.

How byte-level BPE stores text
------------------------------
GPT-2-style tokenizers map each of the 256 byte values to a printable Unicode
character before training, so vocabulary entries are strings even when they
represent arbitrary binary. Decoding reverses that: piece -> characters -> bytes
-> UTF-8 text. :func:`byte_decoder` rebuilds that mapping exactly as the original
encoder defined it.

Document boundaries
-------------------
A flat token dump has no document structure - it is one long stream, and the
original per-document provenance did not survive tokenization. Boundaries are
therefore recovered from the text itself, in priority order:

1. The separator token (``<s>``), where the dump used one between documents.
2. Blank lines in the decoded text - the conventional document break.
3. A hard character cap, so a stretch with neither marker cannot produce one
   unbounded document that the cleaner would reject whole.

Measured on the supplied Bengali dump: the separator appears only in the padded
head, while the dense body carries a newline token roughly every 129 tokens, so
in practice rule 2 does the work. Recovered documents are attributed to a single
synthetic source, because nothing finer is knowable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

#: Documents shorter than this after splitting are merged into the next one, so a
#: stray blank line does not shatter a document into fragments.
MIN_SEGMENT_CHARS = 200

#: Hard cap on a recovered document, in characters. Without a separator the stream
#: would otherwise become one enormous document that the cleaner rejects wholesale.
#: 20k is roughly a long news feature - large enough not to fragment real articles,
#: small enough that a capped segment is still a plausible document rather than an
#: arbitrary slab of concatenated text.
MAX_SEGMENT_CHARS = 20_000

#: Blank line = document break. The dense body of the supplied dump has no blank
#: lines at all, so `_RE_LINE_BREAK` is the practical fallback there.
_RE_DOC_BREAK = re.compile(r"\n\s*\n")
_RE_LINE_BREAK = re.compile(r"\n")


def byte_decoder() -> dict[str, int]:
    """Rebuild the GPT-2 byte-level BPE character-to-byte mapping.

    The encoder assigns every byte a printable Unicode code point: printable ASCII
    and Latin-1 ranges map to themselves, and the remaining 68 byte values are
    shifted into U+0100 and above. Inverting that table recovers raw bytes.

    Returns:
        Mapping from the encoder's characters back to byte values.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = printable[:]
    code_points = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            code_points.append(256 + shift)
            shift += 1
    return {chr(code): byte for byte, code in zip(byte_values, code_points)}


class ByteLevelBpeDecoder:
    """Decodes uint16 token ids back into UTF-8 text.

    Args:
        vocab_file: HuggingFace-style ``vocab.json`` mapping piece -> id.

    Raises:
        ValueError: If the vocabulary cannot be read or is empty.
    """

    def __init__(self, vocab_file: Path) -> None:
        """Load the vocabulary and build the id -> piece and char -> byte tables."""
        import json

        vocab_file = Path(vocab_file)
        with vocab_file.open("r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        if not vocab:
            raise ValueError(f"Vocabulary at {vocab_file} is empty")

        self.vocab_size = len(vocab)
        self._id_to_piece = {index: piece for piece, index in vocab.items()}
        self._byte_decoder = byte_decoder()
        # Special tokens carry no text. They matter twice over here: a dump may
        # contain long padding runs of them, and where one appears between real
        # tokens it is the only surviving marker of a document boundary.
        self.special_ids = frozenset(
            index for piece, index in vocab.items()
            if piece in {"<s>", "</s>", "<pad>", "<unk>", "<mask_token>", "<mask>"}
        )
        self.separator_id = vocab.get("<s>")

    def decode_ids(self, ids: np.ndarray, drop_ids: frozenset[int] | None = None) -> str:
        """Decode an array of token ids into text.

        Unknown ids and characters outside the byte table are skipped rather than
        raising: a single corrupt id should not abort the decoding of a 500M-token
        stream, and the cleaning stage downstream will reject anything malformed.

        Args:
            ids: Token ids to decode.
            drop_ids: Ids to discard before decoding - the special tokens. These
                carry no text, and a dump can contain long runs of them as padding,
                which would otherwise decode into meaningless ``<s><s><s>...``.
        """
        drop = drop_ids if drop_ids is not None else frozenset()
        pieces = []
        for token_id in ids:
            value = int(token_id)
            if value in drop:
                continue
            piece = self._id_to_piece.get(value)
            if piece is not None:
                pieces.append(piece)
        joined = "".join(pieces)
        raw = bytes(self._byte_decoder[ch] for ch in joined if ch in self._byte_decoder)
        return raw.decode("utf-8", errors="replace")

    def stream_documents(
        self,
        bin_file: Path,
        chunk_tokens: int = 4_000_000,
        limit_tokens: int | None = None,
    ) -> Iterator[str]:
        """Decode a token dump and yield recovered documents.

        The file is read in chunks so a multi-gigabyte dump never needs to be held
        in memory. Because a chunk boundary can fall inside a document, the tail of
        each chunk is carried forward and prepended to the next.

        Args:
            bin_file: Path to the uint16 token dump.
            chunk_tokens: Tokens decoded per chunk.
            limit_tokens: Stop after this many tokens; None reads the whole file.

        Yields:
            Recovered document strings.
        """
        bin_file = Path(bin_file)
        total_tokens = bin_file.stat().st_size // 2
        if limit_tokens is not None:
            total_tokens = min(total_tokens, limit_tokens)

        carry = ""
        offset = 0
        while offset < total_tokens:
            count = min(chunk_tokens, total_tokens - offset)
            ids = np.fromfile(bin_file, dtype=np.uint16, count=count, offset=offset * 2)
            if ids.size == 0:
                break
            offset += ids.size

            # Split on the separator token first: in a dump without blank lines it
            # is the only real record of where one document ended and the next
            # began. Blank-line splitting then subdivides what remains.
            segments = []
            if self.separator_id is not None:
                boundaries = np.flatnonzero(ids == self.separator_id)
                pieces_of_ids = np.split(ids, boundaries) if boundaries.size else [ids]
            else:
                pieces_of_ids = [ids]

            decoded = [self.decode_ids(part, self.special_ids) for part in pieces_of_ids]
            decoded = [d for d in decoded if d.strip()]
            if decoded:
                decoded[0] = carry + decoded[0]
            elif carry:
                decoded = [carry]

            for block in decoded:
                parts = _RE_DOC_BREAK.split(block)
                # When a block carries no blank line it is a run of concatenated
                # documents; fall back to single newlines so the merge step below
                # can regroup them into reasonable document-sized units.
                if len(parts) == 1 and len(block) > MAX_SEGMENT_CHARS:
                    parts = _RE_LINE_BREAK.split(block)
                segments.extend(parts)

            # The final segment may be cut off mid-document by the chunk boundary,
            # so hold it back and let the next chunk complete it. On the last chunk
            # there is nothing following, so it is emitted as-is.
            carry = segments.pop() if offset < total_tokens else ""

            yield from _merge_short_segments(segments)

        if carry.strip():
            yield from _merge_short_segments([carry])


def _merge_short_segments(segments: list[str]) -> Iterator[str]:
    """Coalesce very short fragments and split over-long ones.

    Blank-line splitting is imperfect: headlines and datelines become their own
    tiny segments, while text with no blank lines produces enormous ones. This
    merges anything under `MIN_SEGMENT_CHARS` into the following segment and
    splits anything over `MAX_SEGMENT_CHARS` at line boundaries.
    """
    buffer = ""
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        buffer = f"{buffer}\n\n{segment}" if buffer else segment
        if len(buffer) >= MIN_SEGMENT_CHARS:
            if len(buffer) <= MAX_SEGMENT_CHARS:
                yield buffer
            else:
                yield from _split_long(buffer)
            buffer = ""
    if buffer:
        yield buffer


def _split_long(text: str) -> Iterator[str]:
    """Split an over-long block at line boundaries into bounded chunks."""
    current: list[str] = []
    length = 0
    for line in text.split("\n"):
        if length + len(line) > MAX_SEGMENT_CHARS and current:
            yield "\n".join(current)
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        yield "\n".join(current)
