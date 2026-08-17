"""Tokenizer evaluation: fertility, unknown rate, compression, and vocab selection.

Phase 1 asks for the vocabulary size to be *chosen* using fertility and
unknown-token rate on held-out text, so this module measures both and provides
the sweep that turns them into a decision.

Metric definitions used throughout
----------------------------------
**Fertility** - mean subword tokens per whitespace word. 1.0 means every word is
one token; higher means more fragmentation. It is the standard measure of how
well a vocabulary fits a language's morphology, and it is what makes the
resource-tier comparison meaningful: for a fixed vocabulary size, the
lower-resource language typically shows higher fertility because its corpus is
too small to learn good pieces for its productive morphology.

**Unknown rate** - share of tokens emitted as ``<unk>``. With byte fallback this
is near zero by construction, so the honest companion metric is the
*byte-fallback rate*: how often the tokenizer had to spell a character out in raw
bytes, which is the real signal of vocabulary coverage failure.

**Compression** - characters per token and bytes per token. Bytes per token is
what determines Phase 2's bits-per-byte, so it is recorded here for continuity.

All metrics are computed on held-out (validation) text, never on the training
split the vocabulary was fitted to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece as spm

from ..tokenizer.train_tokenizer import TokenizerConfig, TokenizerTrainer, train_from_jsonl

_RE_WORD = re.compile(r"[^\s]+")
#: SentencePiece renders byte-fallback pieces as <0xNN>.
_RE_BYTE_PIECE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


@dataclass
class TokenizerMetrics:
    """Held-out evaluation metrics for one tokenizer.

    Attributes:
        language: ISO code.
        model_type: ``"unigram"`` or ``"bpe"``.
        vocab_size: Configured vocabulary size.
        documents: Documents evaluated.
        words: Whitespace words seen.
        tokens: Subword tokens produced.
        characters: Characters in the evaluated text.
        utf8_bytes: UTF-8 byte length of the evaluated text.
        unk_tokens: Tokens emitted as ``<unk>``.
        byte_fallback_tokens: Tokens emitted as ``<0xNN>`` byte pieces.
        whitespace_byte_tokens: Byte-fallback tokens that encode whitespace
            (overwhelmingly ``<0x0A>``, the newline). Tracked separately because
            they say nothing about how well the vocabulary covers the language.
        single_token_words: Words encoded as exactly one token.
    """

    language: str
    model_type: str
    vocab_size: int
    documents: int = 0
    words: int = 0
    tokens: int = 0
    characters: int = 0
    utf8_bytes: int = 0
    unk_tokens: int = 0
    byte_fallback_tokens: int = 0
    whitespace_byte_tokens: int = 0
    single_token_words: int = 0

    @property
    def fertility(self) -> float:
        """Mean subword tokens per whitespace word."""
        return self.tokens / self.words if self.words else 0.0

    @property
    def unk_rate(self) -> float:
        """Share of produced tokens that are ``<unk>``."""
        return self.unk_tokens / self.tokens if self.tokens else 0.0

    @property
    def byte_fallback_rate(self) -> float:
        """Share of produced tokens that are raw byte pieces, including whitespace."""
        return self.byte_fallback_tokens / self.tokens if self.tokens else 0.0

    @property
    def script_byte_fallback_rate(self) -> float:
        """Byte-fallback rate excluding whitespace bytes - the real coverage metric.

        SentencePiece is trained on one-line-per-sentence input, so it never learns
        a piece for the newline character; every line break in a multi-line document
        therefore costs one ``<0x0A>`` byte token. Measured on the Assamese corpus,
        that artifact accounted for ~85% of all byte-fallback tokens (3.58% of
        tokens, against 0.65% genuine coverage failures). Including it would make a
        formatting property of the corpus look like a vocabulary deficiency, and
        would make the metric grow with document length rather than with how much of
        the language the vocabulary actually covers.
        """
        script_bytes = self.byte_fallback_tokens - self.whitespace_byte_tokens
        return script_bytes / self.tokens if self.tokens else 0.0

    @property
    def chars_per_token(self) -> float:
        """Mean characters represented by one token - the compression rate."""
        return self.characters / self.tokens if self.tokens else 0.0

    @property
    def bytes_per_token(self) -> float:
        """Mean UTF-8 bytes per token; feeds Phase 2's bits-per-byte."""
        return self.utf8_bytes / self.tokens if self.tokens else 0.0

    @property
    def single_token_word_rate(self) -> float:
        """Share of words that need no splitting at all."""
        return self.single_token_words / self.words if self.words else 0.0

    def as_dict(self) -> dict:
        """Return raw counters plus every derived metric."""
        payload = asdict(self)
        payload.update(
            {
                "fertility": self.fertility,
                "unk_rate": self.unk_rate,
                "byte_fallback_rate": self.byte_fallback_rate,
                "script_byte_fallback_rate": self.script_byte_fallback_rate,
                "chars_per_token": self.chars_per_token,
                "bytes_per_token": self.bytes_per_token,
                "single_token_word_rate": self.single_token_word_rate,
            }
        )
        return payload


class TokenizerEvaluator:
    """Measures a trained tokenizer on held-out text.

    Args:
        model_file: Path to the SentencePiece ``.model``.
        language: ISO code, recorded in the metrics.
    """

    def __init__(self, model_file: Path, language: str) -> None:
        """Load the tokenizer and cache its byte-piece and unk ids."""
        self.model_file = Path(model_file)
        self.language = language
        self.processor = TokenizerTrainer.load(self.model_file)
        self.vocab_size = self.processor.get_piece_size()
        self.unk_id = self.processor.unk_id()
        self._byte_ids = frozenset(
            piece_id
            for piece_id in range(self.vocab_size)
            if _RE_BYTE_PIECE.match(self.processor.id_to_piece(piece_id))
        )
        # Byte pieces that encode whitespace - almost entirely <0x0A>, the newline.
        # Separated so coverage can be measured without the line-break artifact.
        self._whitespace_byte_ids = frozenset(
            piece_id
            for piece_id in self._byte_ids
            if chr(int(self.processor.id_to_piece(piece_id)[3:5], 16)).isspace()
        )
        self.model_type = self._infer_model_type()

    def _infer_model_type(self) -> str:
        """Recover the model type from the filename, defaulting to unigram."""
        name = self.model_file.stem.lower()
        return "bpe" if "bpe" in name else "unigram"

    def evaluate(self, texts: Iterable[str]) -> TokenizerMetrics:
        """Compute held-out metrics over a stream of documents.

        Args:
            texts: Held-out document texts.

        Returns:
            Populated :class:`TokenizerMetrics`.
        """
        metrics = TokenizerMetrics(
            language=self.language, model_type=self.model_type, vocab_size=self.vocab_size
        )

        for text in texts:
            if not text.strip():
                continue
            metrics.documents += 1
            metrics.characters += len(text)
            metrics.utf8_bytes += len(text.encode("utf-8"))

            words = _RE_WORD.findall(text)
            metrics.words += len(words)

            ids = self.processor.encode(text, out_type=int)
            metrics.tokens += len(ids)
            metrics.unk_tokens += sum(1 for i in ids if i == self.unk_id)
            metrics.byte_fallback_tokens += sum(1 for i in ids if i in self._byte_ids)
            metrics.whitespace_byte_tokens += sum(1 for i in ids if i in self._whitespace_byte_ids)

            # Per-word encoding is needed for the single-token-word rate; it is
            # sampled rather than exhaustive on long documents to bound cost.
            sample = words if len(words) <= 400 else words[:400]
            if sample:
                encoded = self.processor.encode(sample, out_type=int)
                hits = sum(1 for piece_ids in encoded if len(piece_ids) == 1)
                metrics.single_token_words += round(hits * len(words) / len(sample))

        return metrics

    def examples(self, texts: Sequence[str], max_examples: int = 12) -> list[dict]:
        """Produce tokenization examples for the report.

        Args:
            texts: Candidate sentences.
            max_examples: How many to include.

        Returns:
            A list of dicts with the text, its pieces, ids, and per-example counts.
        """
        results = []
        for text in texts[:max_examples]:
            snippet = " ".join(text.split())[:220]
            if not snippet:
                continue
            pieces = self.processor.encode(snippet, out_type=str)
            ids = self.processor.encode(snippet, out_type=int)
            words = len(_RE_WORD.findall(snippet))
            results.append(
                {
                    "text": snippet,
                    "pieces": pieces,
                    "ids": ids,
                    "n_words": words,
                    "n_tokens": len(pieces),
                    "fertility": len(pieces) / words if words else 0.0,
                    "roundtrip_ok": self.processor.decode(ids).strip() == snippet.strip(),
                }
            )
        return results

    def piece_statistics(self) -> dict:
        """Summarize the vocabulary's composition and piece-length profile."""
        lengths: list[int] = []
        script_pieces = 0
        byte_pieces = 0
        latin_pieces = 0
        whole_word_pieces = 0

        for piece_id in range(self.vocab_size):
            piece = self.processor.id_to_piece(piece_id)
            if _RE_BYTE_PIECE.match(piece):
                byte_pieces += 1
                continue
            body = piece.lstrip("▁")  # SentencePiece word-boundary marker
            if piece.startswith("▁"):
                whole_word_pieces += 1
            lengths.append(len(body))
            if any("ঀ" <= ch <= "৿" for ch in body):
                script_pieces += 1
            if any("a" <= ch.lower() <= "z" for ch in body):
                latin_pieces += 1

        mean_length = sum(lengths) / len(lengths) if lengths else 0.0
        histogram: dict[str, int] = {}
        for length in lengths:
            key = str(length) if length <= 9 else "10+"
            histogram[key] = histogram.get(key, 0) + 1

        return {
            "vocab_size": self.vocab_size,
            "byte_fallback_pieces": byte_pieces,
            "bengali_script_pieces": script_pieces,
            "latin_pieces": latin_pieces,
            "word_initial_pieces": whole_word_pieces,
            "mean_piece_length_chars": mean_length,
            "piece_length_histogram": dict(sorted(histogram.items(), key=lambda kv: (len(kv[0]), kv[0]))),
        }

    def token_frequency(self, texts: Iterable[str], top_k: int = 60) -> dict:
        """Measure token-frequency statistics on held-out text.

        Returns:
            Top-`top_k` tokens with counts and shares, plus coverage figures that
            show how concentrated the distribution is.
        """
        from collections import Counter

        counter: Counter = Counter()
        for text in texts:
            counter.update(self.processor.encode(text, out_type=str))

        total = sum(counter.values())
        if total == 0:
            return {"total_tokens": 0, "distinct_tokens": 0, "top_tokens": [], "coverage": {}}

        ordered = counter.most_common()
        cumulative = 0
        coverage: dict[str, float] = {}
        checkpoints = {100, 500, 1000, 5000, 10000}
        for index, (_, count) in enumerate(ordered, start=1):
            cumulative += count
            if index in checkpoints:
                coverage[f"top_{index}"] = cumulative / total

        return {
            "total_tokens": total,
            "distinct_tokens": len(counter),
            "vocab_utilization": len(counter) / self.vocab_size,
            "top_tokens": [
                {"piece": piece, "count": count, "share": count / total} for piece, count in ordered[:top_k]
            ],
            "coverage": coverage,
        }


def sweep_vocabulary_sizes(
    train_files: Sequence[Path],
    heldout_texts: Sequence[str],
    language: str,
    vocab_sizes: Sequence[int],
    model_types: Sequence[str] = ("unigram", "bpe"),
    output_dir: Path | None = None,
    max_train_lines: int | None = 400_000,
) -> list[dict]:
    """Train candidate tokenizers and measure each on held-out text.

    This is the experiment that justifies the final vocabulary choice: it trains
    one tokenizer per (model_type, vocab_size) pair on the training split and
    evaluates all of them on the same held-out text.

    Args:
        train_files: Training-split JSONL files.
        heldout_texts: Held-out documents for evaluation.
        language: ISO code.
        vocab_sizes: Vocabulary sizes to try.
        model_types: Algorithms to try.
        output_dir: Where candidate models are written. A temporary directory is
            used when None.
        max_train_lines: Cap on training lines, to keep the sweep affordable.

    Returns:
        One metrics dict per candidate, in the order trained.
    """
    import shutil
    import tempfile

    temp_dir = None
    if output_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="spm_sweep_"))
        output_dir = temp_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    try:
        for model_type in model_types:
            for vocab_size in vocab_sizes:
                config = TokenizerConfig(language=language, model_type=model_type, vocab_size=vocab_size)
                try:
                    manifest = train_from_jsonl(train_files, config, output_dir, max_lines=max_train_lines)
                except Exception as exc:  # noqa: BLE001 - a too-large vocab for a small corpus is expected
                    results.append(
                        {
                            "language": language,
                            "model_type": model_type,
                            "vocab_size": vocab_size,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                evaluator = TokenizerEvaluator(Path(manifest["model_file"]), language)
                metrics = evaluator.evaluate(heldout_texts).as_dict()
                metrics["requested_vocab_size"] = vocab_size
                metrics["model_file"] = manifest["model_file"]
                results.append(metrics)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def select_best_config(
    sweep_results: Sequence[dict],
    max_unk_rate: float = 1e-4,
    model_dim: int = 384,
    param_budget: int = 25_000_000,
    max_embedding_share: float = 0.40,
) -> dict | None:
    """Pick the best candidate from a sweep, trading fertility against parameters.

    Why this is not simply "lowest fertility"
    -----------------------------------------
    Fertility falls monotonically with vocabulary size, so minimizing it alone
    always selects the largest candidate swept - the sweep would decide nothing.
    The real trade-off is against parameter cost: with tied input/output
    embeddings, the embedding matrix holds ``vocab_size x model_dim`` parameters,
    and Phase 2's budget is a fixed ~25M. At d_model=384, a 32k vocabulary spends
    12.3M parameters - roughly half the entire budget - on embeddings alone,
    leaving too little for the Transformer blocks that actually do the modelling.

    Selection rule, in priority order:

    1. Discard candidates that failed to train, exceed `max_unk_rate`, or whose
       embedding matrix would exceed `max_embedding_share` of the budget.
    2. Among the rest, maximize fertility improvement per parameter spent, by
       taking the candidate with the best fertility subject to the parameter cap,
       then stepping *down* to any smaller vocabulary within 3% fertility of it.

    The 3% band encodes the judgement that a few percent of sequence length is not
    worth millions of parameters at this scale.

    Args:
        sweep_results: Output of :func:`sweep_vocabulary_sizes`.
        max_unk_rate: Hard ceiling on held-out unknown rate.
        model_dim: Planned Phase 2 model dimension, for the embedding-cost estimate.
        param_budget: Planned total parameter budget.
        max_embedding_share: Largest share of the budget embeddings may consume.

    Returns:
        The winning result dict, or None when no candidate is viable.
    """
    max_embedding_params = param_budget * max_embedding_share

    viable = [
        result
        for result in sweep_results
        if "error" not in result
        and result.get("unk_rate", 1.0) <= max_unk_rate
        and result.get("tokens", 0) > 0
        and result["vocab_size"] * model_dim <= max_embedding_params
    ]
    if not viable:
        # Every candidate busts the parameter cap; fall back to unk-rate viability
        # alone so the sweep still returns something rather than silently failing.
        viable = [
            result
            for result in sweep_results
            if "error" not in result and result.get("unk_rate", 1.0) <= max_unk_rate and result.get("tokens", 0) > 0
        ]
    if not viable:
        return None

    best = min(viable, key=lambda r: r["fertility"])
    threshold = best["fertility"] * 1.03
    near_ties = [r for r in viable if r["fertility"] <= threshold]
    winner = min(near_ties, key=lambda r: (r["vocab_size"], r["fertility"]))
    winner = dict(winner)
    winner["embedding_params"] = winner["vocab_size"] * model_dim
    winner["embedding_share_of_budget"] = winner["embedding_params"] / param_budget
    return winner
