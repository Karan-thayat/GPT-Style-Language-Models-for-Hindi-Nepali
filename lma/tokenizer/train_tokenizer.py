"""Training subword tokenizers from scratch with SentencePiece.

Phase 1 forbids pretrained tokenizers, so every vocabulary here is fitted on the
language's own training split only. The two languages get entirely separate
tokenizers and vocabularies - nothing is shared, not even the special-token ids
by coincidence of a shared file.

Model choice
------------
Both BPE and Unigram are supported behind one interface so the choice can be
made empirically (see :mod:`lma.stats.tokenizer_stats`) rather than by assertion.
Unigram is the usual better fit for agglutinative Indic morphology because its
pruning objective yields more linguistically coherent pieces, while BPE tends to
produce shorter sequences for a given vocabulary size. The selection script
reports fertility and unknown-token rate for both so the report can justify the
final pick.

Character coverage
------------------
Defaults to 0.9995 rather than 1.0. Bengali-script corpora contain a long tail
of stray Latin, CJK, and symbol characters; at full coverage each becomes its own
vocabulary entry, wasting slots. The remainder falls back to byte pieces, so no
input is ever unrepresentable.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator

import sentencepiece as spm

#: Special tokens and their fixed ids, identical in structure per language but
#: fitted on separate corpora. BOS/EOS are needed for Phase 2 sequence framing;
#: PAD keeps batches rectangular.
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<s>", "</s>"
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


@dataclass
class TokenizerConfig:
    """Hyperparameters for one tokenizer training run.

    Attributes:
        language: ISO code, used in output filenames.
        model_type: ``"unigram"`` or ``"bpe"``.
        vocab_size: Target vocabulary size, including special tokens.
        character_coverage: Fraction of training characters the vocabulary must
            cover before falling back to byte pieces.
        input_sentence_size: Cap on sentences sampled for training; SentencePiece
            holds these in memory, so this bounds RAM on a large corpus.
        shuffle_input_sentence: Sample uniformly across the corpus rather than
            taking a prefix, so one large source cannot dominate the vocabulary.
        max_sentence_length: Bytes; longer lines are split by SentencePiece.
        byte_fallback: Encode out-of-vocabulary characters as UTF-8 byte pieces so
            the unknown rate on held-out text is effectively zero.
        split_digits: Give each digit its own piece, which generalizes numbers.
        allow_whitespace_only_pieces: Keep pieces made only of whitespace.
        normalization_rule_name: SentencePiece's own normalizer. Set to
            ``identity`` because :mod:`lma.data.normalize` already normalized the
            corpus, and a second, script-unaware normalization pass risks folding
            the Assamese-specific letters.
        num_threads: Training parallelism.
        seed: Reproducibility seed.
        user_defined_symbols: Symbols forced into the vocabulary intact.
    """

    language: str
    model_type: str = "unigram"
    vocab_size: int = 32_000
    character_coverage: float = 0.9995
    input_sentence_size: int = 4_000_000
    shuffle_input_sentence: bool = True
    max_sentence_length: int = 16_384
    byte_fallback: bool = True
    split_digits: bool = True
    allow_whitespace_only_pieces: bool = True
    normalization_rule_name: str = "identity"
    num_threads: int = 8
    seed: int = 20260812
    user_defined_symbols: list[str] = field(default_factory=lambda: ["।", "॥"])

    def __post_init__(self) -> None:
        """Validate the model type."""
        if self.model_type not in {"unigram", "bpe"}:
            raise ValueError(f"model_type must be 'unigram' or 'bpe', got {self.model_type!r}")


class TokenizerTrainer:
    """Trains and loads a SentencePiece tokenizer for one language."""

    def __init__(self, config: TokenizerConfig) -> None:
        """Store the training configuration."""
        self.config = config

    def _model_prefix(self, output_dir: Path) -> Path:
        """Return the output path prefix for the trained model files."""
        cfg = self.config
        return Path(output_dir) / f"{cfg.language}_{cfg.model_type}_{cfg.vocab_size}"

    def train(self, corpus_files: Iterable[Path], output_dir: Path) -> dict:
        """Fit a tokenizer on the given corpus files.

        Args:
            corpus_files: Plain-text files, one sentence or document per line.
                These must come from the *training* split only.
            output_dir: Directory to write ``.model`` and ``.vocab`` into.

        Returns:
            A manifest dict with the resolved config and output paths.
        """
        cfg = self.config
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._model_prefix(output_dir)
        files = [str(Path(p)) for p in corpus_files]
        if not files:
            raise ValueError("No corpus files supplied for tokenizer training")

        # SentencePiece takes the RNG seed from the environment rather than from
        # TrainerSpec (there is no `random_seed` trainer field), and it is read at
        # trainer construction, so it must be set before the call below. This is
        # what makes sentence sampling - and therefore the vocabulary - reproducible.
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)
        spm.set_random_generator_seed(cfg.seed)

        spm.SentencePieceTrainer.train(
            input=",".join(files),
            model_prefix=str(prefix),
            model_type=cfg.model_type,
            vocab_size=cfg.vocab_size,
            character_coverage=cfg.character_coverage,
            input_sentence_size=cfg.input_sentence_size,
            shuffle_input_sentence=cfg.shuffle_input_sentence,
            max_sentence_length=cfg.max_sentence_length,
            byte_fallback=cfg.byte_fallback,
            split_digits=cfg.split_digits,
            allow_whitespace_only_pieces=cfg.allow_whitespace_only_pieces,
            normalization_rule_name=cfg.normalization_rule_name,
            num_threads=cfg.num_threads,
            train_extremely_large_corpus=False,
            pad_id=PAD_ID, pad_piece=PAD_TOKEN,
            unk_id=UNK_ID, unk_piece=UNK_TOKEN,
            bos_id=BOS_ID, bos_piece=BOS_TOKEN,
            eos_id=EOS_ID, eos_piece=EOS_TOKEN,
            user_defined_symbols=cfg.user_defined_symbols,
            minloglevel=1,
        )

        return {
            "config": asdict(cfg),
            "model_file": str(prefix.with_suffix(".model")),
            "vocab_file": str(prefix.with_suffix(".vocab")),
            "corpus_files": files,
        }

    @staticmethod
    def load(model_file: Path) -> spm.SentencePieceProcessor:
        """Load a trained tokenizer from its ``.model`` file."""
        processor = spm.SentencePieceProcessor()
        processor.load(str(model_file))
        return processor


def write_training_text(
    texts: Iterable[str],
    output_file: Path,
    max_lines: int | None = None,
    max_line_bytes: int = 12_000,
) -> int:
    """Flatten documents into the one-line-per-sentence file SentencePiece wants.

    Documents are split on newlines and long lines are chunked, because
    SentencePiece silently drops lines over ``max_sentence_length`` - which would
    quietly discard the longest documents in the corpus.

    The chunk limit is measured in **UTF-8 bytes, not characters**, because
    SentencePiece's limit is a byte limit. Bengali-script characters occupy three
    bytes each, so a character-based limit would be roughly 3x larger than
    intended and would still lose long lines. The default of 12,000 bytes leaves
    headroom under the trainer's 16,384-byte default.

    Args:
        texts: Document texts.
        output_file: Destination text file.
        max_lines: Stop after writing this many lines.
        max_line_bytes: Hard-wrap lines longer than this at a space boundary.

    Returns:
        Number of lines written.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_file.open("w", encoding="utf-8") as handle:
        for text in texts:
            for raw_line in text.split("\n"):
                line = " ".join(raw_line.split())
                if not line:
                    continue
                for chunk in _chunk_line(line, max_line_bytes):
                    handle.write(chunk + "\n")
                    written += 1
                    if max_lines is not None and written >= max_lines:
                        return written
    return written


def _chunk_line(line: str, byte_limit: int) -> Iterator[str]:
    """Split an over-long line at space boundaries into <= `byte_limit` byte chunks.

    A single word longer than the limit is emitted on its own: breaking mid-word
    would fabricate tokens that do not occur in the language, which is worse for a
    vocabulary than one over-long line that SentencePiece may skip.
    """
    if len(line.encode("utf-8")) <= byte_limit:
        yield line
        return

    current: list[str] = []
    length = 0
    for word in line.split(" "):
        word_bytes = len(word.encode("utf-8")) + 1  # +1 for the joining space
        if length + word_bytes > byte_limit and current:
            yield " ".join(current)
            current, length = [], 0
        current.append(word)
        length += word_bytes
    if current:
        yield " ".join(current)


def train_from_jsonl(
    jsonl_files: Iterable[Path],
    config: TokenizerConfig,
    output_dir: Path,
    max_lines: int | None = None,
) -> dict:
    """Convenience wrapper: JSONL training split in, trained tokenizer out.

    Args:
        jsonl_files: Training-split JSONL files.
        config: Tokenizer hyperparameters.
        output_dir: Where to write the model and vocabulary.
        max_lines: Cap on training lines extracted from the corpus.

    Returns:
        The training manifest, with the line count added.
    """
    from ..data.splits import iter_texts

    tmp_dir = Path(tempfile.mkdtemp(prefix="spm_corpus_"))
    try:
        corpus_txt = tmp_dir / f"{config.language}_spm_input.txt"
        lines = write_training_text(iter_texts(jsonl_files), corpus_txt, max_lines=max_lines)
        if lines == 0:
            raise ValueError("Tokenizer training corpus is empty")
        manifest = TokenizerTrainer(config).train([corpus_txt], output_dir)
        manifest["training_lines"] = lines
        manifest["corpus_files"] = [str(Path(p)) for p in jsonl_files]
        return manifest
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
