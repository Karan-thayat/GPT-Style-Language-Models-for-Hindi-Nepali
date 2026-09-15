#!/usr/bin/env python3
"""Component tests for the Phase 3 reasoning pipeline.

    python scripts/test_reasoning.py

Plain assertions with a counter, matching ``scripts/test_model.py`` - the project
has no test runner dependency, and adding one for Phase 3 alone would be
inconsistent. Every check here corresponds to a property the Phase 3 report
claims, so a claim cannot silently stop being true.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from lma.model import ModelConfig, TransformerLM  # noqa: E402
from lma.reasoning.evaluate import (  # noqa: E402
    above_chance,
    chance_baseline,
    compare_pretrained_finetuned,
    error_analysis,
    score_examples,
)
from lma.reasoning.finetune import (  # noqa: E402
    IGNORE_INDEX,
    FinetuneConfig,
    ReasoningDataset,
    ReasoningFinetuner,
    finetune_lr,
)
from lma.reasoning.generate import (  # noqa: E402
    GeneratorConfig,
    ReasoningDatasetBuilder,
    dataset_statistics,
)
from lma.reasoning.templates import TEMPLATES  # noqa: E402
from lma.tokenizer.train_tokenizer import BOS_ID, EOS_ID  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def tokenizer_for(language: str):
    """Load a language's Phase 1 tokenizer, or None when unavailable."""
    import sentencepiece as spm

    path = {
        "hi": REPO_ROOT / "hindi/tokenizer/hi_unigram_8000.model",
        "ne": REPO_ROOT / "nepali/tokenizer/ne_unigram_8000.model",
    }[language]
    if not path.exists():
        return None
    return spm.SentencePieceProcessor(model_file=str(path))


def test_templates() -> None:
    """Every answer must be a single token in its own language's tokenizer.

    This is the property the whole task format rests on: it makes the finetuning
    loss maskable to the answer span and exact match unambiguous. It is checked
    rather than assumed because an edit to the entity pools could break it
    silently - which happened during development, when four Hindi and ten Nepali
    hand-picked names turned out to fragment into multiple pieces.
    """
    print("\n[templates]")
    for language, templates in TEMPLATES.items():
        tokenizer = tokenizer_for(language)
        if tokenizer is None:
            print(f"  SKIP  {language}: tokenizer not present")
            continue

        fragmented = [
            name
            for name in templates.entities
            if len(tokenizer.encode(name)) != 1
        ]
        check(
            f"[{language}] every entity is a single token",
            not fragmented,
            f"{len(templates.entities)} names" if not fragmented else str(fragmented[:4]),
        )

        byte_fallback = [
            text
            for dimension in templates.dimensions
            for text in (dimension.premise_more, dimension.question_most,
                         dimension.question_least)
            if any(p.startswith("<0x") for p in tokenizer.encode(text, out_type=str))
        ]
        check(
            f"[{language}] no byte fallback in template strings",
            not byte_fallback,
            str(byte_fallback[:2]) if byte_fallback else "",
        )

        # The answer prefix is what the model conditions on before answering, so
        # it must tokenize cleanly too. "उत्तर:" does not - Nepali has no ":"
        # piece - which is why the templates use a dash.
        prefix_pieces = tokenizer.encode(templates.answer_prefix, out_type=str)
        check(
            f"[{language}] answer prefix has no byte fallback",
            not any(p.startswith("<0x") for p in prefix_pieces),
            str(prefix_pieces),
        )

        check(
            f"[{language}] entity pool has no duplicates",
            len(templates.entities) == len(set(templates.entities)),
        )


def test_generation() -> None:
    """The generated corpus must satisfy every leakage property claimed."""
    print("\n[generation]")
    for language in ("hi", "ne"):
        builder = ReasoningDatasetBuilder(
            GeneratorConfig(language=language, total_examples=4000)
        )
        examples = builder.build()
        stats = dataset_statistics(examples, builder)
        checks = stats["leakage_checks"]

        check(
            f"[{language}] no chain-signature overlap between train and eval",
            checks["chain_signature_overlap"] == 0,
            f"{checks['train_chain_signatures']} train structures",
        )
        check(
            f"[{language}] no prompt overlap between train and eval",
            checks["prompt_overlap_train_eval"] == 0,
        )
        check(f"[{language}] no duplicate prompts", checks["duplicate_prompts"] == 0)

        # The defect an earlier design had: an answer the model was never trained
        # to emit makes the example unanswerable rather than difficult.
        check(
            f"[{language}] no evaluation answer is a distractor",
            checks["distractor_answers_in_eval"] == 0,
        )
        check(
            f"[{language}] distractors never appear in training examples",
            not checks["distractors_in_training_examples"],
        )

        mismatched = [
            e
            for e in examples
            if e.answer != (e.entities[0] if e.ask_most else e.entities[-1])
        ]
        check(
            f"[{language}] answer matches the chain ordering",
            not mismatched,
            f"{len(examples)} examples",
        )

        # Shuffled premises: the answer must not be readable off the first word
        # more often than chance, or the task can be solved by copying.
        long_chains = [e for e in examples if e.chain_length == 4]
        if long_chains:
            copyable = sum(
                1 for e in long_chains if e.prompt.split()[0] == e.answer
            ) / len(long_chains)
            check(
                f"[{language}] answer is not copyable from the first word",
                copyable <= 0.25,
                f"{100 * copyable:.1f}% vs 25% chance",
            )

        check(
            f"[{language}] generation is reproducible from the seed",
            [e.text for e in examples]
            == [
                e.text
                for e in ReasoningDatasetBuilder(
                    GeneratorConfig(language=language, total_examples=4000)
                ).build()
            ],
        )


def test_dataset_masking() -> None:
    """Loss must be computed on the answer span only."""
    print("\n[answer-span masking]")
    tokenizer = tokenizer_for("hi")
    if tokenizer is None:
        print("  SKIP  tokenizer not present")
        return

    builder = ReasoningDatasetBuilder(
        GeneratorConfig(language="hi", total_examples=400)
    )
    rows = [e.to_dict() for e in builder.build()]
    dataset = ReasoningDataset(rows, tokenizer, max_length=64)

    check("examples tokenize without being skipped", dataset.skipped == 0,
          f"{len(dataset)} usable")

    statistics = dataset.statistics()
    check(
        "every answer is one token",
        statistics["answer_tokens"]["single_token_fraction"] == 1.0,
    )

    row = dataset.rows[0]
    check("sequence starts with BOS", row["input_ids"][0] == BOS_ID)
    check("sequence ends with EOS", row["input_ids"][-1] == EOS_ID)

    supervised = [label for label in row["labels"] if label != IGNORE_INDEX]
    check(
        "only the answer and EOS are supervised",
        len(supervised) == 2,
        f"{len(supervised)} of {len(row['labels'])} positions",
    )
    answer_id = tokenizer.encode(row["answer"], out_type=int)[0]
    check("the supervised token is the answer", supervised[0] == answer_id)

    # Padding must never contribute to the loss.
    padded_supervised = 0
    for x, y in dataset.batches(16, shuffle=False, seed=0, device="cpu"):
        padded_supervised += int(((y != IGNORE_INDEX) & (x == 0)).sum().item())
    check("padding is never supervised", padded_supervised == 0)


def test_finetune_loop() -> None:
    """Finetuning must reduce loss and resume to identical weights."""
    print("\n[finetuning]")
    import tempfile

    tokenizer = tokenizer_for("hi")
    if tokenizer is None:
        print("  SKIP  tokenizer not present")
        return

    builder = ReasoningDatasetBuilder(
        GeneratorConfig(language="hi", total_examples=1200)
    )
    rows = [e.to_dict() for e in builder.build()]
    train_rows = [r for r in rows if r["split"] == "train"][:400]
    val_rows = [r for r in rows if r["split"] == "validation"][:80]

    def make(out_dir: Path, epochs: int = 2) -> ReasoningFinetuner:
        torch.manual_seed(0)
        model = TransformerLM(
            ModelConfig(
                vocab_size=8000, n_layer=2, n_head=2, n_embd=64,
                block_size=64, dropout=0.0,
            )
        )
        config = FinetuneConfig(
            batch_size=16, epochs=epochs, eval_interval=1000, log_interval=1000,
            warmup_steps=5, amp=False, learning_rate=3e-4,
        )
        return ReasoningFinetuner(
            model,
            ReasoningDataset(train_rows, tokenizer),
            ReasoningDataset(val_rows, tokenizer),
            config,
            out_dir,
            device="cpu",
        )

    straight_dir = Path(tempfile.mkdtemp()) / "straight"
    straight = make(straight_dir)
    before = straight.evaluate()["val_loss"]
    straight.train(resume=False)
    after = straight.evaluate()["val_loss"]
    check("finetuning reduces validation loss", after < before,
          f"{before:.4f} -> {after:.4f}")

    # Both halves must be built with the SAME epoch count. `epochs` sets the
    # cosine decay horizon, so shortening it for the first half would compress
    # the whole schedule and produce genuinely different weights - a difference
    # in configuration, not a resume bug. The interruption is simulated by
    # stopping after one epoch's worth of steps instead.
    resume_dir = Path(tempfile.mkdtemp()) / "resumed"
    first = make(resume_dir, epochs=2)
    first.cfg.epochs = 1  # stop early; the schedule horizon stays at 2 epochs
    first.train(resume=False)
    first.cfg.epochs = 2
    second = make(resume_dir, epochs=2)
    second.train(resume=True)
    resumed = second.evaluate()["val_loss"]
    check(
        "an interrupted run resumes to the same weights",
        abs(after - resumed) < 1e-5,
        f"{after:.6f} vs {resumed:.6f}",
    )

    payload = torch.load(
        resume_dir / "finetuned_latest.pt", map_location="cpu", weights_only=False
    )
    check(
        "checkpoint carries weights, optimizer, step, and configs",
        all(
            key in payload
            for key in ("model", "optimizer", "step", "model_config", "finetune_config")
        ),
    )

    schedule = [finetune_lr(step, 100, FinetuneConfig(warmup_steps=10)) for step in (0, 10, 99)]
    check("learning rate warms up then decays",
          schedule[0] < schedule[1] and schedule[2] < schedule[1])


def test_scoring() -> None:
    """Scoring must be correct at its known extremes."""
    print("\n[scoring]")
    tokenizer = tokenizer_for("hi")
    if tokenizer is None:
        print("  SKIP  tokenizer not present")
        return

    builder = ReasoningDatasetBuilder(
        GeneratorConfig(language="hi", total_examples=600)
    )
    rows = [e.to_dict() for e in builder.build() if e.split == "test"][:60]

    torch.manual_seed(0)
    model = TransformerLM(
        ModelConfig(vocab_size=8000, n_layer=2, n_head=2, n_embd=64,
                    block_size=64, dropout=0.0)
    ).eval()
    results = score_examples(model, rows, tokenizer, device="cpu")

    check("every example is scored", results["examples"] == len(rows))
    check("exact match is a probability", 0.0 <= results["exact_match"] <= 1.0)
    check(
        "constrained accuracy is a probability",
        0.0 <= results["constrained_accuracy"] <= 1.0,
    )

    baseline = chance_baseline(rows)
    check(
        "chance for a two-entity comparison is one half",
        abs(baseline["by_chain_length"].get("2", 0.5) - 0.5) < 1e-9,
    )

    margins = above_chance(results, baseline)
    check(
        "per-length margins cover every length present",
        set(margins["by_chain_length"]) == set(results["by_chain_length"]),
    )

    identical = compare_pretrained_finetuned(results, results)
    check(
        "comparing a result with itself yields zero deltas",
        abs(identical["exact_match"]["delta"]) < 1e-12,
    )

    errors = error_analysis(results, rows)
    check(
        "error categories sum to the errors examined",
        sum(errors["counts"].values()) == errors["errors_examined"],
    )


def test_error_coverage() -> None:
    """The error analysis must count every mistake, not a stored sample.

    A regression test for a real defect: errors were once categorized from the
    first 40 stored predictions only, so a near-perfect model whose few mistakes
    fell outside that window reported an error profile of all zeros. An untrained
    model gets nearly every one of 60 examples wrong, so more errors exist than
    the prediction sample holds, and every one must be counted.
    """
    print("\n[error coverage]")
    tokenizer = tokenizer_for("hi")
    if tokenizer is None:
        print("  SKIP  tokenizer not present")
        return

    rows = [
        e.to_dict()
        for e in ReasoningDatasetBuilder(GeneratorConfig(language="hi", total_examples=900)).build()
        if e.split == "test"
    ][:60]
    torch.manual_seed(0)
    model = TransformerLM(
        ModelConfig(vocab_size=8000, n_layer=2, n_head=2, n_embd=64, block_size=64, dropout=0.0)
    ).eval()
    results = score_examples(model, rows, tokenizer, device="cpu")
    errors = error_analysis(results, rows)

    check("more errors than the stored prediction sample", results["errors"] > 40,
          f"{results['errors']} errors, {len(results['predictions'])} stored")
    check("every error is categorized",
          sum(results["error_counts"].values()) == results["errors"])
    check("error analysis examines every error",
          errors["errors_examined"] == results["errors"])
    check("failure sample is capped and labelled",
          0 < len(results["failures"]) <= 60
          and all("category" in f for f in results["failures"]))


def test_attention_analysis() -> None:
    """Averaged attention and answer attention must be well-formed."""
    print("\n[reasoning attention]")
    from lma.reasoning.evaluate import answer_attention_by_layer, averaged_attention_statistics

    for language in ("hi", "ne"):
        tokenizer = tokenizer_for(language)
        if tokenizer is None:
            print(f"  SKIP  {language}: tokenizer not present")
            continue
        rows = [
            e.to_dict()
            for e in ReasoningDatasetBuilder(
                GeneratorConfig(language=language, total_examples=600)
            ).build()
            if e.split == "test"
        ][:30]
        torch.manual_seed(0)
        model = TransformerLM(
            ModelConfig(vocab_size=8000, n_layer=2, n_head=2, n_embd=64,
                        block_size=64, dropout=0.0)
        ).eval()

        averaged = averaged_attention_statistics(model, rows, tokenizer, device="cpu", limit=20)
        check(f"[{language}] attention averaged over the requested prompts",
              averaged["prompts"] == 20)
        check(f"[{language}] every head classified after averaging",
              sum(averaged["head_groups"].values()) == 4)

        answer = answer_attention_by_layer(model, rows, tokenizer, device="cpu", limit=30)
        check(f"[{language}] answer token located in the prompts",
              answer["prompts"] > 0, f"{answer['prompts']} measured, {answer['skipped']} skipped")
        in_range = all(
            0.0 <= value <= 1.0 + 1e-6
            for series in ("answer_max_head", "answer_mean_head", "other_entity_max_head")
            for value in answer[series]
        )
        check(f"[{language}] attention masses are probabilities", in_range)
        check(f"[{language}] strongest head is at least the head average",
              all(a >= b - 1e-9 for a, b in zip(answer["answer_max_head"], answer["answer_mean_head"])))


def test_config_matches_record() -> None:
    """The documented finetuning config must be the one that actually ran."""
    print("\n[config vs record]")
    import json

    import yaml

    for language, directory in (("hi", "hindi"), ("ne", "nepali")):
        config_path = REPO_ROOT / directory / "configs" / "finetune.yaml"
        record_path = REPO_ROOT / directory / "reports" / "finetune_record.json"
        check(f"[{language}] finetuning config exists", config_path.exists())
        if not (config_path.exists() and record_path.exists()):
            print(f"  SKIP  [{language}] no finetuning record to compare against")
            continue

        documented = FinetuneConfig.from_dict(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))["finetune"]
        ).to_dict()
        recorded = json.loads(record_path.read_text(encoding="utf-8"))["finetune_config"]
        mismatched = sorted(k for k in recorded if documented.get(k) != recorded[k])
        check(f"[{language}] config file matches the recorded run", not mismatched,
              str(mismatched) if mismatched else f"{len(recorded)} hyperparameters")


def test_ablation() -> None:
    """The no-positional-embeddings ablation must be a real ablation."""
    print("\n[positional ablation]")
    standard = ModelConfig(vocab_size=8000)
    ablated = ModelConfig(vocab_size=8000, use_positional_embeddings=False)

    check(
        "ablation removes exactly the positional table",
        standard.n_params - ablated.n_params == standard.block_size * standard.n_embd,
        f"{standard.n_params - ablated.n_params:,} parameters",
    )

    model = TransformerLM(ablated).eval()
    check("no positional embedding is allocated", model.position_embedding is None)
    check(
        "measured parameters match the computed count",
        model.num_parameters() == ablated.n_params,
    )

    # What removing positions actually breaks, stated precisely. Reordering the
    # tokens BEFORE the last position, while keeping the last token fixed:
    #   * one ablated layer sees those tokens as an unordered set, so the last
    #     position's logits cannot change;
    #   * a second layer can tell - each earlier position summarizes a prefix of a
    #     different length, so the causal mask leaks order into deeper layers;
    #   * the standard model sees order directly through its positional table.
    # "The ablated model is permutation invariant" is therefore true of one layer
    # and false of a deep stack, which is why the ablation has something to measure.
    torch.manual_seed(0)
    ids = torch.randperm(8000)[:8].unsqueeze(0)
    reordered = torch.cat([ids[:, :-1].flip(1), ids[:, -1:]], dim=1)

    def last_logits_change(config: ModelConfig) -> float:
        torch.manual_seed(0)
        small = TransformerLM(config).eval()
        with torch.no_grad():
            return float((small(ids)["logits"][0, -1] - small(reordered)["logits"][0, -1]).abs().max())

    tiny = dict(vocab_size=8000, n_head=2, n_embd=64, block_size=16, dropout=0.0)
    one_layer = last_logits_change(ModelConfig(n_layer=1, use_positional_embeddings=False, **tiny))
    two_layer = last_logits_change(ModelConfig(n_layer=2, use_positional_embeddings=False, **tiny))
    positional = last_logits_change(ModelConfig(n_layer=1, use_positional_embeddings=True, **tiny))
    check("one ablated layer cannot see the order of earlier tokens", one_layer < 1e-5,
          f"max change {one_layer:.1e}")
    check("two ablated layers can - the causal mask leaks order", two_layer > 1e-4,
          f"max change {two_layer:.1e}")
    check("the standard model sees order directly", positional > 1e-4,
          f"max change {positional:.1e}")

    # A Phase 2 checkpoint predates the flag and must still load.
    legacy = ModelConfig.from_dict(
        {
            "vocab_size": 8000, "n_layer": 12, "n_head": 6, "n_embd": 384,
            "n_inner": 1536, "block_size": 512, "dropout": 0.1,
            "tie_weights": True, "bias": True,
        }
    )
    check(
        "a pre-ablation checkpoint config still loads",
        legacy.use_positional_embeddings and legacy.n_params == standard.n_params,
    )


def main() -> int:
    """Run every Phase 3 component test."""
    print("=" * 60)
    print("Phase 3 component tests")
    print("=" * 60)

    test_templates()
    test_generation()
    test_dataset_masking()
    test_finetune_loop()
    test_scoring()
    test_error_coverage()
    test_attention_analysis()
    test_config_matches_record()
    test_ablation()

    total = PASSED + FAILED
    print("\n" + "=" * 60)
    print(f"{PASSED}/{total} assertions passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
