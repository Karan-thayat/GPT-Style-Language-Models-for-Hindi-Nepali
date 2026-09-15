#!/usr/bin/env python3
"""Score pretrained against finetuned on the reasoning test set.

Produces everything sections 3.1 and 3.2 of the assignment ask for, for one
language: pretrained vs finetuned exact match on the full synthetic test set, a
per-chain-length breakdown against chance, an error analysis over every mistake,
qualitative successes and failures, and the pretrain-vs-finetune attention
comparison on comparative-reasoning prompts.

Usage::

    python scripts/evaluate_reasoning.py --language hi \\
        --pretrained Phase2-Checkpoints/hindi/checkpoint_best.pt \\
        --finetuned Phase3-Finetuned/hi_finetuned_best.pt \\
        --tokenizer hindi/tokenizer/hi_unigram_8000.model

Both checkpoints are the files linked from the README. The finetuned one is the
checkpoint selected on validation loss, so every number written here describes
exactly the model a reader can download.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from lma.eval import plot_attention_heatmaps, plot_head_summary  # noqa: E402
from lma.reasoning.evaluate import (  # noqa: E402
    above_chance,
    answer_attention_by_layer,
    averaged_attention_statistics,
    chance_baseline,
    compare_pretrained_finetuned,
    error_analysis,
    score_examples,
)
from lma.reasoning.finetune import load_pretrained  # noqa: E402
from lma.reasoning.plots import plot_answer_attention  # noqa: E402
from lma.tokenizer.train_tokenizer import BOS_ID  # noqa: E402
from lma.utils.logging_utils import read_json, write_json  # noqa: E402

LANGUAGES = {"hi": ("hindi", "Model H", "Hindi"), "ne": ("nepali", "Model L", "Nepali")}

#: Checkpoint keys the evaluation never reads. Dropping them straight after
#: loading removes the optimizer state - two thirds of each file - from memory.
UNUSED_PAYLOAD_KEYS = (
    "optimizer", "scaler", "torch_rng", "cuda_rng", "train_sampler_rng", "history",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--language", required=True, choices=sorted(LANGUAGES))
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--finetuned", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=2000,
                        help="Test examples to score (default: all 2000)")
    parser.add_argument("--attention-prompts", type=int, default=200,
                        help="Test prompts the attention statistics are averaged over")
    parser.add_argument("--pretrained-name", default=None,
                        help="How to record the pretrained checkpoint (e.g. its Drive path)")
    parser.add_argument("--finetuned-name", default=None,
                        help="How to record the finetuned checkpoint (e.g. its Drive path)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load(path: Path, device: str):
    """Load a checkpoint for evaluation, discarding the training-only state."""
    model, payload = load_pretrained(path, device)
    for key in UNUSED_PAYLOAD_KEYS:
        payload.pop(key, None)
    gc.collect()
    model.eval()
    return model, payload


def evaluate_one(name: str, model, test: list[dict], tokenizer, args, figures: Path,
                 code: str, language_name: str) -> tuple[dict, dict]:
    """Score one checkpoint and analyse its attention.

    Returns:
        ``(scores, attention)``: the :func:`score_examples` output, and the
        attention statistics averaged over many prompts together with the
        answer-attention measurement.
    """
    device = args.device
    print(f"[{name}] scoring {len(test)} test examples")
    scores = score_examples(model, test, tokenizer, device=device)
    print(f"    exact match {scores['exact_match']:.4f}  "
          f"({scores['errors']} errors: {scores['error_counts']})")

    print(f"[{name}] attention over {args.attention_prompts} test prompts")
    averaged = averaged_attention_statistics(
        model, test, tokenizer, device=device, limit=args.attention_prompts
    )
    answer = answer_attention_by_layer(
        model, test, tokenizer, device=device, limit=args.attention_prompts
    )
    print(f"    head types {averaged['head_groups']}; answer attention measured on "
          f"{answer['prompts']} prompts ({answer['skipped']} skipped)")

    # Heatmaps need one concrete sentence to label their axes; the statistics
    # above are the evidence, this is the illustration.
    prompt = test[0]["prompt"]
    ids = torch.tensor([[BOS_ID, *tokenizer.encode(prompt, out_type=int)]],
                       dtype=torch.long, device=device)
    with torch.no_grad():
        attentions = model(ids, return_attention=True)["attentions"]
    pieces = [tokenizer.id_to_piece(int(t)) for t in ids[0].tolist()]
    last = model.config.n_layer - 1
    plot_attention_heatmaps(
        attentions, pieces,
        layers=[0, last], heads=list(range(min(3, model.config.n_head))),
        out_path=figures / f"reasoning_attention_{name}_{code}.png",
        title=f"{language_name} reasoning prompt - {name} model (layers 0 and {last})",
    )
    plot_head_summary(
        averaged,
        figures / f"reasoning_head_summary_{name}_{code}.png",
        title=(f"{language_name} per-head attention - {name} model, "
               f"averaged over {averaged['prompts']} test prompts"),
    )

    attention = {
        "layer_mean_entropy": averaged["layer_mean_entropy"],
        "layer_mean_distance": averaged["layer_mean_distance"],
        "normalized_entropy": averaged["normalized_entropy"],
        "mean_distance": averaged["mean_distance"],
        "head_groups": averaged["head_groups"],
        "prompts": averaged["prompts"],
        "answer_attention": answer,
    }
    return scores, attention


def main() -> int:
    """Run the full Phase 3 evaluation for one language."""
    args = parse_args()
    directory, label, language_name = LANGUAGES[args.language]
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    data_dir = args.data_dir or (REPO_ROOT / directory / "data" / "reasoning")
    test_path = data_dir / f"{args.language}_reasoning_test.jsonl"
    if not test_path.exists():
        raise SystemExit(
            f"missing {test_path}\nRun: python scripts/generate_reasoning_data.py "
            f"--language {args.language}"
        )
    with test_path.open("r", encoding="utf-8") as handle:
        test = [json.loads(line) for line in handle if line.strip()][: args.limit]

    figures = REPO_ROOT / "report" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    print(f"{label} ({language_name}) - {len(test)} test examples, device {args.device}\n")

    # One model in memory at a time: the two are never needed simultaneously.
    pretrained, pre_payload = load(args.pretrained, args.device)
    parameters = pretrained.num_parameters()
    model_config = pretrained.config.to_dict()
    before, attention_before = evaluate_one(
        "pretrained", pretrained, test, tokenizer, args, figures, args.language, language_name
    )
    del pretrained
    gc.collect()

    finetuned, fine_payload = load(args.finetuned, args.device)
    after, attention_after = evaluate_one(
        "finetuned", finetuned, test, tokenizer, args, figures, args.language, language_name
    )
    del finetuned
    gc.collect()

    baseline = chance_baseline(test)
    margins = above_chance(after, baseline)

    print("\nPer chain length (finetuned):")
    for key, row in sorted(margins["by_chain_length"].items()):
        verdict = "above" if row["above_chance"] else "below"
        print(f"    {key} entities: acc {row['accuracy']:.3f}  chance {row['chance']:.3f}  "
              f"margin {row['margin']:+.3f}  n={row['n']}  ({verdict} chance)")

    plot_answer_attention(
        attention_before["answer_attention"],
        attention_after["answer_attention"],
        figures / f"reasoning_answer_attention_{args.language}.png",
        title=(f"{language_name} ({label}) - attention from the answer position, "
               f"averaged over {attention_after['answer_attention']['prompts']} test prompts"),
    )

    shifts = [
        {"layer": layer, "pretrained": pre, "finetuned": post, "delta": post - pre}
        for layer, (pre, post) in enumerate(zip(
            attention_before["layer_mean_distance"], attention_after["layer_mean_distance"]
        ))
    ]

    dataset_path = REPO_ROOT / directory / "reports" / "reasoning_dataset.json"
    results = {
        "language": args.language,
        "model": label,
        "test_examples": len(test),
        "pretrained_checkpoint": args.pretrained_name or str(args.pretrained),
        "finetuned_checkpoint": args.finetuned_name or str(args.finetuned),
        "pretrained_step": pre_payload.get("step"),
        "finetuned_step": fine_payload.get("step"),
        "finetuned_best_val_loss": fine_payload.get("best_val_loss"),
        "checkpoint_selection": "lowest answer-span validation loss (finetuned_best.pt)",
        "scored_on": args.device,
        "parameters": parameters,
        "model_config": model_config,
        "finetune_config": fine_payload.get("finetune_config"),
        "chance_baseline": baseline,
        "pretrained": before,
        "finetuned": after,
        "comparison": compare_pretrained_finetuned(before, after),
        "above_chance": margins,
        "error_analysis": {
            "pretrained": error_analysis(before, test),
            "finetuned": error_analysis(after, test),
        },
        "attention": {
            "pretrained": attention_before,
            "finetuned": attention_after,
            "distance_shift_by_layer": shifts,
        },
        "attention_prompt": test[0]["prompt"],
        "attention_prompts": attention_after["prompts"],
        "dataset": read_json(dataset_path) if dataset_path.exists() else None,
    }

    out_path = REPO_ROOT / directory / "reports" / "phase3_evaluation.json"
    write_json(out_path, results)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
