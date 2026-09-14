#!/usr/bin/env python3
"""Score pretrained against finetuned on the reasoning test set.

Produces everything section 3.1 and 3.2 of the assignment ask for: pretrained vs
finetuned exact match on the synthetic test set, a per-chain-length breakdown
against chance, an error analysis, qualitative examples, and the pretrain-vs-
finetune attention comparison on a comparative-reasoning prompt.

Usage::

    python scripts/evaluate_reasoning.py --language hi \\
        --pretrained hindi/checkpoints/checkpoint_best.pt \\
        --finetuned hindi/checkpoints_finetuned/finetuned_best.pt \\
        --tokenizer hindi/tokenizer/hi_unigram_8000.model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from lma.eval import (  # noqa: E402
    attention_statistics,
    classify_heads,
    plot_attention_heatmaps,
    plot_head_summary,
)
from lma.reasoning.evaluate import (  # noqa: E402
    above_chance,
    chance_baseline,
    compare_pretrained_finetuned,
    error_analysis,
    score_examples,
)
from lma.reasoning.finetune import load_pretrained  # noqa: E402
from lma.tokenizer.train_tokenizer import BOS_ID  # noqa: E402
from lma.utils.logging_utils import write_json  # noqa: E402

LANGUAGES = {"hi": ("hindi", "Model H", "Hindi"), "ne": ("nepali", "Model L", "Nepali")}


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
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def attention_for_prompt(model, prompt: str, tokenizer, device: str):
    """Collect attention tensors for one reasoning prompt."""
    ids = torch.tensor(
        [[BOS_ID, *tokenizer.encode(prompt, out_type=int)]], dtype=torch.long
    )
    with torch.no_grad():
        output = model(ids.to(device), return_attention=True)
    return output["attentions"], ids


def main() -> int:
    """Run the full Phase 3 evaluation for one language."""
    args = parse_args()
    directory, label, language_name = LANGUAGES[args.language]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f"{label} ({args.language}) - {len(test)} test examples\n")

    pretrained, pre_payload = load_pretrained(args.pretrained, device)
    finetuned, fine_payload = load_pretrained(args.finetuned, device)

    print("[1] Scoring pretrained")
    before = score_examples(pretrained, test, tokenizer, device=device)
    print(f"    exact match: {before['exact_match']:.4f}")

    print("[2] Scoring finetuned")
    after = score_examples(finetuned, test, tokenizer, device=device)
    print(f"    exact match: {after['exact_match']:.4f}")

    baseline = chance_baseline(test)
    comparison = compare_pretrained_finetuned(before, after)
    margins = above_chance(after, baseline)

    print("\n[3] Per chain length (finetuned)")
    for key, row in sorted(margins["by_chain_length"].items()):
        verdict = "above" if row["above_chance"] else "below"
        print(
            f"    {key} entities: acc {row['accuracy']:.3f}  "
            f"chance {row['chance']:.3f}  margin {row['margin']:+.3f}  "
            f"n={row['n']}  ({verdict} chance)"
        )

    # --- attention: pretrained vs finetuned on the same prompt ---------------
    print("\n[4] Attention comparison")
    figures = REPO_ROOT / "report" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    prompt = test[0]["prompt"]

    attention_summary = {}
    for name, model in (("pretrained", pretrained), ("finetuned", finetuned)):
        attentions, ids = attention_for_prompt(model, prompt, tokenizer, device)
        stats = attention_statistics(attentions)
        groups = classify_heads(stats)
        attention_summary[name] = {
            "layer_mean_entropy": stats["layer_mean_entropy"],
            "layer_mean_distance": stats["layer_mean_distance"],
            "normalized_entropy": stats["normalized_entropy"],
            "mean_distance": stats["mean_distance"],
            "head_groups": groups["counts"],
        }
        print(f"    {name:11s} head types: {groups['counts']}")

        pieces = [tokenizer.id_to_piece(int(t)) for t in ids[0].tolist()]
        # Draw up to three heads, but never more than the model has - a narrower
        # configuration (as used by the component tests) would otherwise index
        # past the head axis and crash.
        heads = list(range(min(3, model.config.n_head)))
        plot_attention_heatmaps(
            attentions,
            pieces,
            layers=[0, model.config.n_layer - 1],
            heads=heads,
            out_path=figures / f"reasoning_attention_{name}_{args.language}.png",
            title=f"{language_name} reasoning prompt - {name}",
        )
        plot_head_summary(
            stats,
            figures / f"reasoning_head_summary_{name}_{args.language}.png",
            title=f"{language_name} per-head statistics - {name}",
        )

    # How far each layer's attention moved during finetuning.
    shifts = []
    for layer, (pre, post) in enumerate(
        zip(
            attention_summary["pretrained"]["layer_mean_distance"],
            attention_summary["finetuned"]["layer_mean_distance"],
        )
    ):
        shifts.append({"layer": layer, "pretrained": pre, "finetuned": post,
                       "delta": post - pre})
    attention_summary["distance_shift_by_layer"] = shifts

    results = {
        "language": args.language,
        "model": label,
        "test_examples": len(test),
        "pretrained_checkpoint": str(args.pretrained),
        "finetuned_checkpoint": str(args.finetuned),
        "pretrained_step": pre_payload.get("step"),
        "finetuned_step": fine_payload.get("step"),
        "parameters": finetuned.num_parameters(),
        "model_config": finetuned.config.to_dict(),
        "chance_baseline": baseline,
        "pretrained": before,
        "finetuned": after,
        "comparison": comparison,
        "above_chance": margins,
        "error_analysis": {
            "pretrained": error_analysis(before, test),
            "finetuned": error_analysis(after, test),
        },
        "attention": attention_summary,
        "attention_prompt": prompt,
    }

    out_path = REPO_ROOT / directory / "reports" / "phase3_evaluation.json"
    write_json(out_path, results)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
