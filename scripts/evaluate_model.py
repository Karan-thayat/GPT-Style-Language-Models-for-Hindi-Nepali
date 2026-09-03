"""Run the full Phase 2 evaluation for one trained model.

    python scripts/evaluate_model.py --language hi \
        --checkpoint hindi/checkpoints/checkpoint_best.pt \
        --data-dir /kaggle/input/lma-phase2-tokenized \
        --tokenizer hindi/tokenizer/hi_unigram_8000.model

Produces everything section 2.3 of the assignment asks for: cross-entropy,
perplexity and bits-per-byte on the held-out split; BLEU, chrF and ROUGE-L for
greedy decoding and temperatures 0.5, 1.0 and 1.5; repetition and Distinct-1/2;
and the attention analysis with per-head entropy, mean attention distance, and
heatmaps from an early and a late layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.eval import (  # noqa: E402
    attention_statistics,
    classify_heads,
    evaluate_generations,
    evaluate_language_model,
    plot_attention_heatmaps,
    plot_head_summary,
)
from lma.model import ModelConfig, TransformerLM  # noqa: E402
from lma.train import TokenBinaryDataset  # noqa: E402

#: Decoding settings evaluated for every model, as the assignment specifies.
DECODE_SETTINGS = [
    {"name": "greedy", "greedy": True},
    {"name": "temperature_0.5", "temperature": 0.5},
    {"name": "temperature_1.0", "temperature": 1.0},
    {"name": "temperature_1.5", "temperature": 1.5},
]


def load_model(checkpoint_path: Path, device: str) -> tuple[TransformerLM, dict]:
    """Rebuild a model from a training checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(payload["model_config"])
    model = TransformerLM(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


def build_prompts(
    dataset: TokenBinaryDataset,
    tokenizer,
    count: int,
    prompt_tokens: int,
    continuation_tokens: int,
) -> tuple[list[torch.Tensor], list[str]]:
    """Take held-out prefixes and their true continuations.

    The reference is the text that actually followed the prompt in the held-out
    split, which is what the assignment means by "reference continuations from
    the held-out set".
    """
    prompts, references = [], []
    window = prompt_tokens + continuation_tokens
    stride = window * 4  # spread samples across the split rather than clustering

    for index in range(count):
        start = index * stride
        if start + window >= len(dataset.tokens):
            break
        chunk = [int(t) for t in dataset.tokens[start : start + window]]
        prompts.append(torch.tensor(chunk[:prompt_tokens], dtype=torch.long))
        references.append(tokenizer.decode(chunk[prompt_tokens:]))

    return prompts, references


def generate_continuations(
    model: TransformerLM,
    prompts: list[torch.Tensor],
    tokenizer,
    max_new_tokens: int,
    device: str,
    batch_size: int = 8,
    **decode_kwargs,
) -> list[str]:
    """Generate one continuation per prompt under the given decoding settings."""
    outputs: list[str] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        stacked = torch.stack(batch).to(device)
        generated = model.generate(stacked, max_new_tokens=max_new_tokens, **decode_kwargs)
        # Keep only the newly generated part, not the prompt.
        for row in generated[:, stacked.shape[1] :].tolist():
            outputs.append(tokenizer.decode(row))

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=["hi", "ne"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or REPO_ROOT / (
        "hindi" if args.language == "hi" else "nepali"
    ) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = REPO_ROOT / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model, payload = load_model(args.checkpoint, device)

    print(f"Model {'H' if args.language == 'hi' else 'L'} ({args.language})")
    print(f"  parameters: {model.num_parameters():,}")
    print(f"  trained to step: {payload.get('step', 0):,}")
    print(f"  device: {device}\n")

    results = {
        "language": args.language,
        "checkpoint": str(args.checkpoint),
        "parameters": model.num_parameters(),
        "training_step": payload.get("step"),
        "tokens_seen": payload.get("tokens_seen"),
        "model_config": payload.get("model_config"),
    }

    # --- intrinsic language-modeling metrics -------------------------------
    print("[1] Language-modeling metrics on the held-out split")
    test_data = TokenBinaryDataset(
        args.data_dir / f"{args.language}_test.bin", model.config.block_size
    )
    lm = evaluate_language_model(
        model,
        test_data,
        batch_size=16,
        max_batches=args.eval_batches,
        device=device,
        tokenizer=tokenizer,
    )
    results["language_modeling"] = lm
    print(f"  cross-entropy : {lm['cross_entropy']:.4f} nats/token")
    print(f"  perplexity    : {lm['perplexity']:.2f}")
    print(f"  bits-per-byte : {lm.get('bits_per_byte', float('nan')):.4f}")
    print(f"  tokens        : {lm['tokens']:,}\n")

    # --- generation quality -------------------------------------------------
    print("[2] Generation quality")
    prompts, references = build_prompts(
        test_data, tokenizer, args.samples, args.prompt_tokens, args.continuation_tokens
    )
    print(f"  {len(prompts)} held-out prompts of {args.prompt_tokens} tokens\n")

    results["generation"] = {}
    samples_for_report = {}

    for setting in DECODE_SETTINGS:
        name = setting["name"]
        kwargs = {k: v for k, v in setting.items() if k != "name"}
        hypotheses = generate_continuations(
            model,
            prompts,
            tokenizer,
            max_new_tokens=args.continuation_tokens,
            device=device,
            **kwargs,
        )
        metrics = evaluate_generations(hypotheses, references[: len(hypotheses)])
        results["generation"][name] = metrics
        samples_for_report[name] = hypotheses[:3]

        print(
            f"  {name:16s} BLEU {metrics['bleu']:5.2f}  chrF {metrics['chrf']:5.2f}  "
            f"ROUGE-L {metrics['rouge_l']:5.2f}  rep {metrics['repetition_rate']:.3f}  "
            f"D-1 {metrics['distinct_1']:.3f}  D-2 {metrics['distinct_2']:.3f}"
        )

    results["samples"] = samples_for_report
    print()

    # --- attention analysis -------------------------------------------------
    print("[3] Attention analysis")
    sample_ids = torch.tensor(
        [int(t) for t in test_data.tokens[:64]], dtype=torch.long
    ).unsqueeze(0)
    with torch.no_grad():
        attentions = model(sample_ids.to(device), return_attention=True)["attentions"]

    stats = attention_statistics(attentions)
    grouped = classify_heads(stats)
    results["attention"] = {
        "normalized_entropy": stats["normalized_entropy"],
        "mean_distance": stats["mean_distance"],
        "layer_mean_entropy": stats["layer_mean_entropy"],
        "layer_mean_distance": stats["layer_mean_distance"],
        "head_groups": grouped["counts"],
    }
    print(f"  layers x heads: {stats['n_layer']} x {stats['n_head']}")
    print(f"  head types: {grouped['counts']}")

    tokens = [tokenizer.id_to_piece(int(t)) for t in sample_ids[0].tolist()]
    label = "Hindi (Model H)" if args.language == "hi" else "Nepali (Model L)"

    early_late = [0, model.config.n_layer - 1]
    heatmap_path = figures_dir / f"attention_heatmaps_{args.language}.png"
    plot_attention_heatmaps(
        attentions,
        tokens,
        layers=early_late,
        heads=[0, 1, 2],
        out_path=heatmap_path,
        title=f"Attention patterns - {label}",
    )
    summary_path = figures_dir / f"attention_summary_{args.language}.png"
    plot_head_summary(stats, summary_path, title=f"Per-head statistics - {label}")
    print(f"  wrote {heatmap_path.name} and {summary_path.name}\n")

    # --- persist ------------------------------------------------------------
    out_path = out_dir / "phase2_evaluation.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
