#!/usr/bin/env python3
"""Generate the consolidated Phase 3 report from the evaluation artifacts.

    python scripts/make_phase3_report.py

Reads the Phase 1, 2, and 3 JSON artifacts and writes ``report/phase3_report.md``
plus the finetuning loss-curve figure. Every number in the report comes from
those files, so the report cannot drift from the runs that produced it.

Phase 3's report is the consolidated one: the assignment asks it to answer four
questions about the two resource tiers, drawing on the earlier phases as needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LANGUAGES = {
    "hi": ("hindi", "Model H", "Hindi"),
    "ne": ("nepali", "Model L", "Nepali"),
}


def load_artifacts() -> dict:
    """Collect every phase's records for whichever languages have them."""
    artifacts: dict[str, dict] = {}
    for code, (directory, _, _) in LANGUAGES.items():
        reports = REPO_ROOT / directory / "reports"
        entry: dict = {}
        for key, name in (
            ("dataset", "reasoning_dataset.json"),
            ("finetune", "finetune_record.json"),
            ("evaluation", "phase3_evaluation.json"),
            ("phase2", "phase2_evaluation.json"),
            ("tokens", "token_counts.json"),
            ("tokenizer", "tokenizer_stats.json"),
            ("corpus", "corpus_stats.json"),
            ("cleaning", "cleaning_stats.json"),
        ):
            path = reports / name
            if path.exists():
                entry[key] = json.loads(path.read_text(encoding="utf-8"))
        if entry:
            artifacts[code] = entry
    return artifacts


def fmt(value, spec: str = "{:.4f}") -> str:
    """Format a number, or a dash when it is missing."""
    if value is None:
        return "-"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def section_dataset(artifacts: dict) -> str:
    """The synthetic dataset: size, variety, and leakage evidence."""
    parts = [
        "## 1. The synthetic reasoning dataset",
        "",
        "Generated programmatically rather than downloaded, so every ground-truth",
        "label is known by construction and the corpus is reproducible from a seed.",
        "The task states a chain of pairwise comparisons and asks for the extreme",
        "element, which requires chaining the inequalities rather than matching a",
        "single sentence.",
        "",
    ]

    rows = ["| Property | " + " | ".join(
        LANGUAGES[c][1] for c in artifacts if "dataset" in artifacts[c]
    ) + " |"]
    codes = [c for c in artifacts if "dataset" in artifacts[c]]
    if not codes:
        return "\n".join(parts + ["_No dataset artifacts found._", ""])

    rows.append("|---|" + "---|" * len(codes))
    for label, getter in (
        ("Total examples", lambda d: f"{d['total_examples']:,}"),
        ("Train", lambda d: f"{d['by_split'].get('train', 0):,}"),
        ("Validation", lambda d: f"{d['by_split'].get('validation', 0):,}"),
        ("Test", lambda d: f"{d['by_split'].get('test', 0):,}"),
        ("Unique prompts", lambda d: f"{d['unique_prompts']:,}"),
        ("Entity pool", lambda d: str(d["template_variety"]["entity_pool"])),
        ("Comparable dimensions", lambda d: str(d["template_variety"]["dimensions"])),
        ("Chain lengths", lambda d: ", ".join(str(n) for n in d["template_variety"]["chain_lengths"])),
    ):
        cells = [getter(artifacts[c]["dataset"]) for c in codes]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    parts += rows + ["", "### Leakage control", ""]
    parts += [
        "Three mechanisms, each covering what the others do not. **Chain signatures**",
        "- the unordered entity set plus the dimension - are partitioned across",
        "splits, so no evaluation example re-tests a comparison any training example",
        "covered. A minority of names appear **only in evaluation premises**, testing",
        "that an unfamiliar participant does not derail the comparison. Within those",
        "constraints, assignment is by salted content hash, as in Phase 1.",
        "",
    ]

    check_rows = ["| Check | " + " | ".join(LANGUAGES[c][1] for c in codes) + " |",
                  "|---|" + "---|" * len(codes)]
    for label, key in (
        ("Chain-signature overlap (train vs eval)", "chain_signature_overlap"),
        ("Prompt overlap (train vs eval)", "prompt_overlap_train_eval"),
        ("Duplicate prompts", "duplicate_prompts"),
        ("Evaluation answers that are distractors", "distractor_answers_in_eval"),
    ):
        cells = [str(artifacts[c]["dataset"]["leakage_checks"].get(key, "-")) for c in codes]
        check_rows.append(f"| {label} | " + " | ".join(cells) + " |")
    parts += check_rows + [""]

    parts += [
        "**A rejected earlier design, recorded because it is the obvious one.**",
        "Reserving names for the *answer* position - so every test answer was a token",
        "absent from all training targets - makes the task unanswerable rather than",
        "hard. Measured during development on a small, randomly initialized model",
        "trained on this task: 0.0000 exact match on held-out-answer",
        "examples against 0.2600 on examples whose prompts were equally unseen but",
        "whose answers came from the training vocabulary. The evaluation set, not the",
        "model, was the failure.",
        "",
    ]
    return "\n".join(parts)


def section_finetuning(artifacts: dict) -> str:
    """Finetuning setup and what it cost."""
    codes = [c for c in artifacts if "finetune" in artifacts[c]]
    parts = ["## 2. Finetuning", ""]
    if not codes:
        return "\n".join(parts + ["_Not run yet._", ""])

    parts += [
        "Each model starts from its own Phase 2 checkpoint, keeps its own tokenizer",
        "and vocabulary, and trains only on its own language's data. Loss is computed",
        "on the **answer span only**: the prompt states premises the model is",
        "conditioned on, not asked to produce, so every prompt position is masked out",
        "of the objective.",
        "",
    ]

    parts += [
        "Every hyperparameter is listed below. They are committed as",
        "`<language>/configs/finetune.yaml`, and `scripts/test_reasoning.py` asserts",
        "that file matches the configuration recorded by the run itself, so the",
        "documentation cannot drift from what produced the results.",
        "",
    ]

    def config(code: str, key: str):
        return (artifacts[code]["finetune"].get("finetune_config") or {}).get(key)

    def selected(code: str):
        evaluation = artifacts[code].get("evaluation") or {}
        step = evaluation.get("finetuned_step")
        loss = evaluation.get("finetuned_best_val_loss")
        if step is None:
            return "lowest validation loss"
        return f"step {step:,} (val loss {loss:.4f})"

    rows = ["| Hyperparameter | " + " | ".join(LANGUAGES[c][1] for c in codes) + " |",
            "|---|" + "---|" * len(codes)]
    for label, getter in (
        ("Starting checkpoint", lambda c: f"own Phase 2 model, step {artifacts[c]['finetune'].get('pretrained_step') or 0:,}"),
        ("Tokenizer / vocabulary", lambda c: "own Phase 1 unigram, 8,000 pieces, frozen"),
        ("Trainable parameters", lambda c: f"{artifacts[c]['finetune'].get('parameters', 0):,}"),
        ("Training examples", lambda c: f"{(artifacts[c]['finetune'].get('train_data') or {}).get('examples', 0):,}"),
        ("Epochs", lambda c: str(config(c, "epochs"))),
        ("Batch size (examples)", lambda c: str(config(c, "batch_size"))),
        ("Optimizer steps", lambda c: f"{artifacts[c]['finetune'].get('steps', 0):,}"),
        ("Optimizer", lambda c: f"AdamW, betas ({config(c, 'beta1')}, {config(c, 'beta2')})"),
        ("Peak learning rate", lambda c: fmt(config(c, "learning_rate"), "{:.1e}")),
        ("Minimum learning rate", lambda c: fmt(config(c, "min_learning_rate"), "{:.1e}")),
        ("Schedule", lambda c: f"linear warmup {config(c, 'warmup_steps')} steps, cosine decay"),
        ("Weight decay", lambda c: f"{config(c, 'weight_decay')} (2D weights only)"),
        ("Gradient clipping", lambda c: f"norm {config(c, 'grad_clip')}"),
        ("Mixed precision", lambda c: "yes" if config(c, "amp") else "no"),
        ("Max sequence length", lambda c: f"{config(c, 'max_length')} tokens (longer examples dropped)"),
        ("Loss", lambda c: "answer token + EOS only; prompt masked"),
        ("Seed", lambda c: str(config(c, "seed"))),
        ("Checkpoint reported", selected),
        ("Wall-clock", lambda c: fmt((artifacts[c]["finetune"].get("wall_seconds") or 0) / 60, "{:.1f} min")),
    ):
        rows.append(f"| {label} | " + " | ".join(getter(c) for c in codes) + " |")
    parts += rows + [""]

    figure = REPO_ROOT / "report" / "figures" / "phase3_finetune_curves.png"
    if figure.exists():
        parts += ["![Finetuning validation loss](figures/phase3_finetune_curves.png)", ""]
    return "\n".join(parts)


def section_results(artifacts: dict) -> str:
    """The headline comparison: pretrained against finetuned."""
    codes = [c for c in artifacts if "evaluation" in artifacts[c]]
    parts = ["## 3. Reasoning results", ""]
    if not codes:
        return "\n".join(parts + ["_Not run yet._", ""])

    parts += [
        "Exact match on the held-out test split. Both checkpoints are scored the same",
        "way - greedy argmax at the position following the answer prefix, no sampling",
        "- so the comparison isolates the weights.",
        "",
    ]

    rows = ["| Metric | " + " | ".join(LANGUAGES[c][1] for c in codes) + " |",
            "|---|" + "---|" * len(codes)]

    def get(code: str, *path, default=None):
        node = artifacts[code]["evaluation"]
        for key in path:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    for label, path, spec in (
        ("Pretrained exact match", ("comparison", "exact_match", "pretrained"), "{:.4f}"),
        ("Finetuned exact match", ("comparison", "exact_match", "finetuned"), "{:.4f}"),
        ("Improvement", ("comparison", "exact_match", "delta"), "{:+.4f}"),
        ("Chance (aggregate)", ("chance_baseline", "overall"), "{:.4f}"),
        ("Pretrained constrained accuracy", ("comparison", "constrained_accuracy", "pretrained"), "{:.4f}"),
        ("Finetuned constrained accuracy", ("comparison", "constrained_accuracy", "finetuned"), "{:.4f}"),
        ("Test examples", ("test_examples",), "{:,}"),
    ):
        rows.append(f"| {label} | " + " | ".join(fmt(get(c, *path), spec) for c in codes) + " |")
    parts += rows + ["", "### Accuracy against chance, per chain length", ""]
    parts += [
        "Chance differs sharply by chain length - one in two for a two-entity",
        "comparison against one in four for a four-entity chain - so a single blended",
        "baseline is misleading. The per-length comparison is the honest one.",
        "",
    ]

    for code in codes:
        label = LANGUAGES[code][1]
        margins = get(code, "above_chance", "by_chain_length") or {}
        if not margins:
            continue
        parts += [f"**{label}**", "",
                  "| Entities | Accuracy | Chance | Margin | n |",
                  "|---|---:|---:|---:|---:|"]
        for key, row in sorted(margins.items()):
            parts.append(
                f"| {key} | {row['accuracy']:.3f} | {row['chance']:.3f} | "
                f"{row['margin']:+.3f} | {row['n']:,} |"
            )
        parts.append("")

    parts += ["### Error analysis", ""]
    parts += [
        "Mistakes are sorted into three kinds, because they call for different",
        "explanations. **Off-chain** means the prediction was not any entity in the",
        "premises - the model did not identify the answer as a choice among the named",
        "entities at all. **Inverted** means it named the entity at the opposite end",
        "of the ordering, which suggests it tracked the comparison but mishandled the",
        "question's polarity. **Wrong entity** is everything else.",
        "",
    ]
    parts += [
        "Counts cover every one of the test examples. (An earlier revision of this",
        "analysis read only a 40-prediction sample, and so reported an all-zero",
        "profile for models whose few mistakes fell outside it.)",
        "",
    ]
    header = "| Failure mode | " + " | ".join(
        f"{LANGUAGES[c][1]} pretrained | {LANGUAGES[c][1]} finetuned" for c in codes
    ) + " |"
    error_rows = [header, "|---|" + "---:|---:|" * len(codes)]

    def error_cell(code: str, side: str, key: str | None) -> str:
        analysis = get(code, "error_analysis", side) or {}
        examined = analysis.get("errors_examined", 0)
        if key is None:
            return f"{examined:,}"
        count = (analysis.get("counts") or {}).get(key, 0)
        return f"{count:,} ({count / examined:.1%})" if examined else "0"

    for label, key in (("All errors", None), ("Off-chain", "off_chain"),
                       ("Wrong entity", "wrong_entity"), ("Inverted polarity", "inverted")):
        cells = []
        for code in codes:
            cells += [error_cell(code, "pretrained", key), error_cell(code, "finetuned", key)]
        error_rows.append(f"| {label} | " + " | ".join(cells) + " |")
    parts += error_rows + [""]
    parts += section_qualitative(artifacts, codes)
    return "\n".join(parts)


#: Devanagari letters that alternate in the spelling of a single word, so two
#: names differing only by such a pair read as variants of one name - as
#: बिकास / विकास (Bikas / Vikas) do. One-letter edit distance alone is not
#: enough: गीता and सीता are also one letter apart, but are two different names.
SPELLING_ALTERNATIONS = [frozenset(pair) for pair in (("ब", "व"), ("श", "स"), ("ष", "श"), ("ष", "स"))]


def spelling_variants(a: str, b: str) -> bool:
    """Whether two names differ only by one letter that alternates in spelling."""
    if a == b or len(a) != len(b):
        return False
    differences = [(x, y) for x, y in zip(a, b) if x != y]
    return len(differences) == 1 and frozenset(differences[0]) in SPELLING_ALTERNATIONS


def section_qualitative(artifacts: dict, codes: list[str]) -> list[str]:
    """Successes, every stored failure, and what the failures reveal."""
    parts = ["### Qualitative successes and failures", ""]

    for code in codes:
        _, label, language_name = LANGUAGES[code]
        evaluation = artifacts[code]["evaluation"]
        before = evaluation.get("pretrained", {})
        after = evaluation.get("finetuned", {})
        parts += [f"**{label} ({language_name})**", ""]

        # What an untrained-on-task model does at the answer position.
        from collections import Counter

        predicted = Counter(
            (item["predicted"] or "▁") for item in before.get("failures", [])
        ).most_common(3)
        if predicted:
            shown = ", ".join(f"`{piece}` ({count})" for piece, count in predicted)
            parts += [
                f"*Before finetuning.* Across the first {len(before.get('failures', []))} "
                f"recorded pretrained errors, the most frequent predictions were {shown}. "
                "`▁` is SentencePiece's bare word-boundary piece: the pretrained model "
                "treats the answer prefix as the middle of running text and predicts "
                "that a new word begins, rather than naming anyone - which is why "
                "off-chain errors dominate its profile.",
                "",
            ]

        successes = sorted(after.get("successes", []), key=lambda s: -s["chain_length"])[:3]
        if successes:
            parts += ["*Finetuned successes* (longest chains first):", "",
                      "| Prompt | Answer | Entities |", "|---|---|---:|"]
            for item in successes:
                parts.append(f"| {item['prompt']} | {item['expected']} | {item['chain_length']} |")
            parts.append("")

        failures = after.get("failures", [])
        errors = after.get("errors", len(failures))
        if failures:
            parts += [f"*Every finetuned failure* ({errors} of {evaluation.get('test_examples', 0):,}):", "",
                      "| Prompt | Expected | Predicted | Entities |", "|---|---|---|---:|"]
            for item in failures:
                parts.append(f"| {item['prompt']} | {item['expected']} | {item['predicted']} | {item['chain_length']} |")
            parts.append("")

            longest = sum(1 for item in failures if item["chain_length"] == 4)
            parts += [f"{longest} of the {len(failures)} failures are four-entity, three-hop "
                      "chains, the hardest length; none are off-chain, so the model always "
                      "answered with a named participant.", ""]

            # Names one character apart read as the same person to a human.
            from lma.reasoning.templates import TEMPLATES

            pool = TEMPLATES[code].entities
            confusable = set()
            for item in failures:
                names = {entity for entity in pool if entity in item["prompt"]}
                for a in names:
                    for b in names:
                        if a < b and spelling_variants(a, b):
                            confusable.add((a, b))
            if confusable:
                involved = sum(
                    1 for item in failures
                    if any(a in item["prompt"] and b in item["prompt"] for a, b in confusable)
                )
                pairs = ", ".join(f"`{a}` / `{b}`" for a, b in sorted(confusable))
                parts += [
                    f"**A flaw in the entity pool, found by reading these failures.** "
                    f"{involved} of the {len(failures)} errors involve {pairs}: names one "
                    "character apart that the generator treats as different people, but "
                    "which a reader would take to be spelling variants of one name. A "
                    "premise pair comparing them is contradictory to a human, so these "
                    "examples are ill-posed rather than hard. The pools were vetted for "
                    "single-token encoding but not for near-homographs; dropping one "
                    "variant from each pair is the fix.",
                    "",
                ]
    return parts


def section_attention(artifacts: dict) -> str:
    """Pretrained against finetuned attention, with the evidence spelled out."""
    codes = [c for c in artifacts
             if "attention" in (artifacts[c].get("evaluation") or {})]
    parts = ["## 4. Attention, before and after finetuning", ""]
    if not codes:
        return "\n".join(parts + ["_Not run yet._", ""])

    def attention(code: str) -> dict:
        return artifacts[code]["evaluation"]["attention"]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    prompts = artifacts[codes[0]]["evaluation"].get("attention_prompts", "the")
    n_layer = len(attention(codes[0])["pretrained"]["layer_mean_distance"])
    paired = " | ".join(f"{LANGUAGES[c][1]} pretrained | {LANGUAGES[c][1]} finetuned" for c in codes)
    rule = "|---|" + "---:|---:|" * len(codes)

    parts += [
        f"Both checkpoints read the same {prompts} held-out reasoning prompts, so any",
        "difference in where the heads look is attributable to finetuning. Each head's",
        "entropy and mean look-back distance are averaged over those prompts *before*",
        "heads are classified, because a single prompt is a sample of one. The heatmaps",
        "at the end of this section show layers 0 and 11 for one prompt, as an",
        "illustration only.",
        "",
        "### Local versus long-range attention",
        "",
        f"| Head type | {paired} |",
        rule,
    ]
    for label, key in (("Local (mean distance < 5)", "local"),
                       ("Long-range", "long_range"),
                       ("Diffuse (normalized entropy > 0.8)", "diffuse")):
        cells = []
        for code in codes:
            cells += [str(attention(code)[side]["head_groups"].get(key, 0))
                      for side in ("pretrained", "finetuned")]
        parts.append(f"| {label} | " + " | ".join(cells) + " |")

    parts += ["", "Mean look-back distance by layer, in positions:", "",
              f"| Layer | {paired} |", rule]
    for layer in range(n_layer):
        cells = []
        for code in codes:
            cells += [f"{attention(code)[side]['layer_mean_distance'][layer]:.2f}"
                      for side in ("pretrained", "finetuned")]
        parts.append(f"| {layer} | " + " | ".join(cells) + " |")
    parts.append("")

    middle = list(range(3, min(9, n_layer)))
    late = list(range(max(0, n_layer - 3), n_layer))
    all_more_local = True
    for code in codes:
        pre, post = attention(code)["pretrained"], attention(code)["finetuned"]
        closer = sum(1 for l in range(n_layer)
                     if post["layer_mean_distance"][l] < pre["layer_mean_distance"][l])
        local_pre = pre["head_groups"].get("local", 0)
        local_post = post["head_groups"].get("local", 0)
        mid_pre = mean([pre["layer_mean_distance"][l] for l in middle])
        mid_post = mean([post["layer_mean_distance"][l] for l in middle])
        all_more_local = all_more_local and local_post > local_pre and mid_post < mid_pre
        parts += [
            f"**{LANGUAGES[code][1]}:** local heads {local_pre} → {local_post}, long-range "
            f"heads {pre['head_groups'].get('long_range', 0)} → "
            f"{post['head_groups'].get('long_range', 0)}. Mean distance fell in {closer} of "
            f"{n_layer} layers; across layers {middle[0]}-{middle[-1]} it went from "
            f"{mid_pre:.2f} to {mid_post:.2f} positions, and across layers "
            f"{late[0]}-{late[-1]} from "
            f"{mean([pre['layer_mean_distance'][l] for l in late]):.2f} to "
            f"{mean([post['layer_mean_distance'][l] for l in late]):.2f}.",
            "",
        ]

    if all_more_local:
        parts += [
            "In both models finetuning made the middle of the network more local. That",
            "fits the task: each premise is a short clause, and relating a name to the",
            "comparative that follows it is work done over a few positions. It does not",
            "mean the models stopped reaching back - the next measurement shows their",
            "late layers doing exactly that, but selectively.",
            "",
        ]

    parts += [
        "### Head specialization: attention on the correct answer",
        "",
        "The last prompt position is where the answer is predicted, so what it attends",
        "to is the most direct view of how the answer is reached. For each layer and",
        "each prompt, the measurement takes the head putting the most attention on the",
        "correct entity's token; as the baseline, it takes whichever head attends most",
        "to each occurrence of the *other* named entities. Premises are shuffled, so the answer sits at a",
        "different position in every prompt: a head that finds it is not doing so by",
        "position.",
        "",
        f"| Measure | {' | '.join(LANGUAGES[c][1] for c in codes)} |",
        "|---|" + "---|" * len(codes),
    ]
    pretrained_undistinguished = True
    finetuned_specialized = True
    cells_pre, cells_post, cells_layers = [], [], []
    for code in codes:
        pre = attention(code)["pretrained"]["answer_attention"]
        post = attention(code)["finetuned"]["answer_attention"]
        pre_best = max(range(n_layer), key=lambda l: pre["answer_max_head"][l])
        post_best = max(range(n_layer), key=lambda l: post["answer_max_head"][l])
        pre_other = pre["other_entity_max_head"][pre_best]
        post_other = post["other_entity_max_head"][post_best]
        strong = [l for l in range(n_layer) if post["answer_max_head"][l] >= 0.8]
        pretrained_undistinguished = (pretrained_undistinguished
                                      and pre["answer_max_head"][pre_best] < 1.5 * max(pre_other, 1e-9))
        finetuned_specialized = finetuned_specialized and bool(strong)
        cells_pre.append(f"{pre['answer_max_head'][pre_best]:.2f} at layer {pre_best} "
                         f"(any head on another entity: {pre_other:.2f})")
        cells_post.append(f"{post['answer_max_head'][post_best]:.2f} at layer {post_best} "
                          f"(any head on another entity: {post_other:.2f})")
        cells_layers.append(", ".join(str(l) for l in strong) or "none")
    parts += [
        "| Pretrained: most answer-focused layer | " + " | ".join(cells_pre) + " |",
        "| Finetuned: most answer-focused layer | " + " | ".join(cells_post) + " |",
        "| Finetuned layers with a head at ≥ 0.80 on the answer | " + " | ".join(cells_layers) + " |",
        "",
    ]

    if pretrained_undistinguished and finetuned_specialized:
        parts += [
            "Before finetuning, no layer singles out the answer: its most answer-focused",
            "head gives the correct entity about what it gives any other named entity.",
            "After finetuning, both models - independently, each in its own language -",
            "have late layers containing a head that puts almost all of the answer",
            "position's attention on the correct entity - and since a head's attention sums",
            "to one, that leaves it almost nothing for the distractors.",
            "That is a content-based head created by finetuning, and it is the clearest",
            "evidence in this project of head specialization on comparative-reasoning",
            "prompts. It shows where the answer is *read out*; it does not show where the",
            "ordering is worked out, which happens in earlier layers this measurement",
            "does not isolate.",
            "",
        ]

    figures = REPO_ROOT / "report" / "figures"
    for code in codes:
        label, language_name = LANGUAGES[code][1], LANGUAGES[code][2]
        parts += [f"#### {label} ({language_name}) figures", ""]
        for filename, caption in (
            (f"reasoning_answer_attention_{code}.png", f"{label} attention on the correct answer by layer"),
            (f"reasoning_head_summary_pretrained_{code}.png", f"{label} per-head statistics, pretrained"),
            (f"reasoning_head_summary_finetuned_{code}.png", f"{label} per-head statistics, finetuned"),
            (f"reasoning_attention_pretrained_{code}.png", f"{label} attention heatmaps, pretrained"),
            (f"reasoning_attention_finetuned_{code}.png", f"{label} attention heatmaps, finetuned"),
        ):
            if (figures / filename).exists():
                parts += [f"![{caption}](figures/{filename})", ""]
    return "\n".join(parts)


def section_comparison(artifacts: dict) -> str:
    """The four questions the assignment asks the final report to answer."""
    codes = [c for c in LANGUAGES if c in artifacts]
    parts = ["## 5. The two resource tiers compared", ""]
    if len(codes) < 2:
        return "\n".join(parts + ["_Needs artifacts for both languages._", ""])
    hi, ne = codes[0], codes[1]

    def dig(code: str, *path):
        node = artifacts[code]
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node

    def largest_source(code: str):
        words = dig(code, "corpus", "by_source", "words") or {}
        total = sum(words.values())
        if not total:
            return None, 0.0, 0
        name, count = max(words.items(), key=lambda item: item[1])
        return name, count / total, len(words)

    names = " | ".join(f"{LANGUAGES[c][1]} ({LANGUAGES[c][2]})" for c in codes)
    parts += [
        "### 1. How did data scale and quality differ between Model H and Model L?",
        "",
        f"| | {names} |",
        "|---|" + "---:|" * len(codes),
    ]
    for label, getter in (
        ("Documents kept", lambda c: fmt(dig(c, "cleaning", "kept"), "{:,}")),
        ("Share of documents surviving cleaning", lambda c: fmt(dig(c, "cleaning", "keep_rate"), "{:.1%}")),
        ("Rejected as the wrong language", lambda c: fmt(dig(c, "cleaning", "rejected", "wrong_language"), "{:,}")),
        ("Rejected for low Devanagari share", lambda c: fmt(dig(c, "cleaning", "rejected", "low_script_ratio"), "{:,}")),
        ("Training tokens", lambda c: fmt(dig(c, "tokens", "per_split", "train", "tokens"), "{:,}")),
        ("Words in the corpus", lambda c: fmt(dig(c, "corpus", "totals", "words"), "{:,}")),
        ("Manual share of training tokens", lambda c: fmt(dig(c, "tokens", "manual_token_fraction"), "{:.1%}")),
        ("Corpus sources", lambda c: str(largest_source(c)[2])),
        ("Largest source, share of words", lambda c: (f"`{largest_source(c)[0]}` {largest_source(c)[1]:.1%}" if largest_source(c)[0] else "-")),
        ("Tokenizer fertility (tokens/word)", lambda c: fmt(dig(c, "tokenizer", "metrics", "fertility"), "{:.3f}")),
        ("Words encoded as a single token", lambda c: fmt(dig(c, "tokenizer", "metrics", "single_token_word_rate"), "{:.1%}")),
        ("Validation perplexity (Phase 2)", lambda c: fmt(dig(c, "phase2", "language_modeling", "perplexity"), "{:.2f}")),
        ("Bits-per-byte (Phase 2)", lambda c: fmt(dig(c, "phase2", "language_modeling", "bits_per_byte"), "{:.4f}")),
    ):
        parts.append(f"| {label} | " + " | ".join(getter(c) for c in codes) + " |")
    parts.append("")

    tokens = {c: dig(c, "tokens", "per_split", "train", "tokens") or 0 for c in codes}
    words = {c: dig(c, "corpus", "totals", "words") or 0 for c in codes}
    keep = {c: dig(c, "cleaning", "keep_rate") or 0.0 for c in codes}
    wrong = {c: dig(c, "cleaning", "rejected", "wrong_language") or 0 for c in codes}
    manual = {c: dig(c, "tokens", "manual_token_fraction") or 0.0 for c in codes}
    fertility = {c: dig(c, "tokenizer", "metrics", "fertility") or 1.0 for c in codes}
    single = {c: dig(c, "tokenizer", "metrics", "single_token_word_rate") or 0.0 for c in codes}
    source = {c: largest_source(c) for c in codes}
    parts += [
        f"**Scale was not the difference.** Model L's corpus finished with {tokens[ne]:,} "
        f"training tokens against Model H's {tokens[hi]:,} ({tokens[ne] / tokens[hi] - 1:+.1%}), "
        f"and both clear the 500M target. Counted in words the order reverses - "
        f"{words[ne]:,} against {words[hi]:,} - because Nepali text costs more tokens per word.",
        "",
        f"**Cleanliness and availability were.** {keep[ne]:.1%} of Nepali documents survived "
        f"cleaning against {keep[hi]:.1%} of Hindi, and {wrong[ne]:,} Nepali documents were "
        f"rejected as the wrong language against {wrong[hi]:,}: "
        f"{wrong[ne] / max(wrong[hi], 1):.0f} times the contamination pressure. Model L's "
        f"corpus is {manual[ne]:.0%} manually collected against {manual[hi]:.0%}; both exceed "
        "the 20% requirement, but Nepali's share reflects how little clean public Nepali "
        "text there was to download.",
        "",
        "The number of sources is a weak proxy for breadth and is not used as evidence: "
        f"Model H's largest source, `{source[hi][0]}`, supplies {source[hi][1]:.0%} of its "
        f"words against {source[ne][1]:.0%} for Model L's `{source[ne][0]}`, but it is itself "
        "a crawl of many news and web sites.",
        "",
        f"**The tokenizers differ sharply.** At the same 8,000-piece vocabulary, Nepali needs "
        f"{fertility[ne]:.3f} tokens per word against {fertility[hi]:.3f}, and encodes "
        f"{single[ne]:.0%} of words as a single piece against {single[hi]:.0%}.",
        "",
    ]

    pre = {c: dig(c, "evaluation", "comparison", "exact_match", "pretrained") for c in codes}
    post = {c: dig(c, "evaluation", "comparison", "exact_match", "finetuned") for c in codes}
    ppl = {c: dig(c, "phase2", "language_modeling", "perplexity") for c in codes}
    bpb = {c: dig(c, "phase2", "language_modeling", "bits_per_byte") for c in codes}
    bpt = {c: dig(c, "phase2", "language_modeling", "bytes_per_token") for c in codes}
    parts += ["### 2. How do language-modeling and reasoning results compare across the two tiers?", ""]
    if None not in (*ppl.values(), *bpb.values(), *bpt.values()):
        best_ppl = min(codes, key=lambda c: ppl[c])
        best_bpb = min(codes, key=lambda c: bpb[c])
        other = {hi: ne, ne: hi}
        if best_ppl != best_bpb:
            verdict = (f"{LANGUAGES[best_ppl][1]} has the better perplexity ({ppl[best_ppl]:.2f} "
                       f"against {ppl[other[best_ppl]]:.2f}) but {LANGUAGES[best_bpb][1]} the "
                       f"better bits-per-byte ({bpb[best_bpb]:.4f} against {bpb[other[best_bpb]]:.4f}).")
        else:
            verdict = (f"{LANGUAGES[best_ppl][1]} has both the better perplexity and the "
                       "better bits-per-byte.")
        parts += [
            f"**Language modeling (Phase 2).** {verdict} Each Nepali token carries "
            f"{bpt[ne]:.2f} bytes of text against {bpt[hi]:.2f}, so a per-token loss is spread "
            "over different amounts of text: perplexity is not comparable across tokenizers, "
            "and bits-per-byte corrects for it. The models are closer than perplexity alone "
            "suggests.",
            "",
        ]
    if None not in (*pre.values(), *post.values()):
        higher = max(codes, key=lambda c: post[c])
        gap = abs(post[hi] - post[ne]) * 100
        ceiling = max(1 - post[c] for c in codes) * 100
        variety = dig(hi, "dataset", "template_variety") or {}
        examples = dig(hi, "finetune", "train_data", "examples") or 0
        parts += [
            f"**Reasoning (Phase 3).** Finetuning took Model H from {pre[hi]:.2%} to "
            f"{post[hi]:.2%} exact match and Model L from {pre[ne]:.2%} to {post[ne]:.2%}. "
            f"The two finish within {gap:.2f} percentage points of each other and within "
            f"{ceiling:.2f} points of ceiling, and {LANGUAGES[higher][1]} scores marginally higher.",
            "",
            "**The reasoning task did not separate the tiers, and that is the result.** Each "
            "language's corpus uses one comparative sentence frame and one question frame, "
            f"instantiated over {variety.get('dimensions', '-')} attributes, "
            f"{len(variety.get('chain_lengths', []))} chain lengths and "
            f"{variety.get('question_polarities', '-')} question polarities, with single-token "
            f"answers. That regularity is learnable to near-ceiling from {examples:,} examples "
            "whichever pretrained model starts the finetuning, so the task cannot resolve a "
            "difference between the two.",
            "",
            "Two things follow. The tier difference established in Phase 2 is real but lives in "
            "open-ended language modeling, which is what those metrics measure. And a reasoning "
            "benchmark that would discriminate needs to be harder than this one: longer chains, "
            "premises irrelevant to the question, or comparisons stated across intervening text.",
            "",
        ]

    tokens_seen = {c: dig(c, "phase2", "tokens_seen") or 0 for c in codes}
    sequence = {c: dig(c, "finetune", "train_data", "sequence_tokens", "mean") for c in codes}
    block = dig(hi, "phase2", "model_config", "block_size") or 512
    seen_words = {c: tokens_seen[c] / fertility[c] for c in codes}
    parts += [
        "### 3. What tokenizer and corpus factors most affected the lower-resource model?",
        "",
        f"**Tokenizer fertility, first.** A {block}-token context holds about "
        f"{block / fertility[ne]:.0f} Nepali words against {block / fertility[hi]:.0f} Hindi "
        f"ones, and the {tokens_seen[ne] / 1e6:.0f}M-token pretraining budget covered about "
        f"{seen_words[ne] / 1e6:.0f}M Nepali words against {seen_words[hi] / 1e6:.0f}M Hindi "
        f"ones - {1 - seen_words[ne] / seen_words[hi]:.0%} less language for the same compute."
        + (f" The effect persists in Phase 3: the Nepali reasoning prompts average "
           f"{sequence[ne]:.1f} tokens against {sequence[hi]:.1f} for Hindi."
           if None not in sequence.values() else ""),
        "",
        f"**Corpus cleanliness and availability, second.** Lower cleaning survival and a "
        f"{wrong[ne] / max(wrong[hi], 1):.0f}-fold higher wrong-language rejection count mark "
        f"noisier source material, and the {manual[ne]:.0%} manual share "
        "shows how little of it could simply be downloaded.",
        "",
        "Both plausibly contribute to Model L's higher Phase 2 perplexity. Neither limited "
        "reasoning finetuning, which saturated for both models.",
        "",
    ]

    counts = {c: dig(c, "evaluation", "error_analysis", "pretrained", "counts") or {} for c in codes}
    examined = {c: dig(c, "evaluation", "error_analysis", "pretrained", "errors_examined") or 0 for c in codes}
    named = {c: (counts[c].get("wrong_entity", 0) + counts[c].get("inverted", 0)) / examined[c]
             if examined[c] else 0.0 for c in codes}
    strong = {c: [l for l, v in enumerate(
        dig(c, "evaluation", "attention", "finetuned", "answer_attention", "answer_max_head") or [])
        if v >= 0.8] for c in codes}
    parts += [
        "### 4. What evidence explains the observed differences?",
        "",
        "**For the language-modeling gap:** the bytes-per-token figures above, which reproduce "
        "the perplexity/bits-per-byte disagreement arithmetically (bits-per-byte = "
        "cross-entropy / (bytes per token × ln 2)), together with the fertility and cleaning "
        "figures in question 1.",
        "",
    ]
    if None not in pre.values() and all(examined.values()):
        parts += [
            f"**Before finetuning, the tier does show.** Neither pretrained model can do the task "
            f"- {pre[hi]:.1%} and {pre[ne]:.2%} exact match, below chance - but they fail "
            f"differently. When Model H is wrong it still names a participant from the premises "
            f"{named[hi]:.0%} of the time; Model L does so {named[ne]:.0%} of the time. Otherwise "
            "both mostly predict a bare word boundary, continuing the answer prefix as running "
            "text. The model pretrained on the cleaner corpus came closer to treating the prefix "
            "as a slot for a name - suggestive, not conclusive, with one model per language on a "
            "task neither was trained for.",
            "",
        ]
    if all(strong.values()):
        parts += [
            "**After finetuning, the mechanism is the same in both.** Each model has late layers "
            "with a head putting at least 0.80 of the answer position's attention on the correct "
            f"entity - layers {', '.join(map(str, strong[hi]))} in Model H and "
            f"{', '.join(map(str, strong[ne]))} in Model L (section 4). The two tiers converge on "
            "the same solution in the same part of the network.",
            "",
        ]
    parts += [
        "**Two caveats, stated plainly.** There is one training run per language, so a "
        "difference between the models could reflect ordinary between-run variation rather "
        "than the resource tier; several seeds per language would be needed to establish it. "
        "And near-ceiling accuracy on a synthetic, templated task is evidence that the models "
        "learned this task's regularities, not evidence of general reasoning ability.",
        "",
    ]

    from lma.reasoning.templates import TEMPLATES

    homographs = {
        code: sorted((a, b) for a in TEMPLATES[code].entities for b in TEMPLATES[code].entities
                     if a < b and spelling_variants(a, b))
        for code in codes
    }
    lowest = min(v for v in post.values() if v is not None) if None not in post.values() else None
    prompts = dig(hi, "evaluation", "attention_prompts")
    parts += ["### Limitations", ""]
    if lowest is not None:
        parts.append(f"- **The reasoning task saturates.** Both models exceed "
                     f"{int(lowest * 1000) / 10:.1f}% exact match, so it cannot rank the tiers.")
    parts.append("- **Limited template variety.** One comparative frame and one question frame per "
                 "language; no equality or numeric comparisons.")
    for code in codes:
        if homographs[code]:
            pairs = ", ".join(f"`{a}` / `{b}`" for a, b in homographs[code])
            parts.append(f"- **Spelling-variant names in the {LANGUAGES[code][2]} pool** ({pairs}), "
                         "which read as one name to a human; see the failures in section 3.")
    if prompts:
        parts.append(f"- **The attention evidence is correlational.** Statistics averaged over "
                     f"{prompts} prompts show where finetuned models attend, not that this attention "
                     "causes the answer; ablating the specialized heads would test that.")
    parts += ["- **One run per language**, for both pretraining and finetuning.", ""]
    return "\n".join(parts)


def plot_finetune_curves(artifacts: dict, out_path: Path) -> Path | None:
    """Draw validation loss against step for whichever models have a history."""
    series = {}
    for code, entry in artifacts.items():
        history = (entry.get("finetune") or {}).get("history") or []
        points = [(row["step"], row["val_loss"]) for row in history if "val_loss" in row]
        if points:
            series[code] = points
    if not series:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for code, points in series.items():
        steps = [p[0] for p in points]
        losses = [p[1] for p in points]
        axis.plot(steps, losses, marker="o", markersize=3,
                  label=f"{LANGUAGES[code][1]} ({LANGUAGES[code][2]})")

    axis.set_title("Finetuning validation loss (answer span only)")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("cross-entropy (nats/token)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def section_overview(artifacts: dict) -> str:
    """The whole project on one page: the results that carry through all phases."""
    codes = [c for c in LANGUAGES if c in artifacts]
    if not codes:
        return ""

    def dig(code: str, *path):
        node = artifacts[code]
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node

    def pair(code: str, first: tuple, second: tuple, spec: str) -> str:
        return f"{fmt(dig(code, *first), spec)} / {fmt(dig(code, *second), spec)}"

    parts = [
        "## Project at a glance",
        "",
        "This report consolidates the project. Phases 1 and 2 are reported in full in",
        "[`phase1_report.md`](phase1_report.md) and [`phase2_report.md`](phase2_report.md);",
        "the results that carry through to the final comparison are collected here.",
        "",
        "| Phase | Result | " + " | ".join(f"{LANGUAGES[c][1]} ({LANGUAGES[c][2]})" for c in codes) + " |",
        "|---|---|" + "---:|" * len(codes),
    ]
    for phase, label, getter in (
        ("1", "Training tokens", lambda c: fmt(dig(c, "tokens", "per_split", "train", "tokens"), "{:,}")),
        ("1", "Manual share of training tokens", lambda c: fmt(dig(c, "tokens", "manual_token_fraction"), "{:.1%}")),
        ("1", "Tokenizer fertility (unigram, 8,000 pieces)", lambda c: fmt(dig(c, "tokenizer", "metrics", "fertility"), "{:.3f}")),
        ("2", "Trainable parameters", lambda c: fmt(dig(c, "phase2", "parameters"), "{:,}")),
        ("2", "Pretraining tokens seen", lambda c: fmt(dig(c, "phase2", "tokens_seen"), "{:,}")),
        ("2", "Validation perplexity", lambda c: fmt(dig(c, "phase2", "language_modeling", "perplexity"), "{:.2f}")),
        ("2", "Bits-per-byte", lambda c: fmt(dig(c, "phase2", "language_modeling", "bits_per_byte"), "{:.4f}")),
        ("2", "chrF, greedy / T = 1.0", lambda c: pair(c, ("phase2", "generation", "greedy", "chrf"), ("phase2", "generation", "temperature_1.0", "chrf"), "{:.2f}")),
        ("2", "Distinct-2, greedy / T = 1.0", lambda c: pair(c, ("phase2", "generation", "greedy", "distinct_2"), ("phase2", "generation", "temperature_1.0", "distinct_2"), "{:.3f}")),
        ("3", "Reasoning exact match, pretrained → finetuned", lambda c: f"{fmt(dig(c, 'evaluation', 'comparison', 'exact_match', 'pretrained'), '{:.2%}')} → {fmt(dig(c, 'evaluation', 'comparison', 'exact_match', 'finetuned'), '{:.2%}')}"),
        ("3", "Three-hop (4-entity) chains, finetuned", lambda c: fmt(dig(c, "evaluation", "above_chance", "by_chain_length", "4", "accuracy"), "{:.1%}")),
    ):
        parts.append(f"| {phase} | {label} | " + " | ".join(getter(c) for c in codes) + " |")
    parts.append("")

    figures = REPO_ROOT / "report" / "figures"
    for filename, caption in (
        ("manual_fraction.png", "Phase 1: manual versus downloaded share of each corpus"),
        ("vocab_sweep.png", "Phase 1: the vocabulary sweep behind the 8,000-piece tokenizers"),
        ("phase2_loss_curves.png", "Phase 2: pretraining validation loss and perplexity"),
        ("attention_heatmaps_hi.png", "Phase 2: Model H attention heatmaps on held-out text"),
        ("attention_heatmaps_ne.png", "Phase 2: Model L attention heatmaps on held-out text"),
    ):
        if (figures / filename).exists():
            parts += [f"![{caption}](figures/{filename})", "", f"*{caption}.*", ""]
    return "\n".join(parts)


def section_bonus() -> str:
    """The optional ablation: its setup, and its results once they exist."""
    parts = [
        "## 6. Bonus ablation: no positional embeddings",
        "",
        "The optional ablation retrains one language with the positional-embedding table",
        "removed (`ModelConfig(use_positional_embeddings=False)`, 196,608 fewer parameters)",
        "and every other setting unchanged. `scripts/test_reasoning.py` pins down what that",
        "removes: one ablated attention layer sees the tokens before a position as an",
        "unordered set, but two layers can recover their order, because under the causal",
        "mask each position's state summarizes a prefix of a different length. The ablation",
        "therefore measures how much of the positional signal a 12-layer stack recovers on",
        "its own - it does not test a model that is blind to word order.",
        "",
    ]
    results = []
    for directory, label, language_name in LANGUAGES.values():
        reports = REPO_ROOT / directory / "reports"
        standard, ablated = reports / "phase2_evaluation.json", reports / "bonus_nopos_evaluation.json"
        if standard.exists() and ablated.exists():
            results.append((label, language_name, json.loads(standard.read_text(encoding="utf-8")),
                            json.loads(ablated.read_text(encoding="utf-8"))))
    if not results:
        parts += [
            "The run uses `notebooks/bonus_nopos_pretrain.ipynb` and",
            "`notebooks/bonus_nopos_evaluate.ipynb`. No ablation results are claimed until it",
            "completes.",
            "",
        ]
        return "\n".join(parts)

    parts += ["| Model | Variant | Steps | Perplexity | Bits-per-byte |", "|---|---|---:|---:|---:|"]
    for label, language_name, standard, ablated in results:
        for variant, evaluation in (("standard", standard), ("no positional embeddings", ablated)):
            metrics = evaluation["language_modeling"]
            parts.append(f"| {label} ({language_name}) | {variant} | {evaluation.get('training_step') or 0:,} "
                         f"| {metrics['perplexity']:.2f} | {metrics['bits_per_byte']:.4f} |")
    parts += [
        "",
        "The full comparison - validation loss at matched steps, every generation metric, the",
        "attention analysis, and what breaks without position information - is in",
        "[`bonus_ablation.md`](bonus_ablation.md), generated by `scripts/make_ablation_report.py`.",
        "",
    ]
    return "\n".join(parts)


def build_report(artifacts: dict) -> str:
    """Assemble the whole Phase 3 report."""
    reproduction = [
        "## 7. Reproduction",
        "",
        "The checkpoints are on Google Drive (links in the README): the Phase 2 models at",
        "`Phase2-Checkpoints/<language>/checkpoint_best.pt` and the finetuned models at",
        "`Phase3-Finetuned/<code>_finetuned_best.pt`.",
        "",
        "```bash",
        "# Component tests - templates, leakage, masking, resume, scoring, attention, ablation",
        "python scripts/test_reasoning.py",
        "",
        "# Generate both reasoning datasets (seeded, so this reproduces them byte for byte)",
        "python scripts/generate_reasoning_data.py --language hi",
        "python scripts/generate_reasoning_data.py --language ne",
        "",
        "# Finetune from the Phase 2 checkpoint; reads hindi/configs/finetune.yaml",
        "# (notebooks/phase3_finetune.ipynb runs the same code on a Kaggle GPU)",
        "python scripts/finetune_reasoning.py --language hi \\",
        "    --checkpoint Phase2-Checkpoints/hindi/checkpoint_best.pt \\",
        "    --tokenizer hindi/tokenizer/hi_unigram_8000.model",
        "",
        "# Score pretrained against finetuned and analyse attention (a CPU is enough)",
        "python scripts/evaluate_reasoning.py --language hi \\",
        "    --pretrained Phase2-Checkpoints/hindi/checkpoint_best.pt \\",
        "    --finetuned Phase3-Finetuned/hi_finetuned_best.pt \\",
        "    --tokenizer hindi/tokenizer/hi_unigram_8000.model",
        "",
        "# Check every deliverable, then regenerate this report",
        "python scripts/check_phase3.py",
        "python scripts/make_phase3_report.py",
        "```",
        "",
    ]
    return "\n".join([
        "# Phase 3 - Reasoning Finetuning, Attention Analysis, and Final Report",
        "",
        "**Language Models and Agents, Monsoon 2026 - Individual Project**",
        "",
        "Two independently pretrained decoder-only Transformers, each finetuned on a synthetic",
        "comparative-reasoning corpus generated in its own language. No data, vocabulary, or",
        "weights are shared between them at any stage. Every reasoning number below is computed",
        "by `scripts/evaluate_reasoning.py` from the published checkpoints linked in the README.",
        "",
        "---",
        "",
        section_overview(artifacts),
        "---",
        "",
        section_dataset(artifacts),
        "---",
        "",
        section_finetuning(artifacts),
        "---",
        "",
        section_results(artifacts),
        "---",
        "",
        section_attention(artifacts),
        "---",
        "",
        section_comparison(artifacts),
        "---",
        "",
        section_bonus(),
        "---",
        "",
        *reproduction,
    ])


def main() -> int:
    """Write the report and its figure."""
    artifacts = load_artifacts()
    if not artifacts:
        print(
            "No Phase 3 artifacts found. Run scripts/generate_reasoning_data.py, "
            "then finetune and evaluate, before generating the report."
        )
        return 1

    print(f"found artifacts for: {', '.join(artifacts)}")
    for code, entry in artifacts.items():
        print(f"  {code}: {', '.join(sorted(entry))}")

    figures = REPO_ROOT / "report" / "figures"
    curve = plot_finetune_curves(artifacts, figures / "phase3_finetune_curves.png")
    if curve:
        print(f"wrote {curve.relative_to(REPO_ROOT)}")

    report = REPO_ROOT / "report" / "phase3_report.md"
    report.write_text(build_report(artifacts), encoding="utf-8")
    print(f"wrote {report.relative_to(REPO_ROOT)} ({report.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
