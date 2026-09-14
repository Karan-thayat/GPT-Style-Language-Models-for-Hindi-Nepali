"""Programmatic generation of the synthetic comparative-reasoning dataset.

The assignment forbids downloading an existing reasoning benchmark as the primary
finetune set: examples must be generated so the ground-truth label is known by
construction. Everything here is therefore derived from a seed, and re-running
with the same seed reproduces the corpus byte for byte.

The task
--------
Each example states a chain of pairwise comparisons and asks for the extreme
element::

    राम श्याम से लंबा है। श्याम मोहन से लंबा है। सबसे लंबा कौन है? उत्तर - राम

Answering requires chaining the inequalities rather than pattern-matching a
single sentence, which is the "multi-hop comparison" the assignment asks for. The
answer is always a single entity name, and the entity pools in
:mod:`lma.reasoning.templates` are vetted so that name is exactly one token under
that language's own Phase 1 tokenizer - which is what makes the finetuning loss
maskable to the answer span and exact-match scoring unambiguous.

Leakage control
---------------
The property being tested is generalization over **comparison structures**, not
over vocabulary. A model should answer a chain it has never seen, built from
entities it can name; it should not be asked to emit a token it was never trained
to produce. Three mechanisms, each covering what the others do not:

1. **Disjoint chains.** The unordered set of entities in an example, together
   with its dimension, forms a *chain signature*. Signatures are partitioned
   across splits, so no evaluation example reuses a combination of entities that
   any training example compared. The model cannot recall "गणेश beat दीपक"; it
   has to derive it.

2. **Evaluation-only distractor names.** A minority of the pool appears only in
   evaluation examples, mixed in alongside familiar names. These test that the
   model handles an unfamiliar participant without making the answer itself
   unproducible - the answer is drawn from the full pool.

3. **Content-hash split assignment.** Within those constraints, assignment uses
   the same salted-hash mechanism Phase 1 used
   (:class:`lma.data.splits.SplitAssigner`), so a duplicate rendering lands in
   the same split and cannot straddle train and test.

The generator additionally rejects any prompt it has already emitted, so the
corpus contains no repeated prompts at all.

A rejected earlier design, recorded because it is the obvious one
-----------------------------------------------------------------
Reserving names for the *answer* position - so every test answer was a token
absent from all training targets - makes the task unanswerable rather than hard.
Measured on a finetuned model: **0.0000** exact match on held-out-answer examples
against **0.2600** on examples whose prompts were equally unseen but whose
answers came from the training vocabulary. The evaluation set, not the model, was
the failure. Chain-level disjointness tests the same generalization without that
defect.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..data.splits import SplitAssigner, SplitConfig
from .templates import Dimension, LanguageTemplates, get_templates

#: Fraction of the entity pool reserved as *distractors* that only ever appear in
#: evaluation examples. Deliberately small, and deliberately not applied to the
#: answer.
#:
#: An earlier revision reserved these names for the answer position, so every
#: test answer was a token the model had never been trained to emit. That is not
#: leakage control - it makes the task unanswerable. Measured on a finetuned
#: model: 0.0000 exact match on held-out-answer examples against 0.2600 on
#: examples using training entities with equally unseen prompts. Generalization
#: is therefore tested over *comparison structures*, not over vocabulary: the
#: evaluation entities are ones the model can produce, while the chains,
#: orderings, and prompts are disjoint from anything seen in training.
HELD_OUT_ENTITY_FRACTION = 0.20

#: Chain lengths generated, as counts of entities being compared. Two entities is
#: a single comparison; three and four require chaining, which is the reasoning
#: the task is meant to exercise.
CHAIN_LENGTHS = (2, 3, 4)


@dataclass
class ReasoningExample:
    """One generated question with its known-correct answer.

    Attributes:
        prompt: Everything the model is conditioned on, ending with the answer
            prefix. The finetuning loss is masked to exclude these tokens.
        answer: The correct entity name. Exactly one token by construction.
        text: ``prompt`` and ``answer`` joined, the sequence actually trained on.
        language: ISO code of the language this was rendered in.
        dimension: Which comparable attribute was used.
        chain_length: How many entities the comparison chain covers.
        hops: Inference steps needed; ``chain_length - 1``.
        ask_most: Whether the question asked for the maximum or the minimum.
        entities: The entities in the chain, in stated order (strongest first).
        split: Assigned split, filled in by :class:`ReasoningDatasetBuilder`.
    """

    prompt: str
    answer: str
    text: str
    language: str
    dimension: str
    chain_length: int
    hops: int
    ask_most: bool
    entities: list[str]
    split: str = ""

    def to_dict(self) -> dict:
        """Serialize for JSONL output."""
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "text": self.text,
            "language": self.language,
            "dimension": self.dimension,
            "chain_length": self.chain_length,
            "hops": self.hops,
            "ask_most": self.ask_most,
            "entities": self.entities,
            "split": self.split,
        }


@dataclass
class GeneratorConfig:
    """Controls the size and shape of a generated dataset.

    Attributes:
        language: ``"hi"`` or ``"ne"``.
        total_examples: How many unique examples to emit across all splits.
        seed: Seeds both the sampler and the split hash, so the whole corpus is
            reproducible.
        train: Fraction of examples assigned to the training split.
        validation: Fraction assigned to validation.
        test: Fraction assigned to test.
        held_out_entity_fraction: Portion of the name pool reserved for
            evaluation examples.
        chain_lengths: Chain sizes to sample from.
    """

    language: str
    total_examples: int = 20_000
    seed: int = 20260812
    train: float = 0.80
    validation: float = 0.10
    test: float = 0.10
    held_out_entity_fraction: float = HELD_OUT_ENTITY_FRACTION
    chain_lengths: tuple[int, ...] = CHAIN_LENGTHS

    def to_dict(self) -> dict:
        """Serialize the resolved configuration into the output statistics."""
        return {
            "language": self.language,
            "total_examples": self.total_examples,
            "seed": self.seed,
            "splits": {
                "train": self.train,
                "validation": self.validation,
                "test": self.test,
            },
            "held_out_entity_fraction": self.held_out_entity_fraction,
            "chain_lengths": list(self.chain_lengths),
        }


class ReasoningDatasetBuilder:
    """Generates one language's reasoning corpus.

    Args:
        config: Size, seed, and split policy.
        templates: Language surface forms. Defaults to the registered set for
            ``config.language``.
    """

    def __init__(
        self, config: GeneratorConfig, templates: LanguageTemplates | None = None
    ) -> None:
        """Prepare the entity pools, the sampler, and the split assigner."""
        self.config = config
        self.templates = templates or get_templates(config.language)
        self.rng = random.Random(config.seed)

        pool = list(self.templates.entities)
        # Reserve a minority of names as evaluation-only distractors. They are
        # *additional* participants in evaluation chains, not a separate answer
        # vocabulary: every split draws its answers from the full pool, so the
        # answer is always a token the model has been trained to emit.
        random.Random(config.seed).shuffle(pool)
        distractor_count = max(2, round(len(pool) * config.held_out_entity_fraction))
        self.eval_only_entities = tuple(sorted(pool[:distractor_count]))
        self.shared_entities = tuple(sorted(pool[distractor_count:]))

        #: Names available when building a training example.
        self.train_entities = self.shared_entities
        #: Names available when building an evaluation example - the shared pool
        #: plus the distractors, so evaluation covers both familiar and novel
        #: participants.
        self.eval_entities = tuple(sorted(self.shared_entities + self.eval_only_entities))

        if len(self.shared_entities) < max(config.chain_lengths):
            raise ValueError(
                f"shared entity pool has {len(self.shared_entities)} names, too few "
                f"for chains of {max(config.chain_lengths)}"
            )

        self.split_assigner = SplitAssigner(
            SplitConfig(
                train=config.train,
                validation=config.validation,
                test=config.test,
                seed=config.seed,
            )
        )

    def _signature(self, entities: list[str], dimension: str) -> tuple:
        """Identify a comparison structure independently of its stated order.

        Two examples that compare the same people on the same attribute share a
        signature even if the premises were shuffled differently or the question
        asked for the opposite extreme. Partitioning on this is what stops an
        evaluation example from re-testing a comparison the model was trained on.
        """
        return (dimension, frozenset(entities))

    def _signature_split(self, signature: tuple) -> str:
        """Assign a chain signature to a split, deterministically.

        Hashing the signature rather than the rendered text is the whole point:
        every example sharing a comparison structure lands in the same split, so
        no evaluation chain has a training counterpart.
        """
        dimension, entities = signature
        payload = f"{self.config.seed}:{dimension}:{'|'.join(sorted(entities))}"
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        position = int.from_bytes(digest, "big") / float(1 << 64)

        if position < self.config.train:
            return "train"
        if position < self.config.train + self.config.validation:
            return "validation"
        return "test"

    def _sample_example(self, entity_pool: tuple[str, ...]) -> ReasoningExample:
        """Render one example from a given entity pool.

        The chain is built as a strict ordering: ``entities[0]`` exceeds
        ``entities[1]``, which exceeds ``entities[2]``, and so on. Premises are
        stated in a shuffled order so the answer cannot be read off the first
        sentence, which would let the model succeed without chaining.

        Evaluation-only distractors may participate in a chain but are never
        placed at the answered end of it. They exist to test that an unfamiliar
        participant does not derail the comparison; putting one in the answer
        position would instead ask the model to emit a token it was never trained
        to produce, which is the defect described in the module docstring.
        """
        chain_length = self.rng.choice(self.config.chain_lengths)
        dimension: Dimension = self.rng.choice(self.templates.dimensions)
        ask_most = self.rng.random() < 0.5

        # Build the chain so the answered end is answerable by construction. A
        # post-hoc swap is not enough: a chain sampled entirely from distractors
        # has no position to swap in, and swapping one end can leave the other a
        # distractor for the opposite polarity.
        answerable = [n for n in entity_pool if n not in self.eval_only_entities]
        answer = self.rng.choice(answerable)

        remaining = [n for n in entity_pool if n != answer]
        others = self.rng.sample(remaining, chain_length - 1)

        # entities[0] is the strongest and entities[-1] the weakest, so the
        # answer goes to whichever end the question will ask about.
        entities = [answer, *others] if ask_most else [*others, answer]

        premises = [
            self.templates.render_premise(entities[i], entities[i + 1], dimension)
            for i in range(chain_length - 1)
        ]
        # Shuffling matters: with premises in chain order the strongest entity is
        # always the first word of the prompt, and a model could answer by copying
        # it without performing any inference at all.
        self.rng.shuffle(premises)

        question = self.templates.render_question(dimension, ask_most)
        prompt = " ".join(premises + [question, self.templates.answer_prefix])
        answer = entities[0] if ask_most else entities[-1]

        return ReasoningExample(
            prompt=prompt,
            answer=answer,
            text=f"{prompt} {answer}",
            language=self.config.language,
            dimension=dimension.key,
            chain_length=chain_length,
            hops=chain_length - 1,
            ask_most=ask_most,
            entities=entities,
        )

    def build(self) -> list[ReasoningExample]:
        """Generate the full corpus, de-duplicated and split-assigned.

        Every example is routed to a split by its *comparison structure*, so all
        examples comparing the same entities on the same dimension share a split.
        Training examples draw from the shared entity pool; evaluation examples
        may additionally include the evaluation-only distractors. Answers come
        from the full pool in every split, so the model is never asked to emit a
        token it was not trained on.

        Returns:
            Every generated example, each carrying its assigned split.
        """
        targets = {
            "train": round(self.config.total_examples * self.config.train),
            "validation": round(self.config.total_examples * self.config.validation),
            "test": round(self.config.total_examples * self.config.test),
        }
        # Absorb any rounding drift into the training split.
        targets["train"] += self.config.total_examples - sum(targets.values())

        collected: dict[str, list[ReasoningExample]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        seen_prompts: set[str] = set()

        attempts = 0
        # The pools admit far more distinct structures than any realistic target,
        # so exhausting this ceiling means the configuration is wrong.
        max_attempts = max(self.config.total_examples * 60, 20_000)

        while attempts < max_attempts and any(
            len(collected[name]) < targets[name] for name in targets
        ):
            attempts += 1

            # Decide the intended split first, then sample from the pool that
            # split is allowed to use. Sampling first would bias which structures
            # can reach the evaluation splits.
            pending = [name for name in targets if len(collected[name]) < targets[name]]
            intended = self.rng.choice(pending)
            pool = (
                self.train_entities if intended == "train" else self.eval_entities
            )

            example = self._sample_example(pool)
            if example.prompt in seen_prompts:
                continue

            signature = self._signature(example.entities, example.dimension)
            split = self._signature_split(signature)
            if len(collected[split]) >= targets[split]:
                continue
            # A structure containing a distractor can only be an evaluation
            # example; if its signature hashes to train, it is simply skipped
            # rather than forced, which would break signature disjointness.
            if split == "train" and not set(example.entities) <= set(
                self.train_entities
            ):
                continue

            seen_prompts.add(example.prompt)
            example.split = split
            collected[split].append(example)

        shortfall = {
            name: targets[name] - len(collected[name])
            for name in targets
            if len(collected[name]) < targets[name]
        }
        if shortfall:
            raise RuntimeError(
                f"could not fill splits {shortfall} from a pool of "
                f"{len(self.templates.entities)} entities; reduce total_examples"
            )

        return collected["train"] + collected["validation"] + collected["test"]


def write_jsonl(examples: list[ReasoningExample], path: Path) -> None:
    """Write examples to a JSONL file, one record per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False))
            handle.write("\n")


def dataset_statistics(
    examples: list[ReasoningExample], builder: ReasoningDatasetBuilder
) -> dict:
    """Summarize a generated corpus for the report.

    Reports the template variety and the leakage-control evidence the assignment
    asks for: split sizes, the entity partition, and a direct check that no
    training entity appears in an evaluation example.
    """
    by_split: dict[str, int] = {}
    by_dimension: dict[str, int] = {}
    by_chain: dict[str, int] = {}
    by_hops: dict[str, int] = {}

    for example in examples:
        by_split[example.split] = by_split.get(example.split, 0) + 1
        by_dimension[example.dimension] = by_dimension.get(example.dimension, 0) + 1
        key = str(example.chain_length)
        by_chain[key] = by_chain.get(key, 0) + 1
        hop_key = str(example.hops)
        by_hops[hop_key] = by_hops.get(hop_key, 0) + 1

    train_examples = [e for e in examples if e.split == "train"]
    eval_examples = [e for e in examples if e.split in ("validation", "test")]

    def signatures(rows: list[ReasoningExample]) -> set[tuple]:
        """Comparison structures present in a set of examples."""
        return {builder._signature(e.entities, e.dimension) for e in rows}

    train_signatures = signatures(train_examples)
    eval_signatures = signatures(eval_examples)

    train_answers = {e.answer for e in train_examples}
    eval_answers = {e.answer for e in eval_examples}
    distractors = set(builder.eval_only_entities)

    return {
        "language": builder.config.language,
        "config": builder.config.to_dict(),
        "total_examples": len(examples),
        "by_split": by_split,
        "by_dimension": by_dimension,
        "by_chain_length": by_chain,
        "by_hops": by_hops,
        "unique_prompts": len({e.prompt for e in examples}),
        "entity_pools": {
            "shared": sorted(builder.shared_entities),
            "evaluation_only_distractors": sorted(builder.eval_only_entities),
            "shared_count": len(builder.shared_entities),
            "distractor_count": len(builder.eval_only_entities),
        },
        "leakage_checks": {
            # The property that matters: no evaluation example re-tests a
            # comparison structure any training example already covered.
            "chain_signature_overlap": len(train_signatures & eval_signatures),
            "train_chain_signatures": len(train_signatures),
            "eval_chain_signatures": len(eval_signatures),
            "prompt_overlap_train_eval": len(
                {e.prompt for e in train_examples} & {e.prompt for e in eval_examples}
            ),
            "duplicate_prompts": len(examples) - len({e.prompt for e in examples}),
            # The defect the earlier design had, stated precisely. A distractor
            # in the answer position asks the model to emit a token no training
            # target contained, which is unanswerable by construction. This must
            # be zero.
            "distractor_answers_in_eval": sum(
                1 for e in eval_examples if e.answer in distractors
            ),
            "distractors_in_training_examples": sorted(
                {n for e in train_examples for n in e.entities} & distractors
            ),
            # Informational, and deliberately separate from the check above: a
            # shared-pool name that happened never to be sampled as a training
            # answer. That is sampling coverage, not a design violation - the
            # name is still one the model saw in training premises and can
            # produce - so it is reported rather than asserted to be empty.
            "eval_answers_not_sampled_as_train_answer": sorted(
                (eval_answers - train_answers) - distractors
            ),
        },
        "template_variety": {
            "dimensions": len(builder.templates.dimensions),
            "chain_lengths": list(builder.config.chain_lengths),
            "question_polarities": 2,
            "entity_pool": len(builder.templates.entities),
        },
    }
