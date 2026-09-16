from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rs_vlm.schema import TASK_TYPES, SchemaError, TrainingExample, write_jsonl


DEFAULT_TRAIN_QUOTAS = {"vqa": 15000, "count": 10000, "grounding": 10000, "spatial": 5000}
DEFAULT_EVAL_QUOTAS = {"vqa": 1500, "count": 1000, "grounding": 1000, "spatial": 500}


def parse_quotas(value: str | None, defaults: dict[str, int]) -> dict[str, int]:
    if not value:
        return defaults.copy()
    result: dict[str, int] = {}
    for item in value.split(","):
        task, raw_count = item.split("=", 1)
        task = task.strip()
        if task not in TASK_TYPES:
            raise ValueError(f"unknown quota task: {task}")
        result[task] = int(raw_count)
    return result


def load_examples(paths: Iterable[str | Path]) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    example = TrainingExample.from_dict(json.loads(line)).validate(require_images=False)
                except (json.JSONDecodeError, SchemaError) as exc:
                    raise SchemaError(f"{path}:{line_number}: {exc}") from exc
                examples.append(example)
    return examples


def assert_no_image_leakage(examples: Iterable[TrainingExample]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for example in examples:
        key = (str(example.meta.get("source")), str(example.meta.get("image_id")))
        split = str(example.meta.get("split"))
        previous = seen.setdefault(key, split)
        if previous != split:
            raise SchemaError(f"image leakage for {key}: {previous} and {split}")


def _sample_count(examples: list[TrainingExample], limit: int, zero_ratio: float, rng: random.Random) -> list[TrainingExample]:
    zeros = [example for example in examples if example.solution.get("value") == 0]
    positives = [example for example in examples if example.solution.get("value") != 0]
    rng.shuffle(zeros)
    rng.shuffle(positives)
    desired_zero = min(len(zeros), round(limit * zero_ratio))
    selected = zeros[:desired_zero] + positives[: max(0, limit - desired_zero)]
    if len(selected) < limit:
        used = {id(example) for example in selected}
        remainder = [example for example in examples if id(example) not in used]
        rng.shuffle(remainder)
        selected.extend(remainder[: limit - len(selected)])
    rng.shuffle(selected)
    return selected


def sample_split(
    examples: list[TrainingExample],
    quotas: dict[str, int],
    *,
    seed: int,
    zero_ratio: float = 0.2,
) -> list[TrainingExample]:
    rng = random.Random(seed)
    by_task: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        by_task[example.task_type].append(example)
    selected: list[TrainingExample] = []
    for task, limit in quotas.items():
        candidates = by_task.get(task, [])
        if task == "count":
            current = _sample_count(candidates, limit, zero_ratio, rng)
        else:
            rng.shuffle(candidates)
            current = candidates[:limit]
        selected.extend(current)
    rng.shuffle(selected)
    return selected


def assemble(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    train_quotas: dict[str, int] | None = None,
    eval_quotas: dict[str, int] | None = None,
    grpo_size: int = 8000,
    seed: int = 42,
) -> dict[str, Any]:
    examples = load_examples(inputs)
    assert_no_image_leakage(examples)
    output_dir = Path(output_dir)
    train_quotas = train_quotas or DEFAULT_TRAIN_QUOTAS
    eval_quotas = eval_quotas or DEFAULT_EVAL_QUOTAS
    summary: dict[str, Any] = {"available": Counter(example.task_type for example in examples)}
    written: dict[str, list[TrainingExample]] = {}
    for offset, split in enumerate(("train", "val", "test")):
        candidates = [example for example in examples if example.meta.get("split") == split]
        quotas = train_quotas if split == "train" else eval_quotas
        selected = sample_split(candidates, quotas, seed=seed + offset)
        written[split] = selected
        write_jsonl(output_dir / f"sft_{split}.jsonl", selected)
        summary[split] = Counter(example.task_type for example in selected)

    grpo_candidates = [example for example in written["train"] if example.task_type in {"count", "spatial"}]
    count_limit = round(grpo_size * 0.6)
    grpo = sample_split(
        grpo_candidates,
        {"count": count_limit, "spatial": grpo_size - count_limit},
        seed=seed + 10,
    )
    grpo_records = []
    for example in grpo:
        raw = example.to_dict()
        raw["messages"] = [message for message in raw["messages"] if message.get("role") != "assistant"]
        grpo_records.append(raw)
    write_jsonl(output_dir / "grpo_train.jsonl", grpo_records)
    summary["grpo_train"] = Counter(example.task_type for example in grpo)
    count_only = [record for record in grpo_records if record["task_type"] == "count"]
    write_jsonl(output_dir / "grpo_count_train.jsonl", count_only)
    summary["grpo_count_train"] = Counter(record["task_type"] for record in count_only)
    serializable = {key: dict(value) if isinstance(value, Counter) else value for key, value in summary.items()}
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return serializable
