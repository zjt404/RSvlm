#!/usr/bin/env python3
"""Build relation SFT data with an all-boxes output protocol.

Relation samples always expose every selected box.  Non-relation replay
samples are copied unchanged so the base model's general abilities remain in
the mixed SFT set.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


RELATION_TASK = "conditional_set_grounding"
PROTOCOL_TEXT = (
    "Return exactly one JSON object with no additional text. "
    "Find every target that satisfies the condition. Always include both "
    "count and selected_bboxes. The count must equal the number of boxes in "
    "selected_bboxes. Output every matching target box; if there are no "
    "matching targets, use count 0 and selected_bboxes []. "
    "Normalize every box to 0-1000."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_relation(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    copied["messages"] = [dict(message) for message in row.get("messages", [])]
    if not copied["messages"] or copied["messages"][-1].get("role") != "assistant":
        raise ValueError("relation row must end with an assistant message")
    solution = dict(row.get("solution", {}))
    boxes = [box for box in solution.get("bboxes", []) if isinstance(box, list) and len(box) == 4]
    question = str(copied["messages"][0].get("content", ""))
    question = question.split("\n\nReturn exactly one JSON object", 1)[0].strip()
    copied["messages"][0]["content"] = f"{question}\n\n{PROTOCOL_TEXT}"
    answer = {"count": len(boxes), "selected_bboxes": boxes}
    copied["messages"][-1]["content"] = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    copied["meta"] = {**dict(row.get("meta", {})), "output_protocol": "all_boxes_v1"}
    return copied


def sft_view(row: dict[str, Any]) -> dict[str, Any]:
    copied = normalize_relation(row) if row.get("task_type") == RELATION_TASK else dict(row)
    copied.pop("solution", None)
    return copied


def relation_test_view(row: dict[str, Any]) -> dict[str, Any]:
    return normalize_relation(row)


def replay_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    return [row for row in rows if row.get("meta", {}).get("stage2_source") == "stage1_replay"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-dir", required=True)
    parser.add_argument("--stage2-train", required=True)
    parser.add_argument("--stage2-val", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    context = Path(args.context_dir)
    output = Path(args.output_dir)
    rng = random.Random(args.seed)

    relation_train = read_jsonl(context / "sft_context_train.jsonl")
    relation_val = read_jsonl(context / "sft_context_val.jsonl")
    relation_test = read_jsonl(context / "sft_context_test.jsonl")
    if not all(row.get("task_type") == RELATION_TASK for row in relation_train + relation_val + relation_test):
        raise ValueError("context input contains a non-relation row")

    replay_train = replay_rows(Path(args.stage2_train))
    replay_val = replay_rows(Path(args.stage2_val))
    train = [*relation_train, *replay_train]
    val = [*relation_val, *replay_val]
    rng.shuffle(train)
    rng.shuffle(val)

    write_jsonl(output / "sft_train.jsonl", [sft_view(row) for row in train])
    write_jsonl(output / "sft_val.jsonl", [sft_view(row) for row in val])
    write_jsonl(output / "sft_context_val.jsonl", [relation_test_view(row) for row in relation_val])
    write_jsonl(output / "sft_context_test.jsonl", [relation_test_view(row) for row in relation_test])

    summary = {
        "seed": args.seed,
        "protocol": "all_boxes_v1",
        "relation_task": RELATION_TASK,
        "relation": {
            "train": len(relation_train),
            "val": len(relation_val),
            "test": len(relation_test),
        },
        "stage1_replay": {
            "train": len(replay_train),
            "val": len(replay_val),
            "train_by_task": dict(Counter(row.get("task_type", "unknown") for row in replay_train)),
            "val_by_task": dict(Counter(row.get("task_type", "unknown") for row in replay_val)),
        },
        "final": {
            "train": len(train),
            "val": len(val),
            "test": len(relation_test),
        },
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
