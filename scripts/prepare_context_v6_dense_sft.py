#!/usr/bin/env python3
"""Add complete dense-set supervision to the v5 relation SFT set.

The full ReCon1M training split stores dense targets in ``solution.bboxes``
but its legacy assistant answer may contain count only. This script rebuilds
those training answers as the current all-box protocol and leaves v5 val/test
unchanged for a fair comparison.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL = (
    "Return exactly one JSON object with no additional text. "
    "Find every target that satisfies the condition. Always include both "
    "count and selected_bboxes. The count must equal the number of boxes in "
    "selected_bboxes. Output every matching target box; if there are no "
    "matching targets, use count 0 and selected_bboxes []. "
    "Normalize every box to 0-1000."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_dense(row: dict[str, Any]) -> dict[str, Any]:
    messages = [dict(message) for message in row["messages"]]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("row must end with assistant message")
    solution = row.get("solution", {})
    boxes = solution.get("bboxes", [])
    if not isinstance(boxes, list) or len(boxes) < 11:
        raise ValueError("dense row must contain at least 11 solution boxes")
    checked: list[list[float | int]] = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("invalid solution bbox")
        values = [float(value) for value in box]
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("solution bbox is outside 0-1000")
        checked.append(box)
    question = str(messages[0].get("content", ""))
    question = question.split("\n\nReturn exactly one JSON object", 1)[0].strip()
    messages[0]["content"] = f"{question}\n\n{PROTOCOL}"
    messages[-1]["content"] = json.dumps(
        {"count": len(checked), "selected_bboxes": checked},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    copied = {
        "messages": messages,
        "images": list(row["images"]),
        "task_type": "conditional_set_grounding",
        "meta": {
            **dict(row.get("meta", {})),
            "output_protocol": "all_boxes_v1",
            "augmentation": "dense_complete_set_sft",
            "dense_target_count": len(checked),
        },
    }
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-dir", required=True)
    parser.add_argument("--full-train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    v5 = Path(args.v5_dir)
    output = Path(args.output_dir)
    current_train = read_jsonl(v5 / "sft_train.jsonl")
    full_train = read_jsonl(Path(args.full_train))
    existing_queries = {
        row.get("meta", {}).get("query_id")
        for row in current_train
        if row.get("task_type") == "conditional_set_grounding"
    }
    candidates = []
    for row in full_train:
        meta = row.get("meta", {})
        if meta.get("query_id") in existing_queries:
            continue
        if len(meta.get("target_object_indices", [])) < 11:
            continue
        if len(row.get("solution", {}).get("bboxes", [])) < 11:
            continue
        candidates.append(normalize_dense(row))

    # Keep the original v5 rows and add only genuinely new dense queries.
    train = [*current_train, *candidates]
    random.Random(args.seed).shuffle(train)
    write_jsonl(output / "sft_train.jsonl", train)
    for name in ("sft_val.jsonl", "sft_context_val.jsonl", "sft_context_test.jsonl"):
        write_jsonl(output / name, read_jsonl(v5 / name))

    summary = {
        "protocol": "all_boxes_v1",
        "source_v5_train": len(current_train),
        "added_dense_rows": len(candidates),
        "added_by_family": dict(Counter(row["meta"].get("family") for row in candidates)),
        "added_by_relation": dict(
            Counter(
                ",".join(row["meta"].get("source_relations", []))
                for row in candidates
            )
        ),
        "final_train": len(train),
        "val": len(read_jsonl(v5 / "sft_val.jsonl")),
        "context_val": len(read_jsonl(v5 / "sft_context_val.jsonl")),
        "context_test": len(read_jsonl(v5 / "sft_context_test.jsonl")),
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
