#!/usr/bin/env python3
"""Build target-domain dense all-box GRPO data from ReCon1M."""
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
TARGET_RELATIONS = {"drive-on", "sail-on"}


def valid_box(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return False
    x1, y1, x2, y2 = (float(item) for item in value)
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-targets", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        source = json.loads(line)
        if source.get("task_type") != "conditional_set_grounding":
            continue
        meta = dict(source.get("meta", {}))
        relations = set(meta.get("source_relations", []))
        if len(relations) != 1 or not relations.issubset(TARGET_RELATIONS):
            continue
        boxes = source.get("solution", {}).get("bboxes", [])
        if not isinstance(boxes, list) or len(boxes) < args.min_targets:
            continue
        if not all(valid_box(box) for box in boxes):
            raise ValueError(f"invalid bbox in {meta.get('query_id')}")
        question = str(source["messages"][0]["content"])
        question = question.split("\n\nReturn exactly one JSON object", 1)[0].strip()
        rows.append(
            {
                "messages": [{"role": "user", "content": f"{question}\n\n{PROTOCOL}"}],
                "images": list(source["images"]),
                "task_type": "conditional_dense_all_boxes",
                "solution": {"bboxes": boxes},
                "meta": {
                    **meta,
                    "output_protocol": "all_boxes_v1",
                    "grpo_role": "target_domain_dense_recall",
                    "target_count": len(boxes),
                },
            }
        )

    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": len(rows),
                "families": dict(Counter(row["meta"]["family"] for row in rows)),
                "relations": dict(
                    Counter(",".join(row["meta"]["source_relations"]) for row in rows)
                ),
                "count_min": min((row["meta"]["target_count"] for row in rows), default=None),
                "count_max": max((row["meta"]["target_count"] for row in rows), default=None),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
