#!/usr/bin/env python3
"""Prepare count-only dense conditional-grounding rows for GRPO."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        source = json.loads(line)
        if source.get("task_type") != "conditional_set_grounding":
            continue
        target_count = len(source["meta"]["target_object_indices"])
        if target_count < 5:
            continue
        rows.append(
            {
                "messages": [source["messages"][0]],
                "images": source["images"],
                "task_type": "conditional_dense_count",
                "solution": {"value": target_count},
                "meta": source["meta"],
            }
        )

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
                "difficulties": dict(Counter(row["meta"]["difficulty"] for row in rows)),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
