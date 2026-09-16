#!/usr/bin/env python3
"""Prepare the audited relation-specialisation SFT data with rehearsal replay."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rs_vlm.schema import OUTPUT_INSTRUCTIONS


DEFAULT_FAMILIES = ("road_vehicle", "ship_water", "aircraft_surface")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def mark_source(row: dict[str, Any], source: str) -> dict[str, Any]:
    copied = dict(row)
    copied["meta"] = {**dict(row.get("meta", {})), "stage2_source": source}
    return copied


CONTRASTIVE_EXCLUSIONS = {
    "drive-on": "Exclude vehicles parked at a parking area.",
    "parked-at": "Exclude vehicles driving on a road.",
    "sail-on": "Exclude ships moored or docked at the shore.",
    "dock-at": "Exclude ships sailing on open water.",
    "moor-at": "Exclude ships sailing on open water.",
}


def normalize_context_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the calibrated answer protocol and expose audited distractors in the prompt."""
    copied = dict(row)
    copied["messages"] = [dict(message) for message in row.get("messages", [])]
    solution = row.get("solution", {})
    boxes = list(solution.get("bboxes", []))
    target_count = len(boxes)
    relation = str((row.get("meta", {}).get("source_relations") or [""])[0]).lower()
    question = copied["messages"][0]["content"]
    question = question.split("\n\nReturn exactly one JSON object", 1)[0].strip()
    if row.get("meta", {}).get("distractor_object_indices") and relation in CONTRASTIVE_EXCLUSIONS:
        question += f"\n\nImportant: {CONTRASTIVE_EXCLUSIONS[relation]}"
        copied["meta"] = {
            **dict(row.get("meta", {})),
            "contrastive_example": True,
        }
    question += f"\n\n{OUTPUT_INSTRUCTIONS['conditional_set_grounding']}"
    copied["messages"][0]["content"] = question
    answer = {"count": target_count}
    if target_count < 5:
        answer["selected_bboxes"] = boxes
    copied["messages"][-1]["content"] = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    return copied


def sft_view(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only conversational fields after applying the calibrated protocol."""
    if row.get("task_type") != "conditional_set_grounding":
        copied = dict(row)
    else:
        copied = normalize_context_row(row)
    copied.pop("solution", None)
    return copied


def filtered_context(rows: list[dict[str, Any]], families: set[str]) -> list[dict[str, Any]]:
    return [
        mark_source(row, "audited_relation_context")
        for row in rows
        if row.get("meta", {}).get("family") in families
        and row.get("meta", {}).get("audit_decision") == "accept"
    ]


def stratified_replay(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("task_type", "unknown"))].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    if count > len(rows):
        raise ValueError(f"requested {count} replay rows but only {len(rows)} are available")
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count:
        key = keys[cursor % len(keys)]
        if buckets[key]:
            selected.append(mark_source(buckets[key].pop(), "stage1_replay"))
        cursor += 1
        if cursor > count * len(keys) * 2:
            raise RuntimeError("unable to draw the requested stratified replay sample")
    return selected


def build(args: argparse.Namespace) -> dict[str, Any]:
    context_dir = Path(args.context_dir)
    output_dir = Path(args.output_dir)
    families = set(args.families)
    rng = random.Random(args.seed)
    core_train = filtered_context(read_jsonl(context_dir / "sft_context_train.jsonl"), families)
    core_val = filtered_context(read_jsonl(context_dir / "sft_context_val.jsonl"), families)
    core_test = filtered_context(read_jsonl(context_dir / "sft_context_test.jsonl"), families)
    if not core_train or not core_val or not core_test:
        raise ValueError("each context split must retain at least one accepted audited sample")
    if not 0 <= args.replay_fraction < 0.5:
        raise ValueError("replay_fraction must be in [0, 0.5)")
    replay_train_count = round(len(core_train) * args.replay_fraction / (1 - args.replay_fraction))
    replay_val_count = round(len(core_val) * args.replay_fraction / (1 - args.replay_fraction))
    replay_train = stratified_replay(read_jsonl(Path(args.replay_train)), replay_train_count, rng)
    replay_val = stratified_replay(read_jsonl(Path(args.replay_val)), replay_val_count, rng)
    train_rows = core_train + replay_train
    val_rows = core_val + replay_val
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    write_jsonl(output_dir / "sft_train.jsonl", [sft_view(row) for row in train_rows])
    write_jsonl(output_dir / "sft_val.jsonl", [sft_view(row) for row in val_rows])
    write_jsonl(output_dir / "sft_context_test.jsonl", [normalize_context_row(row) for row in core_test])
    contrastive_train = sum(
        bool(row.get("meta", {}).get("contrastive_example"))
        for row in (normalize_context_row(item) for item in core_train)
    )
    summary = {
        "seed": args.seed,
        "families": sorted(families),
        "replay_fraction_of_final_train": args.replay_fraction,
        "context": {"train": len(core_train), "val": len(core_val), "test": len(core_test)},
        "replay": {
            "train": len(replay_train),
            "val": len(replay_val),
            "train_by_task": dict(Counter(row["task_type"] for row in replay_train)),
            "val_by_task": dict(Counter(row["task_type"] for row in replay_val)),
        },
        "final": {"train": len(train_rows), "val": len(val_rows), "context_test": len(core_test)},
        "contrastive": {"context_train": contrastive_train},
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited relation Stage-2 SFT data with Stage-1 replay")
    parser.add_argument("--context-dir", required=True)
    parser.add_argument("--replay-train", required=True)
    parser.add_argument("--replay-val", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--replay-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
