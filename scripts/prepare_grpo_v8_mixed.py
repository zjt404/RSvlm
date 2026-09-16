#!/usr/bin/env python3
"""Build mixed-density relation GRPO data with generic-capability replay."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTOCOL = (
    "Return exactly one JSON object with no additional text. Find every target "
    "that satisfies the condition. Always include both count and selected_bboxes. "
    "The count must equal the number of boxes in selected_bboxes. If none match, "
    "use count 0 and selected_bboxes []. Normalize every box to 0-1000."
)
BIN_QUOTAS = {"1_4": 0.40, "5_10": 0.30, "11_plus": 0.30}


def valid_box(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return False
    x1, y1, x2, y2 = map(float, value)
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def density_bin(count: int) -> str:
    return "1_4" if count <= 4 else "5_10" if count <= 10 else "11_plus"


def polygon_box(points: Any, width: float, height: float) -> list[int] | None:
    if not isinstance(points, list) or not points:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, IndexError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    box = [round(1000 * min(xs) / width), round(1000 * min(ys) / height), round(1000 * max(xs) / width), round(1000 * max(ys) / height)]
    return box if valid_box(box) else None


def load_instances(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records[str(row["image_id"])] = row
    return records


def select_balanced(rows: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    """Uniformly cycle relations so frequent relations cannot dominate a bin."""
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_relation[row["meta"]["relation"]].append(row)
    for values in by_relation.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    relations = list(by_relation)
    rng.shuffle(relations)
    cursor = 0
    while len(selected) < quota and relations:
        relation = relations[cursor % len(relations)]
        values = by_relation[relation]
        if values:
            selected.append(values.pop())
        if not values:
            relations.remove(relation)
            cursor = 0
        else:
            cursor += 1
    if len(selected) < quota:
        raise RuntimeError(f"only {len(selected)} valid records available for quota {quota}")
    return selected


def relation_rows(source_path: Path, instances: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            if source.get("task_type") != "conditional_set_grounding":
                continue
            solution = source.get("solution", {})
            targets = solution.get("bboxes", [])
            if not isinstance(targets, list) or not targets or not all(valid_box(box) for box in targets):
                continue
            meta = dict(source.get("meta", {}))
            image_id = str(meta.get("image_id", ""))
            instance = instances.get(image_id)
            relations = meta.get("source_relations", [])
            if instance is None or not isinstance(relations, list) or len(relations) != 1:
                continue
            relation = str(relations[0])
            width, height = float(instance["width"]), float(instance["height"])
            boxes = [polygon_box(poly, width, height) for poly in instance.get("polygons", [])]
            target_indices = {int(index) for index in meta.get("target_indices", [])}
            distractor_indices = {int(index) for index in meta.get("distractor_indices", [])}
            ignore_indices = {int(index) for index in meta.get("ignore_indices", [])}

            def indexed(indices: set[int]) -> list[list[int]]:
                return [boxes[index] for index in sorted(indices) if 0 <= index < len(boxes) and boxes[index] is not None]

            question = str(source["messages"][0]["content"]).split("\n\nReturn exactly one JSON object", 1)[0].strip()
            rows.append({
                "messages": [{"role": "user", "content": f"{question}\n\n{PROTOCOL}"}],
                "images": list(source["images"]),
                "task_type": "conditional_all_boxes_v8",
                "solution": {
                    "bboxes": targets,
                    "distractor_bboxes": indexed(distractor_indices),
                    "ignore_bboxes": indexed(ignore_indices),
                    "all_annotated_bboxes": [box for box in boxes if box is not None],
                },
                "meta": {
                    **meta,
                    "relation": relation,
                    "target_count": len(targets),
                    "density_bin": density_bin(len(targets)),
                    "output_protocol": "all_boxes_v8",
                    "grpo_role": "relation_mixed_density",
                },
            })
    return rows


def generic_replay_rows(path: Path, per_task: int, rng: random.Random) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task = row.get("task_type")
            if task not in {"vqa", "count", "grounding", "spatial"}:
                continue
            messages = [message for message in row.get("messages", []) if message.get("role") != "assistant"]
            if not messages or not isinstance(row.get("solution"), dict):
                continue
            by_task[task].append({
                **row,
                "messages": messages,
                "meta": {**dict(row.get("meta", {})), "grpo_role": "generic_replay"},
            })
    output: list[dict[str, Any]] = []
    for task in ("vqa", "count", "grounding", "spatial"):
        rng.shuffle(by_task[task])
        if len(by_task[task]) < per_task:
            raise RuntimeError(f"only {len(by_task[task])} {task} replay rows available")
        output.extend(by_task[task][:per_task])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-input", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--generic-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relation-total", type=int, default=2000)
    parser.add_argument("--generic-per-task", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    candidates = relation_rows(args.relation_input, load_instances(args.instances))
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bin[row["meta"]["density_bin"]].append(row)
    relation_output: list[dict[str, Any]] = []
    for name, ratio in BIN_QUOTAS.items():
        relation_output.extend(select_balanced(by_bin[name], round(args.relation_total * ratio), rng))
    replay_output = generic_replay_rows(args.generic_input, args.generic_per_task, rng)
    output = relation_output + replay_output
    rng.shuffle(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "records": len(output),
        "task_types": dict(Counter(row["task_type"] for row in output)),
        "density": dict(Counter(row.get("meta", {}).get("density_bin", "generic") for row in output)),
        "relations": dict(Counter(row.get("meta", {}).get("relation", "generic") for row in relation_output)),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
