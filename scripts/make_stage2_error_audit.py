#!/usr/bin/env python3
"""Render a small, stratified visual audit set for Stage-2 relation errors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def payload(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2, y2] if 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000 else None


def polygon_box(polygon: Any, width: int, height: int) -> list[float] | None:
    points = []
    for point in polygon or []:
        x, y = (point.get("x"), point.get("y")) if isinstance(point, dict) else point[:2]
        if x is not None and y is not None:
            points.append((float(x), float(y)))
    if not points or width <= 0 or height <= 0:
        return None
    xs, ys = zip(*points)
    return box([1000 * min(xs) / width, 1000 * min(ys) / height, 1000 * max(xs) / width, 1000 * max(ys) / height])


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - inter
    return inter / union if union else 0.0


def match_count(predicted: list[list[float]], truth: list[list[float]]) -> int:
    pairs = sorted(((iou(p, t), pi, ti) for pi, p in enumerate(predicted) for ti, t in enumerate(truth)), reverse=True)
    used_pred, used_truth, matched = set(), set(), 0
    for score, pi, ti in pairs:
        if score < 0.5:
            break
        if pi not in used_pred and ti not in used_truth:
            used_pred.add(pi)
            used_truth.add(ti)
            matched += 1
    return matched


def pixels(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return tuple(round(value * scale / 1000) for value, scale in zip(bbox, (width, height, width, height)))  # type: ignore[return-value]


def render(row: dict[str, Any], output: Path) -> None:
    with Image.open(row["raw"]["input_image"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for bbox in row["target_boxes"]:
        draw.rectangle(pixels(bbox, width, height), outline="#18a558", width=4)
    for bbox in row["distractor_boxes"]:
        draw.rectangle(pixels(bbox, width, height), outline="#d64545", width=4)
    for bbox in row["predicted_boxes"]:
        draw.rectangle(pixels(bbox, width, height), outline="#f2c94c", width=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    instances = {
        row["image_id"]: row
        for line in args.instances.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))
    }
    candidates: list[dict[str, Any]] = []
    for ordinal, line in enumerate(args.predictions.read_text(encoding="utf-8").splitlines()):
        raw = json.loads(line)
        solution = raw["solution"]
        instance = instances[raw["meta"]["image_id"]]
        targets = [item for item in (box(value) for value in solution.get("bboxes", [])) if item]
        parsed = payload(raw.get("prediction", ""))
        predictions = [item for item in (box(value) for value in (parsed or {}).get("selected_bboxes", [])) if item]
        count = (parsed or {}).get("count")
        count = count if type(count) is int and count >= 0 else None
        error = abs(count - len(targets)) if count is not None else len(targets) + 1
        matched = match_count(predictions, targets) if len(targets) < 5 else 0
        kind = "dense_count" if len(targets) >= 5 else "sparse_count"
        candidates.append({
            "ordinal": ordinal, "raw": raw, "target_boxes": targets,
            "distractor_boxes": [
                item for item in (
                    polygon_box(instance["polygons"][index], int(instance["width"]), int(instance["height"]))
                    for index in raw["meta"].get("distractor_object_indices", [])
                    if index < len(instance.get("polygons", []))
                ) if item
            ],
            "predicted_boxes": predictions, "predicted_count": count, "count_error": error,
            "matched": matched, "error_type": kind,
        })

    groups = [
        ("dense_count", lambda row: len(row["target_boxes"]) >= 5, 5),
        ("sparse_count", lambda row: len(row["target_boxes"]) < 5, 5),
        ("grounding_miss", lambda row: len(row["target_boxes"]) < 5 and row["matched"] < len(row["target_boxes"]), 5),
        ("grounding_false_positive", lambda row: len(row["target_boxes"]) < 5 and len(row["predicted_boxes"]) > row["matched"], 5),
    ]
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for name, predicate, quota in groups:
        ranked = sorted((row for row in candidates if predicate(row)), key=lambda row: (row["count_error"], len(row["target_boxes"]) - row["matched"], len(row["predicted_boxes"]) - row["matched"]), reverse=True)
        for row in ranked:
            if row["ordinal"] in used:
                continue
            row = dict(row)
            row["error_type"] = name
            chosen.append(row)
            used.add(row["ordinal"])
            if sum(item["error_type"] == name for item in chosen) == quota:
                break
    for row in sorted(candidates, key=lambda item: item["count_error"], reverse=True):
        if len(chosen) >= 20:
            break
        if row["ordinal"] not in used:
            chosen.append(row)
            used.add(row["ordinal"])

    image_dir = args.output_dir / "images"
    manifest: list[dict[str, Any]] = []
    for index, row in enumerate(chosen, 1):
        meta = row["raw"]["meta"]
        audit_id = f"stage2_error_{index:02d}_{meta['image_id']}"
        image = image_dir / f"{audit_id}.jpg"
        render(row, image)
        manifest.append({
            "audit_id": audit_id, "error_type": row["error_type"], "query_id": meta["query_id"],
            "image_id": meta["image_id"], "predicate": meta["source_relations"][0],
            "family": meta["family"], "difficulty": meta["difficulty"],
            "target_count": len(row["target_boxes"]), "predicted_count": row["predicted_count"],
            "absolute_count_error": row["count_error"], "matched_boxes_at_iou_0_5": row["matched"],
            "image": str(image.resolve()), "review_decision": "pending", "reviewer_notes": "",
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audit_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]) if manifest else [])
        writer.writeheader()
        writer.writerows(manifest)
    (args.output_dir / "README.txt").write_text(
        "Colors: green=ground-truth selected target; red=ground-truth distractor; yellow=model prediction.\n"
        "Review whether target/distractor labels are visually defensible, then whether the model count/boxes are correct.\n",
        encoding="utf-8",
    )
    print(json.dumps({"records": len(manifest), "output": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
