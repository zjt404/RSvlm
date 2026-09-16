#!/usr/bin/env python3
"""Classify unmatched all-box predictions against complete ReCon1M annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return [x1, y1, x2, y2]


def polygon_box(polygon: Any, width: int, height: int) -> list[float] | None:
    points: list[tuple[float, float]] = []
    for point in polygon or []:
        if isinstance(point, dict):
            x, y = point.get("x"), point.get("y")
        else:
            try:
                x, y = point[:2]
            except (TypeError, ValueError):
                continue
        if x is not None and y is not None:
            points.append((float(x), float(y)))
    if not points or width <= 0 or height <= 0:
        return None
    xs, ys = zip(*points)
    return valid_box([
        1000 * min(xs) / width,
        1000 * min(ys) / height,
        1000 * max(xs) / width,
        1000 * max(ys) / height,
    ])


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def parse_prediction(text: str) -> tuple[list[list[float]], int]:
    try:
        payload = json.loads(str(text).strip())
    except json.JSONDecodeError:
        start, end = str(text).find("{"), str(text).rfind("}")
        if start < 0 or end <= start:
            return [], 1
        try:
            payload = json.loads(str(text)[start : end + 1])
        except json.JSONDecodeError:
            return [], 1
    if not isinstance(payload, dict) or not isinstance(payload.get("selected_bboxes"), list):
        return [], 1
    boxes: list[list[float]] = []
    invalid = 0
    for value in payload["selected_bboxes"]:
        candidate = valid_box(value)
        if candidate is None:
            invalid += 1
        else:
            boxes.append(candidate)
    return boxes, invalid


def greedy_target_matches(
    predicted: list[list[float]], target: list[list[float]], threshold: float
) -> tuple[set[int], set[int]]:
    pairs = sorted(
        (
            (iou(pred_box, target_box), pred_index, target_index)
            for pred_index, pred_box in enumerate(predicted)
            for target_index, target_box in enumerate(target)
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_target: set[int] = set()
    for score, pred_index, target_index in pairs:
        if score < threshold:
            break
        if pred_index in used_pred or target_index in used_target:
            continue
        used_pred.add(pred_index)
        used_target.add(target_index)
    return used_pred, used_target


def load_instances(path: Path, required_ids: set[str]) -> dict[str, dict[str, Any]]:
    instances: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image_id = str(row.get("image_id"))
            if image_id in required_ids:
                instances[image_id] = row
                if len(instances) == len(required_ids):
                    break
    return instances


def analyze(path: Path, instances: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    counts: Counter[str] = Counter()
    by_density: dict[str, Counter[str]] = defaultdict(Counter)
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    no_match_proximity: Counter[str] = Counter()

    def proximity_bin(score: float) -> str:
        if score >= 0.4:
            return "0.4_to_0.5"
        if score >= 0.3:
            return "0.3_to_0.4"
        if score >= 0.1:
            return "0.1_to_0.3"
        return "below_0.1"

    for row in rows:
        image_id = str(row["meta"]["image_id"])
        instance = instances[image_id]
        width, height = int(instance["width"]), int(instance["height"])
        labels = instance.get("labels", [])
        object_boxes = [polygon_box(polygon, width, height) for polygon in instance.get("polygons", [])]
        target = [box for value in row.get("solution", {}).get("bboxes", []) if (box := valid_box(value))]
        predicted, invalid = parse_prediction(row.get("prediction", ""))
        matched_pred, _ = greedy_target_matches(predicted, target, threshold)
        target_ids = set(row["meta"].get("target_object_indices", []))
        distractor_ids = set(row["meta"].get("distractor_object_indices", []))
        ignore_ids = set(row["meta"].get("ignore_object_indices", []))
        density = "1_4" if len(target) <= 4 else "5_10" if len(target) <= 10 else "11_plus"

        counts["records"] += 1
        counts["predicted"] += len(predicted) + invalid
        counts["true_positive"] += len(matched_pred)
        counts["invalid"] += invalid
        by_density[density]["invalid"] += invalid

        for pred_index, pred_box in enumerate(predicted):
            if pred_index in matched_pred:
                continue
            best_target_iou = max((iou(pred_box, target_box) for target_box in target), default=0.0)
            if best_target_iou >= threshold:
                category = "duplicate_target"
                object_index = None
            else:
                overlaps = sorted(
                    (
                        (iou(pred_box, object_box), index)
                        for index, object_box in enumerate(object_boxes)
                        if object_box is not None and index not in target_ids
                    ),
                    reverse=True,
                )
                best_iou, object_index = overlaps[0] if overlaps else (0.0, None)
                if best_iou < threshold or object_index is None:
                    category = "no_annotation_match"
                    object_index = None
                    no_match_proximity[f"nearest_any_annotation_{proximity_bin(best_iou)}"] += 1
                    no_match_proximity[f"nearest_target_{proximity_bin(best_target_iou)}"] += 1
                elif object_index in distractor_ids:
                    category = "annotated_distractor"
                elif object_index in ignore_ids:
                    category = "annotated_ignored_object"
                else:
                    category = "other_annotated_object"
            counts[category] += 1
            by_density[density][category] += 1
            if object_index is not None:
                label = str(labels[object_index]) if object_index < len(labels) else "unknown"
                label_counts[category][label] += 1

    false_positive = counts["predicted"] - counts["true_positive"]
    categories = [
        "duplicate_target",
        "annotated_distractor",
        "annotated_ignored_object",
        "other_annotated_object",
        "no_annotation_match",
        "invalid",
    ]
    return {
        "prediction_file": str(path),
        "records": counts["records"],
        "predicted": counts["predicted"],
        "true_positive": counts["true_positive"],
        "false_positive": false_positive,
        "false_positive_breakdown": {
            category: {
                "count": counts[category],
                "fraction_of_false_positive": counts[category] / false_positive if false_positive else 0.0,
            }
            for category in categories
        },
        "by_density": {key: dict(value) for key, value in sorted(by_density.items())},
        "top_labels": {
            category: label_counts[category].most_common(15)
            for category in categories
            if label_counts[category]
        },
        "no_annotation_match_proximity": dict(sorted(no_match_proximity.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    image_ids: set[str] = set()
    for path in args.predictions:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    image_ids.add(str(json.loads(line)["meta"]["image_id"]))
    instances = load_instances(args.instances, image_ids)
    if missing := image_ids - set(instances):
        raise RuntimeError(f"missing {len(missing)} instances")
    result = {
        "iou_threshold": args.iou_threshold,
        "instance_count": len(instances),
        "analyses": [analyze(path, instances, args.iou_threshold) for path in args.predictions],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
