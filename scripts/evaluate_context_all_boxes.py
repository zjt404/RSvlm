#!/usr/bin/env python3
"""Evaluate all-box conditional grounding predictions.

Unlike the legacy evaluator, every record requires a complete box set.  Count
metrics are derived from valid predicted boxes, matching the deployment
contract where the UI counts what it can draw.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_payload(text: str) -> dict[str, Any] | None:
    text = str(text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return box


def extract_boxes(value: Any) -> tuple[list[list[float]], int]:
    if not isinstance(value, list):
        return [], 1
    valid, invalid = [], 0
    for candidate in value:
        box = valid_bbox(candidate)
        if box is None:
            invalid += 1
        else:
            valid.append(box)
    return valid, invalid


def iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / union if union > 0 else 0.0


def best_matches(predicted: list[list[float]], target: list[list[float]], threshold: float) -> list[float]:
    pairs = sorted(
        ((iou(pred, truth), pred_index, truth_index)
         for pred_index, pred in enumerate(predicted)
         for truth_index, truth in enumerate(target)),
        reverse=True,
    )
    used_pred, used_truth, matches = set(), set(), []
    for score, pred_index, truth_index in pairs:
        if score < threshold:
            break
        if pred_index in used_pred or truth_index in used_truth:
            continue
        used_pred.add(pred_index)
        used_truth.add(truth_index)
        matches.append(score)
    return matches


def density_bin(count: int) -> str:
    if count <= 4:
        return "1_4"
    if count <= 10:
        return "5_10"
    return "11_plus"


def empty_stats() -> dict[str, Any]:
    return {"records": 0, "count_error": 0.0, "count_exact": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []}


def render_stats(stats: dict[str, Any]) -> dict[str, Any]:
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    records = stats["records"]
    return {
        "records": records,
        "count_mae": stats["count_error"] / records if records else None,
        "count_exact_accuracy": stats["count_exact"] / records if records else None,
        "grounding_precision_at_iou_0_5": precision,
        "grounding_recall_at_iou_0_5": recall,
        "grounding_f1_at_iou_0_5": f1,
        "matched_mean_iou": sum(stats["ious"]) / len(stats["ious"]) if stats["ious"] else 0.0,
        "true_positive": stats["tp"],
        "false_positive": stats["fp"],
        "false_negative": stats["fn"],
    }


def evaluate(rows: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    valid_json = valid_schema = 0
    grouped: dict[str, dict[str, Any]] = defaultdict(empty_stats)
    overall = empty_stats()
    for row in rows:
        target = [box for box in (valid_bbox(item) for item in row.get("solution", {}).get("bboxes", [])) if box is not None]
        payload = parse_payload(row.get("prediction", ""))
        valid_json += int(payload is not None)
        raw_boxes = payload.get("selected_bboxes") if payload else None
        predicted, invalid_boxes = extract_boxes(raw_boxes)
        schema_ok = (
            payload is not None
            and set(payload) == {"count", "selected_bboxes"}
            and type(payload.get("count")) is int
            and payload["count"] >= 0
            and isinstance(raw_boxes, list)
            and invalid_boxes == 0
            and payload["count"] == len(raw_boxes)
        )
        valid_schema += int(schema_ok)
        matches = best_matches(predicted, target, threshold)
        for stats in (overall, grouped[density_bin(len(target))]):
            stats["records"] += 1
            stats["count_error"] += abs(len(predicted) - len(target))
            stats["count_exact"] += int(len(predicted) == len(target))
            stats["tp"] += len(matches)
            stats["fp"] += len(predicted) - len(matches) + invalid_boxes
            stats["fn"] += len(target) - len(matches)
            stats["ious"].extend(matches)
    result = render_stats(overall)
    result.update({
        "num_records": len(rows),
        "json_valid_rate": valid_json / len(rows) if rows else None,
        "schema_valid_rate": valid_schema / len(rows) if rows else None,
        "iou_threshold": threshold,
        "density_bins": {name: render_stats(stats) for name, stats in sorted(grouped.items())},
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.predictions).open(encoding="utf-8") if line.strip()]
    rendered = json.dumps(evaluate(rows), ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
