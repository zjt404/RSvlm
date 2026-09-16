#!/usr/bin/env python3
"""Evaluate conditional set-grounding predictions."""

from __future__ import annotations

import argparse
import json
import math
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
        bbox = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return bbox


def extract_predicted_bboxes(value: Any) -> tuple[list[list[float]], int]:
    """Return valid predicted boxes and the number of malformed box candidates."""
    if value is None:
        return [], 0
    if not isinstance(value, list):
        return [], 1
    valid: list[list[float]] = []
    invalid_count = 0
    for candidate in value:
        bbox = valid_bbox(candidate)
        if bbox is None:
            invalid_count += 1
        else:
            valid.append(bbox)
    return valid, invalid_count


def iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def best_matches(predicted: list[list[float]], target: list[list[float]], threshold: float) -> list[float]:
    pairs = sorted(
        ((iou(pred, truth), pred_index, truth_index)
         for pred_index, pred in enumerate(predicted)
         for truth_index, truth in enumerate(target)),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matches: list[float] = []
    for score, pred_index, truth_index in pairs:
        if score < threshold:
            break
        if pred_index in used_pred or truth_index in used_truth:
            continue
        used_pred.add(pred_index)
        used_truth.add(truth_index)
        matches.append(score)
    return matches


def evaluate(rows: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    valid_json = 0
    valid_schema = 0
    count_errors: list[float] = []
    count_exact: list[float] = []
    sparse_count_errors: list[float] = []
    sparse_count_exact: list[float] = []
    dense_count_errors: list[float] = []
    dense_count_exact: list[float] = []
    sparse_records = 0
    dense_records = 0
    true_positive = false_positive = false_negative = 0
    matched_iou: list[float] = []
    for row in rows:
        solution = row.get("solution", {})
        target = [bbox for bbox in (valid_bbox(x) for x in solution.get("bboxes", [])) if bbox is not None]
        is_dense = len(target) >= 5
        if is_dense:
            dense_records += 1
        else:
            sparse_records += 1
        payload = parse_payload(row.get("prediction", ""))
        if payload is not None:
            valid_json += 1
        predicted_raw = payload.get("selected_bboxes") if payload else None
        predicted, invalid_box_count = extract_predicted_bboxes(predicted_raw)
        count_ok = payload is not None and type(payload.get("count")) is int and payload.get("count") >= 0
        if is_dense:
            schema_ok = count_ok and set(payload) == {"count"}
        else:
            schema_ok = (
                count_ok
                and isinstance(predicted_raw, list)
                and payload.get("count") == len(predicted_raw)
                and len(predicted) == len(predicted_raw)
            )
        valid_schema += int(schema_ok)

        if schema_ok:
            count_errors.append(abs(payload["count"] - len(target)))
            count_exact.append(float(payload["count"] == len(target)))
            if is_dense:
                dense_count_errors.append(abs(payload["count"] - len(target)))
                dense_count_exact.append(float(payload["count"] == len(target)))
            else:
                sparse_count_errors.append(abs(payload["count"] - len(target)))
                sparse_count_exact.append(float(payload["count"] == len(target)))

        # Dense samples intentionally use count-only responses. Grounding is
        # scored only for sparse samples where boxes are part of the contract.
        if not is_dense:
            matches = best_matches(predicted, target, threshold)
            true_positive += len(matches)
            false_positive += len(predicted) - len(matches) + invalid_box_count
            false_negative += len(target) - len(matches)
            matched_iou.extend(matches)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "num_records": len(rows),
        "json_valid_rate": valid_json / len(rows) if rows else None,
        "schema_valid_rate": valid_schema / len(rows) if rows else None,
        "mode_compliance_rate": valid_schema / len(rows) if rows else None,
        "count_mae": sum(count_errors) / len(count_errors) if count_errors else None,
        "count_exact_accuracy": sum(count_exact) / len(count_exact) if count_exact else None,
        "count_evaluable_records": len(count_errors),
        "sparse_records": sparse_records,
        "dense_records": dense_records,
        "sparse_count_mae": sum(sparse_count_errors) / len(sparse_count_errors) if sparse_count_errors else None,
        "sparse_count_exact_accuracy": sum(sparse_count_exact) / len(sparse_count_exact) if sparse_count_exact else None,
        "dense_count_mae": sum(dense_count_errors) / len(dense_count_errors) if dense_count_errors else None,
        "dense_count_exact_accuracy": sum(dense_count_exact) / len(dense_count_exact) if dense_count_exact else None,
        "grounding_records": sparse_records,
        "grounding_precision_at_iou_0_5": precision,
        "grounding_recall_at_iou_0_5": recall,
        "grounding_f1_at_iou_0_5": f1,
        "matched_mean_iou": sum(matched_iou) / len(matched_iou) if matched_iou else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "iou_threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.predictions).open("r", encoding="utf-8") if line.strip()]
    result = evaluate(rows)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
