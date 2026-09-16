from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rs_vlm.rewards import extract_answer, normalize_spatial_label, parse_json_payload
from rs_vlm.schema import SPATIAL_LABELS


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower().replace("_", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def parse_number(text: str) -> float | None:
    answer = extract_answer(text)
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return float(answer)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(answer))
    return float(match.group()) if match else None


def parse_bbox(text: str) -> list[float] | None:
    payload = parse_json_payload(text)
    value = payload.get("bbox") if payload else None
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


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, list[Any]] = defaultdict(list)
    valid_json: list[float] = []
    for row in records:
        task = row["task_type"]
        prediction = str(row.get("prediction", ""))
        solution = row["solution"]
        valid_json.append(float(parse_json_payload(prediction) is not None))
        if task == "vqa":
            predicted = normalize_text(extract_answer(prediction))
            stats["vqa_correct"].append(float(predicted == normalize_text(solution["answer"])))
        elif task == "count":
            predicted = parse_number(prediction)
            target = float(solution["value"])
            error = abs(predicted - target) if predicted is not None else math.inf
            stats["count_abs_error"].append(error)
            stats["count_sq_error"].append(error * error)
            stats["count_exact"].append(float(error == 0))
            if target == 0:
                stats["count_zero_fp"].append(float(predicted is None or predicted != 0))
        elif task == "grounding":
            predicted = parse_bbox(prediction)
            iou = bbox_iou(predicted, solution["bbox"]) if predicted else 0.0
            stats["grounding_iou"].append(iou)
            stats["grounding_acc50"].append(float(iou >= 0.5))
        elif task == "spatial":
            predicted = normalize_spatial_label(extract_answer(prediction))
            target = normalize_spatial_label(solution["label"])
            stats["spatial_pairs"].append((target, predicted))

    result: dict[str, Any] = {
        "num_records": len(valid_json),
        "json_valid_rate": _mean(valid_json),
    }
    if stats["vqa_correct"]:
        result["vqa"] = {"accuracy": _mean(stats["vqa_correct"]), "n": len(stats["vqa_correct"])}
    if stats["count_abs_error"]:
        finite_abs = [x for x in stats["count_abs_error"] if math.isfinite(x)]
        finite_sq = [x for x in stats["count_sq_error"] if math.isfinite(x)]
        result["count"] = {
            "mae": _mean(finite_abs) if len(finite_abs) == len(stats["count_abs_error"]) else None,
            "rmse": math.sqrt(sum(finite_sq) / len(finite_sq)) if len(finite_sq) == len(stats["count_sq_error"]) else None,
            "exact_accuracy": _mean(stats["count_exact"]),
            "zero_false_positive_rate": _mean(stats["count_zero_fp"]),
            "parse_failure_rate": 1.0 - len(finite_abs) / len(stats["count_abs_error"]),
            "n": len(stats["count_abs_error"]),
        }
    if stats["grounding_iou"]:
        result["grounding"] = {
            "mean_iou": _mean(stats["grounding_iou"]),
            "acc_at_iou_0_5": _mean(stats["grounding_acc50"]),
            "n": len(stats["grounding_iou"]),
        }
    if stats["spatial_pairs"]:
        pairs = stats["spatial_pairs"]
        accuracy = sum(target == pred for target, pred in pairs) / len(pairs)
        f1_scores: list[float] = []
        for label in SPATIAL_LABELS:
            tp = sum(target == label and pred == label for target, pred in pairs)
            fp = sum(target != label and pred == label for target, pred in pairs)
            fn = sum(target == label and pred != label for target, pred in pairs)
            denominator = 2 * tp + fp + fn
            if denominator:
                f1_scores.append(2 * tp / denominator)
        result["spatial"] = {"accuracy": accuracy, "macro_f1": _mean(f1_scores), "n": len(pairs)}
    return result


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rs-vlm prediction JSONL")
    parser.add_argument("predictions", help="JSONL with task_type, solution, and prediction")
    parser.add_argument("--output", help="Optional metrics JSON path")
    args = parser.parse_args()
    metrics = evaluate_records(load_jsonl(args.predictions))
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
