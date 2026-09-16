#!/usr/bin/env python3
"""Compare full-image and tiled inference for v5 all-box grounding.

The same Qwen3-VL model is used for the full image and overlapping crops. Crop
coordinates are mapped back to the original image coordinate system, then
near-duplicate boxes are removed before writing the normal all-box JSONL format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from rs_vlm.inference import InferenceEngine


def load_rows(path: str | Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    return rows[:limit] if limit else rows


def parse_payload(text: str) -> dict[str, Any] | None:
    text = str(text).strip().replace("```json", "").replace("```", "").strip()
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


def boxes_from_payload(text: str) -> list[list[float]]:
    payload = parse_payload(text)
    raw = payload.get("selected_bboxes") if payload else None
    if not isinstance(raw, list):
        return []
    boxes: list[list[float]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 4:
            continue
        try:
            box = [float(value) for value in item]
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = box
        if 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000:
            boxes.append(box)
    return boxes


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - inter
    return inter / union if union > 0 else 0.0


def map_crop_box(box: list[float], crop: tuple[int, int, int, int], size: tuple[int, int]) -> list[float]:
    left, top, right, bottom = crop
    width, height = size
    crop_width, crop_height = right - left, bottom - top
    x1, y1, x2, y2 = box
    return [
        1000 * (left + x1 * crop_width / 1000) / width,
        1000 * (top + y1 * crop_height / 1000) / height,
        1000 * (left + x2 * crop_width / 1000) / width,
        1000 * (top + y2 * crop_height / 1000) / height,
    ]


def crop_boxes(width: int, height: int, scale: float, overlap: float) -> list[tuple[int, int, int, int]]:
    crop_width = max(1, min(width, round(width * scale)))
    crop_height = max(1, min(height, round(height * scale)))
    step_x = max(1, round(crop_width * (1 - overlap)))
    step_y = max(1, round(crop_height * (1 - overlap)))
    xs = list(range(0, max(1, width - crop_width + 1), step_x))
    ys = list(range(0, max(1, height - crop_height + 1), step_y))
    if xs[-1] != width - crop_width:
        xs.append(width - crop_width)
    if ys[-1] != height - crop_height:
        ys.append(height - crop_height)
    return [(x, y, x + crop_width, y + crop_height) for y in ys for x in xs]


def deduplicate(boxes: list[list[float]], threshold: float) -> list[list[float]]:
    kept: list[list[float]] = []
    for box in boxes:
        if all(iou(box, previous) < threshold for previous in kept):
            kept.append(box)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/workspace/zjt/qwen3vl")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-pixels", type=int, default=589824)
    parser.add_argument("--tile-scale", type=float, default=0.6)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--dedup-iou", type=float, default=0.7)
    args = parser.parse_args()

    rows = load_rows(args.dataset, args.limit)
    engine = InferenceEngine(
        args.model,
        args.adapter,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, 1):
            image_path = row["images"][0]
            question = row["messages"][0]["content"].replace("<image>", "").strip()
            with Image.open(image_path).convert("RGB") as image:
                width, height = image.size
                full_boxes = boxes_from_payload(engine.generate(image, question))
                merged = list(full_boxes)
                crops = crop_boxes(width, height, args.tile_scale, args.tile_overlap)
                tile_question = (
                    question
                    + "\nThis is a crop from a larger remote-sensing image. "
                    "Consider targets visible in this crop and return their boxes "
                    "relative to this crop, normalized to 0-1000."
                )
                for crop in crops:
                    tile = image.crop(crop)
                    tile_boxes = boxes_from_payload(engine.generate(tile, tile_question))
                    merged.extend(map_crop_box(box, crop, (width, height)) for box in tile_boxes)
                merged = deduplicate(merged, args.dedup_iou)
            prediction = {"count": len(merged), "selected_bboxes": merged}
            result = {
                "task_type": row["task_type"],
                "solution": row["solution"],
                "prediction": json.dumps(prediction, ensure_ascii=False, separators=(",", ":")),
                "meta": row.get("meta", {}),
                "input_image": image_path,
                "shuffled_image": False,
                "inference_mode": "full_plus_overlapping_tiles",
                "tile_count": len(crops),
            }
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(f"[{index}/{len(rows)}] boxes={len(merged)} tiles={len(crops)}", flush=True)


if __name__ == "__main__":
    main()
