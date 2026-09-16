#!/usr/bin/env python3
"""Render crops for unmatched predictions to visually audit possible missing labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from analyze_all_boxes_false_positives import (
    greedy_target_matches,
    iou,
    load_instances,
    parse_prediction,
    polygon_box,
    valid_box,
)


def density(count: int) -> str:
    return "1-4" if count <= 4 else "5-10" if count <= 10 else "11+"


def pixel_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(box[0] * width / 1000),
        round(box[1] * height / 1000),
        round(box[2] * width / 1000),
        round(box[3] * height / 1000),
    )


def render_crop(item: dict[str, Any], size: int = 256) -> Image.Image:
    with Image.open(item["image"]) as source:
        image = source.convert("RGB")
    width, height = image.size
    box_px = pixel_box(item["box"], width, height)
    bw, bh = max(1, box_px[2] - box_px[0]), max(1, box_px[3] - box_px[1])
    margin = max(24, 3 * max(bw, bh))
    cx, cy = (box_px[0] + box_px[2]) // 2, (box_px[1] + box_px[3]) // 2
    half = min(max(margin, 48), max(width, height) // 2)
    left, top = max(0, cx - half), max(0, cy - half)
    right, bottom = min(width, cx + half), min(height, cy + half)
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    outline = (255, 215, 0)
    local = (box_px[0] - left, box_px[1] - top, box_px[2] - left, box_px[3] - top)
    draw.rectangle(local, outline=outline, width=max(2, crop.width // 80))
    crop.thumbnail((size, size))
    tile = Image.new("RGB", (size, size + 36), "white")
    tile.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    caption = f"{item['predicate']} {item['density']}  IoU={item['nearest_iou']:.2f}"
    ImageDraw.Draw(tile).text((5, size + 5), caption, fill="black")
    return tile


def make_sheet(items: list[dict[str, Any]], output: Path, columns: int = 5) -> None:
    tiles = [render_crop(item) for item in items]
    if not tiles:
        return
    width, height = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "#dddddd")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8") if line.strip()]
    image_ids = {str(row["meta"]["image_id"]) for row in rows}
    instances = load_instances(args.instances, image_ids)
    groups: dict[str, list[dict[str, Any]]] = {"below_0.1": [], "0.4_to_0.5": []}
    for row in rows:
        instance = instances[str(row["meta"]["image_id"])]
        width, height = int(instance["width"]), int(instance["height"])
        object_boxes = [polygon_box(polygon, width, height) for polygon in instance.get("polygons", [])]
        target = [box for value in row["solution"].get("bboxes", []) if (box := valid_box(value))]
        predicted, _ = parse_prediction(row.get("prediction", ""))
        matched, _ = greedy_target_matches(predicted, target, 0.5)
        for index, candidate in enumerate(predicted):
            if index in matched or any(iou(candidate, box) >= 0.5 for box in target):
                continue
            nearest = max((iou(candidate, box) for box in object_boxes if box is not None), default=0.0)
            group = "below_0.1" if nearest < 0.1 else "0.4_to_0.5" if nearest >= 0.4 else None
            if group:
                groups[group].append({
                    "image": row["input_image"],
                    "box": candidate,
                    "nearest_iou": nearest,
                    "predicate": row["meta"]["source_relations"][0],
                    "density": density(len(target)),
                    "query_id": row["meta"]["query_id"],
                })

    rng = random.Random(args.seed)
    manifest: dict[str, list[dict[str, Any]]] = {}
    for name, candidates in groups.items():
        rng.shuffle(candidates)
        selected = candidates[: args.sample_size]
        manifest[name] = selected
        make_sheet(selected, args.output_dir / f"{name}.jpg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: len(items) for name, items in manifest.items()}))


if __name__ == "__main__":
    main()
