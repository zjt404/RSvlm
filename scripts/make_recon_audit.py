from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw


CORE_PREDICATES = (
    "drive-on",
    "parked-at",
    "sail-on",
    "moor-at",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def norm_bbox_from_polygon(polygon: Any, width: int, height: int) -> list[int] | None:
    points = []
    for point in polygon or []:
        if isinstance(point, dict):
            x, y = point.get("x"), point.get("y")
        else:
            x, y = point[:2]
        if x is not None and y is not None:
            points.append((float(x), float(y)))
    if not points or width <= 0 or height <= 0:
        return None
    xs, ys = zip(*points)
    x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(1000 * x1 / width), round(1000 * y1 / height), round(1000 * x2 / width), round(1000 * y2 / height)]


def to_pixels(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(bbox[0] * width / 1000),
        round(bbox[1] * height / 1000),
        round(bbox[2] * width / 1000),
        round(bbox[3] * height / 1000),
    )


def polygon_to_pixels(
    polygon: Any,
    source_width: int,
    source_height: int,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for point in polygon or []:
        if isinstance(point, dict):
            x, y = point.get("x"), point.get("y")
        else:
            x, y = point[:2]
        if x is None or y is None:
            continue
        points.append(
            (
                round(float(x) * image_width / source_width),
                round(float(y) * image_height / source_height),
            )
        )
    return points


def object_info(instance: dict[str, Any], index: int) -> dict[str, Any]:
    width, height = int(instance["width"]), int(instance["height"])
    polygon = instance["polygons"][index]
    return {
        "index": index,
        "label": instance.get("labels", [])[index] if index < len(instance.get("labels", [])) else "unknown",
        "bbox_1000": norm_bbox_from_polygon(polygon, width, height),
        "polygon": polygon,
    }


def relation_anchor_indices(
    instance: dict[str, Any], selected_indices: set[int], predicate: str
) -> set[int]:
    anchors: set[int] = set()
    for relation in instance.get("relations", []):
        if relation.get("predicate") != predicate:
            continue
        subject_index = relation.get("subject_index")
        object_index = relation.get("object_index")
        if subject_index in selected_indices and object_index is not None:
            anchors.add(object_index)
        elif object_index in selected_indices and subject_index is not None:
            anchors.add(subject_index)
    return anchors


def draw_audit_image(
    image_path: Path,
    output_path: Path,
    query: dict[str, Any],
    instance: dict[str, Any],
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    source_width, source_height = int(instance["width"]), int(instance["height"])
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    label_specs: list[tuple[tuple[int, int, int, int], str, str]] = []
    target_ids = set(query["target_object_indices"])
    distractor_ids = set(query["distractor_object_indices"])
    anchor_ids = relation_anchor_indices(instance, target_ids | distractor_ids, query["source_relations"][0])
    seen: set[int] = set()
    for index, color, role in (
        [(i, "#18a558", "target") for i in sorted(target_ids)]
        + [(i, "#d64545", "distractor") for i in sorted(distractor_ids)]
        + [(i, "#246bce", "anchor") for i in sorted(anchor_ids)]
    ):
        if index in seen or index >= len(instance.get("polygons", [])):
            continue
        seen.add(index)
        info = object_info(instance, index)
        if info["bbox_1000"] is None:
            continue
        if role == "anchor":
            points = polygon_to_pixels(info["polygon"], source_width, source_height, width, height)
            if len(points) < 3:
                continue
            rgb = ImageColor.getrgb(color)
            overlay_draw.polygon(points, fill=(*rgb, 45), outline=(*rgb, 255), width=4)
            box = (
                min(x for x, _ in points),
                min(y for _, y in points),
                max(x for x, _ in points),
                max(y for _, y in points),
            )
        else:
            box = to_pixels(info["bbox_1000"], width, height)
            draw.rectangle(box, outline=color, width=3)
        label = f"{role}:{index}:{info['label']}"
        label_specs.append((box, color, label))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box, color, label in label_specs:
        text_box = draw.textbbox((box[0], box[1]), label)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), label, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92, optimize=True)


def make_contact_sheet(paths: list[Path], output: Path, columns: int = 4) -> None:
    if not paths:
        return
    thumb_w, thumb_h = 320, 240
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "white")
    for position, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            x = (position % columns) * thumb_w
            y = (position // columns) * thumb_h
            sheet.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90, optimize=True)


def build_audit(data_dir: Path, image_root: Path, output_dir: Path, per_predicate: int, seed: int) -> dict[str, Any]:
    queries = read_jsonl(data_dir / "queries.jsonl")
    instances = {row["image_id"]: row for row in read_jsonl(data_dir / "instances.jsonl")}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        predicate = query["source_relations"][0]
        if predicate in CORE_PREDICATES:
            grouped[predicate].append(query)
    rng = random.Random(seed)
    selected: list[tuple[str, dict[str, Any]]] = []
    for predicate in CORE_PREDICATES:
        candidates = grouped.get(predicate, [])[:]
        rng.shuffle(candidates)
        conflict = [query for query in candidates if query["difficulty"] == "conflict_distractors"]
        positive_only = [query for query in candidates if query["difficulty"] == "positive_only"]
        conflict_quota = per_predicate // 2
        positive_quota = per_predicate - conflict_quota
        chosen = conflict[:conflict_quota] + positive_only[:positive_quota]
        if len(chosen) < per_predicate:
            remaining = [query for query in candidates if query not in chosen]
            chosen.extend(remaining[: per_predicate - len(chosen)])
        selected.extend((predicate, query) for query in chosen[:per_predicate])

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    manifest: list[dict[str, Any]] = []
    per_predicate_paths: dict[str, list[Path]] = defaultdict(list)
    for ordinal, (predicate, query) in enumerate(selected, 1):
        instance = instances[query["image_id"]]
        image_path = image_root / instance["image_source"]
        audit_id = f"audit_{ordinal:04d}_{safe_name(query['query_id'])}"
        output_image = image_dir / f"{audit_id}.jpg"
        draw_audit_image(image_path, output_image, query, instance)
        target_objects = [object_info(instance, index) for index in query["target_object_indices"]]
        distractor_objects = [object_info(instance, index) for index in query["distractor_object_indices"]]
        anchor_indices = sorted(
            relation_anchor_indices(
                instance,
                set(query["target_object_indices"]) | set(query["distractor_object_indices"]),
                predicate,
            )
        )
        manifest.append(
            {
                "audit_id": audit_id,
                "query_id": query["query_id"],
                "image_id": query["image_id"],
                "predicate": predicate,
                "family": query["family"],
                "expression": query["expression"],
                "difficulty": query["difficulty"],
                "source_relations": query["source_relations"],
                "image": str(output_image.resolve()),
                "source_image": str(image_path.resolve()),
                "target_objects": target_objects,
                "distractor_objects": distractor_objects,
                "anchor_objects": [object_info(instance, index) for index in anchor_indices],
                "decision": "pending",
                "notes": "",
            }
        )
        per_predicate_paths[predicate].append(output_image)

    with (output_dir / "audit_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (output_dir / "audit_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audit_id",
                "query_id",
                "image_id",
                "predicate",
                "family",
                "expression",
                "difficulty",
                "image",
                "decision",
                "notes",
            ],
        )
        writer.writeheader()
        for row in manifest:
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "query_id": row["query_id"],
                    "image_id": row["image_id"],
                    "predicate": row["predicate"],
                    "family": row["family"],
                    "expression": row["expression"],
                    "difficulty": row["difficulty"],
                    "image": row["image"],
                    "decision": row["decision"],
                    "notes": row["notes"],
                }
            )
    for predicate, paths in per_predicate_paths.items():
        make_contact_sheet(paths, output_dir / f"contact_{safe_name(predicate)}.jpg")
    (output_dir / "README.md").write_text(
        "# ReCon1M relation semantic audit\n\n"
        "For each image, inspect the colored overlays and fill `decision` in `audit_index.csv` with `accept`, `ambiguous`, or `reject`; put the reason in `notes`. The same fields are also present in `audit_manifest.jsonl`.\n\n"
        "Green boxes are selected target subjects, red boxes are explicit conflicting distractors, and blue boxes are relation anchors (road, dock, parking area, etc.). The audit checks whether the source relation and the natural-language expression are both supported by the image.\n\n"
        "This is a quality-control artifact, not model output and not a training split.\n",
        encoding="utf-8",
    )
    return {
        "records": len(manifest),
        "by_predicate": {predicate: len(paths) for predicate, paths in per_predicate_paths.items()},
        "conflict_distractor_records": sum(row["difficulty"] == "conflict_distractors" for row in manifest),
        "output_dir": str(output_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a visual semantic audit set for ReCon1M relations")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-predicate", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_audit(args.data_dir, args.image_root, args.output_dir, args.per_predicate, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
