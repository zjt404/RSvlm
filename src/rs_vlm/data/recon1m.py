from __future__ import annotations

"""Build conditional set-grounding data from ReCon1M relation graphs.

The graph annotations are treated as ground truth. Language is generated from
fixed templates in this module; an LLM may paraphrase a query later, but it is
never allowed to invent object ids, boxes, counts, or relations.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from rs_vlm.data.common import deterministic_split
from rs_vlm.schema import make_example, write_jsonl


LABEL_NAMES = [
    "__background__", "van", "small-car", "building", "road", "airplane",
    "block", "parking-lot", "motorboat", "dump-truck", "cargo-truck",
    "dry-cargo-ship", "runway", "container", "water", "intersection",
    "fishing-boat", "other-vehicle", "storage-tank", "airport", "other-ship",
    "harbor", "solar-panel", "pool", "tennis-court", "engineering-ship",
    "crane", "liquid-cargo-ship", "bus", "passenger-ship", "warship",
    "excavator", "storage-tank-group", "bridge", "basketball-court", "trailer",
    "tugboat", "train-carriage", "football-field", "cargo", "baseball-field",
    "boarding_bridge", "greenbelt", "exhaust-fan", "truck-tractor", "factory",
    "construction-site", "roundabout", "terminal", "tractor", "railway",
    "farmland", "stadium", "chimney", "gas-station", "dam", "locomotive",
    "expressway-service-area", "control-tower", "smoke", "helicopter-apron",
]

PREDICATE_NAMES = [
    "__background__", "parked-at", "park-next-to", "close-to", "provide-access-to",
    "inside", "drive-on", "moor-at", "serve", "is-parallel-to", "adjacent-to",
    "belong-to", "sail-on", "pile-up-at", "cross", "supplement", "slow-down",
    "taxi-on", "supply", "cooperate-with", "contain", "power", "link-to",
    "prepared-for", "support", "hoist", "above", "drive-at-the-different-lane",
    "dock-at", "connect", "drive-at-the-same-lane", "border", "equipped-with",
    "separate", "ventilate", "transport", "support-the-construction-of", "manage",
    "placed-on", "sail-by", "lie-under", "park-alone-at", "cultivate", "converge",
    "tow", "provide-shuttle-service-to", "around", "move-away-from", "exit-from",
    "enter", "adjoint-with", "dock-alone-at", "load", "command", "is-symmetric-with",
    "block", "emit", "pass-under", "dig", "pull",
]


VEHICLE_LABELS = {
    "van", "small-car", "dump-truck", "cargo-truck", "other-vehicle",
    "bus", "excavator", "trailer", "tractor", "truck-tractor",
}
SHIP_LABELS = {
    "motorboat", "dry-cargo-ship", "fishing-boat", "other-ship",
    "engineering-ship", "liquid-cargo-ship", "passenger-ship", "warship",
    "tugboat",
}
AIRCRAFT_LABELS = {"airplane", "helicopter"}


PREDICATE_RULES: dict[str, dict[str, Any]] = {
    "drive-on": {
        "family": "road_vehicle",
        "subject_labels": VEHICLE_LABELS,
        "query": "Select every vehicle driving on a road; exclude vehicles parked at a parking area.",
        "conflicts": {"parked-at"},
    },
    "parked-at": {
        "family": "road_vehicle",
        "subject_labels": VEHICLE_LABELS,
        "query": "Select every vehicle parked at a parking area; exclude vehicles driving on a road.",
        "conflicts": {"drive-on"},
    },
    "park-next-to": {
        "family": "vehicle_relation",
        "subject_labels": VEHICLE_LABELS,
        "query": "Select every vehicle parked next to another vehicle.",
        "conflicts": {"drive-on"},
        "enabled_v1": False,
    },
    "sail-on": {
        "family": "ship_water",
        "subject_labels": SHIP_LABELS,
        "query": "Select every ship on open water; exclude ships moored or docked at the shore.",
        "conflicts": {"dock-at", "moor-at", "dock-alone-at"},
    },
    "dock-at": {
        "family": "ship_water",
        "subject_labels": SHIP_LABELS,
        "query": "Select every ship at a dock or berth; exclude ships sailing on open water.",
        "conflicts": {"sail-on"},
    },
    "moor-at": {
        "family": "ship_water",
        "subject_labels": SHIP_LABELS,
        "query": "Select every ship moored at the shore or a berth; exclude ships sailing on open water.",
        "conflicts": {"sail-on"},
    },
    "taxi-on": {
        "family": "aircraft_surface",
        "subject_labels": AIRCRAFT_LABELS,
        "query": "Select every aircraft located on a taxiway or runway.",
        "conflicts": set(),
    },
    "inside": {
        "family": "contained_object",
        "query": "Select every object that is inside another labeled region.",
        "conflicts": set(),
    },
    "adjacent-to": {
        "family": "adjacent_object",
        "query": "Select every object that is adjacent to another labeled object.",
        "conflicts": set(),
    },
    "close-to": {
        "family": "near_object",
        "query": "Select every object that is close to another labeled object.",
        "conflicts": set(),
    },
}

MAX_TARGETS_PER_QUERY = 20
AUDIT_DECISIONS = {"accept", "ambiguous", "reject", "pending"}


def _as_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    if isinstance(value, dict):
        return {str(k): _as_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_python(v) for v in value]
    return value


def _raw_split_map(root: Path) -> dict[str, str]:
    """Read the official ReCon1M image split file next to ``extracted``."""
    candidates = [root / "dataset_split.json", root.parent / "dataset_split.json"]
    split_path = next((path for path in candidates if path.is_file()), None)
    if split_path is None:
        raise FileNotFoundError(
            f"official dataset_split.json not found next to raw root {root}"
        )
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_map: dict[str, str] = {}
    for split in ("train", "val", "validation", "test"):
        values = payload.get(split, [])
        normalized_split = "val" if split == "validation" else split
        if not isinstance(values, list):
            raise ValueError(f"dataset split {split!r} must be a list")
        for value in values:
            image_id = Path(str(value)).stem
            if image_id in split_map and split_map[image_id] != normalized_split:
                raise ValueError(f"image {image_id} appears in multiple official splits")
            split_map[image_id] = normalized_split
    if not split_map:
        raise ValueError(f"official dataset_split.json is empty: {split_path}")
    return split_map


def _parse_raw_label_file(path: Path) -> tuple[list[str], list[list[list[float]]]]:
    """Parse ReCon1M labelTxt: x1 y1 ... x4 y4 class difficult."""
    labels: list[str] = []
    polygons: list[list[list[float]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("imagesource:") or line.startswith("gsd:"):
                continue
            fields = line.split()
            if len(fields) < 10:
                raise ValueError(f"invalid label line {path}:{line_number}: {line!r}")
            try:
                coordinates = [float(value) for value in fields[:8]]
            except ValueError as exc:
                raise ValueError(f"invalid polygon {path}:{line_number}") from exc
            label = fields[8].strip()
            if not label:
                raise ValueError(f"empty object label {path}:{line_number}")
            polygon = [[coordinates[index], coordinates[index + 1]] for index in range(0, 8, 2)]
            labels.append(label)
            polygons.append(polygon)
    return labels, polygons


def _parse_raw_relation_file(path: Path) -> list[dict[str, Any]]:
    """Parse ReCon1M relTxt: subject_index object_index predicate_index."""
    relations: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"invalid relation line {path}:{line_number}: {line!r}")
            try:
                subject, object_index, predicate = (int(value) for value in fields[:3])
            except ValueError as exc:
                raise ValueError(f"invalid relation {path}:{line_number}") from exc
            if not (0 <= predicate < len(PREDICATE_NAMES)):
                raise ValueError(f"unknown predicate index {predicate} in {path}:{line_number}")
            relations.append(
                {
                    "subject_index": subject,
                    "object_index": object_index,
                    "predicate": PREDICATE_NAMES[predicate],
                }
            )
    return relations


def iter_raw_records(root: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the complete ReCon1M annotation rows from extracted raw files.

    The official split is attached to each row as ``_official_split`` so the
    downstream builder can keep train/val/test image-disjoint.
    """
    root = Path(root)
    image_dir, label_dir, relation_dir = root / "images", root / "labelTxt", root / "relTxt"
    if not (image_dir.is_dir() and label_dir.is_dir() and relation_dir.is_dir()):
        raise FileNotFoundError(f"raw ReCon1M root must contain images/labelTxt/relTxt: {root}")
    split_map = _raw_split_map(root)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("Pillow is required to read ReCon1M image dimensions") from exc
    for image_path in sorted(image_dir.glob("*.png")):
        image_id = image_path.stem
        label_path = label_dir / f"{image_id}.txt"
        relation_path = relation_dir / f"{image_id}.txt"
        if not label_path.is_file() or not relation_path.is_file():
            raise FileNotFoundError(f"missing annotation pair for image {image_id}")
        with Image.open(image_path) as image:
            width, height = image.size
        labels, polygons = _parse_raw_label_file(label_path)
        relations = _parse_raw_relation_file(relation_path)
        for relation in relations:
            if relation["subject_index"] >= len(labels) or relation["object_index"] >= len(labels):
                raise ValueError(f"relation index out of range for image {image_id}: {relation}")
        yield {
            "id": image_id,
            "width": width,
            "height": height,
            "labels": labels,
            "polygons": polygons,
            "relations": relations,
            "image": f"images/{image_id}.png",
            "_official_split": split_map.get(image_id),
            "_annotation_file": str(label_path),
        }


def _parquet_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.parquet")))
        elif path.suffix.lower() == ".parquet":
            files.append(path)
    return sorted(set(files))


def iter_records(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    raw_roots = [
        Path(path)
        for path in paths
        if Path(path).is_dir()
        and (Path(path) / "images").is_dir()
        and (Path(path) / "labelTxt").is_dir()
        and (Path(path) / "relTxt").is_dir()
    ]
    if raw_roots:
        for root in raw_roots:
            yield from iter_raw_records(root)
        return
    json_paths = [Path(path) for path in paths if Path(path).suffix.lower() in {".json", ".jsonl"}]
    if json_paths:
        for path in json_paths:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            yield _as_python(json.loads(line))
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = payload if isinstance(payload, list) else payload.get("data", payload.get("records", []))
                yield from (_as_python(row) for row in rows)
        return
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("pyarrow is required to read ReCon1M Parquet files") from exc
    files = _parquet_files(paths)
    if not files:
        raise FileNotFoundError("no Parquet files found")
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                row["_annotation_file"] = str(path)
                yield _as_python(row)


def _image_id(row: dict[str, Any]) -> str:
    for key in ("id", "image_id", "image_source", "filename", "file_name"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return Path(str(value)).stem
    raise ValueError("ReCon1M row has no image identifier")


def _image_source(row: dict[str, Any], image_id: str) -> str:
    for key in ("image", "filename", "file_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.replace("\\", "/")
    return f"{image_id}.jpg"


def _labels(row: dict[str, Any]) -> list[str]:
    values = row.get("labels", row.get("categories", [])) or []
    return [LABEL_NAMES[int(value)] if isinstance(value, int) and 0 <= value < len(LABEL_NAMES) else str(value) for value in values]


def _point_xy(point: Any) -> tuple[float, float] | None:
    point = _as_python(point)
    if isinstance(point, dict):
        for keys in (("x", "y"), ("X", "Y")):
            if keys[0] in point and keys[1] in point:
                return float(point[keys[0]]), float(point[keys[1]])
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def polygon_to_hbox(polygon: Any, width: int, height: int) -> list[int] | None:
    points = [_point_xy(point) for point in (_as_python(polygon) or [])]
    points = [point for point in points if point is not None]
    if not points or width <= 0 or height <= 0:
        return None
    xs, ys = zip(*points)
    x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    normalized = [
        round(1000 * x1 / width),
        round(1000 * y1 / height),
        round(1000 * x2 / width),
        round(1000 * y2 / height),
    ]
    # Very small remote-sensing objects can collapse after 0-1000 quantization.
    # Preserve them with the smallest representable positive extent.
    if normalized[2] <= normalized[0]:
        if normalized[0] >= 1000:
            normalized[0] = 999
        normalized[2] = normalized[0] + 1
    if normalized[3] <= normalized[1]:
        if normalized[1] >= 1000:
            normalized[1] = 999
        normalized[3] = normalized[1] + 1
    return normalized


def _relations(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("relations", row.get("relation", [])) or []
    result: list[dict[str, Any]] = []
    if isinstance(raw, dict) and all(key in raw for key in ("subject_index", "object_index", "predicate")):
        subjects = raw.get("subject_index") or []
        objects = raw.get("object_index") or []
        predicates = raw.get("predicate") or []
        raw = zip(subjects, objects, predicates)
    for relation in raw:
        relation = _as_python(relation)
        if isinstance(relation, dict):
            predicate = relation.get("predicate", relation.get("relation"))
            subject = relation.get("subject_index", relation.get("subject"))
            object_index = relation.get("object_index", relation.get("object"))
        elif isinstance(relation, (list, tuple)) and len(relation) >= 3:
            subject, object_index, predicate = relation[:3]
        else:
            continue
        try:
            if isinstance(predicate, int) and 0 <= predicate < len(PREDICATE_NAMES):
                predicate = PREDICATE_NAMES[predicate]
            predicate = str(predicate).strip().lower()
            subject = int(subject)
            object_index = int(object_index)
        except (TypeError, ValueError):
            continue
        if predicate and subject >= 0 and object_index >= 0:
            result.append({"subject_index": subject, "object_index": object_index, "predicate": predicate})
    return result


def _resolve_image(root: Path, image_source: str, image_id: str) -> Path:
    source = image_source.replace("\\", "/")
    candidates = [root / source, root / Path(source).name, root / f"{image_id}.jpg", root / f"{image_id}.png"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(root.rglob(Path(source).name)) + list(root.rglob(f"{image_id}.*"))
    return (matches[0] if matches else candidates[0]).resolve()


def _has_conflict(subject: int, relations: list[dict[str, Any]], conflicts: set[str]) -> bool:
    return any(r["subject_index"] == subject and r["predicate"] in conflicts for r in relations)


def _subject_allowed(labels: list[str], subject_index: int, rule: dict[str, Any]) -> bool:
    allowed_labels = rule.get("subject_labels")
    return not allowed_labels or (
        0 <= subject_index < len(labels) and labels[subject_index] in allowed_labels
    )


def _query_id_from_audit_row(row: dict[str, str]) -> str | None:
    query_id = str(row.get("query_id", "")).strip()
    if query_id:
        return query_id
    image_id = str(row.get("image_id", "")).strip()
    predicate = str(row.get("predicate", "")).strip().lower()
    if image_id and predicate:
        return f"{image_id}:{predicate}"
    audit_id = str(row.get("audit_id", "")).strip()
    if not audit_id or not predicate:
        return None
    suffix = f"_{predicate}"
    encoded_query = re.sub(r"^audit_\d+_", "", audit_id)
    if not encoded_query.endswith(suffix):
        return None
    image_id = encoded_query[: -len(suffix)]
    return f"{image_id}:{predicate}" if image_id else None


def load_audit_decisions(path: str | Path | None) -> dict[str, str]:
    """Load human audit decisions keyed by stable ReCon1M query id.

    New audit files contain query_id directly. The fallback parser keeps the
    original audit CSV format usable without rewriting the completed review.
    """
    if path is None:
        return {}
    decisions: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = _query_id_from_audit_row(row)
            # Resolved delta-review files carry the authoritative decision in
            # final_decision; retain compatibility with the original audit CSV.
            decision = str(row.get("final_decision") or row.get("decision", "")).strip().lower()
            if not query_id or not decision:
                continue
            if decision not in AUDIT_DECISIONS:
                raise ValueError(f"invalid audit decision {decision!r} for {query_id}")
            previous = decisions.get(query_id)
            if previous is not None and previous != decision:
                raise ValueError(f"conflicting audit decisions for {query_id}: {previous!r} and {decision!r}")
            decisions[query_id] = decision
    return decisions


def build_queries(
    row: dict[str, Any], predicates: Iterable[str] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_id = _image_id(row)
    width, height = int(row.get("width", 0)), int(row.get("height", 0))
    labels = _labels(row)
    polygons = _as_python(row.get("polygons", row.get("polygon", []))) or []
    relations = _relations(row)
    instances = {
        "image_id": image_id,
        "image_source": _image_source(row, image_id),
        "width": width,
        "height": height,
        "labels": labels,
        "polygons": polygons,
        "relations": relations,
        "official_split": row.get("_official_split"),
        "annotation_file": row.get("_annotation_file"),
    }
    queries: list[dict[str, Any]] = []
    selected_predicates = tuple(predicates) if predicates is not None else tuple(PREDICATE_RULES)
    unknown_predicates = set(selected_predicates) - set(PREDICATE_RULES)
    if unknown_predicates:
        raise ValueError(f"unknown predicates: {sorted(unknown_predicates)}")
    for predicate in selected_predicates:
        rule = PREDICATE_RULES[predicate]
        if not rule.get("enabled_v1", True):
            continue
        positive = sorted(
            {
                r["subject_index"]
                for r in relations
                if r["predicate"] == predicate and _subject_allowed(labels, r["subject_index"], rule)
            }
        )
        positive = [index for index in positive if not _has_conflict(index, relations, rule["conflicts"])]
        if not positive or len(positive) > MAX_TARGETS_PER_QUERY:
            continue
        distractors = sorted(
            {
                r["subject_index"]
                for r in relations
                if r["predicate"] in rule["conflicts"]
                and r["subject_index"] not in positive
                and _subject_allowed(labels, r["subject_index"], rule)
            }
        )
        query_id = f"{image_id}:{predicate}"
        queries.append(
            {
                "query_id": query_id,
                "image_id": image_id,
                "expression": rule["query"],
                "target_object_indices": positive,
                "distractor_object_indices": distractors,
                "ignore_object_indices": [],
                "program": {"select": "subject", "filter": {"predicate": predicate}},
                "source_relations": [predicate],
                "family": rule["family"],
                "difficulty": "conflict_distractors" if distractors else "positive_only",
            }
        )
    return instances, queries


def _make_sft_record(instances: dict[str, Any], query: dict[str, Any], image_root: Path, split: str) -> dict[str, Any] | None:
    width, height = instances["width"], instances["height"]
    boxes: list[list[int]] = []
    for index in query["target_object_indices"]:
        if index >= len(instances["polygons"]):
            return None
        bbox = polygon_to_hbox(instances["polygons"][index], width, height)
        if bbox is None:
            return None
        boxes.append(bbox)
    image = _resolve_image(image_root, instances["image_source"], instances["image_id"])
    example = make_example(
        image=image,
        question=query["expression"],
        task_type="conditional_set_grounding",
        solution={
            "target_ids": query["target_object_indices"],
            "distractor_ids": query["distractor_object_indices"],
            "bboxes": boxes,
        },
        answer_payload=(
            {"count": len(boxes), "selected_bboxes": boxes}
            if len(boxes) < 5
            else {"count": len(boxes)}
        ),
        source="ReCon1M",
        image_id=instances["image_id"],
        split=split,
        meta={
            "query_id": query["query_id"],
            "family": query["family"],
            "difficulty": query["difficulty"],
            "program": query["program"],
            "source_relations": query["source_relations"],
            "image_source": instances["image_source"],
            "target_object_indices": query["target_object_indices"],
            "distractor_object_indices": query["distractor_object_indices"],
            "audit_decision": query.get("audit_decision"),
        },
    )
    return example.to_dict()


def scan(paths: Iterable[str | Path], *, max_images: int | None = None) -> dict[str, Any]:
    image_count = 0
    relation_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    for row in iter_records(paths):
        image_count += 1
        for relation in _relations(row):
            relation_counts[relation["predicate"]] += 1
        _, queries = build_queries(row)
        for query in queries:
            query_counts[query["source_relations"][0]] += 1
            family_counts[query["family"]] += 1
        if max_images and image_count >= max_images:
            break
    return {
        "images_scanned": image_count,
        "relation_counts": dict(relation_counts.most_common()),
        "queryable_counts": dict(query_counts.most_common()),
        "family_counts": dict(family_counts.most_common()),
    }


def select_pilot(
    paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    max_images: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Select a small, relation-balanced image subset for a reproducible pilot.

    Selection happens before query generation and keeps complete source rows.
    Images with explicit conflicting relations are preferred because they test
    context-conditioned selection rather than plain object presence.
    """
    rng = random.Random(seed)
    families = ("road_vehicle", "ship_water", "aircraft_surface", "contained_object", "adjacent_object", "near_object")
    candidates: dict[str, list[tuple[int, float, dict[str, Any]]]] = {family: [] for family in families}
    seen: set[str] = set()
    for row in iter_records(paths):
        image_id = _image_id(row)
        if image_id in seen:
            continue
        instances, queries = build_queries(row)
        if not queries:
            continue
        seen.add(image_id)
        by_family: dict[str, list[dict[str, Any]]] = {}
        for query in queries:
            by_family.setdefault(query["family"], []).append(query)
        for family, family_queries in by_family.items():
            if family not in candidates:
                continue
            conflict_count = sum(query["difficulty"] == "conflict_distractors" for query in family_queries)
            score = 100.0 * conflict_count + 10.0 * len(family_queries) + rng.random()
            candidates[family].append((conflict_count, score, row))
    for family in candidates:
        candidates[family].sort(key=lambda item: (-item[0], -item[1]))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    cursor = 0
    while len(selected) < max_images and any(cursor < len(candidates[family]) for family in families):
        for family in families:
            if len(selected) >= max_images:
                break
            if cursor >= len(candidates[family]):
                continue
            row = candidates[family][cursor][2]
            image_id = _image_id(row)
            if image_id not in selected_ids:
                selected.append(row)
                selected_ids.add(image_id)
        cursor += 1
    write_jsonl(output_path, selected)
    return {
        "selected_images": len(selected),
        "output": str(Path(output_path).resolve()),
        "families_available": {family: len(items) for family, items in candidates.items()},
        "seed": seed,
    }


def build_dataset(
    paths: Iterable[str | Path],
    image_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    max_images: int | None = None,
    max_queries: int | None = None,
    audit_index: str | Path | None = None,
    audit_exclude_decisions: Iterable[str] = ("reject",),
    exclude_query_ids: Iterable[str] = (),
    predicates: Iterable[str] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_out: list[dict[str, Any]] = []
    queries_out: list[dict[str, Any]] = []
    sft_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    audit_decisions = load_audit_decisions(audit_index)
    excluded_decisions = {str(value).strip().lower() for value in audit_exclude_decisions}
    invalid_excluded = excluded_decisions - AUDIT_DECISIONS
    if invalid_excluded:
        raise ValueError(f"invalid excluded audit decisions: {sorted(invalid_excluded)}")
    manually_excluded = {str(value).strip() for value in exclude_query_ids if str(value).strip()}
    audit_matched: set[str] = set()
    audit_filtered: Counter[str] = Counter()
    manual_filtered: Counter[str] = Counter()
    seen_images: set[str] = set()
    for row in iter_records(paths):
        instances, queries = build_queries(row, predicates=predicates)
        if instances["image_id"] in seen_images:
            continue
        seen_images.add(instances["image_id"])
        split = instances.get("official_split") or deterministic_split(
            "ReCon1M", instances["image_id"], seed=seed
        )
        instances["split"] = split
        instances_out.append(instances)
        for query in queries:
            if query["query_id"] in manually_excluded:
                manual_filtered[query["source_relations"][0]] += 1
                continue
            audit_decision = audit_decisions.get(query["query_id"])
            if audit_decision is not None:
                audit_matched.add(query["query_id"])
                query["audit_decision"] = audit_decision
            if audit_decision in excluded_decisions:
                audit_filtered[query["source_relations"][0]] += 1
                continue
            query["split"] = split
            query["provenance"] = {"generator": "rs_vlm.data.recon1m", "template_version": "v2"}
            queries_out.append(query)
            record = _make_sft_record(instances, query, Path(image_root), split)
            if record is not None:
                sft_by_split[split].append(record)
            if max_queries and len(queries_out) >= max_queries:
                break
        if max_queries and len(queries_out) >= max_queries:
            break
        if max_images and len(seen_images) >= max_images:
            break
    for records in sft_by_split.values():
        rng.shuffle(records)
    write_jsonl(output_dir / "instances.jsonl", instances_out)
    write_jsonl(output_dir / "queries.jsonl", queries_out)
    for split, records in sft_by_split.items():
        write_jsonl(output_dir / f"sft_context_{split}.jsonl", records)
    summary = {
        "seed": seed,
        "predicates": list(predicates) if predicates is not None else list(PREDICATE_RULES),
        "images": len(instances_out),
        "queries": len(queries_out),
        "queries_by_split": dict(Counter(q["split"] for q in queries_out)),
        "queries_by_family": dict(Counter(q["family"] for q in queries_out)),
        "queries_by_difficulty": dict(Counter(q["difficulty"] for q in queries_out)),
        "sft_records_by_split": {split: len(records) for split, records in sft_by_split.items()},
        "audit": {
            "index": str(Path(audit_index).resolve()) if audit_index else None,
            "decisions_loaded": len(audit_decisions),
            "decisions_matched": len(audit_matched),
            "excluded_decisions": sorted(excluded_decisions),
            "queries_filtered": sum(audit_filtered.values()),
            "queries_filtered_by_predicate": dict(audit_filtered),
            "manually_excluded_query_ids": sorted(manually_excluded),
            "queries_manually_filtered": sum(manual_filtered.values()),
            "queries_manually_filtered_by_predicate": dict(manual_filtered),
        },
        "note": "Boxes are HBB projections of source polygons and normalized to 0-1000; source polygons are retained.",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare conditional set-grounding data from ReCon1M")
    sub = parser.add_subparsers(dest="command", required=True)
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("--annotations", nargs="+", required=True)
    scan_parser.add_argument("--max-images", type=int)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--annotations", nargs="+", required=True)
    select_parser.add_argument("--output", required=True)
    select_parser.add_argument("--max-images", type=int, default=500)
    select_parser.add_argument("--seed", type=int, default=42)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--annotations", nargs="+", required=True)
    build_parser.add_argument("--image-root", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--seed", type=int, default=42)
    build_parser.add_argument("--max-images", type=int)
    build_parser.add_argument("--max-queries", type=int)
    build_parser.add_argument("--audit-index")
    build_parser.add_argument("--audit-exclude-decisions", nargs="+", default=["reject"])
    build_parser.add_argument("--exclude-query-ids", nargs="+", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "scan":
        print(json.dumps(scan(args.annotations, max_images=args.max_images), ensure_ascii=False, indent=2))
    elif args.command == "select":
        print(json.dumps(select_pilot(args.annotations, args.output, max_images=args.max_images, seed=args.seed), ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                build_dataset(
                    args.annotations,
                    args.image_root,
                    args.output_dir,
                    seed=args.seed,
                    max_images=args.max_images,
                    max_queries=args.max_queries,
                    audit_index=args.audit_index,
                    audit_exclude_decisions=args.audit_exclude_decisions,
                    exclude_query_ids=args.exclude_query_ids,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
