from __future__ import annotations

import itertools
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rs_vlm.data.common import build_image_index, deterministic_split
from rs_vlm.schema import TrainingExample, make_example


DIOR_CLASSES = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)

_DIRECTION_BY_INDEX = (
    "east",
    "north_east",
    "north",
    "north_west",
    "west",
    "south_west",
    "south",
    "south_east",
)


def display_name(label: str) -> str:
    replacements = {
        "baseballfield": "baseball field",
        "basketballcourt": "basketball court",
        "Expressway-Service-area": "expressway service area",
        "Expressway-toll-station": "expressway toll station",
        "golffield": "golf field",
        "groundtrackfield": "ground track field",
        "storagetank": "storage tank",
        "tenniscourt": "tennis court",
        "trainstation": "train station",
    }
    return replacements.get(label, label.lower())


def _text(node: ET.Element, path: str, default: str = "") -> str:
    found = node.find(path)
    return found.text.strip() if found is not None and found.text else default


def parse_annotation(path: str | Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    width = int(float(_text(root, "size/width", "800")))
    height = int(float(_text(root, "size/height", "800")))
    filename = _text(root, "filename", f"{Path(path).stem}.jpg")
    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        label = _text(obj, "name")
        box = obj.find("bndbox")
        if not label or box is None:
            continue
        try:
            x1 = max(0.0, float(_text(box, "xmin")))
            y1 = max(0.0, float(_text(box, "ymin")))
            x2 = min(float(width), float(_text(box, "xmax")))
            y2 = min(float(height), float(_text(box, "ymax")))
        except ValueError:
            continue
        if x1 >= x2 or y1 >= y2:
            continue
        objects.append({"label": label, "bbox": [x1, y1, x2, y2]})
    return {"image_id": Path(path).stem, "filename": filename, "width": width, "height": height, "objects": objects}


def spatial_direction(
    first_bbox: list[float],
    second_bbox: list[float],
    *,
    image_diagonal: float,
    min_distance_ratio: float = 0.05,
    boundary_margin_degrees: float = 5.0,
) -> str | None:
    first_x = (first_bbox[0] + first_bbox[2]) / 2
    first_y = (first_bbox[1] + first_bbox[3]) / 2
    second_x = (second_bbox[0] + second_bbox[2]) / 2
    second_y = (second_bbox[1] + second_bbox[3]) / 2
    dx, dy = first_x - second_x, first_y - second_y
    distance = math.hypot(dx, dy)
    if image_diagonal <= 0 or distance / image_diagonal < min_distance_ratio:
        return None
    angle = math.degrees(math.atan2(-dy, dx)) % 360
    boundaries = [(22.5 + 45 * index) % 360 for index in range(8)]
    boundary_distance = min(abs((angle - boundary + 180) % 360 - 180) for boundary in boundaries)
    if boundary_distance < boundary_margin_degrees:
        return None
    index = int((angle + 22.5) // 45) % 8
    return _DIRECTION_BY_INDEX[index]


def _image_for(annotation: dict[str, Any], image_index: dict[str, Path]) -> Path:
    filename = annotation["filename"]
    return image_index.get(filename) or image_index.get(Path(filename).stem) or Path(filename)


def convert_dior(
    root: str | Path,
    *,
    seed: int = 42,
    max_spatial_per_image: int = 2,
) -> Iterable[TrainingExample]:
    root = Path(root)
    # Official DIOR ships both horizontal and oriented bounding-box XML files.
    # v1 generates count/spatial QA from horizontal boxes only, so never scan the
    # parent Annotations directory when the dedicated HBB directory is present.
    annotations_dir = next(
        (
            path
            for path in (
                root / "Annotations" / "Horizontal Bounding Boxes",
                root / "annotations" / "Horizontal Bounding Boxes",
                root / "Annotations",
                root / "annotations",
            )
            if path.exists()
        ),
        root,
    )
    image_root = next((p for p in (root / "JPEGImages", root / "images", root) if p.exists()), root)
    image_index = build_image_index(image_root)
    rng = random.Random(seed)
    for annotation_path in sorted(annotations_dir.rglob("*.xml")):
        annotation = parse_annotation(annotation_path)
        image_id = annotation["image_id"]
        split = deterministic_split("DIOR", image_id, seed=seed)
        image = _image_for(annotation, image_index)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for obj in annotation["objects"]:
            grouped[obj["label"]].append(obj)

        for label, objects in sorted(grouped.items()):
            count = len(objects)
            name = display_name(label)
            yield make_example(
                image=image,
                question=f"How many {name} objects are visible?",
                task_type="count",
                solution={"value": count},
                answer_payload={"answer": count},
                source="DIOR",
                image_id=image_id,
                split=split,
                meta={"class_name": label, "is_zero": False},
            )

        absent = [label for label in DIOR_CLASSES if label not in grouped]
        if absent:
            label = rng.choice(absent)
            name = display_name(label)
            yield make_example(
                image=image,
                question=f"How many {name} objects are visible?",
                task_type="count",
                solution={"value": 0},
                answer_payload={"answer": 0},
                source="DIOR",
                image_id=image_id,
                split=split,
                meta={"class_name": label, "is_zero": True},
            )

        unique = [(label, values[0]) for label, values in grouped.items() if len(values) == 1]
        pairs = list(itertools.permutations(unique, 2))
        rng.shuffle(pairs)
        emitted = 0
        diagonal = math.hypot(annotation["width"], annotation["height"])
        for (first_label, first), (second_label, second) in pairs:
            direction = spatial_direction(first["bbox"], second["bbox"], image_diagonal=diagonal)
            if direction is None:
                continue
            yield make_example(
                image=image,
                question=(
                    f"Where is the unique {display_name(first_label)} relative to the unique "
                    f"{display_name(second_label)}?"
                ),
                task_type="spatial",
                solution={"label": direction},
                answer_payload={"answer": direction},
                source="DIOR",
                image_id=image_id,
                split=split,
                meta={"subject": first_label, "reference": second_label},
            )
            emitted += 1
            if emitted >= max_spatial_per_image:
                break


def class_distribution(examples: Iterable[TrainingExample]) -> Counter[str]:
    return Counter(str(example.meta.get("class_name", "")) for example in examples if example.task_type == "count")
