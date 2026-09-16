from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable


def deterministic_split(
    source: str,
    image_id: str,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> str:
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    digest = hashlib.sha256(f"{seed}:{source}:{image_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def train_val_split(source: str, image_id: str, *, seed: int = 42) -> str:
    digest = hashlib.sha256(f"{seed}:{source}:{image_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if value < 8 / 9 else "val"


def build_image_index(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    index: dict[str, Path] = {}
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        for path in root.rglob(suffix):
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
    return index


def resolve_image(root: str | Path, value: str | list[str], index: dict[str, Path] | None = None) -> Path:
    root = Path(root)
    raw = value[0] if isinstance(value, list) else value
    raw = str(raw).replace("\\", "/")
    direct = root / raw
    if direct.exists():
        return direct.resolve()
    basename = Path(raw).name
    stem = Path(raw).stem
    if index:
        candidate = index.get(basename) or index.get(stem)
        if candidate:
            return candidate.resolve()
    for folder in ("Images_train", "Images_val", "images", "JPEGImages"):
        candidate = root / folder / basename
        if candidate.exists():
            return candidate.resolve()
    return direct.resolve()


def chunks(items: list, size: int) -> Iterable[list]:
    for index in range(0, len(items), size):
        yield items[index : index + size]

