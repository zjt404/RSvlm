from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from rs_vlm.data.common import build_image_index, resolve_image, train_val_split
from rs_vlm.schema import TrainingExample, make_example, read_json_records


_BBOX = re.compile(
    r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
)
_VRSBENCH_BBOX = re.compile(
    r"\{\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(-?\d+(?:\.\d+)?)\s*>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*\}"
)
_TASK_TAG = re.compile(r"^\s*\[(caption|refer|vqa)\]\s*", re.IGNORECASE)
_SHORT_ANSWER_SUFFIX = re.compile(
    r"\s*\.?\s*A short answer to the question is\s*$", re.IGNORECASE
)


def _conversation_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    if isinstance(row.get("messages"), list):
        user = next((m.get("content") for m in row["messages"] if m.get("role") == "user"), None)
        assistant = next((m.get("content") for m in reversed(row["messages"]) if m.get("role") == "assistant"), None)
        if isinstance(user, str) and isinstance(assistant, str):
            return user, assistant
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        user = next((m.get("value") for m in conversations if m.get("from") in {"human", "user"}), None)
        assistant = next((m.get("value") for m in reversed(conversations) if m.get("from") in {"gpt", "assistant"}), None)
        if isinstance(user, str) and isinstance(assistant, str):
            return user, assistant
    question = row.get("question") or row.get("query") or row.get("prompt")
    answer = row.get("answer") or row.get("response") or row.get("solution") or row.get("ground_truth")
    if isinstance(question, str) and answer is not None:
        return question, str(answer)
    return None


def _bbox_from(row: dict[str, Any], answer: str) -> list[int] | None:
    raw = row.get("bbox") or row.get("location") or row.get("obj_bbox")
    if isinstance(raw, str):
        match = _BBOX.search(raw)
        raw = [float(value) for value in match.groups()] if match else None
    if raw is None:
        match = _BBOX.search(answer) or _VRSBENCH_BBOX.search(answer)
        raw = [float(value) for value in match.groups()] if match else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    maximum = max(values)
    if maximum <= 1.0:
        values = [value * 1000 for value in values]
    elif maximum <= 100.0:
        values = [value * 10 for value in values]
    values = [min(1000, max(0, int(round(value)))) for value in values]
    x1, y1, x2, y2 = values
    return values if x1 < x2 and y1 < y2 else None


def _task_hint(row: dict[str, Any], question: str, answer: str, forced_task: str) -> str:
    if forced_task != "auto":
        return forced_task
    metadata = " ".join(
        str(row.get(key, "")) for key in ("task", "task_type", "type", "category", "id")
    ).lower()
    lower_question = question.lower()
    if "[refer]" in lower_question:
        return "grounding"
    if "[caption]" in lower_question:
        return "caption"
    if "[vqa]" in lower_question:
        return "vqa"
    if any(token in metadata for token in ("ground", "refer", "ref_")):
        return "grounding"
    if (_BBOX.search(answer) or _VRSBENCH_BBOX.search(answer)) and any(
        token in lower_question for token in ("locate", "location", "bounding box", "coordinates", "where is")
    ):
        return "grounding"
    if any(token in lower_question for token in ("describe the image", "describe this image", "caption")):
        return "caption"
    return "vqa"


def _clean_question(question: str) -> str:
    cleaned = question.replace("<image>", "").strip()
    cleaned = _TASK_TAG.sub("", cleaned)
    cleaned = cleaned.replace("<p>", "").replace("</p>", "")
    cleaned = _SHORT_ANSWER_SUFFIX.sub("", cleaned)
    return cleaned.strip()


def convert_vrsbench(
    annotation_path: str | Path,
    images_root: str | Path,
    *,
    source_split: str = "train",
    task: str = "auto",
    seed: int = 42,
) -> Iterable[TrainingExample]:
    image_index = build_image_index(images_root)
    for row in read_json_records(annotation_path):
        pair = _conversation_pair(row)
        if pair is None:
            continue
        question, answer = pair
        image_value = (
            row.get("image")
            or row.get("images")
            or row.get("image_path")
            or row.get("file_name")
            or row.get("image_id")
        )
        if not image_value:
            continue
        image = resolve_image(images_root, image_value, image_index)
        # Split by the physical source image, never by QA/referring record id.
        image_id = Path(str(row.get("image_id") or image.stem)).stem
        split = "test" if source_split in {"test", "eval", "validation"} else train_val_split("VRSBench", image_id, seed=seed)
        task_type = _task_hint(row, question, answer, task)
        if task_type == "caption":
            continue
        if task_type == "grounding":
            bbox = _bbox_from(row, answer)
            if bbox is None:
                continue
            cleaned_question = _clean_question(question)
            yield make_example(
                image=image,
                question=cleaned_question,
                task_type="grounding",
                solution={"bbox": bbox},
                answer_payload={"bbox": bbox},
                source="VRSBench",
                image_id=image_id,
                split=split,
                meta={"coordinate_source": "VRSBench_0_100", "original_id": row.get("id")},
            )
        else:
            cleaned_question = _clean_question(question)
            answer_text = answer.strip()
            try:
                existing = json.loads(answer_text)
                if isinstance(existing, dict) and "answer" in existing:
                    answer_text = str(existing["answer"])
            except json.JSONDecodeError:
                pass
            if not answer_text:
                continue
            yield make_example(
                image=image,
                question=cleaned_question,
                task_type="vqa",
                solution={"answer": answer_text},
                answer_payload={"answer": answer_text},
                source="VRSBench",
                image_id=image_id,
                split=split,
                meta={"question_type": row.get("type") or row.get("category"), "original_id": row.get("id")},
            )
