from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


TASK_TYPES = {"vqa", "count", "grounding", "spatial", "conditional_set_grounding"}
SPATIAL_LABELS = (
    "north",
    "north_east",
    "east",
    "south_east",
    "south",
    "south_west",
    "west",
    "north_west",
)

OUTPUT_INSTRUCTIONS = {
    "vqa": (
        'Return exactly one JSON object with no additional text using this schema: '
        '{"answer":"<short answer>"}'
    ),
    "count": (
        'Return exactly one JSON object with no additional text using this schema: '
        '{"answer":0}. Replace 0 with the non-negative integer count.'
    ),
    "grounding": (
        'Return exactly one JSON object with no additional text using this schema: '
        '{"bbox":[x1,y1,x2,y2]}. Normalize coordinates to 0-1000.'
    ),
    "spatial": (
        'Choose exactly one label from north, north_east, east, south_east, south, '
        'south_west, west, north_west. Return exactly one JSON object with no '
        'additional text using this schema: {"answer":"north"}. Replace north with '
        'the selected label.'
    ),
    "conditional_set_grounding": (
        'Return exactly one JSON object with no additional text. First determine how '
        'many targets match the condition. If fewer than five targets match, use '
        '{"count":0,"selected_bboxes":[]} and include every box. If five or more '
        'targets match, use {"count":0} and return only the count. Normalize every '
        'box to 0-1000.'
    ),
}


class SchemaError(ValueError):
    """Raised when a training example violates the project data contract."""


@dataclass(slots=True)
class TrainingExample:
    messages: list[dict[str, str]]
    images: list[str]
    task_type: str
    solution: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self, require_images: bool = True) -> "TrainingExample":
        if self.task_type not in TASK_TYPES:
            raise SchemaError(f"unknown task_type: {self.task_type!r}")
        if len(self.messages) < 2:
            raise SchemaError("messages must contain at least user and assistant turns")
        if self.messages[0].get("role") != "user":
            raise SchemaError("first message must have role=user")
        if self.messages[-1].get("role") != "assistant":
            raise SchemaError("last message must have role=assistant")
        if not all(isinstance(m.get("content"), str) for m in self.messages):
            raise SchemaError("every message must contain string content")
        if require_images and not self.images:
            raise SchemaError("at least one image is required")
        if any(not isinstance(image, str) or not image.strip() for image in self.images):
            raise SchemaError("every image path must be a non-empty string")
        image_tokens = self.messages[0]["content"].count("<image>")
        if image_tokens != len(self.images):
            raise SchemaError("number of <image> tokens must match number of images")
        if not isinstance(self.solution, dict):
            raise SchemaError("solution must be an object")
        self._validate_solution()
        self._validate_assistant_json()
        return self

    def _validate_solution(self) -> None:
        if self.task_type == "count":
            if set(self.solution) != {"value"}:
                raise SchemaError("count solution must contain only value")
            value = self.solution.get("value")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaError("count solution.value must be a non-negative integer")
        elif self.task_type == "spatial":
            if set(self.solution) != {"label"}:
                raise SchemaError("spatial solution must contain only label")
            if self.solution.get("label") not in SPATIAL_LABELS:
                raise SchemaError("spatial solution.label is invalid")
        elif self.task_type == "grounding":
            if set(self.solution) != {"bbox"}:
                raise SchemaError("grounding solution must contain only bbox")
            validate_bbox(self.solution.get("bbox"))
        elif self.task_type == "vqa":
            if set(self.solution) != {"answer"}:
                raise SchemaError("vqa solution must contain only answer")
            if not isinstance(self.solution.get("answer"), str) or not self.solution["answer"].strip():
                raise SchemaError("vqa solution.answer must be a non-empty string")
        elif self.task_type == "conditional_set_grounding":
            required = {"target_ids", "distractor_ids", "bboxes"}
            if set(self.solution) != required:
                raise SchemaError(
                    "conditional_set_grounding solution must contain target_ids, distractor_ids and bboxes"
                )
            for key in ("target_ids", "distractor_ids"):
                values = self.solution[key]
                if not isinstance(values, list) or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in values
                ):
                    raise SchemaError(f"{key} must be a list of non-negative integers")
                if len(values) != len(set(values)):
                    raise SchemaError(f"{key} must not contain duplicate ids")
            boxes = self.solution["bboxes"]
            if not isinstance(boxes, list) or len(boxes) != len(self.solution["target_ids"]):
                raise SchemaError("bboxes must align one-to-one with target_ids")
            for bbox in boxes:
                validate_bbox(bbox)

    def _validate_assistant_json(self) -> None:
        content = self.messages[-1]["content"]
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise SchemaError("assistant content must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise SchemaError("assistant content must be a JSON object")
        if self.task_type == "grounding":
            if set(payload) != {"bbox"}:
                raise SchemaError("grounding assistant JSON must contain only bbox")
            predicted_bbox = validate_bbox(payload.get("bbox"))
            if predicted_bbox != validate_bbox(self.solution["bbox"]):
                raise SchemaError("grounding assistant bbox must match solution.bbox")
        elif self.task_type == "conditional_set_grounding":
            if "count" not in payload:
                raise SchemaError("conditional grounding assistant JSON must contain count")
            count = payload["count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise SchemaError("conditional_set_grounding count must be a non-negative integer")
            expected = [validate_bbox(bbox) for bbox in self.solution["bboxes"]]
            if len(expected) < 5:
                if set(payload) != {"count", "selected_bboxes"}:
                    raise SchemaError(
                        "sparse conditional grounding must contain count and selected_bboxes"
                    )
                boxes = payload["selected_bboxes"]
                if not isinstance(boxes, list) or count != len(boxes):
                    raise SchemaError("count must equal the number of selected_bboxes")
                normalized = [validate_bbox(bbox) for bbox in boxes]
                if count != len(expected) or normalized != expected:
                    raise SchemaError("selected_bboxes must match solution.bboxes in generated training data")
            else:
                if set(payload) != {"count"}:
                    raise SchemaError(
                        "dense conditional grounding must contain only count"
                    )
                if count != len(expected):
                    raise SchemaError("dense conditional grounding count must match solution")
        elif set(payload) != {"answer"}:
            raise SchemaError("assistant JSON must contain only answer")
        elif self.task_type == "count":
            answer = payload["answer"]
            if isinstance(answer, bool) or not isinstance(answer, int) or answer < 0:
                raise SchemaError("count assistant answer must be a non-negative integer")
            if answer != self.solution["value"]:
                raise SchemaError("count assistant answer must match solution.value")
        elif self.task_type == "spatial":
            answer = payload["answer"]
            if answer not in SPATIAL_LABELS:
                raise SchemaError("spatial assistant answer is invalid")
            if answer != self.solution["label"]:
                raise SchemaError("spatial assistant answer must match solution.label")
        elif self.task_type == "vqa":
            answer = payload["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise SchemaError("vqa assistant answer must be a non-empty string")
            if answer != self.solution["answer"]:
                raise SchemaError("vqa assistant answer must match solution.answer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "images": self.images,
            "task_type": self.task_type,
            "solution": self.solution,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainingExample":
        return cls(
            messages=list(raw.get("messages", [])),
            images=[str(p) for p in raw.get("images", [])],
            task_type=str(raw.get("task_type", "")),
            solution=dict(raw.get("solution", {})),
            meta=dict(raw.get("meta", {})),
        )


def validate_bbox(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SchemaError("bbox must contain four coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise SchemaError("bbox coordinates must be numeric")
    x1, y1, x2, y2 = (int(round(float(x))) for x in value)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise SchemaError("bbox must be ordered and normalized to 0..1000")
    return [x1, y1, x2, y2]


def make_example(
    *,
    image: str | Path,
    question: str,
    task_type: str,
    solution: dict[str, Any],
    answer_payload: dict[str, Any],
    source: str,
    image_id: str,
    split: str,
    meta: dict[str, Any] | None = None,
) -> TrainingExample:
    question = question.strip()
    if not question.startswith("<image>"):
        question = f"<image> {question}"
    instruction = OUTPUT_INSTRUCTIONS.get(task_type)
    if instruction is None:
        raise SchemaError(f"unknown task_type: {task_type!r}")
    if instruction not in question:
        question = f"{question}\n\n{instruction}"
    merged_meta = {"source": source, "image_id": str(image_id), "split": split}
    if meta:
        merged_meta.update(meta)
    return TrainingExample(
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": json.dumps(answer_payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        images=[str(Path(image).resolve())],
        task_type=task_type,
        solution=solution,
        meta=merged_meta,
    ).validate(require_images=True)


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    for key in ("data", "records", "annotations", "items"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise SchemaError(f"cannot find a record list in {path}")


def write_jsonl(path: str | Path, records: Iterable[TrainingExample | dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = record.to_dict() if isinstance(record, TrainingExample) else record
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count
