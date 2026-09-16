from __future__ import annotations

import json
import math
import re
from typing import Any

from rs_vlm.schema import SPATIAL_LABELS

try:  # Imported only inside a full ms-swift training environment.
    from swift.rewards import ORM, orms
except ImportError:  # pragma: no cover - exercised on the training server.
    class ORM:  # type: ignore[no-redef]
        pass

    orms: dict[str, Any] = {}


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_ALIASES = {
    "northeast": "north_east",
    "north-east": "north_east",
    "north east": "north_east",
    "northwest": "north_west",
    "north-west": "north_west",
    "north west": "north_west",
    "southeast": "south_east",
    "south-east": "south_east",
    "south east": "south_east",
    "southwest": "south_west",
    "south-west": "south_west",
    "south west": "south_west",
}


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", "")))
    if isinstance(completion, list) and completion:
        return completion_text(completion[-1])
    return str(completion or "")


def parse_json_payload(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    match = _JSON_FENCE.match(candidate)
    if match:
        candidate = match.group(1)
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def json_format_reward(text: str, task_type: str) -> float:
    candidate = text.strip()
    if _JSON_FENCE.fullmatch(candidate):
        return 0.0
    payload = parse_json_payload(candidate)
    if payload is None:
        return 0.0
    if task_type == "grounding":
        bbox = payload.get("bbox")
        valid = isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox)
        return float(valid)
    return float("answer" in payload)


def extract_answer(text: str) -> Any:
    payload = parse_json_payload(text)
    if payload is not None and "answer" in payload:
        return payload["answer"]
    return text.strip()


def count_semantic_reward(text: str, ground_truth: int) -> float:
    answer = extract_answer(text)
    if isinstance(answer, bool):
        return 0.0
    if isinstance(answer, (int, float)):
        prediction = float(answer)
    else:
        match = _NUMBER.search(str(answer))
        if not match:
            return 0.0
        prediction = float(match.group())
    if prediction < 0 or not math.isfinite(prediction):
        return 0.0
    return math.exp(-abs(prediction - ground_truth) / (ground_truth + 1.0))


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def vqa_semantic_reward(text: str, ground_truth: Any) -> float:
    """Exact normalized answer credit for the small generic-capability replay."""
    return float(_normalize_text(extract_answer(text)) == _normalize_text(ground_truth))


def grounding_semantic_reward(text: str, ground_truth: Any) -> float:
    payload = parse_json_payload(text)
    predicted = _valid_bbox(payload.get("bbox")) if payload is not None else None
    target = _valid_bbox(ground_truth)
    return _bbox_iou(predicted, target) if predicted is not None and target is not None else 0.0


def conditional_dense_count_reward(text: str, ground_truth: int) -> float:
    """Reward the count-only contract used by dense conditional-grounding rows."""
    payload = parse_json_payload(text)
    value = payload.get("count") if payload is not None else None
    if type(value) is not int or value < 0:
        return 0.0
    return math.exp(-abs(value - ground_truth) / (ground_truth + 1.0))


def conditional_dense_format_reward(text: str) -> float:
    candidate = text.strip()
    if _JSON_FENCE.fullmatch(candidate):
        return 0.0
    payload = parse_json_payload(candidate)
    value = payload.get("count") if payload is not None else None
    return float(payload is not None and set(payload) == {"count"} and type(value) is int and value >= 0)


def _valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return None
    box = [float(item) for item in value]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return box


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def _greedy_match_count(predicted: list[list[float]], target: list[list[float]], threshold: float = 0.5) -> int:
    return len(_greedy_matches(predicted, target, threshold))


def _greedy_matches(
    predicted: list[list[float]], target: list[list[float]], threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    pairs = sorted(
        (
            (_bbox_iou(pred_box, target_box), pred_index, target_index)
            for pred_index, pred_box in enumerate(predicted)
            for target_index, target_box in enumerate(target)
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_target: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, pred_index, target_index in pairs:
        if score < threshold:
            break
        if pred_index in used_pred or target_index in used_target:
            continue
        used_pred.add(pred_index)
        used_target.add(target_index)
        matches.append((pred_index, target_index, score))
    return matches


def conditional_dense_all_boxes_reward(text: str, ground_truth: list[Any]) -> float:
    """Recall-oriented reward for complete dense conditional grounding.

    The reward remains partially informative for imperfect JSON schemas, but a
    perfect score requires exact all-box formatting and one-to-one localization.
    Duplicate predictions receive an explicit penalty in addition to counting
    as false positives during matching.
    """
    target = [box for box in (_valid_bbox(item) for item in ground_truth) if box is not None]
    if not target:
        return 0.0
    payload = parse_json_payload(text)
    raw_boxes = payload.get("selected_bboxes") if payload is not None else None
    if not isinstance(raw_boxes, list):
        return 0.0
    predicted: list[list[float]] = []
    invalid = 0
    for item in raw_boxes:
        box = _valid_bbox(item)
        if box is None:
            invalid += 1
        else:
            predicted.append(box)

    true_positive = _greedy_match_count(predicted, target)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(target)
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    soft_coverage = (
        sum(max((_bbox_iou(pred_box, target_box) for pred_box in predicted), default=0.0) for target_box in target)
        / len(target)
    )
    count_score = math.exp(-abs(len(predicted) - len(target)) / (len(target) + 1.0))
    schema_ok = (
        payload is not None
        and set(payload) == {"count", "selected_bboxes"}
        and type(payload.get("count")) is int
        and payload["count"] == len(raw_boxes)
        and invalid == 0
        and not _JSON_FENCE.fullmatch(text.strip())
    )
    duplicate_pairs = sum(
        _bbox_iou(predicted[left], predicted[right]) >= 0.9
        for left in range(len(predicted))
        for right in range(left + 1, len(predicted))
    )
    duplicate_penalty = min(1.0, duplicate_pairs / max(1, len(predicted)))
    invalid_penalty = min(1.0, invalid / max(1, len(raw_boxes)))
    reward = (
        0.50 * f2
        + 0.20 * recall
        + 0.15 * soft_coverage
        + 0.10 * count_score
        + 0.05 * float(schema_ok)
        - 0.10 * duplicate_penalty
        - 0.10 * invalid_penalty
    )
    return max(0.0, min(1.0, reward))


def _valid_boxes(values: Any) -> list[list[float]]:
    return [box for box in (_valid_bbox(value) for value in values or []) if box is not None]


def conditional_all_boxes_v8_reward(text: str, solution: dict[str, Any]) -> float:
    """Precision-aware all-box reward with explicit annotation roles.

    ``distractor_bboxes`` are labelled non-targets and receive a strong false-
    positive penalty. ``ignore_bboxes`` are visually plausible but benchmark-
    ambiguous objects: they neither earn a match nor count as an error. This
    keeps the policy from exploiting incomplete labels while preserving the
    strict target-box objective used by the final evaluation.
    """
    target = _valid_boxes(solution.get("bboxes"))
    distractors = _valid_boxes(solution.get("distractor_bboxes"))
    ignored = _valid_boxes(solution.get("ignore_bboxes"))
    annotated = _valid_boxes(solution.get("all_annotated_bboxes"))
    payload = parse_json_payload(text)
    raw_boxes = payload.get("selected_bboxes") if payload is not None else None
    if not isinstance(raw_boxes, list):
        return 0.0

    predicted: list[list[float]] = []
    invalid = 0
    for value in raw_boxes:
        box = _valid_bbox(value)
        if box is None:
            invalid += 1
        else:
            predicted.append(box)

    matches = _greedy_matches(predicted, target)
    matched_pred = {match[0] for match in matches}
    matched_target = {match[1] for match in matches}
    duplicate = hard_false_positive = unknown = ignored_predictions = 0
    for pred_index, box in enumerate(predicted):
        if pred_index in matched_pred:
            continue
        # A second box on an already-covered target is a duplicate, even if
        # it was not selected by greedy one-to-one matching.
        if any(_bbox_iou(box, target_box) >= 0.5 for target_box in target):
            duplicate += 1
        elif any(_bbox_iou(box, ignore_box) >= 0.5 for ignore_box in ignored):
            ignored_predictions += 1
        elif any(_bbox_iou(box, distractor_box) >= 0.5 for distractor_box in distractors):
            hard_false_positive += 1
        elif any(_bbox_iou(box, box_ann) >= 0.5 for box_ann in annotated):
            hard_false_positive += 1
        else:
            unknown += 1

    true_positive = len(matches)
    effective_fp = hard_false_positive + duplicate + invalid + 0.4 * unknown
    effective_predictions = true_positive + effective_fp
    precision = true_positive / effective_predictions if effective_predictions else 1.0
    recall = true_positive / len(target) if target else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_match_iou = sum(match[2] for match in matches) / true_positive if true_positive else 0.0
    count_score = math.exp(
        -abs((len(predicted) - ignored_predictions) - len(target)) / (len(target) + 1.0)
    )
    denominator = max(1, len(predicted) + invalid)
    schema_ok = (
        payload is not None
        and set(payload) == {"count", "selected_bboxes"}
        and type(payload.get("count")) is int
        and payload["count"] == len(raw_boxes)
        and invalid == 0
        and not _JSON_FENCE.fullmatch(text.strip())
    )
    reward = (
        0.30 * f1
        + 0.18 * recall
        + 0.17 * precision
        + 0.12 * mean_match_iou
        + 0.10 * count_score
        + 0.13 * float(schema_ok)
        - 0.18 * hard_false_positive / denominator
        - 0.12 * duplicate / denominator
        - 0.08 * invalid / denominator
        - 0.04 * unknown / denominator
    )
    return max(0.0, min(1.0, reward))


def normalize_spatial_label(value: Any) -> str:
    label = str(value).strip().lower().replace("方向", "")
    label = _ALIASES.get(label, label.replace(" ", "_"))
    return label


def spatial_semantic_reward(text: str, ground_truth: str) -> float:
    prediction = normalize_spatial_label(extract_answer(text))
    target = normalize_spatial_label(ground_truth)
    if prediction == target:
        return 1.0
    if prediction not in SPATIAL_LABELS or target not in SPATIAL_LABELS:
        return 0.0
    pred_idx = SPATIAL_LABELS.index(prediction)
    target_idx = SPATIAL_LABELS.index(target)
    distance = min((pred_idx - target_idx) % 8, (target_idx - pred_idx) % 8)
    return 0.5 if distance == 1 else 0.0


def combined_reward(text: str, task_type: str, solution: dict[str, Any]) -> float:
    if task_type == "vqa":
        semantic = vqa_semantic_reward(text, solution["answer"])
    elif task_type == "grounding":
        semantic = grounding_semantic_reward(text, solution["bbox"])
    elif task_type == "count":
        semantic = count_semantic_reward(text, int(solution["value"]))
    elif task_type == "spatial":
        semantic = spatial_semantic_reward(text, str(solution["label"]))
    elif task_type == "conditional_dense_count":
        semantic = conditional_dense_count_reward(text, int(solution["value"]))
        return 0.9 * semantic + 0.1 * conditional_dense_format_reward(text)
    elif task_type == "conditional_dense_all_boxes":
        return conditional_dense_all_boxes_reward(text, list(solution["bboxes"]))
    elif task_type == "conditional_all_boxes_v8":
        return conditional_all_boxes_v8_reward(text, solution)
    else:
        return 0.0
    return 0.9 * semantic + 0.1 * json_format_reward(text, task_type)


class RemoteSensingTaskReward(ORM):
    """Task-conditioned reward registered as ``rs_task_reward`` in ms-swift."""

    def __call__(
        self,
        completions: list[Any],
        task_type: list[str] | str | None = None,
        solution: list[dict[str, Any]] | dict[str, Any] | None = None,
        **_: Any,
    ) -> list[float]:
        task_types = task_type if isinstance(task_type, list) else [task_type] * len(completions)
        solutions = solution if isinstance(solution, list) else [solution] * len(completions)
        rewards: list[float] = []
        for completion, current_task, current_solution in zip(completions, task_types, solutions):
            if current_task not in {
                "vqa",
                "grounding",
                "count",
                "spatial",
                "conditional_dense_count",
                "conditional_dense_all_boxes",
                "conditional_all_boxes_v8",
            } or not isinstance(current_solution, dict):
                rewards.append(0.0)
                continue
            rewards.append(combined_reward(completion_text(completion), current_task, current_solution))
        return rewards


orms["rs_task_reward"] = RemoteSensingTaskReward
