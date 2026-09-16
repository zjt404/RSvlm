import json

import pytest

from rs_vlm.schema import OUTPUT_INSTRUCTIONS, SchemaError, TrainingExample, make_example


def test_make_count_example_uses_strict_json(tmp_path):
    image = tmp_path / "image.png"
    image.touch()
    example = make_example(
        image=image,
        question="How many airplanes?",
        task_type="count",
        solution={"value": 3},
        answer_payload={"answer": 3},
        source="test",
        image_id="one",
        split="train",
    )
    assert example.messages[0]["content"].startswith("<image>")
    assert OUTPUT_INSTRUCTIONS["count"] in example.messages[0]["content"]
    assert json.loads(example.messages[-1]["content"]) == {"answer": 3}


def test_grounding_bbox_must_be_normalized():
    raw = {
        "messages": [
            {"role": "user", "content": "<image> locate it"},
            {"role": "assistant", "content": '{"bbox":[0,0,1200,500]}'},
        ],
        "images": ["x.png"],
        "task_type": "grounding",
        "solution": {"bbox": [0, 0, 1200, 500]},
    }
    with pytest.raises(SchemaError):
        TrainingExample.from_dict(raw).validate()


def test_count_assistant_schema_and_solution_must_match(tmp_path):
    image = tmp_path / "image.png"
    image.touch()
    raw = make_example(
        image=image,
        question="How many airplanes?",
        task_type="count",
        solution={"value": 3},
        answer_payload={"answer": 3},
        source="test",
        image_id="one",
        split="train",
    ).to_dict()

    raw["messages"][-1]["content"] = '{"count":3}'
    with pytest.raises(SchemaError, match="contain only answer"):
        TrainingExample.from_dict(raw).validate()

    raw["messages"][-1]["content"] = '{"answer":2}'
    with pytest.raises(SchemaError, match="match solution.value"):
        TrainingExample.from_dict(raw).validate()


def test_image_tokens_must_match_images(tmp_path):
    image = tmp_path / "image.png"
    image.touch()
    raw = make_example(
        image=image,
        question="How many airplanes?",
        task_type="count",
        solution={"value": 1},
        answer_payload={"answer": 1},
        source="test",
        image_id="one",
        split="train",
    ).to_dict()
    raw["messages"][0]["content"] = raw["messages"][0]["content"].replace("<image>", "")
    with pytest.raises(SchemaError, match="number of <image> tokens"):
        TrainingExample.from_dict(raw).validate()
