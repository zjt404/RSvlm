import math
import runpy
from pathlib import Path

from rs_vlm.metrics import bbox_iou, evaluate_records


CONTEXT_EVALUATOR = runpy.run_path(
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_context.py"
)


def test_bbox_iou():
    assert bbox_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert bbox_iou([0, 0, 100, 100], [100, 100, 200, 200]) == 0.0


def test_evaluate_all_tasks():
    rows = [
        {"task_type": "vqa", "solution": {"answer": "Airport"}, "prediction": '{"answer":"airport"}'},
        {"task_type": "count", "solution": {"value": 4}, "prediction": '{"answer":3}'},
        {"task_type": "count", "solution": {"value": 0}, "prediction": '{"answer":1}'},
        {"task_type": "grounding", "solution": {"bbox": [0, 0, 100, 100]}, "prediction": '{"bbox":[0,0,100,100]}'},
        {"task_type": "spatial", "solution": {"label": "north"}, "prediction": '{"answer":"north"}'},
    ]
    result = evaluate_records(rows)
    assert result["json_valid_rate"] == 1.0
    assert result["vqa"]["accuracy"] == 1.0
    assert result["count"]["mae"] == 1.0
    assert result["count"]["zero_false_positive_rate"] == 1.0
    assert result["grounding"]["mean_iou"] == 1.0
    assert result["spatial"]["accuracy"] == 1.0
    assert math.isclose(result["spatial"]["macro_f1"], 1.0)


def test_context_evaluator_penalizes_malformed_predictions():
    rows = [
        {
            "solution": {"bboxes": [[0, 0, 10, 10]]},
            "prediction": '{"count":1,"selected_bboxes":[[0,0,10,10]]}',
        },
        {
            "solution": {"bboxes": [[20, 20, 30, 30]]},
            "prediction": '{"count":1,"selected_bboxes":[[20,20,30,30]',
        },
        {
            "solution": {"bboxes": [[40, 40, 50, 50]]},
            "prediction": '{"count":2,"selected_bboxes":[[40,40,50,50],[1,1,1,2]]}',
        },
    ]

    result = CONTEXT_EVALUATOR["evaluate"](rows)

    assert result["json_valid_rate"] == 2 / 3
    assert result["schema_valid_rate"] == 1 / 3
    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert math.isclose(result["grounding_f1_at_iou_0_5"], 2 / 3)
