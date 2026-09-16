from rs_vlm.data.recon1m import build_queries, load_audit_decisions, polygon_to_hbox
from rs_vlm.schema import TrainingExample


def test_polygon_to_hbox_normalizes_and_clips():
    assert polygon_to_hbox([[-10, 10], [50, 10], [50, 110], [-10, 110]], 100, 100) == [0, 100, 500, 1000]


def test_build_queries_uses_explicit_conflicting_relations_as_distractors():
    row = {
        "id": "scene_001.jpg",
        "image_source": "scene_001.jpg",
        "width": 1000,
        "height": 1000,
        "labels": ["other-ship", "harbor", "other-ship", "harbor"],
        "polygons": [
            [[10, 10], [30, 10], [30, 30], [10, 30]],
            [[100, 100], [300, 100], [300, 300], [100, 300]],
            [[500, 500], [520, 500], [520, 520], [500, 520]],
            [[600, 600], [800, 600], [800, 800], [600, 800]],
        ],
        "relations": [
            {"subject_index": 0, "object_index": 1, "predicate": "sail-on"},
            {"subject_index": 2, "object_index": 3, "predicate": "dock-at"},
        ],
    }
    _, queries = build_queries(row)
    sail_query = next(query for query in queries if query["source_relations"] == ["sail-on"])
    assert sail_query["target_object_indices"] == [0]
    assert sail_query["distractor_object_indices"] == [2]


def test_drive_on_does_not_treat_park_next_to_as_a_conflict_and_does_not_emit_park_next_to_query():
    row = {
        "id": "scene_vehicles.jpg",
        "width": 1000,
        "height": 1000,
        "labels": ["small-car", "road", "small-car"],
        "polygons": [
            [[10, 10], [30, 10], [30, 30], [10, 30]],
            [[0, 0], [500, 0], [500, 500], [0, 500]],
            [[40, 10], [60, 10], [60, 30], [40, 30]],
        ],
        "relations": [
            {"subject_index": 0, "object_index": 1, "predicate": "drive-on"},
            {"subject_index": 0, "object_index": 2, "predicate": "park-next-to"},
        ],
    }
    _, queries = build_queries(row)
    drive_query = next(query for query in queries if query["source_relations"] == ["drive-on"])
    assert drive_query["target_object_indices"] == [0]
    assert drive_query["distractor_object_indices"] == []
    assert all(query["source_relations"] != ["park-next-to"] for query in queries)


def test_load_audit_decisions_supports_original_and_new_csv_formats(tmp_path):
    original = tmp_path / "original.csv"
    original.write_text(
        "audit_id,predicate,decision\n"
        "audit_0001_18441_drive-on,drive-on,reject\n",
        encoding="utf-8",
    )
    assert load_audit_decisions(original) == {"18441:drive-on": "reject"}

    current = tmp_path / "current.csv"
    current.write_text(
        "audit_id,query_id,image_id,predicate,decision\n"
        "audit_0001_any,20570:sail-on,20570,sail-on,accept\n",
        encoding="utf-8",
    )
    assert load_audit_decisions(current) == {"20570:sail-on": "accept"}


def test_conditional_set_grounding_schema():
    raw = {
        "messages": [
            {"role": "user", "content": '<image> Select every ship in open water.\n\nReturn exactly one JSON object with no additional text using this schema: {"count":0,"selected_bboxes":[]}. The count must equal the number of selected_bboxes. Normalize every box to 0-1000.'},
            {"role": "assistant", "content": '{"count":1,"selected_bboxes":[[10,10,30,30]]}'},
        ],
        "images": ["/tmp/scene.jpg"],
        "task_type": "conditional_set_grounding",
        "solution": {"target_ids": [0], "distractor_ids": [2], "bboxes": [[10, 10, 30, 30]]},
    }
    assert TrainingExample.from_dict(raw).validate(require_images=False).task_type == "conditional_set_grounding"


def test_dense_conditional_set_grounding_uses_count_only():
    boxes = [[10 + i * 20, 10, 20 + i * 20, 20] for i in range(5)]
    raw = {
        "messages": [
            {"role": "user", "content": "<image> Select every ship on open water."},
            {"role": "assistant", "content": '{"count":5}'},
        ],
        "images": ["/tmp/scene.jpg"],
        "task_type": "conditional_set_grounding",
        "solution": {"target_ids": list(range(5)), "distractor_ids": [], "bboxes": boxes},
    }
    assert TrainingExample.from_dict(raw).validate(require_images=False).task_type == "conditional_set_grounding"
