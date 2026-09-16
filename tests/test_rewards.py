import math

from rs_vlm.rewards import (
    RemoteSensingTaskReward,
    combined_reward,
    conditional_dense_all_boxes_reward,
    conditional_all_boxes_v8_reward,
    conditional_dense_count_reward,
    conditional_dense_format_reward,
    count_semantic_reward,
    json_format_reward,
    parse_json_payload,
    spatial_semantic_reward,
)


def test_json_parser_is_tolerant_but_format_reward_is_strict():
    fenced = '```json\n{"answer":2}\n```'
    assert parse_json_payload(fenced) == {"answer": 2}
    assert json_format_reward('{"answer":2}', "count") == 1.0
    assert json_format_reward(fenced, "count") == 0.0
    assert json_format_reward("2", "count") == 0.0
    assert math.isclose(combined_reward(fenced, "count", {"value": 2}), 0.9)


def test_count_reward_is_normalized_and_dense():
    assert count_semantic_reward('{"answer":20}', 20) == 1.0
    assert math.isclose(count_semantic_reward('{"answer":18}', 20), math.exp(-2 / 21))
    assert count_semantic_reward('{"answer":-1}', 20) == 0.0


def test_conditional_dense_count_reward_requires_count_only_schema():
    assert conditional_dense_count_reward('{"count":12}', 12) == 1.0
    assert math.isclose(conditional_dense_count_reward('{"count":10}', 12), math.exp(-2 / 13))
    assert conditional_dense_count_reward('{"answer":12}', 12) == 0.0
    assert conditional_dense_format_reward('{"count":12}') == 1.0
    assert conditional_dense_format_reward('{"count":12,"selected_bboxes":[]}') == 0.0
    assert combined_reward('{"count":12}', "conditional_dense_count", {"value": 12}) == 1.0


def test_conditional_dense_all_boxes_reward_prefers_recall_and_penalizes_duplicates():
    target = [[0, 0, 100, 100], [200, 200, 300, 300]]
    perfect = '{"count":2,"selected_bboxes":[[0,0,100,100],[200,200,300,300]]}'
    partial = '{"count":1,"selected_bboxes":[[0,0,100,100]]}'
    duplicate = '{"count":2,"selected_bboxes":[[0,0,100,100],[0,0,100,100]]}'
    malformed = '{"count":2,"selected_bboxes":[[0,0,100,100],[1,1,1,2]]}'
    assert conditional_dense_all_boxes_reward(perfect, target) == 1.0
    assert conditional_dense_all_boxes_reward(perfect, target) > conditional_dense_all_boxes_reward(partial, target)
    assert conditional_dense_all_boxes_reward(partial, target) > conditional_dense_all_boxes_reward(duplicate, target)
    assert conditional_dense_all_boxes_reward(malformed, target) < conditional_dense_all_boxes_reward(partial, target)
    assert combined_reward(perfect, "conditional_dense_all_boxes", {"bboxes": target}) == 1.0


def test_v8_all_boxes_penalizes_distractors_but_not_ignored_annotations():
    solution = {
        "bboxes": [[0, 0, 100, 100]],
        "distractor_bboxes": [[200, 200, 300, 300]],
        "ignore_bboxes": [[400, 400, 500, 500]],
        "all_annotated_bboxes": [[0, 0, 100, 100], [200, 200, 300, 300], [400, 400, 500, 500]],
    }
    perfect = '{"count":1,"selected_bboxes":[[0,0,100,100]]}'
    distractor = '{"count":2,"selected_bboxes":[[0,0,100,100],[200,200,300,300]]}'
    ignored = '{"count":2,"selected_bboxes":[[0,0,100,100],[400,400,500,500]]}'
    assert conditional_all_boxes_v8_reward(perfect, solution) == 1.0
    assert conditional_all_boxes_v8_reward(perfect, solution) == conditional_all_boxes_v8_reward(ignored, solution)
    assert conditional_all_boxes_v8_reward(ignored, solution) > conditional_all_boxes_v8_reward(distractor, solution)
    assert combined_reward(perfect, "conditional_all_boxes_v8", solution) == 1.0


def test_spatial_reward_has_adjacent_partial_credit():
    assert spatial_semantic_reward('{"answer":"north_east"}', "north_east") == 1.0
    assert spatial_semantic_reward('{"answer":"north"}', "north_east") == 0.5
    assert spatial_semantic_reward('{"answer":"south_west"}', "north_east") == 0.0


def test_combined_reward_and_swift_adapter():
    assert combined_reward('{"answer":4}', "count", {"value": 4}) == 1.0
    scorer = RemoteSensingTaskReward()
    rewards = scorer(
        ['{"answer":4}', '{"answer":"east"}'],
        task_type=["count", "spatial"],
        solution=[{"value": 4}, {"label": "east"}],
    )
    assert rewards == [1.0, 1.0]
