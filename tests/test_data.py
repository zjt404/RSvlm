import json

from rs_vlm.data.dior import convert_dior, spatial_direction
from rs_vlm.data.vrsbench import convert_vrsbench


def test_spatial_direction_uses_image_coordinates():
    reference = [450, 450, 550, 550]
    assert spatial_direction([450, 100, 550, 200], reference, image_diagonal=1414) == "north"
    assert spatial_direction([800, 450, 900, 550], reference, image_diagonal=1414) == "east"


def test_convert_minimal_dior(tmp_path):
    annotations = tmp_path / "Annotations"
    images = tmp_path / "JPEGImages"
    annotations.mkdir()
    images.mkdir()
    (images / "001.jpg").touch()
    (annotations / "001.xml").write_text(
        """<annotation><filename>001.jpg</filename><size><width>800</width><height>800</height></size>
        <object><name>airplane</name><bndbox><xmin>50</xmin><ymin>50</ymin><xmax>100</xmax><ymax>100</ymax></bndbox></object>
        <object><name>airport</name><bndbox><xmin>500</xmin><ymin>500</ymin><xmax>700</xmax><ymax>700</ymax></bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )
    examples = list(convert_dior(tmp_path, max_spatial_per_image=2))
    assert sum(example.task_type == "count" for example in examples) == 3
    assert any(example.task_type == "spatial" for example in examples)
    assert all(example.meta["split"] in {"train", "val", "test"} for example in examples)


def test_convert_dior_prefers_horizontal_boxes_over_oriented_boxes(tmp_path):
    hbb = tmp_path / "Annotations" / "Horizontal Bounding Boxes"
    obb = tmp_path / "Annotations" / "Oriented Bounding Boxes"
    images = tmp_path / "JPEGImages"
    hbb.mkdir(parents=True)
    obb.mkdir(parents=True)
    images.mkdir()
    (images / "001.jpg").touch()
    hbb_xml = """<annotation><filename>001.jpg</filename><size><width>800</width><height>800</height></size>
    <object><name>airplane</name><bndbox><xmin>50</xmin><ymin>50</ymin><xmax>100</xmax><ymax>100</ymax></bndbox></object>
    </annotation>"""
    obb_xml = """<annotation><filename>001.jpg</filename><size><width>800</width><height>800</height></size>
    <object><name>bridge</name><bndbox><xmin>50</xmin><ymin>50</ymin><xmax>100</xmax><ymax>100</ymax></bndbox></object>
    </annotation>"""
    (hbb / "001.xml").write_text(hbb_xml, encoding="utf-8")
    (obb / "001.xml").write_text(obb_xml, encoding="utf-8")

    examples = list(convert_dior(tmp_path))
    labels = {
        example.meta.get("class_name")
        for example in examples
        if example.task_type == "count" and not example.meta.get("is_zero")
    }
    assert "airplane" in labels
    assert "bridge" not in labels


def test_convert_vrsbench_scales_bbox_to_1000(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "x.png").touch()
    annotations = tmp_path / "ref.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "id": "ref_1",
                    "image": "x.png",
                    "conversations": [
                        {"from": "human", "value": "<image> Locate the airplane."},
                        {"from": "gpt", "value": "[10, 20, 30, 40]"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    examples = list(convert_vrsbench(annotations, images, task="grounding"))
    assert len(examples) == 1
    assert examples[0].solution["bbox"] == [100, 200, 300, 400]


def test_convert_official_vrsbench_referring_format(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "sample.png").touch()
    annotations = tmp_path / "vrsbench.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "id": "Final_Data/v1.2",
                    "image": "sample.png",
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\n[refer] locate <p>the unique airplane</p>",
                        },
                        {"from": "gpt", "value": "{<10><20><50><80>}"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    examples = list(convert_vrsbench(annotations, images))
    assert len(examples) == 1
    assert examples[0].task_type == "grounding"
    assert examples[0].solution == {"bbox": [100, 200, 500, 800]}
    assert "[refer]" not in examples[0].messages[0]["content"]
    assert "<p>" not in examples[0].messages[0]["content"]


def test_convert_official_vrsbench_eval_format(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "sample.png").touch()
    annotations = tmp_path / "eval.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "image_id": "sample.png",
                    "question": "What is visible?",
                    "ground_truth": "airport",
                    "question_id": 7,
                    "type": "scene",
                }
            ]
        ),
        encoding="utf-8",
    )

    examples = list(convert_vrsbench(annotations, images, source_split="test", task="vqa"))
    assert len(examples) == 1
    assert examples[0].solution == {"answer": "airport"}
    assert examples[0].meta["image_id"] == "sample"
    assert examples[0].meta["split"] == "test"
