from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from rs_vlm.schema import make_example, write_jsonl


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture_dir = root / "data" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    image_path = fixture_dir / "smoke.png"
    image = Image.new("RGB", (384, 384), "#203040")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 110, 110), fill="#d8d8d8")
    draw.rectangle((250, 250, 340, 340), fill="#a0a0a0")
    image.save(image_path)

    count = make_example(
        image=image_path,
        question="How many gray squares are visible?",
        task_type="count",
        solution={"value": 2},
        answer_payload={"answer": 2},
        source="fixture",
        image_id="smoke",
        split="train",
    )
    spatial = make_example(
        image=image_path,
        question="Where is the smaller gray square relative to the larger gray square?",
        task_type="spatial",
        solution={"label": "north_west"},
        answer_payload={"answer": "north_west"},
        source="fixture",
        image_id="smoke_spatial",
        split="train",
    )
    write_jsonl(fixture_dir / "sft_smoke.jsonl", [count, spatial])
    grpo_rows = []
    for example in (count, spatial):
        raw = example.to_dict()
        raw["messages"] = raw["messages"][:1]
        grpo_rows.append(raw)
    write_jsonl(fixture_dir / "grpo_smoke.jsonl", grpo_rows)
    print(json.dumps({"image": str(image_path), "sft": 2, "grpo": 2}, indent=2))


if __name__ == "__main__":
    main()
