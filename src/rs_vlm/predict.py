from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rs_vlm.inference import InferenceEngine


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL predictions for a prepared test JSONL")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/workspace/zjt/qwen3vl")
    parser.add_argument("--adapter")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle-images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per response (default: 512).",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        help="Optional maximum input-image pixel count passed to the Qwen3-VL processor.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    images = [row["images"][0] for row in rows]
    if args.shuffle_images:
        shuffled = images.copy()
        random.Random(args.seed).shuffle(shuffled)
        if len(shuffled) > 1 and shuffled == images:
            shuffled = shuffled[1:] + shuffled[:1]
        images = shuffled

    engine = InferenceEngine(
        args.model,
        args.adapter,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (row, image) in enumerate(zip(rows, images), 1):
            question = row["messages"][0]["content"].replace("<image>", "").strip()
            prediction = engine.generate(image, question)
            result = {
                "task_type": row["task_type"],
                "solution": row["solution"],
                "prediction": prediction,
                "meta": row.get("meta", {}),
                "input_image": image,
                "shuffled_image": bool(args.shuffle_images),
            }
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(f"[{index}/{len(rows)}] {row['task_type']}: {prediction}", flush=True)


if __name__ == "__main__":
    main()
