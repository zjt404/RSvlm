#!/usr/bin/env python3
"""One-image interactive chat for the final v8 GRPO adapter."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rs_vlm.inference import InferenceEngine


DEFAULT_ADAPTER = (
    PROJECT_ROOT
    / "outputs/grpo_v8_mixed/v0-20260914-171138/checkpoint-625"
)
BOX_PROMPT = (
    "找出图中所有满足条件的目标。只输出一个 JSON 对象，不要附加文字："
    '{"count":目标数量,"selected_bboxes":[[x1,y1,x2,y2],...]}'
    "。count 必须等于框数量；没有目标时输出 count=0 和空数组；"
    "坐标归一化到 0–1000。"
)
TASK_PROMPTS = {
    "vqa": "只输出一个 JSON 对象：{\"answer\":你的简短答案}。不要附加文字。",
    "count": "只输出一个 JSON 对象：{\"answer\":非负整数}。不要附加文字。",
    "grounding": (
        "只输出一个 JSON 对象：{\"bbox\":[x1,y1,x2,y2]}。"
        "框坐标归一化到 0–1000；不要附加文字。"
    ),
    "spatial": (
        "只输出一个 JSON 对象：{\"answer\":方向标签}。方向标签只能为 "
        "north、north_east、east、south_east、south、south_west、west、north_west 之一；"
        "不要附加文字。"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the final v8 remote-sensing model on one image.")
    parser.add_argument("--image", help="Path to the image. Omit for an interactive prompt.")
    parser.add_argument("--question", help="Question for the image. Omit for an interactive prompt.")
    parser.add_argument("--boxes", action="store_true", help="Append the all-box JSON output contract.")
    parser.add_argument("--task", choices=TASK_PROMPTS, help="Append a generic-task JSON output contract.")
    parser.add_argument("--model", default=str(PROJECT_ROOT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    image = args.image or input("图片路径: ").strip()
    question = args.question or input("问题: ").strip()
    if not image or not Path(image).is_file():
        raise SystemExit(f"找不到图片: {image}")
    if not question:
        raise SystemExit("问题不能为空")
    if args.boxes and args.task:
        raise SystemExit("--boxes 和 --task 不能同时使用")
    if args.boxes:
        question = f"{question}\n\n{BOX_PROMPT}"
    elif args.task:
        question = f"{question}\n\n{TASK_PROMPTS[args.task]}"

    engine = InferenceEngine(args.model, args.adapter, max_new_tokens=args.max_new_tokens)
    print("正在加载模型并推理...\n")
    print(engine.generate(image, question))


if __name__ == "__main__":
    main()
