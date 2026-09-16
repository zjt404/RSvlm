from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import requests


API_URL = os.getenv("RS_VLM_API_URL", "http://127.0.0.1:8000/analyze")


def ask(image, question: str) -> str:
    if image is None or not question.strip():
        return "Please upload an image and enter a question."
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_path = Path(handle.name)
        image.save(handle, format="PNG")
    try:
        with temp_path.open("rb") as image_file:
            response = requests.post(
                API_URL,
                files={"image": ("image.png", image_file, "image/png")},
                data={"question": question},
                timeout=300,
            )
        response.raise_for_status()
        payload = response.json()
        return json.dumps(payload.get("parsed"), ensure_ascii=False, indent=2) if payload.get("parsed") else payload["raw"]
    except requests.RequestException as exc:
        return f"API request failed: {exc}"
    finally:
        temp_path.unlink(missing_ok=True)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Remote Sensing Qwen3-VL") as demo:
        gr.Markdown("# Qwen3-VL Remote Sensing Assistant")
        with gr.Row():
            image = gr.Image(type="pil", label="Remote-sensing image")
            with gr.Column():
                question = gr.Textbox(label="Question", value="What objects are visible? Return JSON only.")
                submit = gr.Button("Analyze", variant="primary")
                output = gr.Code(label="Model response", language="json")
        submit.click(ask, inputs=[image, question], outputs=output)
    return demo


def main() -> None:
    build_demo().launch(
        server_name=os.getenv("DEMO_HOST", "0.0.0.0"),
        server_port=int(os.getenv("DEMO_PORT", "7860")),
        share=False,
    )


if __name__ == "__main__":
    main()

