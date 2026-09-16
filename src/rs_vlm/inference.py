from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from PIL import Image


class InferenceEngine:
    """Lazy Qwen3-VL loader shared by the API and batch predictor."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        *,
        max_new_tokens: int = 512,
        max_pixels: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        if max_pixels is not None and max_pixels <= 0:
            raise ValueError("max_pixels must be positive when provided")
        self.max_pixels = max_pixels
        self.model: Any = None
        self.processor: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self.model = model
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, image: Image.Image | str | Path, question: str) -> str:
        import torch

        with self._lock:
            self.load()
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            processor_kwargs: dict[str, Any] = {}
            if self.max_pixels is not None:
                processor_kwargs["max_pixels"] = self.max_pixels
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **processor_kwargs,
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
            return self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

    def generate_payload(self, image: Image.Image | str | Path, question: str) -> dict[str, Any]:
        raw = self.generate(image, question)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        return {"raw": raw, "parsed": parsed}
