"""Readable LoRA utilities for Qwen3-VL.

The production training command uses ms-swift, which performs these same
operations internally.  This module makes the important PEFT operations
explicit so they can be inspected and reused from Python.

The intended order is::

    model = load_qwen3_vl(...)
    model = freeze_vision_and_aligner(model)
    model = apply_lora(model)

Only language-model ``nn.Linear`` modules are selected as LoRA targets.  This
is deliberately more explicit than passing ``target_modules='all-linear'``:
the Qwen3-VL vision tower and its merger are not adapted in this project.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoraSettings:
    """The LoRA hyperparameters used by the project."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


_VISION_PARTS = (
    "visual",
    "vision_tower",
    "vision_model",
    "vision_encoder",
    "aligner",
)


def is_vision_or_aligner_name(name: str) -> bool:
    """Return whether a module name belongs to the visual branch."""

    parts = name.lower().split(".")
    return any(marker in part for part in parts for marker in _VISION_PARTS)


def freeze_vision_and_aligner(model: Any) -> Any:
    """Freeze Qwen3-VL's visual encoder and visual-language merger.

    The exact nesting can differ between Transformers releases, so the
    function uses parameter names instead of relying on one private class
    layout.  It returns the same model object for convenient chaining.
    """

    for name, parameter in model.named_parameters():
        if is_vision_or_aligner_name(name):
            parameter.requires_grad = False
    return model


def language_linear_targets(model: Any) -> list[str]:
    """Collect full names of language-side Linear modules for PEFT.

    PEFT accepts full module names as target names.  Selecting the full names
    prevents a ``q_proj`` or ``down_proj`` inside the vision tower from being
    adapted accidentally.
    """

    import torch.nn as nn

    targets: list[str] = []
    for name, module in model.named_modules():
        if not name or is_vision_or_aligner_name(name):
            continue
        if isinstance(module, nn.Linear) and not name.endswith("lm_head"):
            targets.append(name)
    if not targets:
        raise ValueError("No language-side Linear modules were found for LoRA")
    return targets


def build_lora_config(model: Any, settings: LoraSettings | None = None) -> Any:
    """Build the PEFT configuration used to inject LoRA adapters."""

    from peft import LoraConfig

    settings = settings or LoraSettings()
    return LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        task_type=settings.task_type,
        target_modules=language_linear_targets(model),
    )


def apply_lora(model: Any, settings: LoraSettings | None = None) -> Any:
    """Freeze the visual branch and attach trainable LoRA weights."""

    from peft import get_peft_model

    freeze_vision_and_aligner(model)
    config = build_lora_config(model, settings)
    adapted_model = get_peft_model(model, config)

    visual_trainables = [
        name
        for name, parameter in adapted_model.named_parameters()
        if parameter.requires_grad and is_vision_or_aligner_name(name)
    ]
    if visual_trainables:
        raise RuntimeError(
            "Visual parameters became trainable after LoRA injection: "
            + ", ".join(visual_trainables[:5])
        )
    return adapted_model


def trainable_parameter_report(model: Any) -> dict[str, int | float]:
    """Return the trainable/total parameter counts printed by smoke tests."""

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }


def save_lora_adapter(model: Any, output_dir: str) -> None:
    """Save only the PEFT adapter weights and configuration."""

    model.save_pretrained(output_dir, safe_serialization=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and save a Qwen3-VL LoRA adapter")
    parser.add_argument("--model", required=True, help="Local Qwen3-VL model directory")
    parser.add_argument("--output", required=True, help="Directory for adapter_model.safetensors")
    args = parser.parse_args()

    import torch
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = apply_lora(model)
    print(trainable_parameter_report(model))
    save_lora_adapter(model, args.output)
    print(f"saved adapter to {args.output}")


if __name__ == "__main__":
    _main()
