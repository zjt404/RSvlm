from __future__ import annotations

from rs_vlm.lora import (
    LoraSettings,
    is_vision_or_aligner_name,
    trainable_parameter_report,
)


class DummyParameter:
    def __init__(self, size: int, requires_grad: bool) -> None:
        self.size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.size


class DummyModel:
    def parameters(self):
        return [DummyParameter(6, False), DummyParameter(3, True)]


def test_lora_defaults_match_training_plan() -> None:
    settings = LoraSettings()
    assert settings.rank == 16
    assert settings.alpha == 32
    assert settings.dropout == 0.05


def test_visual_name_detection() -> None:
    assert is_vision_or_aligner_name("model.visual.blocks.0.attn.q_proj")
    assert is_vision_or_aligner_name("model.aligner.proj")
    assert not is_vision_or_aligner_name("model.language_model.layers.0.self_attn.q_proj")


def test_trainable_parameter_report() -> None:
    report = trainable_parameter_report(DummyModel())
    assert report["total_parameters"] == 9
    assert report["trainable_parameters"] == 3
    assert report["trainable_percent"] == 100.0 * 3 / 9
