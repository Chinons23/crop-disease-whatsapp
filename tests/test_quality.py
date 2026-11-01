import pytest
import torch

from src.inference.quality import ImageQualityGate, denormalize


def test_quality_gate_accepts_a_clear_green_leaf_image():
    images = torch.zeros((1, 3, 16, 16))
    images[:, 0] = 0.1
    images[:, 1] = 0.6
    images[:, 2] = 0.05
    images[:, 1, :, 1::2] = 0.8

    accepted, metrics = ImageQualityGate().assess(images)

    assert accepted.tolist() == [True]
    assert metrics["leaf_fraction"].item() > 0.08


def test_quality_gate_rejects_dark_images_with_actionable_feedback():
    accepted, metrics = ImageQualityGate().assess(torch.zeros((1, 3, 16, 16)))

    assert accepted.tolist() == [False]
    assert any("too dark" in message for message in ImageQualityGate().get_feedback(metrics))


def test_quality_gate_validates_input_shape():
    with pytest.raises(ValueError, match="Expected a \\[B, 3, H, W\\]"):
        ImageQualityGate().assess(torch.zeros((3, 16, 16)))


def test_denormalize_restores_image_range_and_accepts_batches():
    images = torch.zeros((2, 3, 8, 8))

    restored = denormalize(images)

    assert restored.shape == images.shape
    assert torch.allclose(restored[0, :, 0, 0], torch.tensor([0.485, 0.456, 0.406]))
