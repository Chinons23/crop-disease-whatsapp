"""Image-quality checks and evaluation for crop disease photos."""

from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize(
    images: torch.Tensor,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Undo ImageNet normalization for quality checks or visualization."""
    if images.ndim not in (3, 4):
        raise ValueError(f"Expected a [3,H,W] or [B,3,H,W] tensor, got {images.shape}")
    if images.shape[-3] != 3:
        raise ValueError(f"Expected three color channels, got {images.shape[-3]}")

    shape = (1, 3, 1, 1) if images.ndim == 4 else (3, 1, 1)
    mean_tensor = torch.tensor(mean, device=images.device).view(shape)
    std_tensor = torch.tensor(std, device=images.device).view(shape)
    return (images * std_tensor + mean_tensor).clamp(0, 1)


class ImageQualityGate:
    """Rule-based checks for brightness, contrast, blur, and visible foliage."""

    def __init__(
        self,
        min_brightness: float = 0.12,
        max_brightness: float = 0.92,
        min_contrast: float = 0.05,
        min_blur_score: float = 0.002,
        min_leaf_fraction: float = 0.08,
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_blur_score = min_blur_score
        self.min_leaf_fraction = min_leaf_fraction
        self.laplacian_kernel = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ).view(1, 1, 3, 3)

    def to_dict(self) -> dict[str, float]:
        """Return thresholds in a serialization-friendly format."""
        return {
            "min_brightness": self.min_brightness,
            "max_brightness": self.max_brightness,
            "min_contrast": self.min_contrast,
            "min_blur_score": self.min_blur_score,
            "min_leaf_fraction": self.min_leaf_fraction,
        }

    @torch.inference_mode()
    def assess(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Assess RGB images in [0, 1], returning per-image decisions and metrics."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected a [B, 3, H, W] tensor, got {tuple(images.shape)}"
            )
        if not torch.is_floating_point(images):
            raise TypeError("Image quality checks require floating-point images")

        red, green, blue = images[:, 0], images[:, 1], images[:, 2]
        grayscale = 0.299 * red + 0.587 * green + 0.114 * blue
        brightness = grayscale.mean(dim=(1, 2))
        contrast = grayscale.std(dim=(1, 2))

        laplacian = F.conv2d(
            grayscale.unsqueeze(1),
            self.laplacian_kernel.to(images.device, images.dtype),
            padding=1,
        )
        blur_score = laplacian.var(dim=(1, 2, 3))

        excess_green = 2 * green - red - blue
        leaf_fraction = (excess_green > 0.08).float().mean(dim=(1, 2))

        brightness_ok = (
            (brightness >= self.min_brightness)
            & (brightness <= self.max_brightness)
        )
        contrast_ok = contrast >= self.min_contrast
        sharp_enough = blur_score >= self.min_blur_score
        leaf_visible = leaf_fraction >= self.min_leaf_fraction
        accepted = brightness_ok & contrast_ok & sharp_enough & leaf_visible

        metrics = {
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "leaf_fraction": leaf_fraction,
            "brightness_ok": brightness_ok,
            "contrast_ok": contrast_ok,
            "sharp_enough": sharp_enough,
            "leaf_visible": leaf_visible,
        }
        return accepted, metrics

    def get_feedback(
        self,
        metrics: dict[str, torch.Tensor],
        index: int = 0,
    ) -> list[str]:
        """Create actionable feedback for one image's failed checks."""
        if index < 0 or index >= len(metrics["brightness"]):
            raise IndexError("quality metric index is outside the batch")

        messages = []
        if not metrics["brightness_ok"][index].item():
            if metrics["brightness"][index].item() < self.min_brightness:
                messages.append("The photo is too dark. Move to a brighter area.")
            else:
                messages.append(
                    "The photo is over-exposed. Avoid strong glare or direct sun."
                )
        if not metrics["contrast_ok"][index].item():
            messages.append("The photo has very low contrast. Use a plain background.")
        if not metrics["sharp_enough"][index].item():
            messages.append("Hold the phone steady and make sure the leaf is in focus.")
        if not metrics["leaf_visible"][index].item():
            messages.append("Place one affected leaf clearly in the centre of the photo.")
        return messages


@torch.inference_mode()
def evaluate_with_gate(
    model: nn.Module,
    data_loader: DataLoader,
    gate: ImageQualityGate,
    device: torch.device,
) -> pd.DataFrame:
    """Collect predictions and quality metrics for every image in a loader."""
    model.to(device)
    model.eval()
    records: list[pd.DataFrame] = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        accepted, metrics = gate.assess(denormalize(images))
        predictions = model(images).argmax(dim=1)
        records.append(
            pd.DataFrame(
                {
                    "label": labels.cpu().numpy(),
                    "pred": predictions.cpu().numpy(),
                    "accepted": accepted.cpu().numpy(),
                    "brightness": metrics["brightness"].cpu().numpy(),
                    "contrast": metrics["contrast"].cpu().numpy(),
                    "blur_score": metrics["blur_score"].cpu().numpy(),
                    "leaf_fraction": metrics["leaf_fraction"].cpu().numpy(),
                }
            )
        )

    if not records:
        raise ValueError("Quality evaluation data loader is empty")
    return pd.concat(records, ignore_index=True)


def summarise_gate(
    results: pd.DataFrame,
    class_names: list[str],
) -> dict[str, Any]:
    """Summarize rejection and accuracy rates, including rates by class."""
    if results.empty:
        raise ValueError("Cannot summarize an empty quality evaluation")

    correct = results["label"] == results["pred"]
    accepted = results["accepted"].astype(bool)
    rejected = ~accepted
    rejection_by_class = rejected.groupby(results["label"]).mean() * 100
    rejection_by_class.index = [
        class_names[index] for index in rejection_by_class.index
    ]

    return {
        "rejection_rate": rejected.mean() * 100,
        "accepted_accuracy": correct[accepted].mean() * 100 if accepted.any() else None,
        "rejected_accuracy": correct[rejected].mean() * 100 if rejected.any() else None,
        "rejection_by_class": rejection_by_class.sort_values(ascending=False),
    }
