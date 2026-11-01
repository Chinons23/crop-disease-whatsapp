"""Helpers for saving deployment-ready model artifacts."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def save_model_artifact(
    model: nn.Module,
    path: str | Path,
    class_names: list[str],
    backbone: str,
    image_size: int = 224,
    resize_size: int = 256,
    quality_gate: dict[str, float] | None = None,
    metrics: dict[str, float] | None = None,
    seed: int = 42,
) -> Path:
    """Save CPU weights and the preprocessing metadata required for inference."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "architecture": backbone,
        "backbone": backbone,
        "class_names": list(class_names),
        "resize_size": resize_size,
        "image_size": image_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "quality_gate": {
            name: float(value) for name, value in (quality_gate or {}).items()
        },
        "metrics": {
            name: float(value) for name, value in (metrics or {}).items()
        },
        "torch_version": str(torch.__version__),
        "seed": seed,
    }
    torch.save(artifact, artifact_path)
    return artifact_path
