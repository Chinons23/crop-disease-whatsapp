"""Single-image inference for the crop disease classifier."""

from typing import Sequence

import torch
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.models.model import build_model


@torch.inference_mode()
def predict_tensor(
    image: torch.Tensor,
    model: torch.nn.Module,
    class_names: Sequence[str],
    device: str | torch.device,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Predict classes for one already transformed image tensor."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected a [3, H, W] image tensor, got {tuple(image.shape)}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    model = model.to(device)
    model.eval()
    probabilities = torch.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0]
    if probabilities.numel() != len(class_names):
        raise ValueError("class_names count does not match model output classes")
    top_probs, top_indices = probabilities.topk(min(top_k, len(class_names)))
    return [
        (class_names[index], probability.item())
        for index, probability in zip(top_indices.tolist(), top_probs)
    ]


class DiseasePredictor:
    def __init__(
        self,
        checkpoint_path: str,
        class_names: Sequence[str] | None = None,
        backbone: str = "mobilenetv3_small_100",
        image_size: int | None = None,
        resize_size: int | None = None,
        device: str | torch.device | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must contain a state dictionary or model artifact")

        artifact = "model_state_dict" in checkpoint
        if artifact:
            state_dict = checkpoint["model_state_dict"]
            class_names = checkpoint.get("class_names", class_names)
            backbone = checkpoint.get(
                "backbone",
                checkpoint.get("architecture", backbone),
            )
            image_size = checkpoint.get("image_size", image_size)
            resize_size = checkpoint.get("resize_size", resize_size)
        else:
            state_dict = checkpoint

        if not class_names:
            raise ValueError("class_names are required for a weights-only checkpoint")
        self.class_names = list(class_names)
        image_size = image_size or 224
        self.transform = get_eval_transforms(image_size, resize_size or 256)
        self.model = build_model(
            num_classes=len(self.class_names),
            backbone=backbone,
            pretrained=False,
            freeze_backbone=False,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 3) -> list[tuple[str, float]]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return predict_tensor(
            tensor,
            self.model,
            self.class_names,
            self.device,
            top_k=top_k,
        )


def format_prediction_for_whatsapp(
    predictions: list[tuple[str, float]],
) -> str:
    """Formats predictions into a short, farmer-readable WhatsApp message."""
    top_label, top_conf = predictions[0]
    crop, disease = top_label.split("___") if "___" in top_label else (top_label, "")
    disease = disease.replace("_", " ")

    lines = [f"🌿 Most likely: *{disease or 'Healthy'}* ({crop}) — {top_conf * 100:.0f}% confidence"]
    if len(predictions) > 1:
        lines.append("\nOther possibilities:")
        for label, conf in predictions[1:]:
            _, d = label.split("___") if "___" in label else ("", label)
            lines.append(f"  • {d.replace('_', ' ')} — {conf * 100:.0f}%")
    return "\n".join(lines)
