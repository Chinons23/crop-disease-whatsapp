"""Single-image inference for the crop disease classifier."""

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.models.model import build_model


class DiseasePredictor:
    def __init__(
        self,
        checkpoint_path: str,
        class_names: List[str],
        backbone: str = "mobilenetv3_small_100",
        image_size: int = 224,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.transform = get_eval_transforms(image_size)

        self.model = build_model(num_classes=len(class_names), backbone=backbone, pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_idxs = probs.topk(min(top_k, len(self.class_names)))
        return [
            (self.class_names[idx], prob.item())
            for idx, prob in zip(top_idxs.tolist(), top_probs)
        ]


def format_prediction_for_whatsapp(predictions: List[Tuple[str, float]]) -> str:
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
