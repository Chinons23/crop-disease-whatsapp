"""Handlers that connect incoming WhatsApp media to the disease predictor."""

import os
import tempfile

import requests

from src.inference.predict import DiseasePredictor, format_prediction_for_whatsapp

# Load once at import time so it's shared across requests.
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT", "models/checkpoints/best_model.pt")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "configs/class_names.txt")

_predictor = None


def get_predictor() -> DiseasePredictor:
    global _predictor
    if _predictor is None:
        with open(CLASS_NAMES_PATH) as f:
            class_names = [line.strip() for line in f if line.strip()]
        _predictor = DiseasePredictor(checkpoint_path=CHECKPOINT_PATH, class_names=class_names)
    return _predictor


def handle_image_message(media_url: str) -> str:
    """Downloads the incoming WhatsApp image and returns a reply message."""
    try:
        auth = (os.environ.get("TWILIO_ACCOUNT_SID"), os.environ.get("TWILIO_AUTH_TOKEN"))
        response = requests.get(media_url, auth=auth, timeout=15)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        predictor = get_predictor()
        predictions = predictor.predict(tmp_path, top_k=3)
        return format_prediction_for_whatsapp(predictions)

    except Exception as exc:  # noqa: BLE001
        return (
            "Sorry, I couldn't process that image 🙏. "
            "Please try sending a clear, well-lit photo of a single leaf."
        )
