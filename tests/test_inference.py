from PIL import Image
import numpy as np
import torch.nn as nn

from src.inference.artifacts import save_model_artifact
from src.inference import predict as predict_module
from src.inference.predict import DiseasePredictor, format_prediction_for_whatsapp


def test_format_prediction_single_result():
    predictions = [("Tomato___Late_blight", 0.92)]
    message = format_prediction_for_whatsapp(predictions)
    assert "Late blight" in message
    assert "92%" in message


def test_format_prediction_multiple_results():
    predictions = [
        ("Tomato___Late_blight", 0.72),
        ("Tomato___Early_blight", 0.18),
        ("Tomato___healthy", 0.10),
    ]
    message = format_prediction_for_whatsapp(predictions)
    assert "Late blight" in message
    assert "Other possibilities" in message
    assert "Early blight" in message


def test_predictor_loads_deployment_artifact(tmp_path, monkeypatch):
    class_names = ["Tomato___healthy", "Tomato___Late_blight"]

    def build_test_model(num_classes, **_kwargs):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, num_classes),
        )

    monkeypatch.setattr(predict_module, "build_model", build_test_model)
    model = build_test_model(num_classes=len(class_names))
    checkpoint_path = save_model_artifact(
        model=model,
        path=tmp_path / "model.pt",
        class_names=class_names,
        backbone="test_model",
        metrics={"test_macro_f1": np.float64(0.9)},
    )
    image_path = tmp_path / "leaf.jpg"
    Image.new("RGB", (240, 240), color=(60, 120, 40)).save(image_path)

    predictor = DiseasePredictor(
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )
    predictions = predictor.predict(str(image_path), top_k=2)

    assert predictor.class_names == class_names
    assert len(predictions) == 2
    assert all(label in class_names for label, _ in predictions)
