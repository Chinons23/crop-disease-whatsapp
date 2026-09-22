from src.inference.predict import format_prediction_for_whatsapp


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
