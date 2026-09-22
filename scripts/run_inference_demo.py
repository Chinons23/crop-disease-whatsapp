"""Quick CLI demo for running inference on a single image.

Usage:
    python scripts/run_inference_demo.py --image path/to/leaf.jpg
"""

import argparse

from src.inference.predict import DiseasePredictor, format_prediction_for_whatsapp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a leaf image")
    parser.add_argument(
        "--checkpoint", default="models/checkpoints/best_model.pt", help="Path to model weights"
    )
    parser.add_argument(
        "--class-names", default="configs/class_names.txt", help="Path to newline-separated class names"
    )
    args = parser.parse_args()

    with open(args.class_names) as f:
        class_names = [line.strip() for line in f if line.strip()]

    predictor = DiseasePredictor(checkpoint_path=args.checkpoint, class_names=class_names)
    predictions = predictor.predict(args.image, top_k=3)

    print(format_prediction_for_whatsapp(predictions))


if __name__ == "__main__":
    main()
