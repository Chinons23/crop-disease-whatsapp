"""Project entry point for model training.

Run with:
    python -m main
    python -m main --config configs/train_config.yaml
"""

import argparse
import sys
import os
from pathlib import Path


try:
    # Find exactly where main.py is running on Kaggle
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive shells/environments
    current_dir = os.getcwd()

# Add the directory to Python's search path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Print structural diagnostics to logs
print(f"Current Execution Directory: {current_dir}")

try:
    print(f"Directory Contents: {os.listdir(current_dir)}")
except Exception:
    pass

# Now Python knows exactly where to find 'src'
from src.models.train import main as train_model

# Establish the project root safely using the resolved directory
PROJECT_ROOT = Path(current_dir).resolve()
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_config.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the crop disease classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Training YAML config (default: {DEFAULT_CONFIG})",
    )
    arguments = parser.parse_args(argv)
    train_model(str(arguments.config))


if __name__ == "__main__":
    main()