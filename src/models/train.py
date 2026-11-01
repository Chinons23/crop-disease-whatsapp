"""Training and evaluation helpers for the crop disease classifier.

Usage:
    python -m src.models.train --config configs/train_config.yaml
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loaders import build_dataloader, create_plant_village_splits
from src.inference.artifacts import save_model_artifact
from src.inference.quality import ImageQualityGate
from src.models.model import build_model


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    """Resolve relative config paths from the project root, not the shell cwd."""
    resolved_path = Path(path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = Path(project_root) / resolved_path
    return resolved_path.resolve()


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Return classification accuracy as a percentage."""
    if len(y_true) == 0:
        raise ValueError("Cannot calculate accuracy for an empty batch")
    return torch.eq(y_true, y_pred).sum().item() / len(y_true) * 100


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    frozen_module: nn.Module | None = None,
) -> tuple[float, float]:
    """Train one epoch and return sample-weighted loss and accuracy percentage."""
    model.to(device)
    model.train()
    if frozen_module is not None:
        frozen_module.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in tqdm(data_loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training data loader is empty")
    return total_loss / total_samples, total_correct / total_samples * 100


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Collect sample-weighted loss, labels, and predictions for a data loader."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    labels_list: list[torch.Tensor] = []
    predictions_list: list[torch.Tensor] = []

    for images, labels in tqdm(data_loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        labels_list.append(labels.cpu())
        predictions_list.append(logits.argmax(dim=1).cpu())

    if not labels_list:
        raise ValueError("Evaluation data loader is empty")
    y_true = torch.cat(labels_list)
    y_pred = torch.cat(predictions_list)
    return total_loss / len(y_true), y_true, y_pred


def val_step(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    loss, y_true, y_pred = collect_predictions(model, data_loader, loss_fn, device)
    return loss, accuracy_fn(y_true, y_pred)


def eval_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    """Return loss, accuracy, macro-F1, and labels for later analysis."""
    loss, y_true, y_pred = collect_predictions(model, data_loader, loss_fn, device)
    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss,
        "model_acc": accuracy_fn(y_true, y_pred),
        "model_macro_f1": f1_score(
            y_true.numpy(),
            y_pred.numpy(),
            average="macro",
            zero_division=0,
        ),
        "y_true": y_true.numpy(),
        "y_pred": y_pred.numpy(),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    frozen_module: nn.Module | None = None,
    patience: int | None = None,
) -> list[dict[str, float]]:
    """Train with validation-loss checkpointing and optional early stopping."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if patience is not None and patience < 1:
        raise ValueError("patience must be positive when set")

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        train_loss, train_acc = train_step(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            frozen_module,
        )
        val_loss, val_acc = val_step(val_loader, model, loss_fn, device)
        history.append(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%"
        )
        print(
            f"Validation loss: {val_loss:.5f} | "
            f"Validation accuracy: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best epoch: {best_epoch}).")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    print(f"Restored best weights from epoch {best_epoch} (val loss {best_val_loss:.5f}).")
    return history


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Compatibility wrapper returning fractional accuracy for the CLI."""
    loss, accuracy = train_step(model, loader, criterion, optimizer, device)
    return loss, accuracy / 100


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Compatibility wrapper returning fractional accuracy for the CLI."""
    loss, y_true, y_pred = collect_predictions(model, loader, criterion, device)
    return loss, accuracy_fn(y_true, y_pred) / 100


def main(config_path: str) -> None:
    config_file = Path(config_path).expanduser().resolve()
    cfg = load_config(config_file)
    project_root = (
        config_file.parent.parent
        if config_file.parent.name == "configs"
        else config_file.parent
    )
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = create_plant_village_splits(
        root=resolve_project_path(cfg["data_root"], project_root),
        image_size=cfg.get("image_size", 224),
        resize_size=cfg.get("resize_size", 256),
        train_size=cfg.get("train_split", 0.70),
        val_size=cfg.get("val_split", 0.15),
        seed=seed,
    )
    loader_options = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg.get("num_workers", 0),
        "seed": seed,
        "pin_memory": device.type == "cuda",
    }
    train_loader = build_dataloader(splits.train, shuffle=True, **loader_options)
    validation_loader = build_dataloader(splits.validation, **loader_options)
    test_loader = build_dataloader(splits.test, **loader_options)

    backbone = cfg.get("backbone", "mobilenetv3_small_100")
    freeze_backbone = cfg.get("freeze_backbone", False)
    model = build_model(
        num_classes=len(splits.class_names),
        backbone=backbone,
        pretrained=cfg.get("pretrained", True),
        freeze_backbone=freeze_backbone,
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=cfg.get("lr", 3e-4),
    )
    criterion = nn.CrossEntropyLoss()
    frozen_module = (
        model.features
        if freeze_backbone and hasattr(model, "features")
        else None
    )

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        epochs=cfg["epochs"],
        device=device,
        frozen_module=frozen_module,
        patience=cfg.get("patience"),
    )
    test_results = eval_model(model, test_loader, criterion, device)
    print(
        f"Test loss: {test_results['model_loss']:.4f} | "
        f"Test accuracy: {test_results['model_acc']:.2f}% | "
        f"Test macro-F1: {test_results['model_macro_f1']:.4f}"
    )

    checkpoint_path = resolve_project_path(
        cfg.get("checkpoint_path", "models/checkpoints/best_model.pt"),
        project_root,
    )
    saved_path = save_model_artifact(
        model=model,
        path=checkpoint_path,
        class_names=splits.class_names,
        backbone=backbone,
        image_size=cfg.get("image_size", 224),
        resize_size=cfg.get("resize_size", 256),
        quality_gate=ImageQualityGate(**cfg.get("quality_gate", {})).to_dict(),
        metrics={
            "test_accuracy": test_results["model_acc"],
            "test_macro_f1": test_results["model_macro_f1"],
        },
        seed=seed,
    )
    if not saved_path.is_file() or saved_path.stat().st_size == 0:
        raise RuntimeError(f"Model artifact was not written successfully: {saved_path}")
    print(f"Saved final best model to {saved_path} ({saved_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    arguments = parser.parse_args()
    main(arguments.config)
