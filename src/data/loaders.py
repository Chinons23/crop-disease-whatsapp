"""PlantVillage split and DataLoader construction helpers."""

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import PlantVillageDataset
from src.data.splitting import stratified_split_indices
from src.data.transforms import get_eval_transforms, get_train_transforms


@dataclass(frozen=True)
class PlantVillageSplits:
    train: Subset
    validation: Subset
    test: Subset
    class_names: list[str]
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]


def create_plant_village_splits(
    root: str | Path,
    image_size: int = 224,
    resize_size: int = 256,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> PlantVillageSplits:
    """Create train/validation/test subsets with separate train/eval transforms."""
    train_base = PlantVillageDataset(
        root,
        transform=get_train_transforms(image_size),
    )
    eval_base = copy.copy(train_base)
    eval_base.transform = get_eval_transforms(image_size, resize_size)

    train_indices, validation_indices, test_indices = stratified_split_indices(
        train_base.targets,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
    )
    return PlantVillageSplits(
        train=Subset(train_base, train_indices),
        validation=Subset(eval_base, validation_indices),
        test=Subset(eval_base, test_indices),
        class_names=train_base.classes,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy generators inside each DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build a reproducible DataLoader, seeding both shuffling and workers."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
