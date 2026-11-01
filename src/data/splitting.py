"""Reproducible stratified dataset splitting."""

from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split_indices(
    targets: Sequence[int],
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split sample indices into stratified train, validation, and test sets."""
    if len(targets) == 0:
        raise ValueError("targets must contain at least one label")
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1")

    labels = np.asarray(targets)
    if labels.ndim != 1:
        raise ValueError("targets must be a one-dimensional sequence")
    if min(Counter(labels.tolist()).values()) < 3:
        raise ValueError("Each class must have at least 3 samples for a three-way split")

    indices = np.arange(len(labels))
    try:
        train_indices, holdout_indices = train_test_split(
            indices,
            train_size=train_size,
            stratify=labels,
            random_state=seed,
        )
        val_share = val_size / (1 - train_size)
        val_indices, test_indices = train_test_split(
            holdout_indices,
            train_size=val_share,
            stratify=labels[holdout_indices],
            random_state=seed,
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to make stratified splits with these class counts and split sizes"
        ) from exc

    return (
        train_indices.tolist(),
        val_indices.tolist(),
        test_indices.tolist(),
    )
