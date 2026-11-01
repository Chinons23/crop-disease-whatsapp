import numpy as np
import pytest
from PIL import Image

from src.data.loaders import build_dataloader, create_plant_village_splits
from src.data.splitting import stratified_split_indices
from src.models.train import resolve_project_path


def test_stratified_split_indices_are_disjoint_complete_and_reproducible():
    targets = np.repeat(np.arange(3), 20)

    splits = stratified_split_indices(targets, seed=7)
    repeated_splits = stratified_split_indices(targets, seed=7)

    assert splits == repeated_splits
    train_indices, validation_indices, test_indices = splits
    assert set(train_indices).isdisjoint(validation_indices)
    assert set(train_indices).isdisjoint(test_indices)
    assert set(validation_indices).isdisjoint(test_indices)
    assert sorted(train_indices + validation_indices + test_indices) == list(
        range(len(targets))
    )
    for indices in splits:
        counts = np.bincount(targets[indices], minlength=3)
        assert np.all(counts > 0)
        assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize(
    ("train_size", "val_size"),
    [(0, 0.15), (0.7, 0), (0.8, 0.2), (1.1, 0.1)],
)
def test_stratified_split_indices_reject_invalid_fractions(train_size, val_size):
    with pytest.raises(ValueError):
        stratified_split_indices([0, 0, 0, 1, 1, 1], train_size, val_size)


def test_stratified_split_indices_reject_classes_too_small():
    with pytest.raises(ValueError, match="at least 3 samples"):
        stratified_split_indices([0, 0, 1, 1, 1, 1])


def test_plant_village_splits_use_train_and_eval_transforms_separately(tmp_path):
    class_names = ["Apple___healthy", "Corn___rust", "Tomato___blight"]
    for class_name in class_names:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(10):
            Image.new("RGB", (40, 40), color=(index * 10, 100, 40)).save(
                class_dir / f"{index}.png"
            )

    splits = create_plant_village_splits(
        root=tmp_path,
        image_size=32,
        resize_size=36,
        seed=3,
    )
    train_images, train_labels = next(
        iter(build_dataloader(splits.train, batch_size=4, seed=3))
    )
    validation_image, _ = splits.validation[0]

    assert splits.class_names == class_names
    assert train_images.shape == (4, 3, 32, 32)
    assert train_labels.shape == (4,)
    assert validation_image.shape == (3, 32, 32)
    for indices in (
        splits.train_indices,
        splits.validation_indices,
        splits.test_indices,
    ):
        assert set(splits.train.dataset.targets[index] for index in indices) == {
            0,
            1,
            2,
        }


def test_relative_config_paths_resolve_from_project_root(tmp_path):
    project_root = tmp_path / "project"

    assert resolve_project_path(
        "models/checkpoints/best_model.pt",
        project_root,
    ) == project_root / "models" / "checkpoints" / "best_model.pt"
