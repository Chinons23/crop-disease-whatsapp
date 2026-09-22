"""PyTorch Dataset for the PlantVillage crop disease dataset."""

from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class PlantVillageDataset(Dataset):
    """Loads PlantVillage images from a class-per-folder directory layout.

    Expected layout:
        root/
            Apple___Apple_scab/
                img1.jpg
                ...
            Corn___Common_rust/
                ...
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform

        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
