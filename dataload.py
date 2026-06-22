from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class CrackSegmentationDataset(Dataset):
    """Image-mask dataset for binary concrete crack segmentation."""

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                f"No image-mask pairs found in {self.image_dir} and {self.mask_dir}"
            )

    def _collect_samples(self) -> list[tuple[Path, Path]]:
        mask_by_stem = {}
        for mask_path in self.mask_dir.iterdir():
            if mask_path.is_file() and mask_path.suffix.lower() in IMAGE_EXTENSIONS:
                mask_by_stem[mask_path.stem] = mask_path

        samples = []
        missing = []
        for image_path in sorted(self.image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = mask_by_stem.get(image_path.stem)
            if mask_path is None:
                missing.append(image_path.name)
                continue
            samples.append((image_path, mask_path))

        if missing:
            preview = ", ".join(missing[:5])
            print(
                f"[WARN] {len(missing)} images have no mask with the same stem. "
                f"Examples: {preview}"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


CarvanaDataset = CrackSegmentationDataset
