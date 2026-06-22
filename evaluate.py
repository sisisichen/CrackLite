from __future__ import annotations

import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_WORKERS,
    PIN_MEMORY,
    TEST_IMG_DIR,
    TEST_MASK_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
)
from dataload import CrackSegmentationDataset
from model import CrackLite
from utils import check_accuracy, load_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    image_dir = TEST_IMG_DIR if os.path.isdir(TEST_IMG_DIR) else VAL_IMG_DIR
    mask_dir = TEST_MASK_DIR if os.path.isdir(TEST_MASK_DIR) else VAL_MASK_DIR

    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    dataset = CrackSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = CrackLite(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(CHECKPOINT_PATH, model, device=DEVICE)
    metrics = check_accuracy(loader, model, device=DEVICE)

    print("\n==================== CrackLite Evaluation ====================")
    print(f"Image directory : {image_dir}")
    print(f"Mask directory  : {mask_dir}")
    print(f"Samples         : {len(dataset)}")
    print(f"Precision       : {metrics['precision']:.4f}")
    print(f"Recall          : {metrics['recall']:.4f}")
    print(f"F1              : {metrics['f1']:.4f}")
    print(f"Foreground IoU  : {metrics['foreground_iou']:.4f}")
    print(f"Background IoU  : {metrics['background_iou']:.4f}")
    print(f"mIoU            : {metrics['miou']:.4f}")
    print("==============================================================")


if __name__ == "__main__":
    main()
