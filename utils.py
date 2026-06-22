from __future__ import annotations

import os

import torch
import torchvision
from torch.utils.data import DataLoader

from config import CHECKPOINT_PATH, SAVE_PREDS_IMG_DIR, THRESHOLD
from dataload import CrackSegmentationDataset


def save_checkpoint(state: dict, filename: str = CHECKPOINT_PATH):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"=> Saved checkpoint: {filename}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: str = "cuda"):
    print(f"=> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CrackSegmentationDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    val_ds = CrackSegmentationDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def _ensure_bchw(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask.float()


def update_confusion_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = THRESHOLD,
) -> dict[str, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = _ensure_bchw(targets).float()

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    tn = ((1.0 - preds) * (1.0 - targets)).sum()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: dict[str, torch.Tensor], eps: float = 1e-7) -> dict:
    tp = counts["tp"].double()
    fp = counts["fp"].double()
    fn = counts["fn"].double()
    tn = counts["tn"].double()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    foreground_iou = tp / (tp + fp + fn + eps)
    background_iou = tn / (tn + fp + fn + eps)
    miou = 0.5 * (foreground_iou + background_iou)
    pixel_acc = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "precision": float(precision.cpu()),
        "recall": float(recall.cpu()),
        "f1": float(f1.cpu()),
        "foreground_iou": float(foreground_iou.cpu()),
        "background_iou": float(background_iou.cpu()),
        "miou": float(miou.cpu()),
        "pixel_acc": float(pixel_acc.cpu()),
    }


def check_accuracy(
    loader,
    model,
    device="cuda",
    threshold: float = THRESHOLD,
    print_metrics: bool = True,
) -> dict:
    counts = {
        "tp": torch.tensor(0.0, device=device),
        "fp": torch.tensor(0.0, device=device),
        "fn": torch.tensor(0.0, device=device),
        "tn": torch.tensor(0.0, device=device),
    }
    model.eval()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = _ensure_bchw(masks).to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            batch_counts = update_confusion_counts(outputs, masks, threshold)
            for key in counts:
                counts[key] += batch_counts[key]

    metrics = metrics_from_counts(counts)
    if print_metrics:
        print(
            "Validation metrics | "
            f"F1: {metrics['f1']:.4f} | "
            f"mIoU: {metrics['miou']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )
    model.train()
    return metrics


def save_predictions_as_imgs(
    loader,
    model,
    folder: str = SAVE_PREDS_IMG_DIR,
    device="cuda",
    threshold: float = THRESHOLD,
):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    for idx, (images, masks) in enumerate(loader):
        images = images.to(device=device)
        with torch.no_grad():
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["out"]
            preds = (torch.sigmoid(logits) > threshold).float()

        torchvision.utils.save_image(preds, os.path.join(folder, f"pred_{idx}.png"))
        torchvision.utils.save_image(
            _ensure_bchw(masks),
            os.path.join(folder, f"gt_{idx}.png"),
        )

    model.train()
