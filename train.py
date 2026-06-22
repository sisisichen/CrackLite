from __future__ import annotations

import csv
import os
import time
from datetime import datetime

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BOUNDARY_RADIUS,
    CHECKPOINT_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INTERNAL_RESPONSE_DIR,
    LAMBDA_BCE,
    LAMBDA_BOUNDARY,
    LAMBDA_CENTERLINE,
    LAMBDA_CLDICE,
    LAMBDA_DICE,
    LAST_CHECKPOINT_PATH,
    LEARNING_RATE,
    LOAD_MODEL,
    LOSS_CSV_PATH,
    LOSS_FIG_PATH,
    NORMAL_LENGTH,
    NUM_EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY,
    SKELETON_ITERATIONS,
    TANGENT_LENGTH,
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    TRAIN_TIME_LOG_PATH,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
)
from losses import HybridCrackLoss
from model import CrackLite
from utils import check_accuracy, get_loaders, save_checkpoint


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIS_INTERNAL_RESPONSES = True


def save_csv(rows: list[dict], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not rows:
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_internal_response_maps(att_maps: dict, epoch: int, batch_idx: int):
    os.makedirs(INTERNAL_RESPONSE_DIR, exist_ok=True)
    for name, feature in att_maps.items():
        feature_np = feature[0].detach().float().cpu().numpy()
        if feature_np.ndim == 3:
            feature_np = feature_np.mean(axis=0)

        min_v = float(feature_np.min())
        max_v = float(feature_np.max())
        if max_v - min_v < 1e-8:
            continue

        feature_np = (feature_np - min_v) / (max_v - min_v + 1e-8)
        feature_np = (feature_np * 255).astype(np.uint8)
        feature_np = cv2.resize(
            feature_np,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap = cv2.applyColorMap(feature_np, cv2.COLORMAP_JET)
        save_path = os.path.join(
            INTERNAL_RESPONSE_DIR,
            f"epoch{epoch + 1}_batch{batch_idx + 1}_{name}.png",
        )
        cv2.imwrite(save_path, heatmap)


def average_components(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


def train_fn(loader, model, optimizer, criterion, scaler, epoch: int) -> dict[str, float]:
    model.train()
    component_rows = []
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)

    for batch_idx, (data, targets) in enumerate(progress):
        data = data.to(device=DEVICE)
        targets = targets.float()
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.to(device=DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=DEVICE == "cuda"):
            outputs = model(data, return_aux=True)
            loss, components = criterion(outputs, targets, return_components=True)

        if DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        component_rows.append(components)
        progress.set_postfix(loss=f"{components['total']:.4f}")

        if VIS_INTERNAL_RESPONSES and epoch == 0 and batch_idx == 0:
            model.eval()
            with torch.no_grad():
                _, maps = model.forward_with_maps(data)
            model.train()
            save_internal_response_maps(maps, epoch, batch_idx)

    return average_components(component_rows)


def build_transforms():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(
                limit=35,
                p=1.0,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
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
    return train_transform, val_transform


def main():
    train_transform, val_transform = build_transforms()

    model = CrackLite(
        in_channels=3,
        out_channels=1,
        tangent_length=TANGENT_LENGTH,
        normal_length=NORMAL_LENGTH,
    ).to(DEVICE)
    criterion = HybridCrackLoss(
        lambda_bce=LAMBDA_BCE,
        lambda_dice=LAMBDA_DICE,
        lambda_cldice=LAMBDA_CLDICE,
        lambda_centerline=LAMBDA_CENTERLINE,
        lambda_boundary=LAMBDA_BOUNDARY,
        skeleton_iterations=SKELETON_ITERATIONS,
        boundary_radius=BOUNDARY_RADIUS,
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=DEVICE == "cuda")

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    start_epoch = 0
    best_miou = -1.0
    history = []

    if LOAD_MODEL and os.path.exists(LAST_CHECKPOINT_PATH):
        print(f"[INFO] Resuming from {LAST_CHECKPOINT_PATH}")
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        best_miou = checkpoint.get("best_miou", -1.0)
        if "scaler" in checkpoint and DEVICE == "cuda":
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    start_dt = datetime.now()
    print(f"[INFO] Training started at {start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"[INFO] Device: {DEVICE}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_components = train_fn(loader=train_loader, model=model, optimizer=optimizer, criterion=criterion, scaler=scaler, epoch=epoch)
        val_metrics = check_accuracy(val_loader, model, device=DEVICE)

        row = {
            "epoch": epoch + 1,
            "loss_total": train_components.get("total", 0.0),
            "loss_bce": train_components.get("bce", 0.0),
            "loss_dice": train_components.get("dice", 0.0),
            "loss_cldice": train_components.get("cldice", 0.0),
            "loss_centerline": train_components.get("centerline", 0.0),
            "loss_boundary": train_components.get("boundary", 0.0),
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_miou": val_metrics["miou"],
        }
        history.append(row)
        save_csv(history, LOSS_CSV_PATH)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_miou": best_miou,
        }
        if DEVICE == "cuda":
            checkpoint["scaler"] = scaler.state_dict()
        save_checkpoint(checkpoint, LAST_CHECKPOINT_PATH)

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            checkpoint["best_miou"] = best_miou
            save_checkpoint(checkpoint, CHECKPOINT_PATH)
            print(f"[INFO] New best validation mIoU: {best_miou:.4f}")

    end_time = time.time()
    end_dt = datetime.now()
    elapsed_sec = end_time - start_time

    os.makedirs(os.path.dirname(LOSS_FIG_PATH), exist_ok=True)
    if history:
        plt.figure(figsize=(10, 6))
        plt.plot([row["epoch"] for row in history], [row["loss_total"] for row in history], label="Training Loss")
        plt.plot([row["epoch"] for row in history], [row["val_miou"] for row in history], label="Validation mIoU")
        plt.title("CrackLite Training Curve")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend()
        plt.savefig(LOSS_FIG_PATH, dpi=300, bbox_inches="tight")

    os.makedirs(os.path.dirname(TRAIN_TIME_LOG_PATH), exist_ok=True)
    with open(TRAIN_TIME_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("CrackLite Training Time Log\n")
        f.write("---------------------------\n")
        f.write(f"Start time : {start_dt:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"End time   : {end_dt:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Total time : {elapsed_sec:.2f} seconds\n")
        f.write(f"Best mIoU  : {best_miou:.4f}\n")

    print(f"[INFO] Training finished at {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"[INFO] Best validation mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
