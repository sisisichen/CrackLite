from __future__ import annotations

import os
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

from config import (
    CHECKPOINT_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PRED_SAVE_DIR,
    TEST_IMG_DIR,
    THRESHOLD,
    VAL_IMG_DIR,
)
from dataload import IMAGE_EXTENSIONS
from model import CrackLite


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_mem_mb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def load_model() -> CrackLite:
    model = CrackLite(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def image_list(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    image_dir = TEST_IMG_DIR if os.path.isdir(TEST_IMG_DIR) else VAL_IMG_DIR
    os.makedirs(PRED_SAVE_DIR, exist_ok=True)

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

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    mem_before_load = get_mem_mb()
    model = load_model()
    mem_after_load = get_mem_mb()

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    files = image_list(image_dir)
    timings = []
    start_time = time.time()

    with torch.no_grad():
        for image_path in tqdm(files, desc="Predicting"):
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image=np.array(image))["image"].unsqueeze(0).to(DEVICE)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            logits = model(tensor)
            pred = (torch.sigmoid(logits) > THRESHOLD).float()

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            timings.append(time.time() - t0)

            save_path = os.path.join(PRED_SAVE_DIR, f"{image_path.stem}.png")
            torchvision.utils.save_image(pred.squeeze(0), save_path)

    total_time = time.time() - start_time
    avg_time = sum(timings) / len(timings) if timings else 0.0
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if DEVICE == "cuda" else 0.0
    mem_for_inference = max(0.0, peak_mem - mem_after_load)

    print("\n==================== Inference Performance ====================")
    print(f"Image directory        : {image_dir}")
    print(f"Predicted images       : {len(files)}")
    print(f"Total inference time   : {total_time:.2f} s")
    print(f"Average per image      : {avg_time * 1000:.2f} ms")
    print(f"FPS                    : {fps:.2f}")
    print("-------------------- GPU Memory Usage -------------------------")
    print(f"Before model load      : {mem_before_load:.2f} MB")
    print(f"After model load       : {mem_after_load:.2f} MB")
    print(f"Peak allocated memory  : {peak_mem:.2f} MB")
    print(f"Extra inference memory : {mem_for_inference:.2f} MB")
    print("===============================================================")

    speed_log_path = os.path.join(PRED_SAVE_DIR, "predict_speed.txt")
    with open(speed_log_path, "w", encoding="utf-8") as f:
        f.write("CrackLite Inference Speed & Memory Report\n")
        f.write("-----------------------------------------\n")
        f.write(f"Image directory        : {image_dir}\n")
        f.write(f"Predicted images       : {len(files)}\n")
        f.write(f"Total inference time   : {total_time:.2f} s\n")
        f.write(f"Average per image      : {avg_time * 1000:.2f} ms\n")
        f.write(f"FPS                    : {fps:.2f}\n")
        f.write(f"Before model load      : {mem_before_load:.2f} MB\n")
        f.write(f"After model load       : {mem_after_load:.2f} MB\n")
        f.write(f"Peak allocated memory  : {peak_mem:.2f} MB\n")
        f.write(f"Extra inference memory : {mem_for_inference:.2f} MB\n")

    print(f"\nSaved predictions to: {PRED_SAVE_DIR}")


if __name__ == "__main__":
    main()
