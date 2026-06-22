from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def draw_crack(size: int, seed: int) -> tuple[Image.Image, Image.Image]:
    rng = np.random.default_rng(seed)
    background = rng.normal(loc=185, scale=18, size=(size, size)).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(background, mode="L").convert("RGB")
    mask = Image.new("L", (size, size), 0)
    draw_img = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)

    x = int(rng.integers(size * 0.15, size * 0.85))
    y = int(rng.integers(size * 0.05, size * 0.25))
    points = [(x, y)]
    angle = float(rng.uniform(math.pi * 0.2, math.pi * 0.8))
    step = size / 9

    for _ in range(8):
        angle += float(rng.normal(0, 0.22))
        x = int(np.clip(x + math.cos(angle) * step + rng.normal(0, 4), 4, size - 5))
        y = int(np.clip(y + math.sin(angle) * step + rng.normal(0, 4), 4, size - 5))
        points.append((x, y))

    width = int(rng.integers(1, 3))
    draw_img.line(points, fill=(35, 35, 35), width=width)
    draw_mask.line(points, fill=255, width=max(width, 2))

    if rng.random() > 0.45:
        branch_start = points[int(rng.integers(2, len(points) - 2))]
        bx, by = branch_start
        branch = [branch_start]
        branch_angle = angle + float(rng.choice([-1, 1]) * rng.uniform(0.7, 1.1))
        for _ in range(3):
            bx = int(np.clip(bx + math.cos(branch_angle) * step * 0.6, 4, size - 5))
            by = int(np.clip(by + math.sin(branch_angle) * step * 0.6, 4, size - 5))
            branch.append((bx, by))
        draw_img.line(branch, fill=(45, 45, 45), width=1)
        draw_mask.line(branch, fill=255, width=1)

    image = image.filter(ImageFilter.SMOOTH_MORE)
    return image, mask


def write_split(root: Path, split: str, count: int, size: int, seed_offset: int, rows: list[dict]):
    image_dir = root / split / "image"
    mask_dir = root / split / "label"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        name = f"demo_{split}_{idx:03d}.png"
        image, mask = draw_crack(size=size, seed=seed_offset + idx)
        image_path = image_dir / name
        mask_path = mask_dir / name
        image.save(image_path)
        mask.save(mask_path)
        rows.append(
            {
                "split": split,
                "image": str(image_path.relative_to(root)).replace("\\", "/"),
                "mask": str(mask_path.relative_to(root)).replace("\\", "/"),
                "width": size,
                "height": size,
                "source": "synthetic_smoke_test",
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic CrackLite demo dataset.")
    parser.add_argument("--out", default="demo_data", help="Output directory.")
    parser.add_argument("--image_size", type=int, default=128, help="Square image size.")
    parser.add_argument("--train", type=int, default=8, help="Number of training samples.")
    parser.add_argument("--val", type=int, default=2, help="Number of validation samples.")
    parser.add_argument("--test", type=int, default=2, help="Number of test samples.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    args = parser.parse_args()

    root = Path(args.out)
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    write_split(root, "train", args.train, args.image_size, 1000, rows)
    write_split(root, "val", args.val, args.image_size, 2000, rows)
    write_split(root, "test", args.test, args.image_size, 3000, rows)

    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "mask", "width", "height", "source"])
        writer.writeheader()
        writer.writerows(rows)

    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# CrackLite Demo Data\n\nSynthetic mini data for software smoke testing only.\n",
            encoding="utf-8",
        )

    print(f"Generated {len(rows)} samples in {root}")


if __name__ == "__main__":
    main()
