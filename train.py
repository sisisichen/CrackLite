# train.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import csv
import matplotlib.pyplot as plt
import time
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast  # ✅ 正确的 AMP 导入
import numpy as np
import cv2   # ✅ 用于保存热力图

from model import UNET
from utils import (
    save_checkpoint,
    get_loaders,
    check_accuracy,
)
from config import (
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    LOSS_CSV_PATH,
    LOSS_FIG_PATH,
    TRAIN_TIME_LOG_PATH,
    CHECKPOINT_PATH,   # ✅ 新增：用于断点续训
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ================== 训练参数设置 ==================
LEARNIMG_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6
NUM_EPOCHS = 100
NUM_WORKERS = 0
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
PIN_MENORY = True

LOAD_MODEL = True   # ✅ 现在真的会用到，用来控制是否断点续训

# 是否在训练中导出一次注意力热力图
VIS_ATTN = True
ATTN_SAVE_DIR = "vis_attention"
os.makedirs(ATTN_SAVE_DIR, exist_ok=True)


# ================== 保存 CSV ==================
def save_csv(data, headers, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)


# ================== 注意力热力图保存函数 ==================
def save_attention_maps(att_maps, epoch, batch_idx):
    """
    将 forward_with_att 返回的 att_maps 可视化保存为热力图。
    att_maps: dict，例如：
        {
            "stage1_att": [B, C, H, W],
            "stage1_ffn": [B, C, H, W],
            ...
        }
    这里只取 batch 中第 1 张，按通道平均后做 COLORMAP_JET。
    """
    for name, feat in att_maps.items():
        # feat: [B, C, H, W]
        feat_np = feat[0].detach().cpu().numpy()  # 取第 1 张
        # 通道平均 -> [H, W]
        feat_np = feat_np.mean(axis=0)
        # 归一化
        min_v, max_v = feat_np.min(), feat_np.max()
        if max_v - min_v < 1e-8:
            continue
        feat_np = (feat_np - min_v) / (max_v - min_v + 1e-8)
        feat_np = (feat_np * 255).astype(np.uint8)
        # resize 到训练分辨率
        feat_np = cv2.resize(feat_np, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        # 伪彩色
        heatmap = cv2.applyColorMap(feat_np, cv2.COLORMAP_JET)
        save_path = os.path.join(
            ATTN_SAVE_DIR,
            f"epoch{epoch+1}_batch{batch_idx+1}_{name}.png"
        )
        cv2.imwrite(save_path, heatmap)


# ================== 单轮训练 ==================
def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    model.train()
    avg_epoch_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        optimizer.zero_grad()

        # ✅ 仅在 GPU 上启用混合精度，CPU 上正常训练
        if DEVICE == "cuda" and scaler is not None:
            with autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

        avg_epoch_loss += loss.item() / num_batches

        # ========= 注意力热力图：只在第一个 epoch + 第一个 batch 导出一次 =========
        if VIS_ATTN and epoch == 0 and batch_idx == 0:
            model.eval()
            with torch.no_grad():
                _, att_maps = model.forward_with_att(data)
            model.train()
            save_attention_maps(att_maps, epoch, batch_idx)

    return avg_epoch_loss


def main():
    # ========== 数据增强 ==========
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    # ========== 模型 & 优化器 ==========
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNIMG_RATE)

    # ✅ AMP 的 scaler（仅 GPU 用）
    scaler = GradScaler() if DEVICE == "cuda" else None

    # ========== DataLoader ==========
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MENORY,
    )

    # ========== 断点续训：恢复模型状态 ==========
    start_epoch = 0
    losses = []
    headers = ["Epoch", "Loss"]

    if LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
        print(f"[INFO] Found checkpoint at {CHECKPOINT_PATH}, loading for resume...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # 恢复 epoch
        start_epoch = checkpoint.get("epoch", 0)
        print(f"[INFO] Resuming from epoch {start_epoch + 1} / {NUM_EPOCHS}")

        # 恢复 scaler（如果之前存了）
        if "scaler" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        # 恢复历史 loss（如果文件存在）
        if os.path.exists(LOSS_CSV_PATH):
            with open(LOSS_CSV_PATH, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # 跳过表头
                for row in reader:
                    if len(row) >= 2:
                        epoch_i = int(row[0])
                        loss_i = float(row[1])
                        # 只加载已经完成的 epoch 记录
                        if epoch_i <= start_epoch:
                            losses.append((epoch_i, loss_i))
        print(f"[INFO] Loaded {len(losses)} historical loss records.")
    else:
        print("[INFO] No previous checkpoint found, start training from scratch.")

    # ========== 计时开始 ==========
    start_time = time.time()
    start_dt = datetime.now()
    print(f"[INFO] Training started at {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    print("[INFO] Starting training...")
    for epoch in tqdm(range(start_epoch, NUM_EPOCHS), desc="Training Progress"):
        current_epoch = epoch + 1
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        # 记录损失（包含以前加载的 + 当前新训练的）
        losses.append((current_epoch, avg_loss))
        tqdm.write(f'Epoch {current_epoch}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}')

        # ========== 保存 checkpoint（断点信息也一起存） ==========
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,  # ✅ 下次从 epoch+1 继续
        }
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
        save_checkpoint(checkpoint)

        # 验证集检查
        check_accuracy(val_loader, model, device=DEVICE)

    # ========== 计时结束 ==========
    end_time = time.time()
    end_dt = datetime.now()
    elapsed_sec = end_time - start_time
    elapsed_min = elapsed_sec / 60.0
    elapsed_hr = elapsed_sec / 3600.0

    print("[INFO] Training finished. Saving results...")
    print(f"[INFO] Training ended at {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total training time (this run): {elapsed_sec:.2f} seconds "
          f"({elapsed_min:.2f} minutes, {elapsed_hr:.2f} hours)")

    # ========== 保存 Loss CSV ==========
    save_csv(losses, headers, LOSS_CSV_PATH)

    # ========== 生成 Loss 曲线 ==========
    os.makedirs(os.path.dirname(LOSS_FIG_PATH), exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs = [x[0] for x in losses]
    loss_values = [x[1] for x in losses]
    plt.plot(epochs, loss_values, label="Training Loss")
    plt.title("Training Loss over Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(LOSS_FIG_PATH, dpi=300, bbox_inches='tight')

    # ========== 训练时间写入 txt ==========
    os.makedirs(os.path.dirname(TRAIN_TIME_LOG_PATH), exist_ok=True)
    with open(TRAIN_TIME_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("Training Time Log\n")
        f.write("-----------------\n")
        f.write(f"Start time : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time : {elapsed_sec:.2f} seconds\n")
        f.write(f"           = {elapsed_min:.2f} minutes\n")
        f.write(f"           = {elapsed_hr:.2f} hours\n")

    print(f"[INFO] Training time has been saved to {TRAIN_TIME_LOG_PATH}")


if __name__ == "__main__":
    main()
