# utils.py
import os
import torch
import torchvision
from dataload import CarvanaDataset
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_PATH,
    SAVE_PREDS_IMG_DIR,
)


# 保存训练模型参数
def save_checkpoint(state, filename: str = CHECKPOINT_PATH):
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("=> Saving checkpoint to:", filename)
    torch.save(state, filename)


# 加载模型参数
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# 加载参数的预制
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
    # 训练集的dataset
    train_ds = CarvanaDataset(
        image_dir=train_dir,          # 影像路径
        mask_dir=train_maskdir,       # 标签路径
        transform=train_transform,    # 影像增强设置
    )
    # 读取训练数据
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # -------------------------------------------------------------------------------------
    # 验证集的dataset
    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    # 读取验证集数据
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    return train_loader, val_loader


# 检查训练模型的精度
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0                      # 初始化正确率
    num_pixels = 0                       # 初始化总像素
    dice_score = 0                       # 每次训练的得分
    model.eval()

    with torch.no_grad():               # 不进行反向传播的数据记录
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()              # 计算结果大于0.5的为1
            num_correct += (preds == y).sum()          # 统计相同结果的pixel量
            num_pixels += torch.numel(preds)           # 总像素
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)} ")
    model.train()


# 保存预测影像结果（用于训练过程中的可视化）
def save_predictions_as_imgs(
    loader, model, folder: str = SAVE_PREDS_IMG_DIR, device="cuda"
):
    # 确保目录存在
    os.makedirs(folder, exist_ok=True)

    model.eval()     # 评估模式，BN/Dropout 固定
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():           # 不进行反向传播
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # 预测结果
        pred_path = os.path.join(folder, f"pred_{idx}.jpg")
        torchvision.utils.save_image(preds, pred_path)

        # 标签影像
        gt_path = os.path.join(folder, f"gt_{idx}.jpg")
        torchvision.utils.save_image(y.unsqueeze(1), gt_path)

    model.train()     # 切回训练模式
