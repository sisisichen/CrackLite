# predict.py
import os
import time
import torch
from PIL import Image
import torchvision
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from model import UNET
from config import (
    TEST_IMG_DIR,
    PRED_SAVE_DIR,
    CHECKPOINT_PATH,
)

# ---------- 通用：自动创建目录的小工具函数 ----------
def ensure_dir_for_file(file_path: str):
    """
    确保 file_path 对应的上级文件夹存在，
    如果不存在就自动创建。
    """
    dir_name = os.path.dirname(file_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)


# 预测设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

os.makedirs(PRED_SAVE_DIR, exist_ok=True)

# 数据变换
transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

# ---------------------- 显存读取函数 ----------------------
def get_mem_mb():
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        return 0.0


# ========== 模型加载 ==========

model = UNET(in_channels=3, out_channels=1).to(DEVICE)

if DEVICE == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# 加载模型后（但尚未加载权重）的显存
mem_before_load = get_mem_mb()

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 加载权重之后的显存（可认为是“模型显存占用”）
mem_after_load = get_mem_mb()

# 为后续峰值显存统计做初始化
if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()

# 读取图像列表
image_files = [
    f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))
]

# ========== 计时开始 ==========
start_time = time.time()

# 推理时间记录
timings = []

with torch.no_grad():
    for image_file in tqdm(image_files):
        image_path = os.path.join(TEST_IMG_DIR, image_file)
        image = Image.open(image_path).convert("RGB")
        image = transform(image=np.array(image))["image"].unsqueeze(0).to(DEVICE)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        timings.append(t1 - t0)

        save_path = os.path.join(PRED_SAVE_DIR, image_file)
        ensure_dir_for_file(save_path)
        torchvision.utils.save_image(preds.squeeze(0), save_path)

# ========== 计时结束 ==========
total_time = time.time() - start_time
avg_time = sum(timings) / len(timings) if timings else 0
fps = 1 / avg_time if avg_time > 0 else 0

# ========== 显存统计 ==========
if DEVICE == "cuda":
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
else:
    peak_mem = 0.0

mem_before_load = round(mem_before_load, 2)
mem_after_load = round(mem_after_load, 2)
peak_mem = round(peak_mem, 2)
mem_for_inference = max(0.0, round(peak_mem - mem_after_load, 2))

# 控制台输出结果
print("\n==================== Inference Performance ====================")
print(f"预测图像数量          : {len(image_files)}")
print(f"总推理耗时            : {total_time:.2f} 秒")
print(f"平均每张耗时          : {avg_time * 1000:.2f} ms")
print(f"推理速度（FPS）       : {fps:.2f} 张/秒")
print("-------------------- GPU Memory Usage ------------------------")
print(f"模型创建后显存         : {mem_before_load} MB")
print(f"加载权重后显存         : {mem_after_load} MB")
print(f"推理峰值显存（峰值）   : {peak_mem} MB")
print(f"推理额外显存消耗       : {mem_for_inference} MB")
print("===============================================================")

# 保存到文件
speed_log_path = "predictspeed.txt"
with open(speed_log_path, "w", encoding="utf-8") as f:
    f.write("Inference Speed & Memory Report\n")
    f.write("--------------------------------\n")
    f.write(f"预测图像数量          : {len(image_files)}\n")
    f.write(f"总推理耗时            : {total_time:.2f} 秒\n")
    f.write(f"平均每张耗时          : {avg_time * 1000:.2f} ms\n")
    f.write(f"推理速度（FPS）       : {fps:.2f} 张/秒\n")
    f.write("\n[GPU Memory]\n")
    f.write(f"模型创建后显存         : {mem_before_load} MB\n")
    f.write(f"加载权重后显存         : {mem_after_load} MB\n")
    f.write(f"推理峰值显存（峰值）   : {peak_mem} MB\n")
    f.write(f"推理额外显存消耗       : {mem_for_inference} MB\n")

print(f"\n推理速度和显存统计已保存至：{speed_log_path}")
