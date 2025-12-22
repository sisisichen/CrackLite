import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


# 定义数据集的读取
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 过滤出有效的图片文件（假设只考虑.jpg和.png）
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpg'))]

    def __len__(self):  # 读取数据集中的影像数量
        return len(self.images)

    def __getitem__(self, index):
        try:
            # 读取图片和标签路径
            image_name = self.images[index]
            image_path = os.path.join(self.image_dir, image_name)
            mask_path = os.path.join(self.mask_dir, image_name.replace(".jpg", ".jpg"))  # 假设标签文件为.png格式

            # 检查文件是否存在且是文件
            if not (os.path.isfile(image_path) and os.path.isfile(mask_path)):
                raise FileNotFoundError(f"Image or mask file not found at {image_path} or {mask_path}")

            # 打开图片并进行处理
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # 转换为numpy数组，并调整掩码数据类型及值域范围
            image = np.array(image)
            mask = np.array(mask, dtype=np.float32) / 255.0  # 将255缩放到1

            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            return image, mask

        except Exception as e:
            print(f"Error processing sample {index}: {str(e)}")
            # 跳过该样本，并返回固定格式的值（如 None）
            return None, None