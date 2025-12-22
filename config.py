# config.py（位于 E:\Baoyuyang\lite\unet\config.py）
import os

# ====================== 项目根目录 ======================
# 当前文件 config.py 在 unet 目录中，所以退回上一层
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ====================== 数据路径 ======================
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "image")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "label")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "image")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "label")

# 测试/预测路径（可单独改 test）
TEST_IMG_DIR = VAL_IMG_DIR

# ====================== 模型与输出路径 ======================
UNET_DIR = os.path.join(BASE_DIR, "CrackLite")  # 这次 BASE_DIR 已正确指向 lite 目录

CHECKPOINT_DIR = os.path.join(UNET_DIR, "checkpoint")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth.tar")

LOSS_CSV_PATH = os.path.join(UNET_DIR, "loss.csv")
LOSS_FIG_PATH = os.path.join(UNET_DIR, "loss.png")

PRED_SAVE_DIR = os.path.join(UNET_DIR, "test")
SAVE_PREDS_IMG_DIR = os.path.join(BASE_DIR, "save_image")

TRAIN_TIME_LOG_PATH = os.path.join(UNET_DIR, "train_time.txt")
