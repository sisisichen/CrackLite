import os


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.environ.get("CRACKLITE_DATA_DIR", os.path.join(PROJECT_DIR, "data"))

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "image")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "label")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "image")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "label")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "image")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "label")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pth.tar")
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_last.pth.tar")

LOSS_CSV_PATH = os.path.join(OUTPUT_DIR, "loss.csv")
LOSS_FIG_PATH = os.path.join(OUTPUT_DIR, "loss.png")
TRAIN_TIME_LOG_PATH = os.path.join(OUTPUT_DIR, "train_time.txt")
PRED_SAVE_DIR = os.path.join(OUTPUT_DIR, "predictions")
SAVE_PREDS_IMG_DIR = os.path.join(OUTPUT_DIR, "debug_predictions")
INTERNAL_RESPONSE_DIR = os.path.join(OUTPUT_DIR, "internal_responses")

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
BATCH_SIZE = 6
NUM_EPOCHS = 100
NUM_WORKERS = 0
PIN_MEMORY = True
LEARNING_RATE = 1e-3
THRESHOLD = 0.5
LOAD_MODEL = False

TANGENT_LENGTH = 15
NORMAL_LENGTH = 5
LAMBDA_BCE = 1.0
LAMBDA_DICE = 1.0
LAMBDA_CLDICE = 0.5
LAMBDA_CENTERLINE = 0.3
LAMBDA_BOUNDARY = 0.3
BOUNDARY_RADIUS = 2
SKELETON_ITERATIONS = 20
