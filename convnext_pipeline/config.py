import os
import torch


if torch.cuda.is_available():
    print(f"✅ GPU khả dụng: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    print("❌ Không có GPU, sẽ sử dụng CPU")
    DEVICE = torch.device("cpu")


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 30
NUM_CLASSES = 8
TRAIN_DIR = None  # Đường dẫn thư mục train
VAL_DIR = None    # Đường dẫn thư mục val

MODEL_SAVE_PATH_FUSE = None  # Đường dẫn lưu model fuse
BEST_MODEL_PATH_FUSE = None  # Đường dẫn lưu best model fuse
EXCEL_PATH_FUSE = None  # Đường dẫn file excel fuse
CONFUSION_MATRIX_PATH_FUSE = None  # Đường dẫn confusion matrix fuse
METRICS_PLOT_PATH_FUSE = None  # Đường dẫn metrics plot fuse

RESULTS_DIR_IMAGE = None  # Đường dẫn thư mục kết quả image only
MODEL_SAVE_PATH_IMAGE = None  # Đường dẫn lưu model image only
BEST_MODEL_PATH_IMAGE = None  # Đường dẫn lưu best model image only
EXCEL_PATH_IMAGE = None  # Đường dẫn file excel image only
CONFUSION_MATRIX_PATH_IMAGE = None  # Đường dẫn confusion matrix image only
METRICS_PLOT_PATH_IMAGE = None  # Đường dẫn metrics plot image only
# Cross Attention
RESULTS_DIR_CROSS_ATTENTION = None  # Đường dẫn thư mục cross attention
MODEL_SAVE_PATH_CROSS_ATTENTION = None  # Đường dẫn lưu model cross attention
BEST_MODEL_PATH_CROSS_ATTENTION = None  # Đường dẫn lưu best model cross attention
EXCEL_PATH_CROSS_ATTENTION = None  # Đường dẫn file excel cross attention
CONFUSION_MATRIX_PATH_CROSS_ATTENTION = None  # Đường dẫn confusion matrix cross attention
METRICS_PLOT_PATH_CROSS_ATTENTION = None  # Đường dẫn metrics plot cross attention

GT_CSV_PATH = None  # Đường dẫn file ground truth csv
GT_METADATA_CSV = None  # Đường dẫn file metadata csv
SHUTDOWN_COMMAND = "shutdown /s /t 180"


def ensure_parent_dir(file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


