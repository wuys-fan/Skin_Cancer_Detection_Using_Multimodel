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
EPOCHS_FINETUNE = 15
NUM_CLASSES = 8
TRAIN_DIR = r'D:\Study\Do_An_Chuyen_Nganh\data\split_data\ISIC_2019\train'
VAL_DIR = r'D:\Study\Do_An_Chuyen_Nganh\data\split_data\ISIC_2019\val'

MODEL_SAVE_PATH_FUSE = r'D:\Study\Do_An_Chuyen_Nganh\results\convnext_metadata\convnext_base_no_sampler_metadata_isic2019.pth'
BEST_MODEL_PATH_FUSE = r'D:\Study\Do_An_Chuyen_Nganh\results\convnext_metadata\convnext_base_best_model_no_sampler_metadata_isic2019.pth'
EXCEL_PATH_FUSE = r'D:\Study\Do_An_Chuyen_Nganh\results\convnext_metadata\training_results.xlsx'
CONFUSION_MATRIX_PATH_FUSE = os.path.join(
    r'D:\Study\Do_An_Chuyen_Nganh\results\convnext_metadata',
    'confusion_matrix_convnext_base_no_sampler.png'
)
METRICS_PLOT_PATH_FUSE = os.path.join(
    r'D:\Study\Do_An_Chuyen_Nganh\results\convnext_metadata',
    'training_metrics_convnext_base_no_sampler.png'
)

RESULTS_DIR_IMAGE = r'D:\Study\Do_An_Chuyen_Nganh\results\convnext'
MODEL_SAVE_PATH_IMAGE = os.path.join(
    RESULTS_DIR_IMAGE,
    'convnext_base_image_only.pth'
)
BEST_MODEL_PATH_IMAGE = os.path.join(
    RESULTS_DIR_IMAGE,
    'convnext_base_best_model_no_sampler_isic2019.pth'
)
EXCEL_PATH_IMAGE = os.path.join(
    RESULTS_DIR_IMAGE,
    'training_results_image_only.xlsx'
)
CONFUSION_MATRIX_PATH_IMAGE = os.path.join(
    RESULTS_DIR_IMAGE,
    'confusion_matrix_convnext_base_image_only.png'
)
METRICS_PLOT_PATH_IMAGE = os.path.join(
    RESULTS_DIR_IMAGE,
    'training_metrics_convnext_base_image_only.png'
)

GT_CSV_PATH = r'D:\Study\Do_An_Chuyen_Nganh\data\ISIC_2019\ISIC_2019_Training_GroundTruth.csv'
GT_METADATA_CSV = r'D:\Study\Do_An_Chuyen_Nganh\data\ISIC_2019\ISIC_2019_Training_Metadata.csv'
SHUTDOWN_COMMAND = "shutdown /s /t 180"


def ensure_parent_dir(file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


