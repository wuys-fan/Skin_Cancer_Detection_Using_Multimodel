import os
import sys
from tokenize import String

# Add current directory to path for absolute imports when running as script
if __name__ == "__main__":
    # Get the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
from torch import optim

# Handle both direct execution and module import
try:
    # Try relative imports first (when running as module)
    from .config import (
        DEVICE,
        EPOCHS_FINETUNE,
        EPOCHS_WARMUP,
        MODEL_SAVE_PATH_FUSE,
        MODEL_SAVE_PATH_IMAGE,
        MODEL_SAVE_PATH_CROSS_ATTENTION,
        NUM_CLASSES,
        SHUTDOWN_COMMAND,
        BEST_MODEL_PATH_FUSE,
        BEST_MODEL_PATH_IMAGE,
        BEST_MODEL_PATH_CROSS_ATTENTION,
        EXCEL_PATH_FUSE,
        EXCEL_PATH_IMAGE,
        EXCEL_PATH_CROSS_ATTENTION,
        CONFUSION_MATRIX_PATH_FUSE,
        CONFUSION_MATRIX_PATH_IMAGE,
        CONFUSION_MATRIX_PATH_CROSS_ATTENTION,
        METRICS_PLOT_PATH_FUSE,
        METRICS_PLOT_PATH_IMAGE,
        METRICS_PLOT_PATH_CROSS_ATTENTION,
        ensure_parent_dir,
    )
    from .data import create_dataloaders
    from .evaluation import evaluate_predictions, plot_confusion_matrix, plot_training_metrics
    from .modeling import build_fusion_model, build_model, build_cross_attention_model, load_checkpoint
    from .trainer import merge_histories, train_model
except ImportError:
    # Fall back to absolute imports (when running as script)
    from config import (
        DEVICE,
        EPOCHS_FINETUNE,
        EPOCHS_WARMUP,
        MODEL_SAVE_PATH_FUSE,
        MODEL_SAVE_PATH_IMAGE,
        MODEL_SAVE_PATH_CROSS_ATTENTION,
        NUM_CLASSES,
        SHUTDOWN_COMMAND,
        BEST_MODEL_PATH_FUSE,
        BEST_MODEL_PATH_IMAGE,
        BEST_MODEL_PATH_CROSS_ATTENTION,
        EXCEL_PATH_FUSE,
        EXCEL_PATH_IMAGE,
        EXCEL_PATH_CROSS_ATTENTION,
        CONFUSION_MATRIX_PATH_FUSE,
        CONFUSION_MATRIX_PATH_IMAGE,
        CONFUSION_MATRIX_PATH_CROSS_ATTENTION,
        METRICS_PLOT_PATH_FUSE,
        METRICS_PLOT_PATH_IMAGE,
        METRICS_PLOT_PATH_CROSS_ATTENTION,
        ensure_parent_dir,
    )
    from data import create_dataloaders
    from evaluation import evaluate_predictions, plot_confusion_matrix, plot_training_metrics
    from modeling import build_fusion_model, build_model, build_cross_attention_model, load_checkpoint
    from trainer import merge_histories, train_model

def _freeze_backbone(model):
    """Freeze backbone, only train classifier. Works with CustomConvNeXt and ConvNeXtMetadataFusion."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    # For fusion model, also unfreeze meta_branch
    if hasattr(model, 'meta_branch'):
        for param in model.meta_branch.parameters():
            param.requires_grad = True


def _unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def _build_optimizer(model):
    """Build optimizer compatible with CustomConvNeXt and ConvNeXtMetadataFusion."""
    # Check if it's fusion model (has backbone) or regular model (has features)
    if hasattr(model, 'backbone'):
        # Fusion model (concat)
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.meta_branch.parameters(), 'lr': 3e-5},
            {'params': model.classifier.parameters(), 'lr': 3e-5}
        ]
    else:
        # Regular model
        param_groups = [
            {'params': model.features.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 3e-5}
        ]
    return optim.AdamW(param_groups, weight_decay=0.01)


def _build_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

model_mode ="fusion"
def run_training(model_type: str = model_mode):
    """
    Run training pipeline.
    
    Args:
        model_type: Type of model to train. Options: 'image', 'fusion', 'cross_attention'
    """
    use_metadata = model_type in ['fusion', 'cross_attention']
    train_ds, val_ds, train_loader, val_loader = create_dataloaders(use_metadata=use_metadata)

    if model_type == 'fusion':
        print(f"FUSION MODE: Image + Metadata (Missing Indicator 13 chiều) - Method: Concat")
        model = build_fusion_model(num_classes=NUM_CLASSES)
        best_model_path = BEST_MODEL_PATH_FUSE
        model_save_path = MODEL_SAVE_PATH_FUSE
        excel_path = EXCEL_PATH_FUSE
        confusion_path = CONFUSION_MATRIX_PATH_FUSE
        metrics_path = METRICS_PLOT_PATH_FUSE
    elif model_type == 'cross_attention':
        print(f"CROSS ATTENTION MODE: Image + Metadata (Missing Indicator 13 chiều) - Method: Cross Attention")
        model = build_cross_attention_model(num_classes=NUM_CLASSES)
        best_model_path = BEST_MODEL_PATH_CROSS_ATTENTION
        model_save_path = MODEL_SAVE_PATH_CROSS_ATTENTION
        excel_path = EXCEL_PATH_CROSS_ATTENTION
        confusion_path = CONFUSION_MATRIX_PATH_CROSS_ATTENTION
        metrics_path = METRICS_PLOT_PATH_CROSS_ATTENTION
    else:
        print("IMAGE-ONLY MODE: Image without Metadata")
        model = build_model(num_classes=NUM_CLASSES)
        best_model_path = BEST_MODEL_PATH_IMAGE
        model_save_path = MODEL_SAVE_PATH_IMAGE
        excel_path = EXCEL_PATH_IMAGE
        confusion_path = CONFUSION_MATRIX_PATH_IMAGE
        metrics_path = METRICS_PLOT_PATH_IMAGE
    
    model, resume_training = load_checkpoint(model, DEVICE, best_model_path, model_save_path)
    criterion = nn.CrossEntropyLoss()

    if resume_training:
        print("🔥 Tiếp tục fine-tuning từ checkpoint...")
        print(f"🔥 đang sử dụng mode {model_mode}")
        _unfreeze_all(model)
        optimizer = _build_optimizer(model)
        scheduler = _build_scheduler(optimizer)
        history, y_true_final, y_pred_final = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            DEVICE,
            scheduler,
            epochs=EPOCHS_FINETUNE,
            best_model_path=best_model_path,
            excel_path=excel_path
        )
    else:

        print("🔥 Bắt đầu warm-up training...")
        print(f"🔥 đang sử dụng mode {model_mode}")
        _freeze_backbone(model)
        # For fusion model, need to include meta_branch in warmup optimizer
        if hasattr(model, 'meta_branch'):
            warmup_params = list(model.classifier.parameters()) + list(model.meta_branch.parameters())
        else:
            warmup_params = model.classifier.parameters()
        warmup_optimizer = optim.AdamW(warmup_params, lr=3e-4, weight_decay=0.01)
        history_warmup, _, _ = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            warmup_optimizer,
            DEVICE,
            epochs=EPOCHS_WARMUP,
            best_model_path=best_model_path,
            excel_path=excel_path
        )

        print("🔧 Bắt đầu fine-tuning toàn bộ mô hình...")
        _unfreeze_all(model)
        optimizer = _build_optimizer(model)
        scheduler = _build_scheduler(optimizer)
        history_finetune, y_true_final, y_pred_final = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            DEVICE,
            scheduler,
            epochs=EPOCHS_FINETUNE,
            best_model_path=best_model_path,
            excel_path=excel_path
        )
        history = merge_histories(history_warmup, history_finetune)

    ensure_parent_dir(model_save_path)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model đã được lưu tại: {model_save_path}")

    evaluate_predictions(y_true_final, y_pred_final)
    # Get class names from dataset
    if hasattr(val_ds, 'classes'):
        class_names = val_ds.classes
    else:
        # For metadata dataset, use CLASSES from data.py
        try:
            from .data import CLASSES
        except ImportError:
            from data import CLASSES
        class_names = CLASSES
    plot_confusion_matrix(y_true_final, y_pred_final, class_names, save_path=confusion_path)
    plot_training_metrics(history, save_path=metrics_path)

    os.system(SHUTDOWN_COMMAND)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ConvNeXt Training Pipeline')
    parser.add_argument('--mode', type=str, default='image', 
                        choices=['image', 'fusion', 'cross_attention'],
                        help='Training mode: image (image-only), fusion (concat metadata), cross_attention (cross attention metadata)')
    args = parser.parse_args()
    
    run_training(model_type=args.mode)
