import os

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score

from .config import CONFUSION_MATRIX_PATH_FUSE, METRICS_PLOT_PATH_FUSE
from .config import ensure_parent_dir


def evaluate_predictions(y_true, y_pred):
    final_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    final_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-8)

    print(f'Precision: {final_precision:.4f}')
    print(f'Recall: {final_recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

    return final_precision, final_recall, f1_score


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str = CONFUSION_MATRIX_PATH_FUSE):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.xticks(rotation=45, ha='right')
    plt.title('Confusion Matrix - ConvNeXt Base', pad=20, fontsize=16)
    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu confusion matrix tại: {save_path}")


def plot_training_metrics(history, save_path: str = METRICS_PLOT_PATH_FUSE):
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title('Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(history['precision'], label='Train Precision', linewidth=2)
    plt.plot(history['val_precision'], label='Val Precision', linewidth=2)
    plt.title('Precision', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(history['recall'], label='Train Recall', linewidth=2)
    plt.plot(history['val_recall'], label='Val Recall', linewidth=2)
    plt.title('Recall', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ metrics tại: {save_path}")


