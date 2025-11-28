import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score
from torch import optim
from tqdm import tqdm

from .config import BEST_MODEL_PATH_FUSE, EXCEL_PATH_FUSE
from .config import ensure_parent_dir


def _init_history() -> Dict[str, List[float]]:
    return {
        'accuracy': [], 'val_accuracy': [],
        'precision': [], 'val_precision': [],
        'recall': [], 'val_recall': [],
        'loss': [], 'val_loss': []
    }


def save_history_to_excel(history: Dict[str, List[float]], excel_path: str = EXCEL_PATH_FUSE) -> None:
    df = pd.DataFrame({
        'Epoch': list(range(1, len(history['accuracy']) + 1)),
        'Train_Accuracy': history['accuracy'],
        'Val_Accuracy': history['val_accuracy'],
        'Train_Loss': history['loss'],
        'Val_Loss': history['val_loss'],
        'Train_Precision': history['precision'],
        'Val_Precision': history['val_precision'],
        'Train_Recall': history['recall'],
        'Val_Recall': history['val_recall']
    })
    ensure_parent_dir(excel_path)
    df.to_excel(excel_path, index=False)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    epochs: int = 25,
    best_model_path: str = BEST_MODEL_PATH_FUSE,
    excel_path: str = EXCEL_PATH_FUSE
) -> Tuple[Dict[str, List[float]], List[int], List[int]]:
    history = _init_history()
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        y_true_train, y_pred_train = [], []

        # ← TRAIN LOOP ĐÃ SỬA
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            # Tự động phát hiện chế độ
            if len(batch) == 2:           # image-only → (images, labels)
                inputs, labels = batch
                meta = None
            else:                         # fusion → (images, meta, labels)
                inputs, meta, labels = batch

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if meta is not None:
                meta = meta.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, meta) if meta is not None else model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

        train_loss = running_loss / total
        train_acc = correct / total
        train_precision = precision_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        train_recall = recall_score(y_true_train, y_pred_train, average='weighted', zero_division=0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    inputs, labels = batch
                    meta = None
                else:
                    inputs, meta, labels = batch

                inputs = inputs.to(device)
                labels = labels.to(device)
                if meta is not None:
                    meta = meta.to(device)

                outputs = model(inputs, meta) if meta is not None else model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_precision = precision_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
        val_recall = recall_score(y_true_val, y_pred_val, average='weighted', zero_division=0)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ensure_parent_dir(best_model_path)
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Mô hình tốt nhất đã được lưu với độ chính xác: {best_val_acc:.4f}")

        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['recall'].append(train_recall)
        history['val_recall'].append(val_recall)

        save_history_to_excel(history, excel_path)
        print(f"💾 Đã lưu kết quả vào {excel_path}")

    return history, y_true_val, y_pred_val


def merge_histories(hist1: Dict[str, List[float]], hist2: Dict[str, List[float]]):
    return {k: hist1[k] + hist2[k] for k in hist1}


