import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import NUM_CLASSES


class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        convnext = models.convnext_base(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(convnext.children())[:-1])
        num_features = 1024
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(num_features),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=NUM_CLASSES):
    return CustomConvNeXt(num_classes=num_classes)


def load_checkpoint(model, device, best_model_path, model_save_path):
    resume_training = False

    if os.path.exists(best_model_path):
        print(f"🔄 Tìm thấy checkpoint: {best_model_path}")
        print("📥 Đang load model để tiếp tục training...")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            resume_training = True
            print("✅ Đã load model thành công! Tiếp tục training từ checkpoint.")
        except Exception as exc:
            print(f"⚠️ Không thể load checkpoint: {exc}")
            print("🔄 Sẽ bắt đầu training từ đầu với pretrained weights...")
    elif os.path.exists(model_save_path):
        print(f"🔄 Tìm thấy checkpoint: {model_save_path}")
        print("📥 Đang load model để tiếp tục training...")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            resume_training = True
            print("✅ Đã load model thành công! Tiếp tục training từ checkpoint.")
        except Exception as exc:
            print(f"⚠️ Không thể load checkpoint: {exc}")
            print("🔄 Sẽ bắt đầu training từ đầu với pretrained weights...")
    else:
        print("ℹ️ Không tìm thấy checkpoint. Bắt đầu training từ đầu với pretrained weights...")

    return model.to(device), resume_training


class ConvNeXtMetadataFusion(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        convnext = models.convnext_base(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(convnext.children())[:-1])
        
        self.meta_branch = nn.Sequential(
            nn.Linear(13, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.GELU()
        )
        
        # Simple concatenation
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024 + 256),
            nn.Dropout(0.4),
            nn.Linear(1024 + 256, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, meta):
        x = self.backbone(x)  # (B, 1024, H, W) - typically (B, 1024, 7, 7) for 224x224 input
        # Global average pooling and flatten
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # (B, 1024, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 1024)
        m = self.meta_branch(meta)  # (B, 256)
        
        # Simple concatenation
        x = torch.cat([x, m], dim=1)  # (B, 1024 + 256)
        
        return self.classifier(x)


def build_fusion_model(num_classes=8):
    return ConvNeXtMetadataFusion(num_classes=num_classes)


