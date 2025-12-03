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
class ConvNeXtCrossAttention(nn.Module):
    def __init__(self, 
                 num_meta_features=13,
                 embed_dim=1024,        # dùng cùng chiều với ConvNeXt
                 num_heads=8,
                 num_classes=8,
                 dropout=0.3):

        super().__init__()

        # 1. Backbone ConvNeXt
        convnext = models.convnext_base(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(convnext.children())[:-1])   # (B, 1024, 7, 7)

        # 2. Metadata projection → 1024-dim
        self.meta_branch = nn.Sequential(
            nn.Linear(num_meta_features, 128),
                nn.LayerNorm(128), nn.GELU(),

                nn.Linear(128, 256),
                nn.LayerNorm(256), nn.GELU(),

                nn.Linear(256, 512),
                nn.LayerNorm(512), nn.GELU(),

                nn.Linear(512, embed_dim),   # Up to 1024
                nn.LayerNorm(embed_dim)
        )
        # 3. Positional embedding cho 49 patches
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim))

        # 4. Cross-attention (metadata → image patches)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 5. Feed-forward network (Transformer style)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 6. Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, meta):
        B = x.size(0)

        # ── 1. Image features → patches ────────────────────────────────
        img = self.backbone(x)                  # (B, 1024, 7, 7)
        img = img.flatten(2).transpose(1, 2)    # (B, 49, 1024)

        # Add positional embedding
        img = img + self.pos_embed

        # ── 2. Metadata token ─────────────────────────────────────────
        meta_token = self.meta_branch(meta).unsqueeze(1)  # (B, 1, 1024)

        # ── 3. Cross Attention ────────────────────────────────────────
        # Query = meta, Key/Value = patches
        attn_out, _ = self.cross_attn(meta_token, img, img)

        # ── Residual + Norm ───────────────────────────────────────────
        x1 = self.norm1(meta_token + attn_out)

        # ── 4. Feed-forward block ─────────────────────────────────────
        x2 = self.norm2(x1 + self.ffn(x1))

        fused = x2.squeeze(1)  # (B, 1024)

        # ── 5. Classifier ─────────────────────────────────────────────
        return self.classifier(fused)
def build_cross_attention_model(num_classes=8):
    return ConvNeXtCrossAttention(num_classes=num_classes)
