# data.py (THAY THẾ TOÀN BỘ NỘI DUNG CŨ)
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from .config import TRAIN_DIR, VAL_DIR, GT_CSV_PATH, GT_METADATA_CSV, BATCH_SIZE, IMAGE_SIZE

# 7 vị trí phổ biến nhất
VALID_SITES = ['torso', 'lower extremity', 'upper extremity', 'head/neck',
               'posterior torso', 'anterior torso', 'lateral torso']

# ISIC 2019 có 8 classes (loại bỏ UNK)
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

class ISIC2019MetadataDataset(Dataset):
    def __init__(self, root_dir: str, labels_csv: str, metadata_csv: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        df_labels = pd.read_csv(labels_csv)
        df_meta   = pd.read_csv(metadata_csv)

        df_labels = df_labels[df_labels['UNK'] == 0].copy()
        df_labels['target'] = df_labels[CLASSES].idxmax(axis=1).map(
            {c: i for i, c in enumerate(CLASSES)}
        )

        df = df_labels[['image', 'target']].merge(
            df_meta[['image', 'age_approx', 'anatom_site_general', 'sex']],
            on='image',
            how='left'
        )

        self.info_dict = {
            row['image']: {
                'label': int(row['target']),
                'meta':  self._encode(row)
            } for _, row in df.iterrows()
        }

        self.samples = []
        for cls in CLASSES:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path): continue
            for f in os.listdir(cls_path):
                if f.lower().endswith(('.jpg', '.jpeg')):
                    img_id = f.rsplit('.', 1)[0]
                    if img_id in self.info_dict:
                        self.samples.append((os.path.join(cls_path, f), img_id))
    
    def _encode(self, row):
        meta = []
        # Age
        meta.extend([0.0, 1.0] if pd.isna(row['age_approx']) else [row['age_approx']/90.0, 0.0])
        # Sex
        if pd.isna(row['sex']) or row['sex'] not in ['male','female']:
            meta.extend([0.0, 0.0, 1.0])
        else:
            f = 1.0 if row['sex']=='female' else 0.0
            m = 1.0 if row['sex']=='male' else 0.0
            meta.extend([f, m, 0.0])
        # Location
        site = row['anatom_site_general']
        if pd.isna(site) or site not in VALID_SITES:
            meta.extend([0.0]*7 + [1.0])
        else:
            onehot = [1.0 if site == s else 0.0 for s in VALID_SITES]
            meta.extend(onehot + [0.0])
        return np.array(meta, dtype=np.float32)  # (13,)
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, img_id = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        info = self.info_dict[img_id]
        return img, torch.tensor(info['meta']), torch.tensor(info['label'], dtype=torch.long)

def build_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

def create_dataloaders(use_metadata=False):
    if use_metadata:
        transform = build_transforms()
        train_ds = ISIC2019MetadataDataset(TRAIN_DIR, GT_CSV_PATH, GT_METADATA_CSV, transform['train'])
        val_ds = ISIC2019MetadataDataset(VAL_DIR, GT_CSV_PATH, GT_METADATA_CSV, transform['val'])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        return train_ds, val_ds, train_loader, val_loader
    else:
        # Giữ nguyên code cũ của bạn (ImageFolder)
        from torchvision import datasets
        transform = build_transforms()
        train_ds = datasets.ImageFolder(TRAIN_DIR, transform['train'])
        val_ds = datasets.ImageFolder(VAL_DIR, transform['val'])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        return train_ds, val_ds, train_loader, val_loader
