"""Data loading for FA-ICGA-FM.

Supports:
  - APTOS 2023 FA+ICGA (Alibaba Tianchi, 58K images, 24 conditions)
  - Synthetic FA-like images for smoke testing when real data unavailable
  - RECOVERY-FA19 for vessel segmentation (8 UWF-FA images)
  - IDRiD FA for lesion evaluation
"""
import os
import glob
import json
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IMG_SIZE = 224
APTOS_CLASSES = 24
FA_MEAN = [0.485, 0.456, 0.406]
FA_STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(FA_MEAN, FA_STD),
])

EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(FA_MEAN, FA_STD),
])

MAE_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(FA_MEAN, FA_STD),
])


class SyntheticFADataset(Dataset):
    """Synthetic FA-like images for smoke testing when real data unavailable."""

    def __init__(self, n=500, n_classes=APTOS_CLASSES, transform=None, seed=42,
                 modal="fa", phase_labels=False):
        self.n = n
        self.n_classes = n_classes
        self.transform = transform or TRAIN_TF
        self.seed = seed
        self.modal = modal
        self.phase_labels = phase_labels
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, n_classes, size=n)
        self.phases = rng.integers(0, 4, size=n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        label = int(self.labels[idx])
        if self.modal == "icga":
            base = rng.uniform(0.02, 0.08, (IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
        else:
            base = rng.uniform(0.05, 0.15, (IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
        cx, cy = rng.integers(80, 144, size=2)
        for r in range(5, 50, 5):
            pts = int(2 * np.pi * r * 3)
            for i in range(pts):
                angle = 2 * np.pi * i / pts
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
                    bright = 0.6 + 0.4 * (label / self.n_classes)
                    base[y, x] = np.clip([bright, bright * 0.9, bright * 0.7], 0, 1)
        img = Image.fromarray((base * 255).astype(np.uint8))
        img = self.transform(img)
        if self.phase_labels:
            return img, label, int(self.phases[idx])
        return img, label


class APTOSDataset(Dataset):
    """APTOS 2023 FA+ICGA dataset (Alibaba Tianchi, 58K images, 24 conditions).

    Directory structure expected:
        root/
          images/    (FA images named <case_id>_FA.jpg)
          icga/      (ICGA images named <case_id>_ICGA.jpg, optional)
          labels.csv (columns: image_id, label, phase, modality)
    """

    def __init__(self, root, split="train", transform=None, modality="fa",
                 require_icga_pair=False, phase_labels=False):
        self.root = Path(root)
        self.transform = transform or (TRAIN_TF if split == "train" else EVAL_TF)
        self.modality = modality
        self.require_icga_pair = require_icga_pair
        self.phase_labels = phase_labels

        label_path = self.root / "labels.csv"
        if not label_path.exists():
            raise FileNotFoundError(f"Labels CSV not found at {label_path}")

        import pandas as pd
        df = pd.read_csv(label_path)
        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        else:
            total = len(df)
            train_end = int(total * 0.7)
            val_end = int(total * 0.85)
            if split == "train":
                df = df.iloc[:train_end].reset_index(drop=True)
            elif split == "val":
                df = df.iloc[train_end:val_end].reset_index(drop=True)
            else:
                df = df.iloc[val_end:].reset_index(drop=True)

        if "modality" in df.columns:
            df = df[df["modality"].str.lower() == modality].reset_index(drop=True)

        self.records = df.to_dict("records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img_id = str(r["image_id"])
        img_dir = self.root / ("icga" if self.modality == "icga" else "images")
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{img_id}.png"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = int(r.get("label", 0))
        if self.phase_labels:
            phase = int(r.get("phase", 0))
            return img, label, phase
        return img, label


class CrossModalPairDataset(Dataset):
    """FA+ICGA paired dataset for cross-modal contrastive training (item C1).

    Returns (fa_img, icga_img, label) triplets for same-patient positive pairs.
    """

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform or TRAIN_TF

        import pandas as pd
        label_path = self.root / "labels.csv"
        df = pd.read_csv(label_path)
        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        fa_ids = set(df[df["modality"].str.lower() == "fa"]["image_id"].astype(str))
        icga_ids = set(df[df["modality"].str.lower() == "icga"]["image_id"].astype(str))
        paired = fa_ids & icga_ids
        self.pairs = [p for p in paired]
        self.label_map = {str(r["image_id"]): int(r.get("label", 0))
                          for _, r in df.iterrows()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pid = self.pairs[idx]
        fa_img = Image.open(self.root / "images" / f"{pid}.jpg").convert("RGB")
        icga_img = Image.open(self.root / "icga" / f"{pid}.jpg").convert("RGB")
        fa_img = self.transform(fa_img)
        icga_img = self.transform(icga_img)
        label = self.label_map.get(pid, 0)
        return fa_img, icga_img, label


class SyntheticCrossModalDataset(Dataset):
    """Synthetic FA+ICGA pairs for smoke testing item C1."""

    def __init__(self, n=200, n_classes=APTOS_CLASSES, transform=None, seed=7):
        self.n = n
        self.n_classes = n_classes
        self.transform = transform or TRAIN_TF
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, n_classes, size=n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        fa = SyntheticFADataset(n=1, n_classes=self.n_classes,
                                transform=self.transform, seed=idx)[0][0]
        icga = SyntheticFADataset(n=1, n_classes=self.n_classes, modal="icga",
                                  transform=self.transform, seed=idx + 10000)[0][0]
        return fa, icga, label


def make_loaders(data_root, batch=32, workers=4, smoke=False, require_real=False):
    """Build train/val/test DataLoaders.

    Returns dict: {train, val, test, cross_modal_train, n_classes, is_synthetic}.
    """
    has_real = data_root and (Path(data_root) / "labels.csv").exists()
    if require_real and not has_real:
        raise RuntimeError(f"Real data not found at {data_root}")

    if not has_real:
        n = 300 if smoke else 2000
        n_classes = APTOS_CLASSES
        train_ds = SyntheticFADataset(n=n, transform=TRAIN_TF)
        val_ds   = SyntheticFADataset(n=n // 5, transform=EVAL_TF, seed=1)
        test_ds  = SyntheticFADataset(n=n // 5, transform=EVAL_TF, seed=2)
        cm_ds    = SyntheticCrossModalDataset(n=n // 4)
        is_syn   = True
    else:
        n_classes = APTOS_CLASSES
        train_ds = APTOSDataset(data_root, "train", TRAIN_TF)
        val_ds   = APTOSDataset(data_root, "val",   EVAL_TF)
        test_ds  = APTOSDataset(data_root, "test",  EVAL_TF)
        try:
            cm_ds = CrossModalPairDataset(data_root, "train", TRAIN_TF)
        except Exception:
            cm_ds = None
        is_syn = False

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                          num_workers=workers, pin_memory=True, drop_last=True)

    return {
        "train": loader(train_ds, shuffle=True),
        "val":   loader(val_ds),
        "test":  loader(test_ds),
        "cross_modal": loader(cm_ds) if cm_ds else None,
        "n_classes": n_classes,
        "is_synthetic": is_syn,
        "n_train": len(train_ds),
    }
