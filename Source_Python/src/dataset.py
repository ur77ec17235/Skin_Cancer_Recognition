import os
from typing import Optional, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DermDataset(Dataset):
    """Dataset for dermoscopy images plus patient metadata.

    Expects a dataframe with columns: isic_id (filename), label (binary), and optional metadata columns.
    """

    def __init__(self, df: pd.DataFrame, image_dir: str, meta_cols: Optional[List[str]] = None,
                 transforms: Optional[A.Compose] = None, preload: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = image_dir
        self.meta_cols = meta_cols or []
        self.transforms = transforms
        self.preload = preload

        if 'isic_id' not in self.df.columns:
            raise ValueError("DataFrame must have 'isic_id' column with image filenames")

        # If filenames stored without .jpg, append if necessary
        def _norm_name(x):
            if isinstance(x, str) and not x.lower().endswith(('.jpg', '.jpeg', '.png')):
                return x + '.jpg'
            return x

        self.df['isic_id'] = self.df['isic_id'].apply(_norm_name)

        self.images = None
        if self.preload:
            self._preload_images()

    def _preload_images(self):
        self.images = []
        for fname in self.df['isic_id']:
            path = os.path.join(self.image_dir, fname)
            img = cv2.imread(path)
            if img is None:
                # create a black placeholder
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.preload and self.images is not None:
            img = self.images[idx]
        else:
            path = os.path.join(self.image_dir, row['isic_id'])
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented['image']
        else:
            img = ToTensorV2()(image=img)['image']

        meta = None
        if len(self.meta_cols) > 0:
            values = []
            for c in self.meta_cols:
                v = row.get(c, 0)
                if pd.isna(v):
                    v = 0
                values.append(float(v))
            meta = torch.tensor(values, dtype=torch.float32)

        label = None
        if 'benign_malignant' in self.df.columns or 'label' in self.df.columns:
            # support both naming conventions
            if 'label' in self.df.columns:
                lab = row['label']
            else:
                lab = row['benign_malignant']
            # convert to binary 0/1
            if isinstance(lab, str):
                lab = 1.0 if lab.lower() in ('malignant', 'malign', '1', 'true', 'yes') else 0.0
            elif pd.isna(lab):
                lab = 0.0
            label = torch.tensor(float(lab), dtype=torch.float32)

        return img, meta, label
