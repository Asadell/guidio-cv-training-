"""
Dataset loader untuk dataset_master_seg/ hasil script 04.

Struktur:
  dataset_master_seg/images/{train,valid,test}/*.jpg
  dataset_master_seg/masks/{train,valid,test}/*.png  (nilai piksel 0/1/2)
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SidewalkSegDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 512, augment: bool = False):
        self.img_dir  = Path(root) / "images" / split
        self.mask_dir = Path(root) / "masks"  / split
        self.img_size = img_size
        self.augment  = augment and (split == "train")
        self.samples  = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        if not self.samples:
            raise FileNotFoundError(
                f"Tidak ada gambar di {self.img_dir}\n"
                f"Jalankan scripts/04_convert_coco_segmentation_to_mask.py dulu."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        img  = Image.open(self.img_dir  / f"{stem}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{stem}.png")

        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment and np.random.rand() < 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img_np = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_t  = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()
        return img_t, mask_t
