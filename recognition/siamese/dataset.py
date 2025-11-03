# ISIC 2020 (preprocessed, 256x256) dataset utils
# Provides:
#   - ISICTable: read & split & balance
#   - ISICImageDataset: (image, label, idx)
#   - ISICTripletDataset: (anchor, positive, negative, anchor_label)
#   - get_loaders(): build DataLoaders for triplet-training & classifier

import os
import random
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    # parameters are stored in params.py
    from params import (
        DATAPATH, CSV_NAME, IMG_DIR, SEED,
        TRAIN_FRAC, VAL_FRAC, TEST_FRAC, USE_GROUP_SPLIT,
        BATCH_TRIPLET, BATCH_CLASSIF, NUM_WORKERS, MEAN, STD
    )
except Exception:
    DATAPATH = "./dataset"
    CSV_NAME = "train-metadata.csv"
    IMG_DIR = "train-image"
    SEED = 42
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.1, 0.2
    USE_GROUP_SPLIT = False
    BATCH_TRIPLET, BATCH_CLASSIF = 64, 64
    NUM_WORKERS = 4
    MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


# ---------- Core table utils ----------
class ISICTable:
    """Load metadata, materialize filepaths, split and balance to 1:1."""
    def __init__(self, root: str, csv_name: str = CSV_NAME, image_dir: str = IMG_DIR):
        self.root = Path(root)

        df = pd.read_csv(self.root / csv_name)
        if df.columns[0].lower().startswith("unnamed"):
            df = df.drop(columns=[df.columns[0]])

        # normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        df = df[["isic_id", "patient_id", "target"]]

        # directly map .jpg filepaths
        img_dir_path = self.root / image_dir / "image"
        df["filepath"] = df["isic_id"].astype(str).apply(
            lambda x: str(img_dir_path / f"{x}.jpg")
        )

        # keep only existing files
        df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
        df["target"] = df["target"].astype(int)

        if len(df) == 0:
            raise RuntimeError(
                f"No .jpg images found in {img_dir_path}. "
                "Check directory level and filename consistency."
            )

        self.df = df
        print(f"[INFO] Loaded {len(df)} samples from {csv_name}")

    def _split_no_group(self, train: float, val: float, seed: int):
        y = self.df["target"].values
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train, random_state=seed)
        train_idx, temp_idx = next(sss.split(self.df, y))
        temp = self.df.iloc[temp_idx]
        y_temp = temp["target"].values
        sss2 = StratifiedShuffleSplit(
            n_splits=1, train_size=val / (1.0 - train), random_state=seed
        )
        val_rel, test_rel = next(sss2.split(temp, y_temp))
        val_idx = temp.index[val_rel]
        test_idx = temp.index[test_rel]
        return (
            self.df.loc[train_idx].reset_index(drop=True),
            self.df.loc[val_idx].reset_index(drop=True),
            self.df.loc[test_idx].reset_index(drop=True),
        )

    def _split_with_group(self, train: float, val: float, seed: int):
        y = self.df["target"].values
        groups = self.df["patient_id"].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
        train_idx, temp_idx = next(gss.split(self.df, y, groups))
        temp = self.df.iloc[temp_idx]
        y_temp = temp["target"].values
        groups_temp = temp["patient_id"].astype(str).values
        gss2 = GroupShuffleSplit(
            n_splits=1, train_size=val / (1.0 - train), random_state=seed
        )
        val_rel, test_rel = next(gss2.split(temp, y_temp, groups_temp))
        val_idx = temp.index[val_rel]
        test_idx = temp.index[test_rel]
        return (
            self.df.loc[train_idx].reset_index(drop=True),
            self.df.loc[val_idx].reset_index(drop=True),
            self.df.loc[test_idx].reset_index(drop=True),
        )

    def split(self, train=TRAIN_FRAC, val=VAL_FRAC, test=TEST_FRAC,
              use_group: bool = USE_GROUP_SPLIT, seed: int = SEED):
        assert abs(train + val + test - 1.0) < 1e-6, "Fractions must sum to 1."
        if use_group and "patient_id" in self.df.columns:
            return self._split_with_group(train, val, seed)
        return self._split_no_group(train, val, seed)
    
    @staticmethod
    def balance_1to1(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
        pos = df[df["target"] == 1]
        neg = df[df["target"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            return df.reset_index(drop=True)
        if len(pos) < len(neg):
            neg = neg.sample(n=len(pos), random_state=seed)
        else:
            pos = pos.sample(n=len(neg), random_state=seed)
        out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)
        return out.reset_index(drop=True)
    

# ---------- Image dataset ----------
class ISICImageDataset(Dataset):
    """Return (image, label, index) for classifier or embedding extraction."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.tfm = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
        label = int(row["target"])
        return img, label, i


# ---------- Triplet dataset ----------
class ISICTripletDataset(Dataset):
    """Return (anchor, positive, negative, anchor_label) for triplet loss."""
    def __init__(self, df: pd.DataFrame, transform=None, seed: int = SEED):
        self.df = df.reset_index(drop=True)
        self.tfm = transform
        self.by_cls = {
            0: self.df[self.df["target"] == 0].index.tolist(),
            1: self.df[self.df["target"] == 1].index.tolist(),
        }
        random.seed(seed)

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, idx: int):
        path = self.df.iloc[idx]["filepath"]
        img = Image.open(path).convert("RGB")
        return self.tfm(img) if self.tfm else img

    def __getitem__(self, i: int):
        anc_row = self.df.iloc[i]
        y = int(anc_row["target"])
        same = [j for j in self.by_cls[y] if j != i]
        pos_idx = random.choice(same) if same else i
        neg_idx = random.choice(self.by_cls[1 - y])
        anc = self._load(i)
        pos = self._load(pos_idx)
        neg = self._load(neg_idx)
        return anc, pos, neg, y
    

# ---------- Transforms ----------
def build_transforms(image_size: int = 256):
    # add color jitter for increased robustness
    color_jitter = T.ColorJitter(
        brightness=0.1,   
        contrast=0.1,     
        saturation=0.05,   
        hue=0.02          
    )

    train_tfm = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        color_jitter,     
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    eval_tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    return train_tfm, eval_tfm


# ---------- Loaders ----------
def get_loaders(
    dataroot: str = DATAPATH,
    balance_each_split: bool = True,
    use_group_split: bool = USE_GROUP_SPLIT,
    batch_triplet: int = BATCH_TRIPLET,
    batch_classif: int = BATCH_CLASSIF,
    num_workers: int = NUM_WORKERS,
):
    """
    Returns:
        dict with keys:
          'triplet_train', 'triplet_val',
          'classif_train', 'classif_val', 'classif_test'
        Each value is a DataLoader.
    """
    table = ISICTable(dataroot, CSV_NAME, IMG_DIR)
    tr_df, va_df, te_df = table.split(
        train=TRAIN_FRAC, val=VAL_FRAC, test=TEST_FRAC,
        use_group=use_group_split, seed=SEED
    )

    if balance_each_split:
        tr_df = ISICTable.balance_1to1(tr_df, seed=SEED)
        va_df = ISICTable.balance_1to1(va_df, seed=SEED)
        te_df = ISICTable.balance_1to1(te_df, seed=SEED)

    tfm_train, tfm_eval = build_transforms(image_size=256)

    # Triplet loaders (train + val)
    ds_triplet = ISICTripletDataset(tr_df, transform=tfm_train, seed=SEED)
    dl_triplet = DataLoader(
        ds_triplet, batch_size=batch_triplet, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    ds_triplet_val = ISICTripletDataset(va_df, transform=tfm_eval, seed=SEED)
    dl_triplet_val = DataLoader(
        ds_triplet_val, batch_size=batch_triplet, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    # Classifier loaders (feature extractor -> classifier)
    ds_tr_cls = ISICImageDataset(tr_df, transform=tfm_train)
    ds_va_cls = ISICImageDataset(va_df, transform=tfm_eval)
    ds_te_cls = ISICImageDataset(te_df, transform=tfm_eval)

    dl_tr_cls = DataLoader(
        ds_tr_cls, batch_size=batch_classif, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    dl_va_cls = DataLoader(
        ds_va_cls, batch_size=batch_classif, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    dl_te_cls = DataLoader(
        ds_te_cls, batch_size=batch_classif, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return {
        "triplet_train": dl_triplet,
        "triplet_val":   dl_triplet_val,   # add validation loader
        "classif_train": dl_tr_cls,
        "classif_val":   dl_va_cls,
        "classif_test":  dl_te_cls,
    }
