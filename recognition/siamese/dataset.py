# dataset.py
# ---------------------------------------------
# ISIC 2020 (preprocessed, 256x256) dataset utils
# Provides:
#   - ISICTable: read & split & balance
#   - ISICImageDataset: (image, label, idx)
#   - ISICTripletDataset: (anchor, positive, negative, anchor_label)
#   - get_loaders(): build DataLoaders for triplet-training & classifier
#
# Expected config.py symbols (define them there):
#   DATAPATH:       root dir, e.g., "./dataset"
#   CSV_NAME:       "train-metadata.csv"
#   IMG_DIR:        "train-image"
#   SEED:           42
#   TRAIN_FRAC:     0.7
#   VAL_FRAC:       0.1
#   TEST_FRAC:      0.2
#   USE_GROUP_SPLIT: True or False   (use patient_id to avoid leakage)
#   BATCH_TRIPLET:  64               (each sample returns 3 images)
#   BATCH_CLASSIF:  64
#   NUM_WORKERS:    4
#   MEAN:           [0.5, 0.5, 0.5]  (or ImageNet means)
#   STD:            [0.5, 0.5, 0.5]
# ---------------------------------------------

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
    # Config is user-defined; fallback defaults if missing for quick testing
    from config import (
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


#  Core table utils 
class ISICTable:
    """Load metadata, materialize filepaths, split and balance to 1:1."""
    def __init__(self, root: str, csv_name: str = CSV_NAME, image_dir: str = IMG_DIR):
        self.root = Path(root)
        df = pd.read_csv(self.root / csv_name)
        # Materialize filepaths; filter rows whose file doesn't exist
        df["filepath"] = df["isic_id"].astype(str).apply(
            lambda x: str(self.root / image_dir / f"{x}.jpg")
        )
        df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
        # Ensure dtypes
        df["target"] = df["target"].astype(int)
        self.df = df

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

    