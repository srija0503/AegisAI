"""
datasets/dataset_loader.py
───────────────────────────
PyTorch Dataset and DataLoader wrapper around the processed Parquet files.
Used by rl_agent/trainer.py for offline RL training.

Also provides a StreamingLoader for simulating online traffic arrival
(used in simulation/ module).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.feature_extraction import FEATURE_NAMES, df_to_tensors

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RL_TRAIN_DIR = ROOT / "data/processed/rl_training"


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class NetworkFlowDataset(Dataset):
    """
    PyTorch Dataset over a processed network flow Parquet file.

    Each item: (features: Tensor[41], label: Tensor[])
    """

    def __init__(
        self,
        parquet_path: Path | str,
        feature_names: list[str] = FEATURE_NAMES,
        label_col: str = "label",
    ) -> None:
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {parquet_path}\n"
                "Run: python datasets/preprocess.py && python datasets/train_test_split.py"
            )

        df = pd.read_parquet(parquet_path)
        log.info(f"Loaded dataset: {parquet_path.name} ({len(df):,} samples)")

        X, y = df_to_tensors(df)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.attack_categories: list[str] = (
            df["attack_category"].tolist() if "attack_category" in df.columns
            else ["unknown"] * len(df)
        )

        self.n_features = self.X.shape[1]
        self.n_classes  = int(self.y.max().item()) + 1
        log.info(
            f"  Features: {self.n_features}, Classes: {self.n_classes}, "
            f"Attack rate: {y.mean():.2%}"
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def get_class_weights(self) -> torch.Tensor:
        """Inverse frequency class weights for weighted loss."""
        counts = torch.bincount(self.y, minlength=2).float()
        weights = 1.0 / (counts + 1e-8)
        return weights / weights.sum()


# ─── DataLoader factory ──────────────────────────────────────────────────────

def get_dataloaders(
    dataset_prefix: str = "nslkdd",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """
    Returns dict with keys 'train', 'val', 'test' → DataLoaders.

    Args:
        dataset_prefix: 'nslkdd' or 'cicids2017'
        batch_size:     mini-batch size
        num_workers:    DataLoader worker processes (0 = main thread)
        pin_memory:     pin memory to GPU (set True if CUDA available)
    """
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        path = RL_TRAIN_DIR / f"{dataset_prefix}_{split}.parquet"
        dataset = NetworkFlowDataset(path)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return loaders


# ─── Streaming loader (for simulation) ───────────────────────────────────────

class StreamingFlowLoader:
    """
    Yields one flow at a time from the dataset, simulating real-time arrival.
    Used by simulation/node_simulator.py.
    """

    def __init__(self, parquet_path: Path | str, shuffle: bool = True, seed: int = 42) -> None:
        df = pd.read_parquet(parquet_path)
        if shuffle:
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        X, y = df_to_tensors(df)
        self._X = X
        self._y = y
        self._categories: list[str] = (
            df["attack_category"].tolist() if "attack_category" in df.columns
            else ["unknown"] * len(df)
        )
        self._index = 0
        self._total = len(df)

    def __len__(self) -> int:
        return self._total

    def __iter__(self) -> Iterator[dict]:
        self._index = 0
        return self

    def __next__(self) -> dict:
        if self._index >= self._total:
            raise StopIteration
        item = {
            "features": self._X[self._index],
            "label": int(self._y[self._index]),
            "attack_category": self._categories[self._index],
            "index": self._index,
        }
        self._index += 1
        return item

    def reset(self) -> None:
        self._index = 0

    def has_next(self) -> bool:
        return self._index < self._total


# ─── Quick dataset info ───────────────────────────────────────────────────────

def dataset_info(dataset_prefix: str = "nslkdd") -> None:
    """Print summary statistics for all splits of a dataset."""
    print(f"\n{'═'*55}")
    print(f"  Dataset: {dataset_prefix}")
    print(f"{'═'*55}")
    for split in ("train", "val", "test"):
        path = RL_TRAIN_DIR / f"{dataset_prefix}_{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            attack_pct = df["label"].mean() * 100 if "label" in df.columns else -1
            print(
                f"  {split:5s}: {len(df):7,} samples | "
                f"attack: {attack_pct:.1f}% | "
                f"cols: {len(df.columns)}"
            )
        else:
            print(f"  {split:5s}: ✗ not found at {path}")
    print()


if __name__ == "__main__":
    dataset_info("nslkdd")
