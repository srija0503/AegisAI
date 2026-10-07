"""
datasets/train_test_split.py
─────────────────────────────
Stratified train/validation/test split for RL training data.

Reads processed Parquet files from data/processed/traffic_features/
Writes split Parquets to data/processed/rl_training/

Usage:
    python datasets/train_test_split.py --dataset nsl-kdd
    python datasets/train_test_split.py --dataset nsl-kdd --val-size 0.1 --test-size 0.1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data/processed/traffic_features"
RL_TRAIN_DIR = ROOT / "data/processed/rl_training"


def load_processed(name: str) -> pd.DataFrame:
    path = FEATURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run preprocess.py first."
        )
    df = pd.read_parquet(path)
    log.info(f"Loaded {name}: {len(df):,} rows")
    return df


def balance_classes(df: pd.DataFrame, strategy: str = "undersample", label_col: str = "label") -> pd.DataFrame:
    """Balance attack vs normal classes.

    strategy: 'undersample' | 'oversample' | 'none'
    """
    if strategy == "none":
        return df

    counts = df[label_col].value_counts()
    log.info(f"  Before balancing: {counts.to_dict()}")

    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    minority_df = df[df[label_col] == minority_class]
    majority_df = df[df[label_col] == majority_class]

    if strategy == "undersample":
        majority_resampled = resample(
            majority_df, n_samples=len(minority_df), random_state=42
        )
        df_balanced = pd.concat([majority_resampled, minority_df])
    elif strategy == "oversample":
        minority_resampled = resample(
            minority_df, n_samples=len(majority_df), random_state=42, replace=True
        )
        df_balanced = pd.concat([majority_df, minority_resampled])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    log.info(f"  After balancing : {df_balanced[label_col].value_counts().to_dict()}")
    return df_balanced


def split_and_save(
    df: pd.DataFrame,
    out_prefix: str,
    val_size: float = 0.10,
    test_size: float = 0.10,
    balance: str = "undersample",
) -> None:
    """Stratified split → save train/val/test Parquets."""

    label_col = "label"
    RL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Balance training portion first, keep val/test natural distribution
    test_frac = test_size
    train_val_df, test_df = train_test_split(
        df, test_size=test_frac, stratify=df[label_col], random_state=42
    )
    val_frac_of_remaining = val_size / (1.0 - test_frac)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_frac_of_remaining,
        stratify=train_val_df[label_col],
        random_state=42,
    )

    # Balance only the training set
    train_df = balance_classes(train_df, strategy=balance)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    for split_name, split_df in splits.items():
        out_path = RL_TRAIN_DIR / f"{out_prefix}_{split_name}.parquet"
        split_df.to_parquet(out_path, index=False, compression="snappy")
        log.info(
            f"  [{split_name}] {len(split_df):,} rows → {out_path.name}  "
            f"(attack={split_df[label_col].sum():,}, "
            f"normal={(split_df[label_col]==0).sum():,})"
        )

    # Print class report
    log.info("\n── Class Balance Report ─────────────────────────────")
    for split_name, split_df in splits.items():
        pct_attack = split_df[label_col].mean() * 100
        log.info(f"  {split_name:5s}: {len(split_df):7,} rows  |  attack: {pct_attack:.1f}%")

    if "attack_category" in df.columns:
        log.info("\n── Attack Category Distribution (train) ────────────")
        log.info(train_df["attack_category"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified train/val/test split for RL training data.")
    parser.add_argument("--dataset", choices=["nsl-kdd", "cicids2017"], default="nsl-kdd")
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument(
        "--balance",
        choices=["undersample", "oversample", "none"],
        default="undersample",
        help="Class balancing strategy for training set",
    )
    args = parser.parse_args()

    if args.dataset == "nsl-kdd":
        # NSL-KDD already has canonical train/test; we combine and re-split for val
        train_df = load_processed("nslkdd_train.parquet")
        test_df  = load_processed("nslkdd_test.parquet")
        df = pd.concat([train_df, test_df], ignore_index=True)
        prefix = "nslkdd"
    else:
        dfs = []
        for fname in ["CICIDS2017_DDoS.parquet", "CICIDS2017_Wednesday.parquet"]:
            try:
                dfs.append(load_processed(fname))
            except FileNotFoundError:
                log.warning(f"Skipping missing file: {fname}")
        if not dfs:
            raise RuntimeError("No CICIDS2017 processed files found.")
        df = pd.concat(dfs, ignore_index=True)
        prefix = "cicids2017"

    split_and_save(df, prefix, args.val_size, args.test_size, args.balance)
    log.info("✅ Split complete.")


if __name__ == "__main__":
    main()
