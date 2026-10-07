"""
datasets/preprocess.py
──────────────────────
Cleans and normalizes raw NSL-KDD / CICIDS2017 CSV files.

Steps:
  1. Load raw CSV / TXT files
  2. Rename columns to a unified schema
  3. Encode categorical features (protocol_type, service, flag)
  4. Remove duplicates and null rows
  5. Normalize numeric features (MinMax by default)
  6. Map multi-class labels → binary (normal=0, attack=1) + keep original label
  7. Save processed Parquet to data/processed/traffic_features/

Usage:
    python datasets/preprocess.py --dataset nsl-kdd
    python datasets/preprocess.py --dataset cicids2017
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ─── NSL-KDD column schema ────────────────────────────────────────────────────
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]

# Attack categories in NSL-KDD → grouped into types
ATTACK_GROUPS = {
    "normal": "NORMAL",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpattack": "R2L", "snmpguess": "R2L", "worm": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}


# ─── NSL-KDD preprocessor ────────────────────────────────────────────────────

def load_nsl_kdd(split: str = "train") -> pd.DataFrame:
    if split == "train":
        path = ROOT / "data/raw/intrusion/KDDTrain+.txt"
    else:
        path = ROOT / "data/raw/intrusion/KDDTest+.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"NSL-KDD file not found: {path}. "
            "Run: python datasets/download_dataset.py --dataset nsl-kdd"
        )

    log.info(f"Loading NSL-KDD {split} split from {path}")
    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    log.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def preprocess_nsl_kdd(df: pd.DataFrame, scaler: MinMaxScaler | None = None) -> tuple[pd.DataFrame, MinMaxScaler]:
    df = df.copy()

    # 1. Drop difficulty column
    if "difficulty" in df.columns:
        df.drop(columns=["difficulty"], inplace=True)

    # 2. Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    log.info(f"  Removed {before - len(df):,} duplicate rows")

    # 3. Map label → binary + attack category
    df["label_raw"] = df["label"].str.lower().str.strip()
    df["attack_category"] = df["label_raw"].map(ATTACK_GROUPS).fillna("Unknown")
    df["label"] = (df["label_raw"] != "normal").astype(int)  # 0=normal, 1=attack

    # 4. Encode categorical features
    encoders: dict[str, LabelEncoder] = {}
    for col in NSL_KDD_CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # 5. Separate feature columns
    feature_cols = [c for c in df.columns if c not in ("label", "label_raw", "attack_category")]

    # 6. Fill NaN in features
    df[feature_cols] = df[feature_cols].fillna(0)

    # 7. Normalize
    if scaler is None:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols].astype(float))
    else:
        df[feature_cols] = scaler.transform(df[feature_cols].astype(float))

    log.info(f"  Final shape: {df.shape}")
    log.info(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
    log.info(f"  Attack categories:\n{df['attack_category'].value_counts().to_string()}")

    return df, scaler


# ─── CICIDS2017 preprocessor ─────────────────────────────────────────────────

def load_cicids2017(csv_file: str) -> pd.DataFrame:
    path = ROOT / "data/raw/network_traffic" / csv_file
    if not path.exists():
        raise FileNotFoundError(f"CICIDS2017 file not found: {path}")

    log.info(f"Loading CICIDS2017 from {path}")
    df = pd.read_csv(path, low_memory=False)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    log.info(f"  Loaded {len(df):,} rows")
    return df


def preprocess_cicids2017(df: pd.DataFrame, scaler: MinMaxScaler | None = None) -> tuple[pd.DataFrame, MinMaxScaler]:
    df = df.copy()

    # 1. Drop identifier columns if present
    id_cols = ["Flow ID", " Source IP", " Source Port", " Destination IP",
                " Destination Port", " Timestamp"]
    df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True, errors="ignore")

    # 2. Normalize label column name
    label_col = None
    for c in df.columns:
        if "label" in c.lower():
            label_col = c
            break
    if label_col is None:
        raise ValueError("No label column found in CICIDS2017 dataset")
    df.rename(columns={label_col: "label_raw"}, inplace=True)

    # 3. Binary label
    df["label"] = (df["label_raw"].str.upper().str.strip() != "BENIGN").astype(int)
    df["attack_category"] = df["label_raw"].str.strip()

    # 4. Drop inf / NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    log.info(f"  Dropped {before - len(df):,} rows with inf/NaN")

    # 5. Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    log.info(f"  Removed {before - len(df):,} duplicates")

    feature_cols = [c for c in df.columns if c not in ("label", "label_raw", "attack_category")]

    # 6. Normalize
    if scaler is None:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols].astype(float))
    else:
        df[feature_cols] = scaler.transform(df[feature_cols].astype(float))

    log.info(f"  Final shape: {df.shape}")
    log.info(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    return df, scaler


# ─── Save helper ──────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    size_mb = out_path.stat().st_size / (1024 ** 2)
    log.info(f"  Saved → {out_path}  ({size_mb:.1f} MB)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw intrusion detection datasets.")
    parser.add_argument(
        "--dataset",
        choices=["nsl-kdd", "cicids2017", "all"],
        default="nsl-kdd",
    )
    args = parser.parse_args()

    out_dir = ROOT / "data/processed/traffic_features"

    if args.dataset in ("nsl-kdd", "all"):
        log.info("═" * 50)
        log.info("Preprocessing NSL-KDD (train) …")
        df_train, scaler = preprocess_nsl_kdd(load_nsl_kdd("train"))
        save_parquet(df_train, out_dir / "nslkdd_train.parquet")

        log.info("Preprocessing NSL-KDD (test) …")
        df_test, _ = preprocess_nsl_kdd(load_nsl_kdd("test"), scaler=scaler)
        save_parquet(df_test, out_dir / "nslkdd_test.parquet")

    if args.dataset in ("cicids2017", "all"):
        log.info("═" * 50)
        for csv_file in ["CICIDS2017_DDoS.csv", "CICIDS2017_Wednesday.csv"]:
            try:
                log.info(f"Preprocessing CICIDS2017: {csv_file} …")
                df, scaler_c = preprocess_cicids2017(load_cicids2017(csv_file))
                out_name = csv_file.replace(".csv", ".parquet")
                save_parquet(df, out_dir / out_name)
            except FileNotFoundError as e:
                log.warning(str(e))

    log.info("✅ Preprocessing complete.")


if __name__ == "__main__":
    main()
