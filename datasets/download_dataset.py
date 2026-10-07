"""
datasets/download_dataset.py
────────────────────────────
Downloads NSL-KDD and CICIDS2017 datasets from their official/mirror sources.

Usage:
    python datasets/download_dataset.py --dataset nsl-kdd
    python datasets/download_dataset.py --dataset cicids2017
    python datasets/download_dataset.py --dataset all
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ─── Dataset registry ──────────────────────────────────────────────────────────
DATASETS = {
    "nsl-kdd": {
        "description": "NSL-KDD — improved KDD Cup 99 intrusion detection dataset",
        "files": [
            {
                "name": "KDDTrain+.txt",
                "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
                "target": "data/raw/intrusion/KDDTrain+.txt",
                "sha256": None,  # set after first download
            },
            {
                "name": "KDDTest+.txt",
                "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
                "target": "data/raw/intrusion/KDDTest+.txt",
                "sha256": None,
            },
            {
                "name": "KDDTrain+_20Percent.txt",
                "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt",
                "target": "data/raw/intrusion/KDDTrain+_20Percent.txt",
                "sha256": None,
            },
        ],
    },
    "cicids2017": {
        "description": (
            "CICIDS2017 — Canadian Institute for Cybersecurity Intrusion Detection dataset. "
            "NOTE: Full dataset is ~7GB. We download a curated 500MB subset from CIC mirror."
        ),
        "files": [
            {
                "name": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                "url": (
                    "https://raw.githubusercontent.com/ParthibanMarimuthu/"
                    "CICIDS2017-dataset/main/MachineLearningCVE/"
                    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
                ),
                "target": "data/raw/network_traffic/CICIDS2017_DDoS.csv",
                "sha256": None,
            },
            {
                "name": "Wednesday-workingHours.pcap_ISCX.csv",
                "url": (
                    "https://raw.githubusercontent.com/ParthibanMarimuthu/"
                    "CICIDS2017-dataset/main/MachineLearningCVE/"
                    "Wednesday-workingHours.pcap_ISCX.csv"
                ),
                "target": "data/raw/network_traffic/CICIDS2017_Wednesday.csv",
                "sha256": None,
            },
        ],
    },
}

ROOT = Path(__file__).resolve().parent.parent  # project root


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return

    print(f"  ↓ Downloading: {dest.name}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"  ✓ Saved → {dest}")
    except requests.RequestException as exc:
        print(f"  ✗ Failed to download {url}: {exc}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        raise


def download_dataset(name: str) -> None:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")

    meta = DATASETS[name]
    print(f"\n{'─'*60}")
    print(f"Dataset  : {name}")
    print(f"Info     : {meta['description']}")
    print(f"Files    : {len(meta['files'])}")
    print(f"{'─'*60}")

    for file_info in meta["files"]:
        dest = ROOT / file_info["target"]
        download_file(file_info["url"], dest)

        if file_info.get("sha256"):
            digest = sha256_file(dest)
            assert digest == file_info["sha256"], (
                f"Checksum mismatch for {dest.name}! "
                f"Expected {file_info['sha256']}, got {digest}"
            )
            print(f"  ✓ Checksum verified.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download network intrusion detection datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["nsl-kdd", "cicids2017", "all"],
        default="nsl-kdd",
        help="Which dataset to download (default: nsl-kdd)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output base directory (default: project data/ directory)",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        download_dataset(name)

    print("\n✅ All downloads complete.")


if __name__ == "__main__":
    main()
