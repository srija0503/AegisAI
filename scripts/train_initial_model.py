"""
scripts/train_initial_model.py
───────────────────────────────
Trains the initial DQN RL agent on the NSL-KDD dataset.
Run AFTER: make data && make preprocess && make split

Usage:
    python scripts/train_initial_model.py
    python scripts/train_initial_model.py --episodes 100 --dataset nsl-kdd
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl_agent.trainer import RLTrainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train initial DQN RL agent.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--dataset", choices=["nsl-kdd", "cicids2017"], default="nsl-kdd")
    parser.add_argument("--config", type=str, default="config/rl_config.yaml")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    args = parser.parse_args()

    import torch
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device_str}")

    prefix = "nslkdd" if args.dataset == "nsl-kdd" else "cicids2017"
    train_path = f"./data/processed/rl_training/{prefix}_train.parquet"
    val_path = f"./data/processed/rl_training/{prefix}_val.parquet"

    config = TrainerConfig(
        total_episodes=args.episodes,
        max_steps_per_episode=200,
        eval_interval=10,
        checkpoint_interval=25,
        min_replay_size=200,
    )

    trainer = RLTrainer(
        config=config,
        train_path=train_path,
        val_path=val_path,
    )

    print(f"Starting training on {args.dataset} for {args.episodes} episodes...")
    results = trainer.train()
    print("Training finished! Results:", results)


if __name__ == "__main__":
    main()
