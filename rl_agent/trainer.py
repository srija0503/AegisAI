"""
rl_agent/trainer.py
───────────────────
Full training orchestrator for the RL Firewall Agent.
Executes multi-episode Double Dueling DQN training with Prioritized Experience Replay,
periodic validation evaluations, early stopping, and checkpoint generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from datasets.feature_extraction import df_to_tensors
from rl_agent.agent import DQNAgent
from rl_agent.checkpoint import CheckpointManager
from rl_agent.environment import FirewallEnv
from rl_agent.evaluator import EvaluationMetrics, Evaluator
from rl_agent.reward import RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    total_episodes: int = 500
    max_steps_per_episode: int = 500
    eval_interval: int = 50
    checkpoint_interval: int = 50
    checkpoint_dir: str = "./rl_agent/models/checkpoints/"
    log_dir: str = "./rl_agent/logs/"
    early_stopping_patience: int = 100
    min_replay_size: int = 500
    batch_size: int = 64
    learning_rate: float = 0.0001
    gamma: float = 0.99
    target_update_freq: int = 200
    use_per: bool = True


class RLTrainer:
    """
    Manages the training cycle of the DQNAgent on NSL-KDD / network flow data.
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        train_path: str = "./data/processed/rl_training/nslkdd_train.parquet",
        val_path: str = "./data/processed/rl_training/nslkdd_val.parquet",
    ) -> None:
        self.config = config or TrainerConfig()
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
        self.evaluator = Evaluator()

        # Load Training & Validation data
        self.train_X, self.train_y = self._load_data(train_path)
        self.val_X, self.val_y = self._load_data(val_path)

        # Environment & Agent
        self.env = FirewallEnv(
            X=self.train_X,
            y=self.train_y,
            max_steps_per_episode=self.config.max_steps_per_episode,
        )

        self.agent = DQNAgent(
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            target_update_freq=self.config.target_update_freq,
            use_per=self.config.use_per,
            replay_capacity=50000,
        )

        self.best_f1 = 0.0
        self.history: list[dict[str, Any]] = []

    def _load_data(self, path_str: str) -> tuple[np.ndarray, np.ndarray]:
        path = Path(path_str)
        if path.exists():
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            return df_to_tensors(df)
        # Synthetic fallback if data not available
        logger.warning(f"Data file {path} not found. Using synthetic dataset.")
        from rl_agent.state import STATE_DIM
        X = np.random.randn(2000, STATE_DIM).astype(np.float32)
        y = np.random.randint(0, 2, size=2000, dtype=np.int32)
        return X, y

    def warm_up(self) -> None:
        """Pre-populate experience replay with initial transitions."""
        obs, _ = self.env.reset()
        for _ in range(self.config.min_replay_size):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.agent.memory.push(obs, action, reward, next_obs, done)
            if done:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

    def train(self, total_episodes: Optional[int] = None) -> dict[str, Any]:
        """Execute full training loop across episodes."""
        episodes = total_episodes or self.config.total_episodes
        self.warm_up()

        patience_counter = 0

        for episode in range(1, episodes + 1):
            obs, _ = self.env.reset()
            episode_losses = []
            done = False

            while not done:
                action = self.agent.act(obs, explore=True)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                loss = self.agent.step(obs, action, reward, next_obs, done)
                if loss is not None:
                    episode_losses.append(loss)

                obs = next_obs

            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            ep_metrics = {
                "episode": episode,
                "reward": self.env.episode_reward,
                "accuracy": info["accuracy"],
                "precision": info["precision"],
                "recall": info["recall"],
                "loss": avg_loss,
                "epsilon": self.agent.policy.epsilon,
            }
            self.history.append(ep_metrics)

            # Validation evaluation
            if episode % self.config.eval_interval == 0:
                val_eval = self.evaluator.evaluate(self.agent, self.val_X, self.val_y)
                val_f1 = val_eval.f1_score

                logger.info(
                    f"Ep {episode:04d}/{episodes} | Rew: {self.env.episode_reward:.1f} | "
                    f"Val Acc: {val_eval.accuracy:.1%} | Val F1: {val_f1:.3f} | "
                    f"Eps: {self.agent.policy.epsilon:.3f}"
                )

                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    patience_counter = 0
                    self.checkpoint_manager.save(
                        model_or_agent=self.agent,
                        filename="best_model.pt",
                        episode=episode,
                        metrics=val_eval.to_dict(),
                    )
                else:
                    patience_counter += self.config.eval_interval
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered at episode {episode}")
                        break

            # Scheduled checkpoint
            if episode % self.config.checkpoint_interval == 0:
                self.checkpoint_manager.save(
                    model_or_agent=self.agent,
                    filename=f"checkpoint_ep_{episode:04d}.pt",
                    episode=episode,
                )

        # Final save of latest model
        self.checkpoint_manager.save(
            model_or_agent=self.agent,
            filename="latest_model.pt",
            episode=episodes,
        )

        return {
            "total_episodes_completed": len(self.history),
            "best_f1": self.best_f1,
            "final_epsilon": self.agent.policy.epsilon,
        }
