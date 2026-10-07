"""
rl_agent/checkpoint.py
──────────────────────
Model checkpointing and Federated Learning export/import interface.
Provides the exact dictionary format contract required by Member 3 (Federated Learning):
{"model_state_dict": OrderedDict, "round": int, "metrics": {"accuracy": float, "f1": float}, "node_id": str}
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from rl_agent.agent import DQNAgent


class CheckpointManager:
    """
    Manages saving, loading, and federated learning serialization of RL models.
    """

    def __init__(self, checkpoint_dir: str = "./rl_agent/models/checkpoints/") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model_or_agent: Union[DQNAgent, nn.Module],
        filename: str = "best_model.pt",
        episode: int = 0,
        metrics: Optional[dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Path:
        """Save full training checkpoint with metadata."""
        model = model_or_agent.q_net if isinstance(model_or_agent, DQNAgent) else model_or_agent
        opt = optimizer or (model_or_agent.optimizer if isinstance(model_or_agent, DQNAgent) else None)

        payload = {
            "model_state_dict": model.state_dict(),
            "episode": episode,
            "metrics": metrics or {},
            "optimizer_state_dict": opt.state_dict() if opt else None,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(payload, save_path)
        return save_path

    def load(
        self,
        model_or_agent: Union[DQNAgent, nn.Module],
        checkpoint_path: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Load checkpoint weights into model/agent."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        data = torch.load(path, map_location=device)
        model = model_or_agent.q_net if isinstance(model_or_agent, DQNAgent) else model_or_agent

        if isinstance(data, dict) and "model_state_dict" in data:
            model.load_state_dict(data["model_state_dict"])
            if isinstance(model_or_agent, DQNAgent) and "optimizer_state_dict" in data and data["optimizer_state_dict"]:
                try:
                    model_or_agent.optimizer.load_state_dict(data["optimizer_state_dict"])
                except Exception:
                    pass
                # Sync target net as well
                model_or_agent.target_net.load_state_dict(model.state_dict())
            return data
        elif isinstance(data, dict):
            # Assume it's a raw state dict
            model.load_state_dict(data)
            return {"model_state_dict": data}
        elif isinstance(data, nn.Module):
            model.load_state_dict(data.state_dict())
            return {"model_state_dict": data.state_dict()}

        return {}

    # ─── Member 3: Federated Learning Interface Contract ──────────────────────

    @staticmethod
    def export_for_federation(
        model_or_agent: Union[DQNAgent, nn.Module],
        round_num: int = 1,
        metrics: Optional[dict[str, float]] = None,
        node_id: str = "edge_node_01",
    ) -> dict[str, Any]:
        """
        Exports local edge model weights for Federated Aggregator (Member 3).
        Payload contains NO raw packets or logs, preserving privacy.
        """
        model = model_or_agent.q_net if isinstance(model_or_agent, DQNAgent) else model_or_agent

        # Detached CPU clone of state_dict
        state_dict_clone = OrderedDict()
        for k, v in model.state_dict().items():
            state_dict_clone[k] = v.detach().cpu().clone()

        default_metrics = {"accuracy": 0.0, "f1": 0.0}
        if metrics:
            default_metrics.update(metrics)

        return {
            "model_state_dict": state_dict_clone,
            "round": int(round_num),
            "metrics": default_metrics,
            "node_id": str(node_id),
        }

    @staticmethod
    def import_from_federation(
        model_or_agent: Union[DQNAgent, nn.Module],
        aggregated_payload: dict[str, Any],
    ) -> None:
        """
        Loads the globally aggregated model weights sent from Member 3's FL Server.
        """
        model = model_or_agent.q_net if isinstance(model_or_agent, DQNAgent) else model_or_agent
        state_dict = aggregated_payload.get("model_state_dict", aggregated_payload)
        model.load_state_dict(state_dict)

        if isinstance(model_or_agent, DQNAgent):
            model_or_agent.target_net.load_state_dict(model.state_dict())
