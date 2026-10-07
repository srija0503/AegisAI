"""
federated_learning/simulation/evaluate_global_model.py
─────────────────────────────────────────────────────
Evaluation harness for global DQN model performance.
Measures policy decision accuracy, MSE loss, and Q-value stability
across validation traffic distributions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from federated_learning.simulation.simulate_clients import generate_synthetic_flow_data
from rl_agent.models.dqn import DuelingDQN
from rl_agent.state import STATE_DIM

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    num_eval_samples: int = 500,
    seed: int = 999,
) -> dict[str, float]:
    """
    Evaluates model on synthetic validation traffic.
    Returns: {"mean_loss": float, "mean_q_value": float, "accuracy": float}
    """
    model.eval()
    states, actions, targets = generate_synthetic_flow_data(
        num_samples=num_eval_samples,
        state_dim=STATE_DIM,
        seed=seed,
    )

    s_t = torch.from_numpy(states).float()
    a_t = torch.from_numpy(actions).long()
    target_t = torch.from_numpy(targets).float()

    criterion = nn.SmoothL1Loss()

    with torch.no_grad():
        q_values = model(s_t)
        chosen_q = q_values.gather(1, a_t.unsqueeze(1)).squeeze(1)
        loss = criterion(chosen_q, target_t).item()
        mean_q = float(q_values.mean().item())

        # Predicted best action vs target action
        pred_actions = q_values.argmax(dim=1).numpy()
        acc = float((pred_actions == actions).mean())

    return {
        "mean_loss": round(float(loss), 4),
        "mean_q_value": round(mean_q, 4),
        "action_accuracy": round(acc, 4),
    }


def compare_models(
    model_before: nn.Module,
    model_after: nn.Module,
) -> dict[str, Any]:
    """Compares pre-FL and post-FL model performance metrics."""
    metrics_before = evaluate_model(model_before)
    metrics_after = evaluate_model(model_after)

    return {
        "before": metrics_before,
        "after": metrics_after,
        "loss_delta": round(metrics_after["mean_loss"] - metrics_before["mean_loss"], 4),
        "accuracy_gain": round(metrics_after["action_accuracy"] - metrics_before["action_accuracy"], 4),
    }
