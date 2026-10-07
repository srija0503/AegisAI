"""
rl_agent/evaluator.py
─────────────────────
Rigorous evaluation and benchmarking suite for the RL Firewall Agent.
Measures detection accuracy, precision, recall, F1, confusion matrix,
and sub-millisecond inference latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from rl_agent.action import FirewallAction
from rl_agent.agent import DQNAgent


@dataclass
class EvaluationMetrics:
    total_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: dict[str, int]
    avg_latency_ms: float
    rag_trigger_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "confusion_matrix": self.confusion_matrix,
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "rag_trigger_rate": round(self.rag_trigger_rate, 4),
        }


class Evaluator:
    """
    Evaluates policy performance and timing benchmarks against ground-truth datasets.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def evaluate(
        self,
        model_or_agent: Union[DQNAgent, nn.Module],
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 512,
    ) -> EvaluationMetrics:
        """
        Evaluate full dataset in mini-batches.
        """
        model = model_or_agent.q_net if isinstance(model_or_agent, DQNAgent) else model_or_agent
        model.eval()

        n = len(X)
        all_actions = []
        total_time = 0.0

        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch_x = torch.from_numpy(X[i : i + batch_size]).float().to(self.device)
                t0 = time.perf_counter()
                q_vals = model(batch_x)
                actions = q_vals.argmax(dim=-1).cpu().numpy()
                t1 = time.perf_counter()

                total_time += (t1 - t0)
                all_actions.extend(actions)

        actions_arr = np.array(all_actions)
        y_arr = y.astype(bool)

        # Action mapping:
        # ALLOW = 0 (predicted benign)
        # BLOCK = 1 (predicted attack)
        # TRIGGER_RAG = 2 (treated as defensive escalation / threat catch)
        pred_attack = (actions_arr == int(FirewallAction.BLOCK)) | (actions_arr == int(FirewallAction.TRIGGER_RAG))

        tp = int(np.sum(pred_attack & y_arr))
        fp = int(np.sum(pred_attack & (~y_arr)))
        tn = int(np.sum((~pred_attack) & (~y_arr)))
        fn = int(np.sum((~pred_attack) & y_arr))

        rag_triggers = int(np.sum(actions_arr == int(FirewallAction.TRIGGER_RAG)))

        accuracy = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        avg_latency_ms = (total_time / n) * 1000.0 if n > 0 else 0.0
        rag_rate = rag_triggers / n if n > 0 else 0.0

        return EvaluationMetrics(
            total_samples=n,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            avg_latency_ms=avg_latency_ms,
            rag_trigger_rate=rag_rate,
        )
