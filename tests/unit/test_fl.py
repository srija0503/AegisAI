"""
tests/unit/test_fl.py
─────────────────────
Unit tests for Member 2 Phase 3: Federated Learning Engine.
Validates ModelSerializer, WeightValidator, FedAvg, WeightedFedAvg,
ClientSelector, FederatedServer, FederatedClient, and full simulated FL rounds.
"""

import numpy as np
import pytest
import torch

from federated_learning.aggregation.fedavg import FedAvg
from federated_learning.aggregation.weighted_fedavg import WeightedFedAvg
from federated_learning.aggregator import ModelAggregator
from federated_learning.client import FederatedClient
from federated_learning.client_selection import ClientSelector, SelectionStrategy
from federated_learning.coordinator import FederatedCoordinator
from federated_learning.model_serializer import ModelSerializer
from federated_learning.server import FederatedServer
from federated_learning.simulation.evaluate_global_model import evaluate_model
from federated_learning.simulation.simulate_round import run_simulated_fl_round
from federated_learning.weight_validator import WeightValidator
from rl_agent.models.dqn import DuelingDQN


def test_model_serializer_roundtrip():
    model = DuelingDQN()
    weights = model.state_dict()

    serialized = ModelSerializer.serialize_weights(weights)
    assert len(serialized) > 0
    checksum = ModelSerializer.compute_checksum(serialized)
    assert len(checksum) == 64

    # Deserialize
    restored = ModelSerializer.deserialize_weights(serialized)
    for k in weights.keys():
        assert torch.allclose(weights[k], restored[k])

    summary = ModelSerializer.get_model_summary(weights)
    assert summary["total_parameters"] > 0
    assert summary["total_layers"] > 0


def test_weight_validator_nan_and_norm_rejection():
    validator = WeightValidator(max_l2_norm_diff=10.0)
    model = DuelingDQN()
    base_weights = model.state_dict()

    # 1. Valid identical weights
    report = validator.validate_weights(base_weights, base_weights)
    assert report.is_valid is True

    # 2. Corrupted NaN weights
    nan_weights = {k: v.clone() for k, v in base_weights.items()}
    first_key = list(nan_weights.keys())[0]
    nan_weights[first_key][0] = float("nan")

    report_nan = validator.validate_weights(nan_weights, base_weights)
    assert report_nan.is_valid is False
    assert "NaN" in report_nan.reason

    # 3. Poisoned / extreme update norm
    poisoned_weights = {k: v.clone() for k, v in base_weights.items()}
    poisoned_weights[first_key] += 50.0  # Huge deviation

    report_poison = validator.validate_weights(poisoned_weights, base_weights)
    assert report_poison.is_valid is False
    assert "exceeds threshold" in report_poison.reason


def test_fedavg_and_weighted_fedavg():
    model1 = DuelingDQN()
    model2 = DuelingDQN()

    w1 = {k: v.clone() for k, v in model1.state_dict().items()}
    w2 = {k: v.clone() for k, v in model2.state_dict().items()}

    # FedAvg with equal samples (50, 50) -> should be exact midpoint
    agg = FedAvg.aggregate([(w1, 50), (w2, 50)])
    for k in w1.keys():
        expected = (w1[k].float() + w2[k].float()) / 2.0
        assert torch.allclose(agg[k].float(), expected, atol=1e-5)

    # WeightedFedAvg with custom alpha
    w_fedavg = WeightedFedAvg(server_momentum=0.0)
    w_agg = w_fedavg.aggregate([(w1, 1.0), (w2, 3.0)])
    for k in w1.keys():
        expected = (w1[k].float() * 0.25) + (w2[k].float() * 0.75)
        assert torch.allclose(w_agg[k].float(), expected, atol=1e-5)


def test_client_selector():
    selector = ClientSelector(strategy=SelectionStrategy.RANDOM, seed=123)
    clients = ["node_01", "node_02", "node_03", "node_04"]

    selected = selector.select_clients(clients, fraction=0.5, min_clients=1)
    assert len(selected) == 2

    # Round Robin
    rr_selector = ClientSelector(strategy=SelectionStrategy.ROUND_ROBIN)
    rr1 = rr_selector.select_clients(clients, fraction=0.5)
    rr2 = rr_selector.select_clients(clients, fraction=0.5)
    assert rr1 != rr2


def test_simulated_round_and_coordinator():
    res = run_simulated_fl_round(strategy="fedavg", num_nodes=3)

    assert res["status"] == "completed"
    assert res["round_id"] == 1
    assert len(res["participating_clients"]) == 3
    assert res["metrics"]["accepted_clients"] == 3
    assert res["metrics"]["rejected_clients"] == 0


def test_evaluate_global_model():
    model = DuelingDQN()
    metrics = evaluate_model(model, num_eval_samples=50)

    assert "mean_loss" in metrics
    assert "mean_q_value" in metrics
    assert "action_accuracy" in metrics
    assert 0.0 <= metrics["action_accuracy"] <= 1.0
