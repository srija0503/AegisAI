"""
federated_learning package
──────────────────────────
Distributed Federated Learning architecture for the Quantum-Secure Federated Firewall.
Coordinates collaborative edge model improvement without sharing raw traffic data.
"""

from federated_learning.aggregation.fedavg import FedAvg
from federated_learning.aggregation.weighted_fedavg import WeightedFedAvg
from federated_learning.aggregator import ModelAggregator
from federated_learning.client import FederatedClient
from federated_learning.client_selection import ClientSelector, SelectionStrategy
from federated_learning.coordinator import FederatedCoordinator
from federated_learning.model_serializer import ModelSerializer
from federated_learning.rounds import FLRound, RoundManager, RoundStatus
from federated_learning.server import FederatedServer
from federated_learning.weight_validator import ValidationReport, WeightValidator

__all__ = [
    "ClientSelector",
    "FedAvg",
    "FederatedClient",
    "FederatedCoordinator",
    "FederatedServer",
    "FLRound",
    "ModelAggregator",
    "ModelSerializer",
    "RoundManager",
    "RoundStatus",
    "SelectionStrategy",
    "ValidationReport",
    "WeightValidator",
    "WeightedFedAvg",
]
