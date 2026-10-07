"""
federated_learning/aggregation package
"""

from federated_learning.aggregation.fedavg import FedAvg
from federated_learning.aggregation.weighted_fedavg import WeightedFedAvg

__all__ = [
    "FedAvg",
    "WeightedFedAvg",
]
