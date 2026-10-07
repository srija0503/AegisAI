"""
federated_learning/simulation package
"""

from federated_learning.simulation.evaluate_global_model import (
    compare_models,
    evaluate_model,
)
from federated_learning.simulation.simulate_clients import (
    create_simulated_clients,
    generate_synthetic_flow_data,
)
from federated_learning.simulation.simulate_round import run_simulated_fl_round

__all__ = [
    "compare_models",
    "create_simulated_clients",
    "evaluate_model",
    "generate_synthetic_flow_data",
    "run_simulated_fl_round",
]
