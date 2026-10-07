"""
Reinforcement Learning Agent package for the Quantum-Secure Federated Firewall.
"""

from rl_agent.action import ACTION_NAMES, FirewallAction, NUM_ACTIONS
from rl_agent.agent import DQNAgent
from rl_agent.checkpoint import CheckpointManager
from rl_agent.evaluator import EvaluationMetrics, Evaluator
from rl_agent.inference import RLInferenceEngine, RLInferenceResult
from rl_agent.models.dqn import DuelingDQN, StandardDQN
from rl_agent.models.policy_network import ActorCriticPolicy
from rl_agent.policy import EpsilonGreedyPolicy
from rl_agent.replay_buffer import PrioritizedReplayBuffer, UniformReplayBuffer
from rl_agent.reward import RewardCalculator, RewardConfig
from rl_agent.state import STATE_DIM, StateProcessor

try:
    from rl_agent.environment import FirewallEnv
    from rl_agent.trainer import RLTrainer, TrainerConfig
except ImportError:
    FirewallEnv = None  # type: ignore
    RLTrainer = None  # type: ignore
    TrainerConfig = None  # type: ignore

__all__ = [
    "ACTION_NAMES",
    "ActorCriticPolicy",
    "CheckpointManager",
    "DQNAgent",
    "DuelingDQN",
    "EpsilonGreedyPolicy",
    "EvaluationMetrics",
    "Evaluator",
    "FirewallAction",
    "FirewallEnv",
    "NUM_ACTIONS",
    "PrioritizedReplayBuffer",
    "RLInferenceEngine",
    "RLInferenceResult",
    "RLTrainer",
    "RewardCalculator",
    "RewardConfig",
    "STATE_DIM",
    "StandardDQN",
    "StateProcessor",
    "TrainerConfig",
    "UniformReplayBuffer",
]
