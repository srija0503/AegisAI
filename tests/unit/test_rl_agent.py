"""
tests/unit/test_rl_agent.py
───────────────────────────
Unit tests for the Reinforcement Learning Agent (Phase 3):
- Action & State Spaces
- Reward Shaping
- Gymnasium Environment (FirewallEnv)
- Neural Architectures (DuelingDQN, StandardDQN, ActorCriticPolicy)
- Experience Replay (SumTree, PrioritizedReplayBuffer, UniformReplayBuffer)
- Epsilon Greedy Policy
- Double DQNAgent Training Steps
- Evaluator & Benchmark Metrics
- CheckpointManager & Federated Learning Export/Import
- RLInferenceEngine
"""

import collections
import pytest
import numpy as np
import torch

from rl_agent import (
    ACTION_NAMES,
    ActorCriticPolicy,
    CheckpointManager,
    DQNAgent,
    DuelingDQN,
    EpsilonGreedyPolicy,
    Evaluator,
    FirewallAction,
    FirewallEnv,
    NUM_ACTIONS,
    PrioritizedReplayBuffer,
    RLInferenceEngine,
    RLTrainer,
    RewardCalculator,
    RewardConfig,
    STATE_DIM,
    StandardDQN,
    StateProcessor,
    TrainerConfig,
    UniformReplayBuffer,
)


# ─── Action & State Tests ─────────────────────────────────────────────────────

def test_action_space():
    assert NUM_ACTIONS == 3
    assert FirewallAction.ALLOW == 0
    assert FirewallAction.BLOCK == 1
    assert FirewallAction.TRIGGER_RAG == 2
    assert ACTION_NAMES[0] == "ALLOW"
    assert ACTION_NAMES[1] == "BLOCK"
    assert ACTION_NAMES[2] == "TRIGGER_RAG"


def test_state_processor():
    processor = StateProcessor(clip_min=-3.0, clip_max=3.0)
    raw = np.array([np.nan, 10.0, -10.0, 0.5])  # Short and has NaNs/outliers
    processed = processor.process(raw)

    assert len(processed) == STATE_DIM
    assert processed[0] == 0.0  # NaN replaced with 0.0
    assert processed[1] == 3.0  # Clipped to max
    assert processed[2] == -3.0 # Clipped to min
    assert processed[3] == 0.5

    t = processor.to_tensor(raw)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (1, STATE_DIM)


# ─── Reward Shaping Tests ─────────────────────────────────────────────────────

def test_reward_calculator():
    calc = RewardCalculator()

    # True Positive: blocked an attack
    r_tp = calc.compute(FirewallAction.BLOCK, is_attack=True)
    assert r_tp == 1.0

    # False Positive: blocked benign user
    r_fp = calc.compute(FirewallAction.BLOCK, is_attack=False)
    assert r_fp == -1.0

    # True Negative: allowed benign user
    r_tn = calc.compute(FirewallAction.ALLOW, is_attack=False)
    assert r_tn == 0.2

    # False Negative: allowed attack through (worst outcome)
    r_fn = calc.compute(FirewallAction.ALLOW, is_attack=True)
    assert r_fn == -2.0

    # RAG trigger on zero-day attack
    r_rag = calc.compute(FirewallAction.TRIGGER_RAG, is_attack=True, is_ambiguous_zero_day=True)
    assert r_rag == calc.config.rag_trigger_cost + calc.config.rag_true_reward

    # Latency penalty
    r_lat = calc.compute(FirewallAction.BLOCK, is_attack=True, latency_ms=10.0)
    assert r_lat < 1.0


# ─── Gymnasium Environment Tests ──────────────────────────────────────────────

def test_firewall_env():
    # Synthetic small dataset
    X = np.random.randn(50, STATE_DIM).astype(np.float32)
    y = np.array([0, 1] * 25, dtype=np.int32)

    env = FirewallEnv(X=X, y=y, max_steps_per_episode=10)
    obs, info = env.reset()

    assert obs.shape == (STATE_DIM,)
    assert env.action_space.n == 3

    # Step through
    next_obs, reward, term, trunc, step_info = env.step(FirewallAction.BLOCK)
    assert next_obs.shape == (STATE_DIM,)
    assert isinstance(reward, float)
    assert "accuracy" in step_info

    # Run episode to completion
    for _ in range(9):
        _, _, term, _, _ = env.step(FirewallAction.ALLOW)
    assert term is True


# ─── Model Architectures Tests ────────────────────────────────────────────────

def test_dueling_dqn():
    model = DuelingDQN(state_dim=STATE_DIM, action_dim=NUM_ACTIONS, hidden_dims=(64, 32))
    dummy_input = torch.randn(4, STATE_DIM)
    q_vals = model(dummy_input)

    assert q_vals.shape == (4, NUM_ACTIONS)


def test_standard_dqn():
    model = StandardDQN(state_dim=STATE_DIM, action_dim=NUM_ACTIONS, hidden_dims=(64, 32))
    dummy_input = torch.randn(4, STATE_DIM)
    q_vals = model(dummy_input)

    assert q_vals.shape == (4, NUM_ACTIONS)


def test_actor_critic_policy():
    policy = ActorCriticPolicy(state_dim=STATE_DIM, action_dim=NUM_ACTIONS, hidden_dims=(64, 32))
    dummy_input = torch.randn(4, STATE_DIM)
    action, log_prob, entropy, value = policy.get_action_and_value(dummy_input)

    assert action.shape == (4,)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4, 1)


# ─── Replay Buffer Tests ──────────────────────────────────────────────────────

def test_prioritized_replay_buffer():
    buf = PrioritizedReplayBuffer(capacity=100, beta_start=0.4, beta_end=1.0)
    s = np.zeros(STATE_DIM, dtype=np.float32)

    for i in range(20):
        buf.push(s, action=i % 3, reward=float(i), next_state=s, done=False)

    assert len(buf) == 20

    states, actions, rewards, next_states, dones, weights, indices = buf.sample(batch_size=8)
    assert states.shape == (8, STATE_DIM)
    assert actions.shape == (8,)
    assert weights.shape == (8,)
    assert len(indices) == 8

    # Priority update
    td_errs = np.ones(8) * 0.5
    buf.update_priorities(indices, td_errs)


def test_uniform_replay_buffer():
    buf = UniformReplayBuffer(capacity=100)
    s = np.zeros(STATE_DIM, dtype=np.float32)

    for i in range(15):
        buf.push(s, action=i % 3, reward=float(i), next_state=s, done=False)

    assert len(buf) == 15
    states, actions, rewards, next_states, dones = buf.sample(batch_size=5)
    assert states.shape == (5, STATE_DIM)


# ─── Exploration Policy Tests ─────────────────────────────────────────────────

def test_epsilon_greedy_policy():
    pol = EpsilonGreedyPolicy(epsilon_start=1.0, epsilon_end=0.1, decay_steps=10)
    assert pol.epsilon == 1.0

    for _ in range(5):
        pol.step()
    assert pol.epsilon == pytest.approx(0.55)

    for _ in range(10):
        pol.step()
    assert pol.epsilon == pytest.approx(0.1)  # Clamped to min

    # Confidence calculation
    q = torch.tensor([1.0, 5.0, 2.0])
    act, conf, probs = pol.select_action_with_confidence(q)
    assert act == 1
    assert conf > 0.8
    assert len(probs) == 3


# ─── DQNAgent Tests ───────────────────────────────────────────────────────────

def test_dqn_agent_step_and_learn():
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=NUM_ACTIONS,
        hidden_dims=(32, 16),
        batch_size=4,
        replay_capacity=50,
        use_per=False,
    )

    s = np.random.randn(STATE_DIM).astype(np.float32)
    action = agent.act(s, explore=False)
    assert action in (0, 1, 2)

    # Fill buffer to batch_size
    loss = None
    for _ in range(6):
        loss = agent.step(s, 1, 1.0, s, False)

    assert loss is not None
    assert isinstance(loss, float)


# ─── Evaluator Tests ──────────────────────────────────────────────────────────

def test_evaluator():
    evaluator = Evaluator()
    model = DuelingDQN(state_dim=STATE_DIM, action_dim=NUM_ACTIONS, hidden_dims=(32, 16))

    X = np.random.randn(100, STATE_DIM).astype(np.float32)
    y = np.random.randint(0, 2, size=100, dtype=np.int32)

    metrics = evaluator.evaluate(model, X, y, batch_size=32)
    assert metrics.total_samples == 100
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert "TP" in metrics.confusion_matrix
    assert metrics.avg_latency_ms >= 0.0


# ─── Checkpoint & Federated Learning Contract Tests ───────────────────────────

def test_checkpoint_and_federated_contract(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir=str(ckpt_dir))

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=NUM_ACTIONS,
        hidden_dims=(32, 16),
    )

    # Save and load checkpoint
    saved_path = manager.save(agent, filename="test_model.pt", episode=10)
    assert saved_path.exists()

    agent2 = DQNAgent(state_dim=STATE_DIM, action_dim=NUM_ACTIONS, hidden_dims=(32, 16))
    data = manager.load(agent2, str(saved_path))
    assert data["episode"] == 10

    # ─── Test Member 3 Federated Learning Interface Contract ───────────────────
    fl_payload = manager.export_for_federation(
        model_or_agent=agent,
        round_num=3,
        metrics={"accuracy": 0.94, "f1": 0.92},
        node_id="edge_firewall_node_1",
    )

    assert isinstance(fl_payload, dict)
    assert "model_state_dict" in fl_payload
    assert isinstance(fl_payload["model_state_dict"], collections.OrderedDict)
    assert fl_payload["round"] == 3
    assert fl_payload["metrics"]["accuracy"] == 0.94
    assert fl_payload["node_id"] == "edge_firewall_node_1"

    # Test importing federated weights back
    manager.import_from_federation(agent2, fl_payload)


# ─── Inference Engine Tests ───────────────────────────────────────────────────

def test_rl_inference_engine():
    model = DuelingDQN(state_dim=STATE_DIM, action_dim=NUM_ACTIONS, hidden_dims=(32, 16))
    engine = RLInferenceEngine(model_or_path=model, confidence_threshold=0.85)

    feat = np.random.randn(STATE_DIM).astype(np.float32)
    res = engine.predict(feat)

    assert res.action in (FirewallAction.ALLOW, FirewallAction.BLOCK, FirewallAction.TRIGGER_RAG)
    assert res.action_name in ("ALLOW", "BLOCK", "TRIGGER_RAG")
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.probabilities) == 3
    assert res.latency_ms >= 0.0
