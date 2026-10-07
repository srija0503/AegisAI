"""
conftest.py — shared pytest fixtures.
"""
import numpy as np
import pandas as pd
import pytest

from datasets.feature_extraction import FEATURE_NAMES


@pytest.fixture
def sample_flow_df(n=100):
    """Create a synthetic labeled flow DataFrame for testing."""
    rng = np.random.default_rng(seed=42)
    data = {feat: rng.random(n).astype(np.float32) for feat in FEATURE_NAMES}
    data["label"] = rng.integers(0, 2, n).astype(np.int32)
    data["label_raw"] = ["normal" if l == 0 else "neptune" for l in data["label"]]
    data["attack_category"] = ["NORMAL" if l == 0 else "DoS" for l in data["label"]]
    return pd.DataFrame(data)


@pytest.fixture
def balanced_flow_df():
    """50/50 balanced benign/attack DataFrame."""
    rng = np.random.default_rng(seed=7)
    n = 200
    data = {feat: rng.random(n).astype(np.float32) for feat in FEATURE_NAMES}
    data["label"] = ([0] * 100 + [1] * 100)
    data["label_raw"] = ["normal"] * 100 + ["neptune"] * 100
    data["attack_category"] = ["NORMAL"] * 100 + ["DoS"] * 100
    return pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
