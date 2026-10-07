"""
tests/unit/test_feature_extraction.py
───────────────────────────────────────
Unit tests for the feature extraction module.
Run: pytest tests/unit/test_feature_extraction.py -v
"""

import numpy as np
import pandas as pd
import pytest

from datasets.feature_extraction import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FlowStats,
    OfflineFeatureExtractor,
    OnlineFeatureExtractor,
    df_to_tensors,
    get_feature_names,
)


# ─── Tests for FEATURE_NAMES / FEATURE_DIM ────────────────────────────────────

def test_feature_dim_is_41():
    assert FEATURE_DIM == 41

def test_feature_names_length():
    assert len(FEATURE_NAMES) == 41

def test_get_feature_names_returns_copy():
    names = get_feature_names()
    names.append("extra")
    assert len(FEATURE_NAMES) == 41, "get_feature_names should return a copy"


# ─── Tests for OfflineFeatureExtractor ────────────────────────────────────────

class TestOfflineFeatureExtractor:
    def setup_method(self):
        self.extractor = OfflineFeatureExtractor()

    def _make_sample_df(self, n: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(seed=42)
        data = {feat: rng.random(n) for feat in FEATURE_NAMES}
        data["label"] = rng.integers(0, 2, n)
        data["label_raw"] = ["normal" if l == 0 else "neptune" for l in data["label"]]
        data["attack_category"] = ["NORMAL" if l == 0 else "DoS" for l in data["label"]]
        return pd.DataFrame(data)

    def test_extract_single_row(self):
        df = self._make_sample_df(3)
        row = df.iloc[0]
        vec = self.extractor.extract(row)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (FEATURE_DIM,)
        assert vec.dtype == np.float32

    def test_batch_extract_shape(self):
        df = self._make_sample_df(100)
        arr = self.extractor.batch_extract(df)
        assert arr.shape == (100, FEATURE_DIM)
        assert arr.dtype == np.float32

    def test_batch_extract_handles_missing_cols(self):
        df = pd.DataFrame({"duration": [1.0, 2.0], "src_bytes": [0.5, 0.3]})
        arr = self.extractor.batch_extract(df)
        assert arr.shape == (2, FEATURE_DIM)

    def test_extract_zeros_for_missing_features(self):
        row = pd.Series({"duration": 0.5})
        vec = self.extractor.extract(row)
        # All features except duration should be 0
        assert vec[0] == pytest.approx(0.5, abs=1e-6)
        assert vec[1] == pytest.approx(0.0, abs=1e-6)


# ─── Tests for df_to_tensors ──────────────────────────────────────────────────

class TestDfToTensors:
    def _make_labeled_df(self, n: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(seed=0)
        data = {feat: rng.random(n) for feat in FEATURE_NAMES}
        data["label"] = rng.integers(0, 2, n)
        return pd.DataFrame(data)

    def test_returns_tuple_of_arrays(self):
        df = self._make_labeled_df()
        X, y = df_to_tensors(df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_X_shape(self):
        df = self._make_labeled_df(50)
        X, y = df_to_tensors(df)
        assert X.shape == (50, FEATURE_DIM)

    def test_y_shape(self):
        df = self._make_labeled_df(50)
        X, y = df_to_tensors(df)
        assert y.shape == (50,)

    def test_y_dtype(self):
        df = self._make_labeled_df()
        _, y = df_to_tensors(df)
        assert y.dtype == np.int32

    def test_no_label_column(self):
        rng = np.random.default_rng(seed=1)
        df = pd.DataFrame({feat: rng.random(10) for feat in FEATURE_NAMES})
        X, y = df_to_tensors(df)
        assert (y == 0).all(), "Without label col, y should be all zeros"


# ─── Tests for FlowStats ──────────────────────────────────────────────────────

class TestFlowStats:
    def test_duration_positive(self):
        import time
        flow = FlowStats()
        time.sleep(0.01)
        flow.last_time = time.time()
        assert flow.duration >= 0.0

    def test_default_values(self):
        flow = FlowStats()
        assert flow.src_bytes == 0
        assert flow.dst_bytes == 0
        assert flow.protocol == "other"


# ─── Tests for OnlineFeatureExtractor ─────────────────────────────────────────

class TestOnlineFeatureExtractor:
    def test_extract_from_flow_shape(self):
        extractor = OnlineFeatureExtractor()
        flow = FlowStats(src_ip="192.168.1.1", dst_ip="10.0.0.1", protocol="tcp")
        vec = extractor.extract_from_flow(flow, [])
        assert vec.shape == (FEATURE_DIM,)

    def test_extract_from_flow_dtype(self):
        extractor = OnlineFeatureExtractor()
        flow = FlowStats()
        vec = extractor.extract_from_flow(flow, [])
        assert vec.dtype == np.float32

    def test_land_feature_same_ip(self):
        extractor = OnlineFeatureExtractor()
        flow = FlowStats(src_ip="10.0.0.1", dst_ip="10.0.0.1")
        vec = extractor.extract_from_flow(flow, [])
        # land feature is at index 6
        assert vec[6] == pytest.approx(1.0, abs=1e-6)

    def test_land_feature_diff_ip(self):
        extractor = OnlineFeatureExtractor()
        flow = FlowStats(src_ip="10.0.0.1", dst_ip="10.0.0.2")
        vec = extractor.extract_from_flow(flow, [])
        assert vec[6] == pytest.approx(0.0, abs=1e-6)
