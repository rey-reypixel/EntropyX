"""
Regression tests for src/inference_engine.py, pinning the exact fixes from
the v2 audit: C1 (anomaly-gate unit mismatch), C2 (threshold>1.0 silent kill
switch), C6 (event-count gate used a feature-magnitude sum instead of the
real event count), C7 (batch_classify called a nonexistent method).

Run from the project root:  .venv/Scripts/python.exe -m pytest tests/ -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference_engine import RansomwareInferenceEngine  # noqa: E402


# Loading the real XGBoost + autoencoder is the slow part of these tests -
# build the engine once and reuse it, resetting only what each test mutates.
@pytest.fixture(scope="module")
def engine():
    eng = RansomwareInferenceEngine()
    yield eng


class _FixedReconstruction:
    """Stand-in for the autoencoder scaler + model that forces an exact,
    known MSE, so detect_anomaly()'s arithmetic can be pinned precisely
    without depending on the real model's actual (unpredictable) output."""

    def __init__(self, mse):
        self.mse = mse
        self._sqrt_mse = mse ** 0.5

    def transform(self, features_df):
        return np.zeros((1, 12))

    def predict(self, features_scaled, verbose=0):
        # mean((0 - c)^2) over 12 identical entries == c^2 == mse
        return np.full((1, 12), self._sqrt_mse)


class _FakeAggregator:
    """Stand-in for EventAggregator with independently controllable
    'feature magnitudes' vs 'real event count' - the exact two quantities
    C6 conflated."""

    def __init__(self, features, event_count, window_seconds=30, elapsed_seconds=5):
        self._features = np.array(features, dtype=float)
        self._event_count = event_count
        self._window_seconds = window_seconds
        self._elapsed_seconds = elapsed_seconds
        self.reset_called = False

    def get_aggregated_features(self):
        return self._features

    def get_window_stats(self):
        return {
            "event_count": self._event_count,
            "window_seconds": self._window_seconds,
            "elapsed_seconds": self._elapsed_seconds,
        }

    def reset_window(self):
        self.reset_called = True


# --------------------------------------------------------------------- C2

def test_anomaly_threshold_must_be_positive(monkeypatch):
    """anomaly_threshold <= 0 must fail loudly at construction time, not
    silently produce a division by zero or a sign-flipped comparison later."""
    monkeypatch.setattr(
        RansomwareInferenceEngine,
        "load_config",
        lambda self: {"ml": {"anomaly_threshold": 0}},
    )
    with pytest.raises(ValueError, match="anomaly_threshold"):
        RansomwareInferenceEngine()

    monkeypatch.setattr(
        RansomwareInferenceEngine,
        "load_config",
        lambda self: {"ml": {"anomaly_threshold": -0.1}},
    )
    with pytest.raises(ValueError, match="anomaly_threshold"):
        RansomwareInferenceEngine()


# --------------------------------------------------------------------- C1

def test_detect_anomaly_is_not_capped_at_one(engine, monkeypatch):
    """The old code did min(mse/threshold, 1.0). A reconstruction error 10x
    the threshold must be reported as ~10.0, not clipped to 1.0 - otherwise
    every ransomware sample (whose errors run 100-1000x goodware, per
    autoencoder_results.csv) looks identical to a barely-anomalous one."""
    threshold = engine.anomaly_threshold
    fixed = _FixedReconstruction(mse=10 * threshold)
    monkeypatch.setattr(engine, "autoencoder_scaler", fixed)
    monkeypatch.setattr(engine, "autoencoder", fixed)

    score = engine.detect_anomaly(features=np.zeros(12))
    assert score == pytest.approx(10.0, rel=1e-6)


@pytest.mark.parametrize(
    "mse_as_fraction_of_threshold, expected_is_anomaly",
    [
        # The exact case that demonstrates the v1 bug: raw MSE at 55% of the
        # threshold. Under the OLD code: normalized = min(0.55, 1.0) = 0.55,
        # then compared AGAIN to the raw threshold value (~0.5417, which is
        # numerically < 0.55) -> is_anomaly = True. That is a false positive:
        # raw MSE (0.55 * threshold) is still BELOW the threshold.
        (0.55, False),
        # Comfortably below the threshold - both old and new code agree here.
        (0.10, False),
        # Exactly at the threshold boundary.
        (1.0, False),
        # Just over the threshold - must be True under both raw-MSE semantics
        # and the fixed code's anomaly_score > 1.0 check.
        (1.01, True),
        # Far over the threshold (typical E/L-class scale).
        (300.0, True),
    ],
)
def test_is_anomaly_matches_raw_mse_vs_threshold(
    engine, monkeypatch, mse_as_fraction_of_threshold, expected_is_anomaly
):
    """is_anomaly must equal (raw MSE > threshold), computed consistently in
    one unit - not a capped/normalized score compared against a raw value."""
    threshold = engine.anomaly_threshold
    mse = mse_as_fraction_of_threshold * threshold
    fixed = _FixedReconstruction(mse=mse)
    monkeypatch.setattr(engine, "autoencoder_scaler", fixed)
    monkeypatch.setattr(engine, "autoencoder", fixed)

    score = engine.detect_anomaly(features=np.zeros(12))
    is_anomaly = bool(score > 1.0)

    assert is_anomaly is expected_is_anomaly, (
        f"mse={mse:.6f} threshold={threshold:.6f} "
        f"(mse {'>' if mse > threshold else '<='} threshold) "
        f"but is_anomaly={is_anomaly}"
    )


# --------------------------------------------------------------------- C6

def test_event_count_gate_ignores_feature_magnitude(engine, monkeypatch):
    """A single event with a huge command_line value must NOT clear the
    min_events gate. Only the real event count may."""
    assert engine.min_events == 10, "test assumes the shipped default of 10"

    # command_line=5000 alone would have cleared the OLD np.sum(features)
    # gate (5000 >= 10), even though only 2 real events occurred.
    huge_feature_few_events = _FakeAggregator(
        features=[0, 0, 0, 0, 0, 5000, 0, 0, 0, 0, 0, 0],
        event_count=2,
    )
    monkeypatch.setattr(engine, "aggregator", huge_feature_few_events)

    result = engine.classify_aggregated()

    assert result is None, "must be gated: only 2 real events occurred"
    assert huge_feature_few_events.reset_called


def test_event_count_gate_allows_real_activity_through(engine, monkeypatch):
    """Sanity check for the other direction: enough real events, tiny
    feature magnitudes, must still classify."""
    small_features_enough_events = _FakeAggregator(
        features=[1, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 3],  # small values
        event_count=12,
    )
    monkeypatch.setattr(engine, "aggregator", small_features_enough_events)

    result = engine.classify_aggregated()

    assert result is not None
    assert result["label"] in ("G", "E", "L")
    assert small_features_enough_events.reset_called


# --------------------------------------------------------------------- C7

def test_batch_classify_removed(engine):
    """batch_classify called self.classify_event(), which never existed -
    a guaranteed AttributeError with no caller anywhere in the codebase and
    no principled per-single-event semantics in a pipeline that classifies
    aggregated 30s windows. Removed rather than papered over; this test
    documents that it is gone on purpose, not missing by accident."""
    assert not hasattr(engine, "batch_classify")
