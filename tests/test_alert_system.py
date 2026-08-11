"""
Regression test for monitor/alert_system.py (v2 audit finding C5).

MALICIOUS_ACTIVITY_TRIGGERS.txt used to claim "Both models must agree for
highest confidence alerts", but analyze_classification()'s if/elif chain
checks the XGBoost classifier and the autoencoder in fully independent
branches - either one alone can produce a CRITICAL alert. That is the
correct, intentional behavior (it is the entire point of running two
independent models - see please_study_this.txt Section 2 - and it's exactly
what the live demo relies on: the classifier says Goodware while the
autoencoder alone fires CRITICAL on the same attack). The documentation was
wrong, not the code - fixed there instead. This test pins the actual
priority-ordered behavior so it cannot silently drift in either direction.

Run from the project root:  .venv/Scripts/python.exe -m pytest tests/ -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.alert_system import AlertSystem, AlertLevel  # noqa: E402


@pytest.fixture()
def alert_system(tmp_path):
    # log_dir override keeps this test out of the real monitor/logs/ files.
    system = AlertSystem(log_dir=str(tmp_path))
    calls = []
    system.trigger_alert = lambda level, event_data, classification_result, message: (
        calls.append(level)
    )
    system._calls = calls
    return system


def _event():
    return {"source": "process", "event": "created", "process": "test.exe", "pid": 1}


CASES = [
    # (label, confidence, is_anomaly, anomaly_score) -> expected level or None
    pytest.param(
        "E", 0.90, True, 0.95, AlertLevel.CRITICAL,
        id="classifier-critical-wins-priority-even-if-anomaly-also-critical",
    ),
    pytest.param(
        "G", 0.50, True, 0.95, AlertLevel.CRITICAL,
        id="anomaly-alone-triggers-critical-when-classifier-says-goodware"
            "-(the exact dual-model demo scenario)",
    ),
    pytest.param(
        "E", 0.75, False, 0.0, AlertLevel.WARNING,
        id="classifier-warning-below-critical-threshold",
    ),
    pytest.param(
        "G", 0.50, True, 0.75, AlertLevel.WARNING,
        id="anomaly-alone-triggers-warning-when-classifier-says-goodware",
    ),
    pytest.param(
        "G", 0.95, False, 0.0, AlertLevel.INFO,
        id="confident-goodware-logs-info",
    ),
    pytest.param(
        "G", 0.50, False, 0.0, None,
        id="unremarkable-goodware-triggers-nothing",
    ),
]


@pytest.mark.parametrize("label, confidence, is_anomaly, anomaly_score, expected", CASES)
def test_alert_priority_order(
    alert_system, label, confidence, is_anomaly, anomaly_score, expected
):
    classification_result = {
        "label": label,
        "confidence": confidence,
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
    }

    alert_system.analyze_classification(_event(), classification_result)

    if expected is None:
        assert alert_system._calls == []
    else:
        assert alert_system._calls == [expected]


def test_either_model_alone_is_sufficient_for_critical(alert_system):
    """Direct pin of the documented (and now doc-corrected) design: the two
    CRITICAL paths are independent, not an AND of both models agreeing."""
    classifier_only = {"label": "E", "confidence": 0.90, "is_anomaly": False, "anomaly_score": 0.0}
    anomaly_only = {"label": "G", "confidence": 0.10, "is_anomaly": True, "anomaly_score": 0.95}

    alert_system.analyze_classification(_event(), classifier_only)
    alert_system.analyze_classification(_event(), anomaly_only)

    assert alert_system._calls == [AlertLevel.CRITICAL, AlertLevel.CRITICAL]
