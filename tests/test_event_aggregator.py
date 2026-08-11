"""
Regression tests for the directory_enumerated window-reset fix (v2 audit
finding C8).

Before: monitor/file_monitor.py tracked "have I seen this directory before"
in a set that lived for the entire monitoring SESSION, so a directory could
only ever be counted once - after the first 30-second window, this signal
always contributed 0, despite directory_enumerated being the model's single
highest-importance feature (27.4%).

After: src/event_aggregator.py (which owns window lifecycle) tracks it
itself, scoped to the current WINDOW, and resets it in reset_window() - so
the same directory can legitimately be "new" again in a later window.

src/collect_normal_data.py has a parallel implementation (by design, per
BUG #4's fix, to keep offline reprocessing consistent with live inference) -
tested here too so the two cannot silently diverge again.

Run from the project root:  .venv/Scripts/python.exe -m pytest tests/ -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.event_aggregator import EventAggregator  # noqa: E402
from src.collect_normal_data import _apply_event_to_counts, _empty_feature_counts  # noqa: E402


def _file_event(path, event_type="created"):
    return {"source": "file", "event": event_type, "details": {"path": path}}


# --------------------------------------------------------------- live aggregator

def test_same_directory_twice_in_one_window_counts_once():
    agg = EventAggregator(window_seconds=30)
    agg.add_event(_file_event(r"C:\demo\folder_0\a.txt"))
    agg.add_event(_file_event(r"C:\demo\folder_0\b.txt"))  # same dir, second file

    features = agg.get_aggregated_features()
    directory_enumerated = features[6]  # index per FEATURES_12 / feature_names order
    assert directory_enumerated == 1


def test_different_directories_in_one_window_each_count():
    agg = EventAggregator(window_seconds=30)
    for i in range(5):
        agg.add_event(_file_event(rf"C:\demo\folder_{i}\a.txt"))

    features = agg.get_aggregated_features()
    assert features[6] == 5


def test_directory_can_be_new_again_after_reset_window():
    """This is the exact bug: the same directory touched in window 1 must be
    countable again in window 2, once reset_window() has run - it must NOT
    stay permanently "seen" for the life of the whole session."""
    agg = EventAggregator(window_seconds=30)

    agg.add_event(_file_event(r"C:\demo\folder_0\a.txt"))
    assert agg.get_aggregated_features()[6] == 1

    agg.reset_window()

    agg.add_event(_file_event(r"C:\demo\folder_0\b.txt"))  # same directory, new window
    assert agg.get_aggregated_features()[6] == 1, (
        "directory_enumerated must reset per window, not stay zero forever "
        "after the first window touches a given directory"
    )


def test_reset_window_clears_seen_dirs_state():
    agg = EventAggregator(window_seconds=30)
    agg.add_event(_file_event(r"C:\demo\folder_0\a.txt"))
    assert agg.seen_dirs_this_window  # non-empty before reset
    agg.reset_window()
    assert agg.seen_dirs_this_window == set()


# --------------------------------------------------------- offline pipeline (collect_normal_data.py)

def test_offline_pipeline_matches_live_semantics_within_one_window():
    feature_counts = _empty_feature_counts()
    seen_dirs = set()

    _apply_event_to_counts(feature_counts, "file", "created", {"path": r"C:\demo\folder_0\a.txt"}, seen_dirs)
    _apply_event_to_counts(feature_counts, "file", "modified", {"path": r"C:\demo\folder_0\b.txt"}, seen_dirs)
    _apply_event_to_counts(feature_counts, "file", "created", {"path": r"C:\demo\folder_1\c.txt"}, seen_dirs)

    assert feature_counts["directory_enumerated"] == 2  # folder_0 once, folder_1 once
    assert feature_counts["file_created"] == 2
    assert feature_counts["file_read"] == 1


def test_offline_pipeline_directory_reappears_after_caller_resets_seen_dirs():
    """Confirms the caller-owned reset pattern in
    process_event_log_to_training_format works the same way the live
    aggregator's reset_window() does."""
    feature_counts = _empty_feature_counts()
    seen_dirs = set()
    _apply_event_to_counts(feature_counts, "file", "created", {"path": r"C:\demo\folder_0\a.txt"}, seen_dirs)
    assert feature_counts["directory_enumerated"] == 1

    # Simulate closing the window: fresh feature_counts AND fresh seen_dirs,
    # exactly as process_event_log_to_training_format now does.
    feature_counts = _empty_feature_counts()
    seen_dirs = set()
    _apply_event_to_counts(feature_counts, "file", "created", {"path": r"C:\demo\folder_0\b.txt"}, seen_dirs)
    assert feature_counts["directory_enumerated"] == 1
