import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import threading


# ==========================================================
# EVENT AGGREGATOR
# ==========================================================

class EventAggregator:
    """
    Aggregates system events over time windows to match ML training data format.
    The ML models were trained on behavioral statistics (counts), not single events.
    """
    
    def __init__(self, window_seconds=30):
        self.window_seconds = window_seconds
        self.events_buffer = []
        self.lock = threading.Lock()
        self.window_start_time = datetime.now()
        
        # Feature counters for current window
        self.feature_counts = {
            "file_read": 0,
            "file_created": 0,
            "regkey_written": 0,
            "regkey_read": 0,
            "apistats": 0,
            "command_line": 0,
            "directory_enumerated": 0,
            "dll_loaded": 0,
            "tree_command_line": 0,
            "arguments": 0,
            "urls": 0,
            "proc_pid": 0
        }
    
    def add_event(self, event_data):
        """
        Add an event to the aggregation buffer.
        
        Args:
            event_data: Dictionary with source, event, process, pid, details
        """
        with self.lock:
            event_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.events_buffer.append(event_data)
            self._update_feature_counts(event_data)
    
    def _update_feature_counts(self, event_data):
        """
        Update feature counts based on event type.
        """
        source = event_data.get("source", "")
        event_type = event_data.get("event", "")
        details = event_data.get("details", {})
        
        if source == "file":
            if event_type == "created":
                self.feature_counts["file_created"] += 1
            elif event_type == "modified":
                self.feature_counts["file_read"] += 1
            # Cost-free supplementary directory_enumerated signal from real
            # file-system activity (see bugs_debugs.txt BUG #9) - supplements
            # the process-level open_files()-based proxy, which is too
            # expensive to poll continuously and only samples at process birth.
            if details.get("new_directory"):
                self.feature_counts["directory_enumerated"] += 1
            # "deleted" events are intentionally not mapped to any of the
            # 12 trained features (there is no deletion-specific feature
            # in the model). Previously this was miscounted as
            # file_created, which conflated deletion with creation
            # behavior. Deletions are still logged to events.csv for
            # record-keeping. See bugs_debugs.txt BUG #2.

        elif source == "process":
            # Only count actual process-creation events toward proc_pid.
            # "activity" events (periodic re-sampling of dll_loaded/apistats/
            # directory_enumerated for already-tracked processes - see
            # bugs_debugs.txt BUG #3) must not inflate this counter, or it
            # would no longer represent "number of processes started".
            if event_type == "created":
                self.feature_counts["proc_pid"] += 1
            if "command_line" in details:
                self.feature_counts["command_line"] += len(details["command_line"])
            if "tree_command_line" in details:
                self.feature_counts["tree_command_line"] += details["tree_command_line"]
            if "arguments" in details:
                self.feature_counts["arguments"] += details["arguments"]
            if "dll_loaded" in details:
                self.feature_counts["dll_loaded"] += details["dll_loaded"]
            if "apistats" in details:
                self.feature_counts["apistats"] += details["apistats"]
            if "directory_enumerated" in details:
                self.feature_counts["directory_enumerated"] += details["directory_enumerated"]
        
        elif source == "network":
            self.feature_counts["urls"] += 1
        
        elif source == "registry":
            if event_type == "written":
                self.feature_counts["regkey_written"] += 1
            elif event_type == "read":
                self.feature_counts["regkey_read"] += 1
    
    def get_aggregated_features(self):
        """
        Get aggregated feature vector for current time window.
        
        Returns:
            numpy array of 12 features matching training data format
        """
        with self.lock:
            features = np.array([
                self.feature_counts["file_read"],
                self.feature_counts["file_created"],
                self.feature_counts["regkey_written"],
                self.feature_counts["regkey_read"],
                self.feature_counts["apistats"],
                self.feature_counts["command_line"],
                self.feature_counts["directory_enumerated"],
                self.feature_counts["dll_loaded"],
                self.feature_counts["tree_command_line"],
                self.feature_counts["arguments"],
                self.feature_counts["urls"],
                self.feature_counts["proc_pid"]
            ], dtype=float)
            
            return features
    
    def reset_window(self):
        """
        Reset the aggregation window (call after classification).
        """
        with self.lock:
            self.events_buffer = []
            for key in self.feature_counts:
                self.feature_counts[key] = 0
            self.window_start_time = datetime.now()
    
    def get_window_stats(self):
        """
        Get statistics about current window.
        """
        with self.lock:
            elapsed = (datetime.now() - self.window_start_time).total_seconds()
            return {
                "event_count": len(self.events_buffer),
                "feature_counts": self.feature_counts.copy(),
                "window_seconds": self.window_seconds,
                "elapsed_seconds": elapsed,
                "window_start_time": self.window_start_time.strftime("%Y-%m-%d %H:%M:%S")
            }


# ==========================================================
# GLOBAL INSTANCE
# ==========================================================

aggregator = None


def get_aggregator(window_seconds=30):
    """
    Get or create the global aggregator instance.
    """
    global aggregator
    if aggregator is None:
        aggregator = EventAggregator(window_seconds)
    return aggregator
