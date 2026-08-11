import csv
import json
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================================
# LOG DIRECTORY
# ==========================================================

LOG_DIR = "monitor/logs"

os.makedirs(LOG_DIR, exist_ok=True)

EVENT_LOG = os.path.join(LOG_DIR, "events.csv")

# ==========================================================
# CREATE CSV
# ==========================================================

if not os.path.exists(EVENT_LOG):

    with open(EVENT_LOG, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "source",
            "event",
            "process",
            "pid",
            "details"
        ])


# ==========================================================
# EVENT COLLECTOR CLASS
# ==========================================================

class EventCollector:

    def __init__(self, enable_ml=True):

        self.event_count = 0
        self.enable_ml = enable_ml
        self.auto_classify = True
        self.print_events = False
        
        # Initialize ML components if enabled
        self.inference_engine = None
        self.alert_system = None
        
        if self.enable_ml:
            try:
                from src.inference_engine import get_inference_engine
                from alert_system import get_alert_system
                
                self.inference_engine = get_inference_engine()
                self.alert_system = get_alert_system()
                
                print("[ML] Inference engine and alert system initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize ML components: {e}")
                print("[WARNING] Running in data collection mode only")
                self.enable_ml = False

    # ------------------------------------------------------

    def add_event(
        self,
        source,
        event,
        process="Unknown",
        pid=-1,
        details=None
    ):

        if details is None:
            details = {}

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.event_count += 1

        if self.print_events:
            print(
                f"[EVENT {self.event_count}] "
                f"{source.upper()} | "
                f"{event.upper()} | "
                f"{process}"
            )

        # Log to CSV
        with open(
            EVENT_LOG,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                timestamp,
                source,
                event,
                process,
                pid,
                json.dumps(details)
            ])
        
        # Run ML classification if enabled
        if self.enable_ml and self.inference_engine and self.alert_system:
            try:
                event_data = {
                    "source": source,
                    "event": event,
                    "process": process,
                    "pid": pid,
                    "details": details
                }
                
                # Add event to aggregator
                self.inference_engine.add_event(event_data)
                
                # Check if we should classify (time window threshold)
                if self.auto_classify:
                    if self.inference_engine.should_classify():
                        classification = self.inference_engine.classify_aggregated()
                        
                        if classification and classification.get("label") != "Unknown":
                            # Trigger alert based on classification
                            self.alert_system.analyze_classification(event_data, classification)
                
            except Exception as e:
                print(f"[ERROR] ML classification failed: {e}")

    # ------------------------------------------------------

    def get_total_events(self):

        return self.event_count


# ==========================================================
# GLOBAL OBJECT
# ==========================================================

collector = EventCollector()


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    collector.add_event(
        source="file",
        event="created",
        process="notepad.exe",
        pid=1234,
        details={
            "path": r"C:\test\a.txt"
        }
    )

    collector.add_event(
        source="network",
        event="connection",
        process="chrome.exe",
        pid=5678,
        details={
            "remote_ip": "8.8.8.8",
            "port": 443
        }
    )

    print(
        "\nTotal Events:",
        collector.get_total_events()
    )