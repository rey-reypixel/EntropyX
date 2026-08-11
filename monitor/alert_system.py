import json
import os
from datetime import datetime
from enum import Enum
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================================
# ALERT LEVELS
# ==========================================================

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ==========================================================
# ALERT SYSTEM
# ==========================================================

class AlertSystem:
    """
    Alert system for ransomware detection warnings.
    """
    
    def load_config(self):
        """
        Load config from JSON file to read alert thresholds.
        """
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load config in alert system: {e}")
        return {}

    def __init__(self, log_dir=None):
        # Load configuration settings
        self.config = self.load_config()
        
        # Override log_dir if provided, otherwise load from config or fallback
        if log_dir is None:
            log_dir = self.config.get("alerts", {}).get("log_dir", "monitor/logs")
            
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.alert_log = os.path.join(log_dir, "alerts.csv")
        self.alert_json = os.path.join(log_dir, "alerts.json")

        self._init_alert_log()

        self.alert_count = 0
        self.critical_count = 0
        self.warning_count = 0

        # Wire up the prevention system so CRITICAL alerts can trigger an
        # automated response. Previously PreventionSystem existed as a
        # standalone class with no caller anywhere in the codebase - see
        # bugs_debugs.txt BUG #5. Stays a no-op unless the user explicitly
        # sets prevention.enabled=true in config.json (default: false).
        self.prevention = None
        try:
            from src.prevention_system import PreventionSystem
            self.prevention = PreventionSystem(self.config)
        except Exception as e:
            print(f"[WARNING] Failed to initialize prevention system: {e}")
    
    def _init_alert_log(self):
        """
        Initialize alert log file with headers.
        """
        if not os.path.exists(self.alert_log):
            with open(self.alert_log, "w", newline="", encoding="utf-8") as f:
                f.write("timestamp,level,source,process,pid,classification,confidence,message,details\n")
    
    def _serialize_datetime(self, obj):
        """
        Recursively convert datetime objects to strings for JSON serialization.
        """
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, dict):
            return {k: self._serialize_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item) for item in obj]
        else:
            return obj
    
    def trigger_alert(self, level, event_data, classification_result, message):
        """
        Trigger an alert based on detection results.
        
        Args:
            level: AlertLevel enum (INFO, WARNING, CRITICAL)
            event_data: Original event data
            classification_result: ML classification result
            message: Alert message
        """
        self.alert_count += 1
        
        if level == AlertLevel.CRITICAL:
            self.critical_count += 1
        elif level == AlertLevel.WARNING:
            self.warning_count += 1
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare alert details
        alert_details = {
            "event_data": self._serialize_datetime(event_data),
            "classification": self._serialize_datetime(classification_result)
        }
        
        # Log to CSV
        with open(self.alert_log, "a", newline="", encoding="utf-8") as f:
            details_json = json.dumps(alert_details).replace('"', '""')
            f.write(f'"{timestamp}","{level.value}","{event_data.get("source", "")}",'
                    f'"{event_data.get("process", "")}",{event_data.get("pid", -1)},'
                    f'"{classification_result.get("label", "Unknown")}",'
                    f'{classification_result.get("confidence", 0):.4f},'
                    f'"{message}",'
                    f'"{details_json}"\n')
        
        # Log to JSON for easier parsing
        alert_entry = {
            "id": self.alert_count,
            "timestamp": timestamp,
            "level": level.value,
            "source": event_data.get("source", ""),
            "process": event_data.get("process", ""),
            "pid": event_data.get("pid", -1),
            "classification": classification_result.get("label", "Unknown"),
            "confidence": classification_result.get("confidence", 0),
            "message": message,
            "details": self._serialize_datetime(alert_details)
        }
        
        # Append to JSON file
        alerts = []
        if os.path.exists(self.alert_json):
            with open(self.alert_json, "r", encoding="utf-8") as f:
                try:
                    alerts = json.load(f)
                except:
                    alerts = []
        
        alerts.append(alert_entry)
        
        with open(self.alert_json, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2)
        
        # Print alert to console
        self._print_alert(level, alert_entry)

        # Hand off to the prevention system on CRITICAL alerts. No-op unless
        # prevention.enabled=true in config.json.
        if level == AlertLevel.CRITICAL and self.prevention is not None:
            try:
                self.prevention.handle_critical_alert({
                    "pid": event_data.get("pid", -1),
                    "process": event_data.get("process", "Unknown"),
                    "label": classification_result.get("label", "Unknown"),
                    "confidence": classification_result.get("confidence", 0),
                    "message": message
                })
            except Exception as e:
                print(f"[WARNING] Prevention system failed to handle alert: {e}")

    def _print_alert(self, level, alert_entry):
        """
        Print alert to console with formatting.
        """
        if level == AlertLevel.INFO:
            return  # Suppress normal info alerts from clogging console
            
        separator = "="*60
        
        try:
            if level == AlertLevel.CRITICAL:
                print(f"\n{separator}")
                print(f"🚨 {level.value} ALERT #{alert_entry['id']}")
                print(f"{separator}")
            elif level == AlertLevel.WARNING:
                print(f"\n⚠️  {level.value} ALERT #{alert_entry['id']}")
                print(f"{separator}")
            else:
                print(f"\nℹ️  {level.value} ALERT #{alert_entry['id']}")
                print(f"{separator}")
            
            print(f"Time: {alert_entry['timestamp']}")
            print(f"Source: {alert_entry['source']}")
            print(f"Process: {alert_entry['process']} (PID: {alert_entry['pid']})")
            print(f"Classification: {alert_entry['classification']}")
            print(f"Confidence: {alert_entry['confidence']:.2%}")
            print(f"Message: {alert_entry['message']}")
            
            if "anomaly_score" in alert_entry["details"]["classification"]:
                anomaly = alert_entry["details"]["classification"]["anomaly_score"]
                print(f"Anomaly Score: {anomaly:.4f}")
            
            print(separator + "\n")
        except UnicodeEncodeError:
            # Safe ASCII fallback for legacy Windows consoles
            if level == AlertLevel.CRITICAL:
                print(f"\n{separator}")
                print(f"[!] {level.value} ALERT #{alert_entry['id']}")
                print(f"{separator}")
            elif level == AlertLevel.WARNING:
                print(f"\n{separator}")
                print(f"[*] {level.value} ALERT #{alert_entry['id']}")
                print(f"{separator}")
            else:
                print(f"\n{separator}")
                print(f"[-] {level.value} ALERT #{alert_entry['id']}")
                print(f"{separator}")
            
            print(f"Time: {alert_entry['timestamp']}")
            print(f"Source: {alert_entry['source']}")
            print(f"Process: {alert_entry['process']} (PID: {alert_entry['pid']})")
            print(f"Classification: {alert_entry['classification']}")
            print(f"Confidence: {alert_entry['confidence']:.2%}")
            print(f"Message: {alert_entry['message']}")
            
            if "anomaly_score" in alert_entry["details"]["classification"]:
                anomaly = alert_entry["details"]["classification"]["anomaly_score"]
                print(f"Anomaly Score: {anomaly:.4f}")
            
            print(separator + "\n")
    
    def analyze_classification(self, event_data, classification_result):
        """
        Analyze classification result and trigger appropriate alerts.
        
        Args:
            event_data: Original event data
            classification_result: ML classification result
        """
        label = classification_result.get("label", "Unknown")
        confidence = classification_result.get("confidence", 0)
        
        # Check for anomaly detection
        is_anomaly = classification_result.get("is_anomaly", False)
        anomaly_score = classification_result.get("anomaly_score", 0)
        
        # Load thresholds from config
        alerts_config = self.config.get("alerts", {})
        critical_threshold = alerts_config.get("critical_threshold", 0.85)
        warning_threshold = alerts_config.get("warning_threshold", 0.70)
        
        # CRITICAL: Ransomware detected with very high confidence
        if label in ["E", "L"] and confidence >= critical_threshold:
            self.trigger_alert(
                AlertLevel.CRITICAL,
                event_data,
                classification_result,
                f"RANSOMWARE DETECTED: {label} with {confidence:.1%} confidence"
            )
        
        # CRITICAL: Very high anomaly score (zero-day threat, score > 90%)
        elif is_anomaly and anomaly_score > 0.9:
            self.trigger_alert(
                AlertLevel.CRITICAL,
                event_data,
                classification_result,
                f"ZERO-DAY THREAT: High anomaly score {anomaly_score:.2f}"
            )
        
        # WARNING: Suspicious activity (confidence > warning_threshold)
        elif label in ["E", "L"] and confidence >= warning_threshold:
            self.trigger_alert(
                AlertLevel.WARNING,
                event_data,
                classification_result,
                f"Suspicious activity: {label} with {confidence:.1%} confidence"
            )
        
        # WARNING: Moderate anomaly (anomaly score > 70%)
        elif is_anomaly and anomaly_score > 0.7:
            self.trigger_alert(
                AlertLevel.WARNING,
                event_data,
                classification_result,
                f"Anomalous behavior detected: score {anomaly_score:.2f}"
            )
        
        # INFO: Normal activity
        elif label == "G":
            # Only log info for significant events
            if confidence > 0.9:
                self.trigger_alert(
                    AlertLevel.INFO,
                    event_data,
                    classification_result,
                    "Normal activity confirmed"
                )
    
    def get_alert_summary(self):
        """
        Get summary of alerts.
        """
        return {
            "total_alerts": self.alert_count,
            "critical_alerts": self.critical_count,
            "warning_alerts": self.warning_count,
            "info_alerts": self.alert_count - self.critical_count - self.warning_count
        }


# ==========================================================
# GLOBAL INSTANCE
# ==========================================================

alert_system = None


def get_alert_system():
    """
    Get or create the global alert system instance.
    """
    global alert_system
    if alert_system is None:
        alert_system = AlertSystem()
    return alert_system


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":
    # Test the alert system
    alerts = get_alert_system()
    
    # Test with a critical alert
    test_event = {
        "source": "file",
        "event": "modified",
        "process": "ransomware.exe",
        "pid": 9999,
        "details": {"path": "C:\\important\\file.doc"}
    }
    
    test_classification = {
        "label": "E",
        "confidence": 0.95,
        "probabilities": {"G": 0.02, "E": 0.95, "L": 0.03}
    }
    
    alerts.analyze_classification(test_event, test_classification)
    
    print("\nAlert Summary:")
    print(json.dumps(alerts.get_alert_summary(), indent=2))
