import joblib
import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
from datetime import datetime

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from event_aggregator import get_aggregator


# ==========================================================
# INFERENCE ENGINE
# ==========================================================

class RansomwareInferenceEngine:
    """
    Real-time inference engine for ransomware detection using trained ML models.
    """
    
    def load_config(self):
        """
        Load config from JSON file to read ML thresholds and parameters.
        """
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load config in inference engine: {e}")
        return {}

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.autoencoder = None
        self.autoencoder_scaler = None
        
        # Load configuration settings
        self.config = self.load_config()
        ml_config = self.config.get("ml", {})
        
        self.classification_threshold = ml_config.get("classification_threshold", 0.85)
        self.anomaly_threshold = ml_config.get("anomaly_threshold", 0.5)
        if self.anomaly_threshold <= 0:
            raise ValueError(
                f"ml.anomaly_threshold must be > 0 (got {self.anomaly_threshold}). "
                "It is a raw reconstruction-error (MSE) calibration value, taken "
                "from evaluate_autoencoder.py's percentile-search output - not a "
                "normalized 0-1 score. A value <= 0 would make detect_anomaly() "
                "divide by zero or by a negative number."
            )
        
        agg_config = ml_config.get("aggregation", {})
        window_seconds = agg_config.get("window_seconds", 30)
        self.min_events = agg_config.get("min_events", 10)
        
        self.aggregator = get_aggregator(window_seconds=window_seconds)
        self.feature_names = [
            "file_read",
            "file_created",
            "regkey_written",
            "regkey_read",
            "apistats",
            "command_line",
            "directory_enumerated",
            "dll_loaded",
            "tree_command_line",
            "arguments",
            "urls",
            "proc_pid"
        ]
        self.load_models()
    
    def load_models(self):
        """
        Load trained ML models and scalers.
        """
        print("Loading ML models...")
        
        try:
            # Load XGBoost model
            self.model = joblib.load("models/practical_xgboost.pkl")
            print("  [LOADED] XGBoost Model")
            
            # Load scaler
            self.scaler = joblib.load("models/practical_scaler.pkl")
            print("  [LOADED] Feature Scaler")
            
            # Load label encoder
            self.label_encoder = joblib.load("models/label_encoder.pkl")
            print("  [LOADED] Label Encoder")
            
            # Try to load autoencoder (optional)
            if os.path.exists("models/autoencoder_model.keras"):
                from tensorflow.keras.models import load_model
                self.autoencoder = load_model("models/autoencoder_model.keras")
                self.autoencoder_scaler = joblib.load("models/autoencoder_scaler.pkl")
                print("  [LOADED] Autoencoder Model")
            
            print("\nAll models loaded successfully.\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            raise
    
    def add_event(self, event_data):
        """
        Add an event to the aggregator for time-window classification.
        
        Args:
            event_data: Dictionary containing event information
        """
        self.aggregator.add_event(event_data)
    
    def should_classify(self):
        """
        Check if we should classify based on event count or time window.
        """
        stats = self.aggregator.get_window_stats()
        # Classify if we have at least min_events OR time window expired
        return stats["event_count"] >= self.min_events or stats["elapsed_seconds"] >= stats["window_seconds"]
    
    def _clean_datetime(self, obj):
        """
        Recursively convert datetime objects to strings for JSON serialization.
        """
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, dict):
            return {k: self._clean_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_datetime(item) for item in obj]
        else:
            return obj
    
    def classify_aggregated(self):
        """
        Classify aggregated events from the current time window.
        This matches the training data format (behavioral statistics).
        
        Returns:
            Dictionary with classification results, or None if not enough events
        """
        try:
            # Get aggregated features
            features = self.aggregator.get_aggregated_features()

            # Gate on the REAL event count (same definition should_classify()
            # uses), not a sum of feature magnitudes. The previous version
            # summed the feature vector itself, which mixes counts (proc_pid,
            # file_created, ...) with magnitudes like command_line (cumulative
            # character length, easily in the thousands) - a single event
            # could trivially clear the "min_events" bar. See v2 audit C6.
            event_count = self.aggregator.get_window_stats()["event_count"]

            # Check if we have enough events to perform classification
            if event_count < self.min_events:
                # Reset the window to avoid stale events carrying over
                self.aggregator.reset_window()
                return None
            
            # Convert to DataFrame with feature names to avoid sklearn warning
            features_df = pd.DataFrame([features], columns=self.feature_names)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Predict using XGBoost
            probabilities = self.model.predict_proba(features_scaled)[0]
            classes = self.label_encoder.classes_
            
            # Identify class indices dynamically
            g_idx = list(classes).index("G") if "G" in classes else -1
            e_idx = list(classes).index("E") if "E" in classes else -1
            l_idx = list(classes).index("L") if "L" in classes else -1
            
            # Determine target label and confidence based on thresholds
            e_prob = probabilities[e_idx] if e_idx != -1 else 0.0
            l_prob = probabilities[l_idx] if l_idx != -1 else 0.0
            g_prob = probabilities[g_idx] if g_idx != -1 else 0.0
            
            max_ransom_prob = max(e_prob, l_prob)
            if max_ransom_prob >= self.classification_threshold:
                if e_prob >= l_prob:
                    label = "E"
                    confidence = e_prob
                else:
                    label = "L"
                    confidence = l_prob
            else:
                label = "G"
                confidence = g_prob
            
            result = {
                "label": label,
                "confidence": float(confidence),
                "probabilities": {
                    self.label_encoder.classes_[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "window_stats": {
                    "event_count": self.aggregator.get_window_stats()["event_count"],
                    "window_seconds": self.aggregator.get_window_stats()["window_seconds"],
                    "elapsed_seconds": self.aggregator.get_window_stats()["elapsed_seconds"]
                }
            }
            
            # Remove any datetime objects from result before returning
            result = self._clean_datetime(result)
            
            # If autoencoder is available, also check for anomalies
            if self.autoencoder is not None:
                anomaly_score = self.detect_anomaly(features)
                result["anomaly_score"] = float(anomaly_score)
                # anomaly_score is already expressed in units of the
                # threshold (raw MSE / threshold, uncapped - see
                # detect_anomaly's docstring), so ">1.0" IS "raw MSE exceeded
                # the calibrated threshold". Comparing it against
                # self.anomaly_threshold again (a second time, in a different
                # unit) was the C1 bug - it made is_anomaly trigger on ~19%
                # of goodware instead of the calibrated ~5%.
                result["is_anomaly"] = bool(anomaly_score > 1.0)
            
            # Reset window after classification
            self.aggregator.reset_window()
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Classification failed: {e}")
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def detect_anomaly(self, features):
        """
        Detect anomalies using autoencoder reconstruction error.

        Args:
            features: Raw feature vector

        Returns:
            Anomaly score = raw MSE / calibrated threshold. NOT capped at 1.0:
            a score of 1.0 means the reconstruction error exactly equals the
            calibrated threshold, and values above 1.0 preserve how far past
            it the error actually is. is_anomaly (in classify_aggregated) is
            defined as anomaly_score > 1.0, which is exactly mse > threshold
            in the original MSE units - see bugs_debugs.txt BUG #6 / the v2
            audit's C1/C2 findings for why capping this at 1.0 and then
            comparing it against the raw threshold again was wrong: it made
            is_anomaly trigger on ~19% of goodware instead of the calibrated
            ~5%, and made any anomaly_threshold > 1.0 silently disable
            is_anomaly forever, since the capped score could never exceed 1.0.
        """
        try:
            # Convert to DataFrame with feature names to avoid sklearn warning
            features_df = pd.DataFrame([features], columns=self.feature_names)

            # Scale features
            features_scaled = self.autoencoder_scaler.transform(features_df)

            # Reconstruct
            reconstructed = self.autoencoder.predict(features_scaled, verbose=0)

            # Calculate reconstruction error
            mse = np.mean(np.square(features_scaled - reconstructed))

            # Express error in units of the calibrated threshold (see docstring
            # above for why this must not be capped at 1.0).
            threshold = self.anomaly_threshold
            anomaly_score = mse / threshold

            return anomaly_score

        except Exception as e:
            print(f"[ERROR] Anomaly detection failed: {e}")
            return 0.0


# ==========================================================
# GLOBAL INSTANCE
# ==========================================================

inference_engine = None


def get_inference_engine():
    """
    Get or create the global inference engine instance.
    """
    global inference_engine
    if inference_engine is None:
        inference_engine = RansomwareInferenceEngine()
    return inference_engine


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":
    # Test the inference engine
    engine = get_inference_engine()
    
    # Test with multiple events to trigger aggregation
    print("Testing event aggregation and classification...")
    
    for i in range(12):
        test_event = {
            "source": "file",
            "event": "modified",
            "process": "test.exe",
            "pid": 1234,
            "details": {
                "path": f"C:\\test\\file{i}.txt"
            }
        }
        engine.add_event(test_event)
        print(f"Added event {i+1}")
    
    # Check if ready to classify
    if engine.should_classify():
        result = engine.classify_aggregated()
        print("\nClassification Result:")
        print(json.dumps(result, indent=2))
    else:
        print("\nNot enough events to classify")
