# WatchRansom - Ransomware Detection and Prevention System

A proactive ransomware detection system using behavioral analysis, anomaly detection, and artificial intelligence to identify malicious encryption activities before system compromise.

## Features

- **Real-Time Monitoring**: Monitors file system, process, and network activities
- **AI-Powered Detection**: Uses XGBoost for supervised classification and Autoencoder for anomaly detection
- **Behavioral Analysis**: Tracks 13 key behavioral features indicative of ransomware
- **Alert System**: Multi-level alerting (INFO, WARNING, CRITICAL) with detailed logging
- **Zero-Day Detection**: Anomaly detection for unknown threats
- **Configurable**: JSON-based configuration for easy customization

## Architecture

```
WatchRansom/
├── monitor/              # Real-time monitoring components
│   ├── file_monitor.py      # File system event monitoring
│   ├── process_monitor.py   # Process creation and activity
│   ├── network_monitor.py   # Network connection tracking
│   ├── registry_monitor.py   # Registry change detection (placeholder)
│   ├── event_collector.py    # Centralized event aggregation
│   ├── alert_system.py       # Alerting and notification system
│   └── start_monitoring.py  # Monitoring orchestration
├── src/                  # ML training and inference
│   ├── preprocessing.py      # Data preprocessing pipeline
│   ├── train_xgboost.py      # Full-feature XGBoost training
│   ├── train_practical_xgboost.py  # Pruned XGBoost training
│   ├── train_autoencoder.py  # Autoencoder for anomaly detection
│   ├── evaluate_autoencoder.py    # Autoencoder evaluation
│   ├── evaluate_isolation_forest.py # Isolation Forest evaluation
│   ├── inference_engine.py   # Real-time ML inference
│   └── feature_importance.py # Feature importance analysis
├── models/               # Trained ML models and scalers
├── datasets/             # Training datasets
├── monitor/logs/         # Runtime logs (events, alerts)
├── config.json           # System configuration
└── main.py               # Main entry point
```

## Installation

### Prerequisites

- Python 3.8+
- Windows OS (for registry monitoring and process monitoring)

### Dependencies

```bash
pip install watchdog psutil pandas numpy scikit-learn xgboost tensorflow joblib
```

## Quick Start

### 1. Train ML Models (First Time Only)

```bash
# Preprocess data
python src/preprocessing.py

# Train practical XGBoost model
python src/train_practical_xgboost.py

# Train autoencoder for anomaly detection
python src/train_autoencoder.py
```

### 2. Start Monitoring

```bash
# Start with default settings (monitors test_folder)
python main.py

# Monitor a specific folder
python main.py --folder "C:\path\to\monitor"

# Disable ML classification (data collection only)
python main.py --no-ml

# Use custom configuration
python main.py --config custom.json
```

### 3. Monitor Logs

- **Events**: `monitor/logs/events.csv` - All collected system events
- **Alerts**: `monitor/logs/alerts.csv` - Detection alerts
- **Alerts JSON**: `monitor/logs/alerts.json` - Alerts in JSON format

## Configuration

Edit `config.json` to customize the system:

```json
{
  "monitoring": {
    "enabled_monitors": ["file", "process", "network"],
    "monitor_folder": "test_folder"
  },
  "ml": {
    "enabled": true,
    "classification_threshold": 0.7,
    "anomaly_threshold": 0.5
  },
  "alerts": {
    "enabled": true,
    "critical_threshold": 0.7,
    "warning_threshold": 0.5
  }
}
```

## Detection Logic

### Supervised Classification (XGBoost)

- **Classes**: G (Goodware), E (Encryption Ransomware), L (Locker Ransomware)
- **Features**: 13 behavioral features including file operations, registry changes, API calls, network activity
- **Threshold**: >70% confidence triggers CRITICAL alert

### Anomaly Detection (Autoencoder)

- Trained ONLY on goodware to learn normal behavior patterns
- High reconstruction error indicates anomalous (potentially malicious) activity
- Enables zero-day threat detection

### Alert Levels

- **CRITICAL**: Ransomware detected with high confidence (>70%) or high anomaly score (>0.8)
- **WARNING**: Suspicious activity with moderate confidence (>50%) or moderate anomaly (>0.5)
- **INFO**: Normal activity confirmed

## Behavioral Features

The system monitors 13 key features:

1. `file_read` - File read operations
2. `file_created` - File creation events
3. `regkey_written` - Registry key modifications
4. `regkey_read` - Registry key reads
5. `apistats` - API call statistics
6. `command_line` - Command line arguments
7. `directory_enumerated` - Directory enumeration (reconnaissance)
8. `dll_loaded` - DLL loading activity
9. `tree_command_line` - Process tree command lines
10. `arguments` - Number of process arguments
11. `urls` - Network connections/URLs
12. `proc_pid` - Process identification

## Usage Examples

### Data Collection Mode

Collect system events without ML classification:

```bash
python main.py --no-ml
```

### Custom Folder Monitoring

Monitor a specific directory:

```bash
python main.py --folder "C:\Users\Documents"
```

### Silent Mode (No Console Alerts)

Modify `config.json`:

```json
{
  "alerts": {
    "console_output": false
  }
}
```

## Model Training

### Preprocessing

```bash
python src/preprocessing.py
```

This performs:
- Data cleaning and validation
- Label encoding (G, E, L)
- Feature scaling with StandardScaler
- Train-test split (80/20)

### Training XGBoost

```bash
python src/train_practical_xgboost.py
```

Outputs:
- `models/practical_xgboost.pkl` - Trained model
- `models/practical_scaler.pkl` - Feature scaler
- Accuracy metrics and classification report

### Training Autoencoder

```bash
python src/train_autoencoder.py
```

Outputs:
- `models/autoencoder_model.keras` - Neural network model
- `models/autoencoder_scaler.pkl` - Feature scaler
- Training history and validation metrics

### Evaluation

```bash
# Evaluate autoencoder
python src/evaluate_autoencoder.py

# Evaluate isolation forest
python src/evaluate_isolation_forest.py

# Feature importance
python src/feature_importance.py
```

## Troubleshooting

### ML Components Failed to Load

If you see "Failed to initialize ML components":
- Ensure models are trained (check `models/` directory)
- Verify all dependencies are installed
- Run with `--no-ml` flag for data collection mode

### Permission Denied Errors

Some monitors require administrator privileges:
- Run terminal/command prompt as Administrator
- Process and network monitoring need elevated permissions

### High False Positive Rate

Adjust thresholds in `config.json`:
```json
{
  "ml": {
    "classification_threshold": 0.8,
    "anomaly_threshold": 0.6
  }
}
```

## Future Enhancements

- [ ] Implement registry monitoring
- [ ] Add email/SMS alert notifications
- [ ] Web dashboard for real-time monitoring
- [ ] Automatic threat response (process termination, file quarantine)
- [ ] Integration with SIEM systems
- [ ] Multi-platform support (Linux, macOS)

## License

[Business Source License 1.1](LICENSE) — source-available, **not** open source.

Free for non-production use: personal projects, evaluation, research, and teaching.
**Any production or commercial use requires a paid commercial license.**
On **2030-08-19** this version converts automatically to Apache License 2.0.

Commercial and production licensing: **mahashreyaa@gmail.com**

## Disclaimer

This software is provided as-is for ransomware detection research. Always test in a controlled environment before production deployment.
