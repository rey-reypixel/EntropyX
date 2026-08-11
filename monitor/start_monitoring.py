import threading
import sys
import os
import json

# Add monitor directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add project root to path for src.* imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_monitor import start as start_file_monitor
from process_monitor import start as start_process_monitor
from network_monitor import start as start_network_monitor
from registry_monitor import RegistryMonitor
from event_collector import collector
from alert_system import AlertLevel


def _load_config():
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ==========================================================
# MONITORING ORCHESTRATOR
# ==========================================================

def main(folders=None):
    """
    Main entry point to start all monitoring components in parallel.
    
    Args:
        folders: List of folder paths to monitor (if None, loads from config)
    """
    
    print("\n" + "="*60)
    print(" WATCHRANSOM - RANSOMWARE DETECTION SYSTEM")
    print("="*60)
    print("\nStarting monitoring components...\n")
    
    # Get folder paths to monitor
    if folders is None:
        # Try to load from config or use default
        if len(sys.argv) > 1:
            folders = [sys.argv[1]]
        else:
            # Default to test_folder if no config
            folders = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_folder")]
    
    # Create threads for each monitor
    threads = []
    
    # File Monitor Threads (one per folder)
    for folder_path in folders:
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder not found: {folder_path}")
            continue
        
        file_thread = threading.Thread(
            target=start_file_monitor,
            args=(folder_path,),
            name=f"FileMonitor-{os.path.basename(folder_path)}",
            daemon=True
        )
        threads.append(file_thread)
    
    # Process Monitor Thread
    process_thread = threading.Thread(
        target=start_process_monitor,
        name="ProcessMonitor",
        daemon=True
    )
    threads.append(process_thread)
    
    # Network Monitor Thread
    network_thread = threading.Thread(
        target=start_network_monitor,
        name="NetworkMonitor",
        daemon=True
    )
    threads.append(network_thread)
    
    # Registry Monitor (uses class-based approach)
    registry_monitor = RegistryMonitor(poll_interval=5)
    
    # Start all threads
    print("Starting monitor threads...\n")
    for thread in threads:
        thread.start()
        print(f"  [STARTED] {thread.name}")
    
    # Start registry monitor
    registry_monitor.start()
    print(f"  [STARTED] RegistryMonitor")
    
    print("\n" + "="*60)
    print(" ALL MONITORS ACTIVE")
    print("="*60)
    print(f"\nMonitoring {len(folders)} folder(s):")
    for folder in folders:
        print(f"  - {folder}")
    print("\nSystem-wide monitors:")
    print("  - Process Monitor")
    print("  - Network Monitor")
    print("  - Registry Monitor")
    print(f"\nEvent log: monitor/logs/events.csv")
    print("\nPress Ctrl+C to stop all monitors\n")
    print("="*60 + "\n")
    
    # Keep main thread alive
    import time
    try:
        # Disable auto classification during the 30-second loop
        collector.auto_classify = False

        # Early-warning indicators (see bugs_debugs.txt BUG #5 follow-up):
        # PreventionSystem.check_early_warning_indicators() existed but was
        # never called anywhere. Wired in here as a lightweight, non-
        # destructive heads-up check that runs independent of
        # prevention.enabled (it only depends on
        # prevention.early_warning_detection, true by default) - it never
        # terminates processes or touches the filesystem/network, it only
        # flags threshold-crossing behavior within the current scan window
        # before the ML classification result is available.
        prevention = None
        try:
            from src.prevention_system import PreventionSystem
            prevention = PreventionSystem(_load_config())
        except Exception as e:
            print(f"[WARNING] Early-warning system unavailable: {e}")

        while True:
            # Check if any thread has died
            for thread in threads:
                if not thread.is_alive():
                    print(f"[WARNING] {thread.name} has stopped unexpectedly")
            
            # Reset collector event count and aggregator before scanning
            collector.event_count = 0
            if collector.enable_ml and collector.inference_engine:
                collector.inference_engine.aggregator.reset_window()
            
            print("\n" + "="*60)
            print(" [+] WATCHRANSOM ACTIVE SCAN WINDOW STARTED")
            print("="*60)
            print("Collecting system activities for 30 seconds...")
            
            duration_seconds = 30
            for remaining in range(duration_seconds, 0, -1):
                sys.stdout.write(f"\r[SCANNING] {remaining:2d}s remaining | Collected Events: {collector.get_total_events()}")
                sys.stdout.flush()
                time.sleep(1)
                
            print("\r[SCANNING] Scan complete! Analyzing behavioral telemetry...      \n")

            # Early-warning check MUST run before classify_aggregated(), which
            # resets the aggregation window at the end of the call.
            if prevention is not None and collector.enable_ml and collector.inference_engine:
                early_warnings = prevention.check_early_warning_indicators(
                    collector.inference_engine.aggregator
                )
                if early_warnings:
                    print("[EARLY WARNING] Possible early-stage suspicious activity:")
                    for w in early_warnings:
                        print(f"  - {w}")
                    if collector.alert_system:
                        collector.alert_system.trigger_alert(
                            AlertLevel.WARNING,
                            {"source": "system", "event": "early_warning", "process": "Aggregated Scan", "pid": -1},
                            {"label": "EARLY_WARNING", "confidence": 0.0},
                            "; ".join(early_warnings)
                        )

            # Retrieve classification/anomaly results
            classification = None
            if collector.enable_ml and collector.inference_engine:
                classification = collector.inference_engine.classify_aggregated()
            
            if classification is None:
                print("\n" + "="*60)
                print(" [+] WATCHRANSOM SCAN COMPLETE")
                print("="*60)
                print(" Status: SAFE (No events collected)")
                print(" Anomaly Level: 0.00")
                print(" Recommendation: System is normal.")
                print("="*60 + "\n")
            else:
                label = classification.get("label", "Unknown")
                confidence = classification.get("confidence", 0.0)
                anomaly_score = classification.get("anomaly_score", 0.0)
                event_count = classification.get("window_stats", {}).get("event_count", 0)
                
                label_map = {"G": "Goodware (Normal)", "E": "Encryption Ransomware", "L": "Locker Ransomware"}
                friendly_label = label_map.get(label, label)
                
                status = "SAFE"
                if label in ["E", "L"] and confidence > 0.85:
                    status = "[CRITICAL] RANSOMWARE DETECTED"
                elif anomaly_score > 0.9:
                    status = "[CRITICAL] ZERO-DAY THREAT DETECTED"
                elif label in ["E", "L"] and confidence > 0.7:
                    status = "[WARNING] SUSPICIOUS ACTIVITY"
                elif anomaly_score > 0.7:
                    status = "[WARNING] ANOMALOUS BEHAVIOR"
                
                print("\n" + "="*60)
                print(" [+] WATCHRANSOM SCAN COMPLETE")
                print("="*60)
                print(f" Status: {status}")
                print(f" Events Analyzed: {event_count}")
                print(f" Classification: {friendly_label} ({confidence:.2%})")
                print(f" Anomaly Score: {anomaly_score:.4f}")
                print("="*60 + "\n")
                
                # Trigger alert system logs and prints
                if collector.alert_system:
                    summary_event = {
                        "source": "system",
                        "event": "scan_complete",
                        "process": "Aggregated Scan",
                        "pid": -1,
                        "details": {}
                    }
                    collector.alert_system.analyze_classification(summary_event, classification)
            
            # Prompt user to scan again
            try:
                choice = input("Do you want to run the scan again? (y/n): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                break
                
            if choice not in ["y", "yes"]:
                break
                
        print("\nExiting WatchRansom. Thank you!")
        
        # Stop registry monitor
        registry_monitor.stop()
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print(" SHUTTING DOWN MONITORS")
        print("="*60)
        print(f"\nTotal events collected: {collector.get_total_events()}")
        
        # Stop registry monitor
        registry_monitor.stop()
        
        print("Monitoring stopped successfully.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
