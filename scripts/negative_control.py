#!/usr/bin/env python3
"""
Negative control run (v2 audit task #9, item 2 - tests B8).

demo/DEMO_RUNBOOK.md and please_study_this.txt report that during the
live demo, the autoencoder fired CRITICAL (anomaly_score = 1.0) in all 3
pre-demo test runs, and treat this as evidence the autoencoder
independently caught the simulated attack when the classifier missed it.

But bugs_debugs.txt FINDING #12 also documents that process_monitor.py
is system-wide and observes the monitoring process's OWN heavy ML
library imports (dll_loaded ~11,000+), and detect_anomaly() scales
features through a StandardScaler fitted on goodware - an extreme
single-feature outlier can saturate the reconstruction error regardless
of whether an actual attack occurred. Nothing in the original demo
testing ever checked what the anomaly score looks like with NO attack
running, under the same conditions (same folder, same monitor-self-noise).
This script is that check.

Setup: runs demo/simulate_ransomware_behavior.py --setup (creates the
same demo/victim_files folder the real demo monitors) for environment
parity with the original 3 demo runs, then starts the real monitor
stack (file/process/network/registry) pointed at it - but never calls
--attack. Ambient system activity (this machine's own background
processes, this very Python process, OS churn) is what generates events;
nothing is deliberately touched inside demo/victim_files.

Run from the project root:  .venv/Scripts/python.exe scripts/negative_control.py
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "monitor"))

WINDOW_SECONDS = 35  # 5s buffer over the 30s aggregation window


def main():
    print(f"\n{'=' * 70}\n NEGATIVE CONTROL RUN (no attack simulator invoked)\n{'=' * 70}\n")

    print("[SETUP] Recreating demo/victim_files/ (same folder the real demo "
          "monitors) for environment parity - no attack will be run against it.")
    subprocess.run(
        [sys.executable, "demo/simulate_ransomware_behavior.py", "--setup"],
        check=True,
    )

    from file_monitor import start as start_file_monitor
    from process_monitor import start as start_process_monitor
    from network_monitor import start as start_network_monitor
    from registry_monitor import RegistryMonitor
    from event_collector import collector

    target_folder = os.path.join(ROOT, "demo", "victim_files")

    collector.auto_classify = False  # we'll classify manually, once, at the end
    collector.event_count = 0
    if collector.enable_ml and collector.inference_engine:
        collector.inference_engine.aggregator.reset_window()

    threads = [
        threading.Thread(target=start_file_monitor, args=(target_folder,),
                          name="FileMonitor-victim_files", daemon=True),
        threading.Thread(target=start_process_monitor, name="ProcessMonitor", daemon=True),
        threading.Thread(target=start_network_monitor, name="NetworkMonitor", daemon=True),
    ]
    registry_monitor = RegistryMonitor(poll_interval=5)

    print(f"[START] Monitoring {target_folder} (file) + system-wide process/network/registry")
    print(f"[START] NO attack simulator will be invoked. Collecting for {WINDOW_SECONDS}s...")
    for t in threads:
        t.start()
    registry_monitor.start()

    for remaining in range(WINDOW_SECONDS, 0, -1):
        sys.stdout.write(f"\r[COLLECTING] {remaining:2d}s remaining | events so far: {collector.get_total_events()}")
        sys.stdout.flush()
        time.sleep(1)
    print(f"\r[COLLECTING] done. Total events: {collector.get_total_events()}          ")

    registry_monitor.stop()

    classification = None
    if collector.enable_ml and collector.inference_engine:
        classification = collector.inference_engine.classify_aggregated()

    print(f"\n{'=' * 70}\n RESULT\n{'=' * 70}\n")
    print(json.dumps(classification, indent=2))

    result = {
        "generated_by": "scripts/negative_control.py",
        "purpose": (
            "Test whether the demo's reported anomaly_score=1.0 reflects "
            "genuine attack detection or self-noise (v2 audit finding B8), "
            "by running the same monitor stack against the same folder "
            "with no attack invoked."
        ),
        "window_seconds": WINDOW_SECONDS,
        "target_folder": target_folder,
        "attack_simulator_invoked": False,
        "total_raw_events_collected": collector.get_total_events(),
        "classification_result": classification,
    }

    if classification is not None:
        anomaly_score = classification.get("anomaly_score")
        is_anomaly = classification.get("is_anomaly")
        result["summary"] = (
            f"label={classification.get('label')} "
            f"confidence={classification.get('confidence'):.4f} "
            f"anomaly_score={anomaly_score} is_anomaly={is_anomaly}"
        )
        if is_anomaly:
            result["interpretation"] = (
                "is_anomaly=True with NO attack running. This supports B8: "
                "the demo's anomaly-path CRITICAL alert is plausibly "
                "self-noise (this monitoring process's own heavy ML "
                "library imports), not attack detection. The dual-model "
                "narrative needs re-examination."
            )
        else:
            result["interpretation"] = (
                "is_anomaly=False with no attack running - the anomaly "
                "path did NOT fire under ambient conditions in this run. "
                "This is evidence AGAINST the self-noise explanation for "
                "THIS run, though a single run is not conclusive either "
                "way; repeat runs would strengthen the conclusion."
            )
    else:
        result["summary"] = "Not enough events collected to classify (window gate not reached)."
        result["interpretation"] = "Inconclusive - re-run with a longer window or busier ambient conditions."

    os.makedirs("results", exist_ok=True)
    with open("results/negative_control.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\n[SAVED] results/negative_control.json")

    print("\n[CLEANUP] Removing demo/victim_files/ (leaves no leftover demo state)")
    subprocess.run(
        [sys.executable, "demo/simulate_ransomware_behavior.py", "--cleanup"],
        check=True,
    )


if __name__ == "__main__":
    main()
