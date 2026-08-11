#!/usr/bin/env python3
"""
Re-runs the live demo's attack scenario (v2 audit, Finding #12 fix
verification) with process_monitor.py's own-PID exclusion in place, to
pair against the negative control re-run in results/negative_control.json.

Mirrors demo/DEMO_RUNBOOK.md's live sequence: start the monitor, let it
run for a few seconds, then launch the attack simulator partway through
the same 30s+ window, and record the classification at the end.

Run from the project root:  .venv/Scripts/python.exe scripts/finding12_attack_verification.py
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

WINDOW_SECONDS = 35
ATTACK_AT_SECOND = 5  # matches DEMO_RUNBOOK.md: "a few seconds into the window"


def main():
    print(f"\n{'=' * 70}\n FINDING #12 FIX VERIFICATION - ATTACK RUN\n{'=' * 70}\n")

    print("[SETUP] Recreating demo/victim_files/")
    subprocess.run([sys.executable, "demo/simulate_ransomware_behavior.py", "--setup"], check=True)

    from file_monitor import start as start_file_monitor
    from process_monitor import start as start_process_monitor
    from network_monitor import start as start_network_monitor
    from registry_monitor import RegistryMonitor
    from event_collector import collector

    target_folder = os.path.join(ROOT, "demo", "victim_files")

    collector.auto_classify = False
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

    print(f"[START] Monitoring {target_folder}. Attack simulator launches at t={ATTACK_AT_SECOND}s. "
          f"Total window: {WINDOW_SECONDS}s.")
    for t in threads:
        t.start()
    registry_monitor.start()

    attack_launched = False
    for remaining in range(WINDOW_SECONDS, 0, -1):
        elapsed = WINDOW_SECONDS - remaining
        if elapsed >= ATTACK_AT_SECOND and not attack_launched:
            print(f"\n[ATTACK] Launching simulate_ransomware_behavior.py --attack at t={elapsed}s")
            subprocess.run([sys.executable, "demo/simulate_ransomware_behavior.py", "--attack"], check=True)
            attack_launched = True
            print()
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
        "generated_by": "scripts/finding12_attack_verification.py",
        "purpose": (
            "Pairs with results/negative_control.json (re-run after the "
            "Finding #12 fix) to test whether excluding the monitor's own "
            "PID from process_monitor.py resampling restores a genuine "
            "dual-model detection result during an actual attack."
        ),
        "fix_applied": "process_monitor.py excludes MY_PID (os.getpid()) from all resampling",
        "window_seconds": WINDOW_SECONDS,
        "attack_launched_at_second": ATTACK_AT_SECOND,
        "target_folder": target_folder,
        "attack_simulator_invoked": True,
        "total_raw_events_collected": collector.get_total_events(),
        "classification_result": classification,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/finding12_attack_verification.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\n[SAVED] results/finding12_attack_verification.json")

    print("\n[CLEANUP] Removing demo/victim_files/")
    subprocess.run([sys.executable, "demo/simulate_ransomware_behavior.py", "--cleanup"], check=True)


if __name__ == "__main__":
    main()
