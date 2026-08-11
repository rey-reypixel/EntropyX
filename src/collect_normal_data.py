#!/usr/bin/env python3
"""
Collect normal Windows activity data for ML training.
This script processes existing event logs and aggregates them into training format.
"""

import sys
import os
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================================
# PROCESS EVENT LOGS INTO TRAINING FORMAT
# ==========================================================

def _empty_feature_counts():
    return {
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


def _apply_event_to_counts(feature_counts, source, event_type, details):
    """
    Update feature_counts in place based on a single event. Mirrors the
    live aggregation logic in src/event_aggregator.py so offline
    reprocessing and live inference stay consistent.
    """
    if source == 'file':
        if event_type == 'created':
            feature_counts["file_created"] += 1
        elif event_type == 'modified':
            feature_counts["file_read"] += 1
        if details.get('new_directory'):
            feature_counts["directory_enumerated"] += 1
        # "deleted" is intentionally not counted - see bugs_debugs.txt BUG #2

    elif source == 'process':
        # Only "created" events represent an actual new process; "activity"
        # events (delta re-sampling, see bugs_debugs.txt BUG #3) must not
        # inflate proc_pid.
        if event_type == 'created':
            feature_counts["proc_pid"] += 1
        if 'command_line' in details:
            feature_counts["command_line"] += len(str(details['command_line']))
        if 'tree_command_line' in details:
            try:
                feature_counts["tree_command_line"] += int(details['tree_command_line'])
            except (ValueError, TypeError):
                pass
        if 'arguments' in details:
            try:
                feature_counts["arguments"] += int(details['arguments'])
            except (ValueError, TypeError):
                pass
        if 'dll_loaded' in details:
            try:
                feature_counts["dll_loaded"] += int(details['dll_loaded'])
            except (ValueError, TypeError):
                pass
        if 'apistats' in details:
            try:
                feature_counts["apistats"] += int(details['apistats'])
            except (ValueError, TypeError):
                pass
        if 'directory_enumerated' in details:
            try:
                feature_counts["directory_enumerated"] += int(details['directory_enumerated'])
            except (ValueError, TypeError):
                pass

    elif source == 'network':
        feature_counts["urls"] += 1

    elif source == 'registry':
        if event_type == 'written':
            feature_counts["regkey_written"] += 1
        elif event_type == 'read':
            feature_counts["regkey_read"] += 1


def process_event_log_to_training_format(event_log_path="monitor/logs/events.csv",
                                         output_file="datasets/normal_activity.csv",
                                         window_seconds=30):
    """
    Process collected event logs and aggregate into training format.

    Streams the event log row by row instead of loading it into memory
    (see bugs_debugs.txt BUG #4 - the log can be gigabytes in size after
    a long monitoring session, and a full in-memory list of every event
    caused a MemoryError). Assumes event_log_path is written in
    chronological order, which holds here since monitor/event_collector.py
    appends events as they occur from a single writer.

    Args:
        event_log_path: Path to events.csv file
        output_file: Where to save the processed training data
        window_seconds: Time window for aggregation (default 30 seconds)
    """

    print(f"\n{'='*60}")
    print(f" PROCESSING EVENT LOGS TO TRAINING FORMAT")
    print(f"{'='*60}")
    print(f"Input: {event_log_path}")
    print(f"Output: {output_file}")
    print(f"Window: {window_seconds} seconds")
    print(f"{'='*60}\n")

    if not os.path.exists(event_log_path):
        print(f"[ERROR] Event log not found: {event_log_path}")
        print("Run the monitoring system first to collect events.")
        return

    print("Streaming event log (single pass, chronological order assumed)...")

    collected_samples = []
    feature_counts = _empty_feature_counts()
    current_window_end = None
    event_count = 0
    total_events = 0
    skipped = 0

    with open(event_log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_events += 1

            try:
                timestamp = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue

            if current_window_end is None:
                current_window_end = timestamp + timedelta(seconds=window_seconds)

            if timestamp >= current_window_end:
                # Close out the current window
                if event_count > 0:
                    sample = {"family": "G"}
                    sample.update(feature_counts)
                    collected_samples.append(sample)
                    if len(collected_samples) % 100 == 0:
                        print(f"  ...{len(collected_samples)} windows aggregated so far "
                              f"({total_events} raw events read)")

                # Start a fresh window beginning at this event's timestamp
                feature_counts = _empty_feature_counts()
                event_count = 0
                current_window_end = timestamp + timedelta(seconds=window_seconds)

            source = row.get('source', '')
            event_type = row.get('event', '')
            try:
                details = json.loads(row.get('details', '{}'))
            except (json.JSONDecodeError, TypeError):
                details = {}

            _apply_event_to_counts(feature_counts, source, event_type, details)
            event_count += 1

    if skipped:
        print(f"[WARNING] Skipped {skipped} rows with unparseable timestamps")

    print(f"Total events read: {total_events}")

    # Save last window if it has events
    if event_count > 0:
        sample = {"family": "G"}
        sample.update(feature_counts)
        collected_samples.append(sample)

    print(f"Total aggregated windows: {len(collected_samples)}")

    if not collected_samples:
        print("[ERROR] No valid windows produced from log file.")
        return

    # Save collected data
    if collected_samples:
        df = pd.DataFrame(collected_samples)
        
        # Check if file exists to append or create new
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            print(f"\n[SAVED] Appended {len(collected_samples)} samples to {output_file}")
            print(f"[TOTAL] {len(combined_df)} samples in dataset")
        else:
            df.to_csv(output_file, index=False)
            print(f"\n[SAVED] {len(collected_samples)} samples to {output_file}")
    else:
        print("\n[WARNING] No samples collected from event log.")
    
    print(f"\n{'='*60}")
    print(" PROCESSING COMPLETE")
    print(f"{'='*60}")


# ==========================================================
# SIMPLIFIED COLLECTION (RUN MONITORING FIRST)
# ==========================================================

def collect_normal_activity(duration_minutes=30, output_file="datasets/normal_activity.csv"):
    """
    Instructions for collecting normal activity.
    """
    
    print(f"\n{'='*60}")
    print(f" NORMAL ACTIVITY DATA COLLECTION")
    print(f"{'='*60}")
    print(f"\nStep-by-step instructions:")
    print(f"\n1. Run the monitoring system:")
    print(f"   python main.py --folder test_folder --no-ml")
    print(f"\n2. Use your computer normally for {duration_minutes} minutes")
    print(f"   (browse web, open files, run normal applications)")
    print(f"\n3. Stop the monitoring system (Ctrl+C)")
    print(f"\n4. Process the collected events:")
    print(f"   python src/collect_normal_data.py --process")
    print(f"\n5. This will convert events.csv to training format")
    print(f"{'='*60}\n")


# ==========================================================
# MERGE WITH EXISTING DATASET
# ==========================================================

def merge_datasets(normal_file="datasets/normal_activity.csv", 
                   ransomware_file="datasets/ransomware_dataset.csv",
                   output_file="datasets/balanced_dataset.csv"):
    """
    Merge collected normal activity with existing ransomware dataset.
    """
    
    print(f"\n{'='*60}")
    print(f" MERGING DATASETS")
    print(f"{'='*60}")
    
    # Load normal activity
    if not os.path.exists(normal_file):
        print(f"[ERROR] Normal activity file not found: {normal_file}")
        print("Run collect_normal_activity() first.")
        return
    
    normal_df = pd.read_csv(normal_file)
    print(f"Normal samples: {len(normal_df)}")
    
    # Load ransomware dataset
    if os.path.exists(ransomware_file):
        ransomware_df = pd.read_csv(ransomware_file)
        # Fix headers like in training script
        ransomware_df.columns = ransomware_df.iloc[0]
        ransomware_df = ransomware_df[1:]
        ransomware_df.reset_index(drop=True, inplace=True)
        
        # Select only the features we need
        selected_features = [
            "family", "proc_pid", "file_read", "file_created", 
            "regkey_written", "regkey_read", "apistats", "command_line",
            "directory_enumerated", "dll_loaded", "tree_command_line",
            "arguments", "urls"
        ]
        
        ransomware_df = ransomware_df[selected_features].copy()
        # Keep family column as string, convert other columns to numeric
        family_col = ransomware_df["family"]
        ransomware_df = ransomware_df.drop("family", axis=1)
        ransomware_df = ransomware_df.apply(pd.to_numeric, errors="coerce")
        ransomware_df["family"] = family_col
        ransomware_df["family"] = ransomware_df["family"].fillna("G")
        print(f"Ransomware samples: {len(ransomware_df)}")
    else:
        print(f"[WARNING] Ransomware dataset not found: {ransomware_file}")
        ransomware_df = pd.DataFrame()
    
    # Combine
    combined_df = pd.concat([normal_df, ransomware_df], ignore_index=True)
    print(f"Combined samples: {len(combined_df)}")
    
    # Show distribution
    print(f"\nClass distribution:")
    print(combined_df["family"].value_counts())
    
    # Save
    combined_df.to_csv(output_file, index=False)
    print(f"\n[SAVED] Balanced dataset to {output_file}")
    print(f"{'='*60}\n")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect normal Windows activity for ML training")
    parser.add_argument("--collect", action="store_true", help="Show instructions for collecting normal activity")
    parser.add_argument("--process", action="store_true", help="Process existing event logs into training format")
    parser.add_argument("--duration", type=int, default=30, help="Collection duration in minutes")
    parser.add_argument("--merge", action="store_true", help="Merge with existing dataset")
    parser.add_argument("--output", type=str, default="datasets/normal_activity.csv", help="Output file for normal data")
    
    args = parser.parse_args()
    
    if args.collect:
        collect_normal_activity(duration_minutes=args.duration, output_file=args.output)
    elif args.process:
        process_event_log_to_training_format(output_file=args.output)
    elif args.merge:
        merge_datasets(normal_file=args.output)
    else:
        print("Usage:")
        print("  python src/collect_normal_data.py --collect --duration 30")
        print("  python src/collect_normal_data.py --process")
        print("  python src/collect_normal_data.py --merge")
