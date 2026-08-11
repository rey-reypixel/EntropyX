from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from event_collector import collector
import time
import os


# ==========================================================
# FILE SYSTEM EVENT HANDLER
# ==========================================================

import json

def load_config():
    """
    Load config from JSON file to get ignore patterns and log files.
    """
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load config in file monitor: {e}")
    return {}


class FileMonitorHandler(FileSystemEventHandler):
    def __init__(self, ignore_patterns=None, log_files=None):
        super().__init__()
        self.ignore_patterns = ignore_patterns or []
        self.log_files = log_files or []

        # Normalize and add standard ignore patterns to prevent self-monitoring
        self.ignore_patterns.extend([".git", "__pycache__", "node_modules", "monitor/logs", "monitor\\logs"])

        # Cost-free supplementary signal for "directory_enumerated" (see
        # bugs_debugs.txt BUG #9). The process-level open_files()-based proxy in
        # process_monitor.py is too expensive to poll continuously, so it only
        # samples at process birth. This tracks distinct directories touched by
        # real (watchdog, event-driven - no polling cost) file activity within
        # the monitored folder, which is exactly the kind of signal ransomware
        # reconnaissance/scanning would produce, and adds it as a second source
        # for the same feature. Grows for the life of the monitoring session
        # (a directory counts as "new" once, not once per window) - a burst of
        # activity across many previously-untouched directories still produces
        # a large signal within whatever window it happens in.
        self.seen_dirs = set()

    def _is_new_directory(self, file_path):
        directory = os.path.dirname(os.path.abspath(file_path))
        if directory in self.seen_dirs:
            return False
        self.seen_dirs.add(directory)
        return True

    def should_ignore(self, path):
        if not path:
            return True
            
        # Normalize path separators
        norm_path = os.path.abspath(path).replace('\\', '/')
        
        # Check against configured log files
        for log_file in self.log_files:
            abs_log = os.path.abspath(log_file).replace('\\', '/')
            if abs_log in norm_path:
                return True
                
        # Check against ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in norm_path or pattern.replace('\\', '/') in norm_path:
                return True
                
        return False

    def on_created(self, event):
        if event.is_directory or self.should_ignore(event.src_path):
            return

        collector.add_event(
            source="file",
            event="created",
            process="Unknown",
            pid=-1,
            details={
                "path": event.src_path,
                "new_directory": 1 if self._is_new_directory(event.src_path) else 0
            }
        )

    def on_modified(self, event):
        if event.is_directory or self.should_ignore(event.src_path):
            return

        collector.add_event(
            source="file",
            event="modified",
            process="Unknown",
            pid=-1,
            details={
                "path": event.src_path,
                "new_directory": 1 if self._is_new_directory(event.src_path) else 0
            }
        )

    def on_deleted(self, event):
        if event.is_directory or self.should_ignore(event.src_path):
            return

        collector.add_event(
            source="file",
            event="deleted",
            process="Unknown",
            pid=-1,
            details={
                "path": event.src_path
            }
        )

    def on_moved(self, event):
        if event.is_directory or self.should_ignore(event.src_path) or self.should_ignore(event.dest_path):
            return

        collector.add_event(
            source="file",
            event="renamed",
            process="Unknown",
            pid=-1,
            details={
                "old_path": event.src_path,
                "new_path": event.dest_path
            }
        )


# ==========================================================
# START FILE MONITOR
# ==========================================================

def start(folder_path):

    if not os.path.exists(folder_path):

        print(f"[ERROR] Folder not found:\n{folder_path}")
        return

    print("\n===================================")
    print(" File Monitor Started")
    print("===================================")
    print(folder_path)
    print()

    # Load config settings
    config = load_config()
    file_monitor_cfg = config.get("monitoring", {}).get("file_monitor", {})
    ignore_patterns = file_monitor_cfg.get("ignore_patterns", [])
    
    # Identify configured log file locations to ignore
    log_files = []
    logging_config = config.get("logging", {})
    for key, val in logging_config.items():
        if isinstance(val, str):
            log_files.append(val)
            
    # Include default log files
    log_files.extend([
        "monitor/logs/events.csv",
        "monitor/logs/alerts.csv",
        "monitor/logs/alerts.json",
        "monitor/logs/network_log.csv",
        "monitor/logs/process_log.csv"
    ])

    event_handler = FileMonitorHandler(ignore_patterns=ignore_patterns, log_files=log_files)

    observer = Observer()

    observer.schedule(
        event_handler,
        folder_path,
        recursive=file_monitor_cfg.get("recursive", True)
    )

    observer.start()

    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        observer.stop()

    observer.join()