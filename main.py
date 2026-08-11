#!/usr/bin/env python3
"""
WatchRansom - Ransomware Detection and Prevention System
Main entry point for the application
"""

import sys
import os
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import monitor.start_monitoring


# ==========================================================
# CONFIGURATION LOADER
# ==========================================================

def load_config(config_path="config.json"):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "monitoring": {
            "enabled_monitors": ["file", "process", "network"],
            "monitor_folder": "test_folder"
        },
        "ml": {
            "enabled": True
        },
        "alerts": {
            "enabled": True
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                print(f"[CONFIG] Loaded configuration from {config_path}")
                return config
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}")
            print("[CONFIG] Using default configuration")
    
    return default_config


# ==========================================================
# MAIN ENTRY POINT
# ==========================================================

def main():
    """
    Main entry point for WatchRansom.
    """
    parser = argparse.ArgumentParser(
        description="WatchRansom - Ransomware Detection and Prevention System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Start with default settings
  python main.py --folder C:\\test        # Monitor specific folder
  python main.py --no-ml                  # Disable ML classification
  python main.py --config custom.json     # Use custom config
        """
    )
    
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder path to monitor (overrides config)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Disable ML classification (data collection mode only)"
    )
    
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Disable alert system"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    folders_to_monitor = None
    if args.folder:
        folders_to_monitor = [args.folder]
    elif "monitor_folders" in config["monitoring"]:
        folders_to_monitor = config["monitoring"]["monitor_folders"]
    elif "monitor_folder" in config["monitoring"]:
        folders_to_monitor = [config["monitoring"]["monitor_folder"]]
    
    if args.no_ml:
        config["ml"]["enabled"] = False
    
    if args.no_alerts:
        config["alerts"]["enabled"] = False
    
    # Print startup banner
    print("\n" + "="*70)
    print(" " * 15 + "WATCHRANSOM DETECTION SYSTEM")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Folders to Monitor: {len(folders_to_monitor) if folders_to_monitor else 1}")
    if folders_to_monitor:
        for folder in folders_to_monitor:
            print(f"    - {folder}")
    print(f"  ML Classification: {'ENABLED' if config['ml']['enabled'] else 'DISABLED'}")
    print(f"  Alert System: {'ENABLED' if config['alerts']['enabled'] else 'DISABLED'}")
    print(f"  Active Monitors: {', '.join(config['monitoring']['enabled_monitors'])}")
    print("\n" + "="*70 + "\n")
    
    # Start monitoring with folders
    try:
        monitor.start_monitoring.main(folders=folders_to_monitor)
    except KeyboardInterrupt:
        print("\nShutdown complete.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
