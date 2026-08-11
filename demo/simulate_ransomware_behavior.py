#!/usr/bin/env python3
"""
WatchRansom DEMO ONLY - safe ransomware-behavior simulator.

Does NOT encrypt, damage, or touch anything outside its own dedicated demo
victim folder (demo/victim_files/, created and populated by this script).
No real cryptography is used to lock anything - files are just overwritten
with random bytes and renamed with a ".locked" extension, then the process
exits. This exists purely to generate REAL, live behavioral signal (mass
file creation/read, directory enumeration, crypto-module loading) so the
actual WatchRansom monitor + ML pipeline can detect it end-to-end during a
live demo, without running anything that resembles real malware.

Usage:
    python demo/simulate_ransomware_behavior.py --setup     # create victim files (run once, before the demo)
    python demo/simulate_ransomware_behavior.py --attack     # run the simulated "attack" (run live, during the demo)
    python demo/simulate_ransomware_behavior.py --cleanup    # delete the demo victim folder afterward
"""

import os
import sys
import time
import random
import string
import argparse
import hashlib          # real crypto-API module load -> real 'apistats' signal
import shutil

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "victim_files")
N_FILES = 400
N_SUBDIRS = 8


def setup():
    print(f"[SETUP] Creating {N_FILES} dummy victim files across {N_SUBDIRS} subfolders in:")
    print(f"        {DEMO_DIR}")

    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    subdirs = []
    for i in range(N_SUBDIRS):
        d = os.path.join(DEMO_DIR, f"folder_{i}")
        os.makedirs(d, exist_ok=True)
        subdirs.append(d)

    for i in range(N_FILES):
        target_dir = subdirs[i % N_SUBDIRS]
        path = os.path.join(target_dir, f"document_{i}.txt")
        content = "".join(random.choices(string.ascii_letters + " \n", k=200))
        with open(path, "w") as f:
            f.write(content)

    print(f"[SETUP] Done. {N_FILES} files ready across {N_SUBDIRS} folders.")
    print("[SETUP] Point WatchRansom's monitor_folders at this demo/ directory "
          "(or its parent) before starting the monitor.")


def attack():
    if not os.path.exists(DEMO_DIR):
        print("[ERROR] Victim files not found - run with --setup first.")
        sys.exit(1)

    print("[ATTACK] Simulated ransomware behavior starting...")
    print("[ATTACK] (This is a DEMO - only touches files inside demo/victim_files/)")
    time.sleep(1)

    # Genuine crypto-API-capable module usage - triggers the real 'apistats'
    # proxy in process_monitor.py (crypto DLL load detection), not simulated.
    hasher = hashlib.sha256()

    all_files = []
    for root, _, files in os.walk(DEMO_DIR):
        for fname in files:
            all_files.append(os.path.join(root, fname))

    random.shuffle(all_files)
    print(f"[ATTACK] Enumerating {len(all_files)} files across {N_SUBDIRS} directories...")

    count = 0
    for path in all_files:
        try:
            # "Read" the original file
            with open(path, "rb") as f:
                data = f.read()
            hasher.update(data)

            # Overwrite the ORIGINAL file in place - this is what actually
            # triggers a watchdog "modified" event -> file_read signal. (A
            # plain read() alone does not touch mtime and produces no
            # filesystem notification at all - confirmed via end-to-end
            # testing before the demo. Note: event_aggregator.py also does
            # not currently map watchdog "renamed"/on_moved events to any
            # feature, so a plain rename() would not produce file_created
            # signal either - writing a genuinely new file, below, is what's
            # needed.)
            fake_encrypted = bytes(random.getrandbits(8) for _ in range(len(data) or 32))
            with open(path, "wb") as f:
                f.write(fake_encrypted)

            # Create a new .locked file (real file_created signal) and
            # remove the now-encrypted original.
            locked_path = path + ".locked"
            with open(locked_path, "wb") as f:
                f.write(fake_encrypted)
            os.remove(path)

            count += 1
            if count % 25 == 0:
                print(f"[ATTACK] ...{count}/{len(all_files)} files processed")
        except Exception as e:
            continue

    print(f"[ATTACK] Done. {count} files 'encrypted' (simulation only, demo/victim_files/).")
    print(f"[ATTACK] Digest (proof real crypto module ran): {hasher.hexdigest()[:16]}...")


def cleanup():
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
        print(f"[CLEANUP] Removed {DEMO_DIR}")
    else:
        print("[CLEANUP] Nothing to clean up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WatchRansom demo attack simulator (safe, self-contained)")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--attack", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.attack:
        attack()
    elif args.cleanup:
        cleanup()
    else:
        parser.print_help()
