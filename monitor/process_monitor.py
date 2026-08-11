import psutil
import time

from event_collector import collector


# ==========================================================
# PROCESS MONITOR
# ==========================================================

# DLLs commonly used for cryptographic / encryption API calls.
# Used as a live-host proxy for the "apistats" feature, which in the
# original Cuckoo Sandbox training data counted actual API call
# volume from in-sandbox hooking (not reproducible on a live host
# without kernel-level API hooking). See bugs_debugs.txt BUG #1.
CRYPTO_API_DLLS = (
    "advapi32.dll",
    "bcrypt.dll",
    "bcryptprimitives.dll",
    "crypt32.dll",
    "ncrypt.dll",
    "wintrust.dll",
)


def _get_dll_loaded_count(proc):
    """
    Count DLLs/modules loaded by a process (Windows).
    Used for the 'dll_loaded' feature.
    """
    try:
        return len(proc.memory_maps())
    except Exception:
        return 0


MAX_TREE_DEPTH = 5


def _get_tree_command_line_length(proc):
    """
    Proxy for 'tree_command_line': walk up the process's ancestor chain
    (parent, grandparent, ...) up to MAX_TREE_DEPTH levels, summing the
    character length of each ancestor's command line. A process spawned
    by a long, complex chain of ancestor commands (a common pattern for
    obfuscated/staged locker ransomware - see MALICIOUS_ACTIVITY_TRIGGERS.txt)
    produces a larger value than a process launched directly by the shell
    or a normal parent application.

    Uses only psutil.Process.parent()/.cmdline() (cheap metadata lookups,
    no memory_maps()/open_files()), so - unlike directory_enumerated
    (BUG #9) - this is safe to compute for every newly-created process
    without a per-cycle cost problem; it is still only computed once, at
    process creation, matching how the plain 'command_line' feature is
    already handled.
    """
    total = 0
    try:
        current = proc.parent()
        depth = 0
        while current is not None and depth < MAX_TREE_DEPTH:
            try:
                cmdline = current.cmdline()
                total += len(" ".join(cmdline)) if cmdline else 0
            except Exception:
                pass
            try:
                current = current.parent()
            except Exception:
                break
            depth += 1
    except Exception:
        pass
    return total


def _get_crypto_api_proxy(proc):
    """
    Count loaded modules that match known cryptographic API DLLs, as a
    proxy for 'apistats' (cryptographic API call volume). This is an
    approximation: it detects that crypto-capable modules are loaded,
    not the actual number of API calls made.
    """
    try:
        maps = proc.memory_maps()
    except Exception:
        return 0

    count = 0
    for m in maps:
        path_lower = m.path.lower() if m.path else ""
        if any(dll in path_lower for dll in CRYPTO_API_DLLS):
            count += 1
    return count


def _get_directory_enumerated_proxy(proc):
    """
    Count distinct parent directories among a process's currently open
    files, as a proxy for 'directory_enumerated' (directory scanning /
    reconnaissance behavior).

    COST WARNING (see bugs_debugs.txt BUG #9): benchmarked at ~148ms
    average per process on this machine (psutil.Process.open_files() is
    expensive on Windows - it must query and resolve every open file
    handle a process owns). A full pass across ~266 processes took
    ~18-52s, 5-25x longer than the entire poll interval it would need to
    run within. Because of this, this function is ONLY called once, at
    process creation (see the `is_new` branch below) - it must NOT be
    called from the continuous per-cycle resampling loop the way
    dll_loaded/apistats are (those use the much cheaper memory_maps(),
    ~18ms/process).
    """
    try:
        open_files = proc.open_files()
    except Exception:
        return 0

    dirs = set()
    for f in open_files:
        try:
            dirs.add(f.path.rsplit("\\", 1)[0].rsplit("/", 1)[0])
        except Exception:
            continue
    return len(dirs)

def start():

    print("\n===================================")
    print(" Process Monitor Started")
    print("===================================\n")

    # Initialize with all currently running processes to avoid a startup false-positive surge.
    # Maps pid -> last-observed (dll_loaded, apistats, directory_enumerated) snapshot, so that
    # for processes we already reported as "created" we can emit only the DELTA in follow-up
    # cycles, instead of never re-sampling them again.
    #
    # Why this matters (see bugs_debugs.txt BUG #3): dll_loaded/apistats/directory_enumerated
    # were previously sampled exactly once, at the instant a process is first observed - often
    # before it has finished loading its own dependencies or opened any target files. A
    # long-running process (which is exactly what a real ransomware payload looks like: one
    # process performing thousands of file/API operations over time) would contribute almost
    # nothing on these three features. Re-sampling tracked processes every cycle and emitting
    # deltas lets these features actually grow as a process's real behavior escalates.
    seen_processes = {}
    for pid in psutil.pids():
        seen_processes[pid] = (0, 0, 0)

    # Resampling already-tracked processes (memory_maps() per process, see BUG #3) is
    # real cost multiplied by total process count. Benchmarked on this machine (266
    # processes): memory_maps() alone averages ~18ms/process (~4.7s per full pass);
    # process_iter() itself costs ~2.2s. Resampling every 4th cycle (~8s) keeps a full
    # pass comfortably under its own interval even in the worst case (~6.9s used of an
    # 8s budget), while still giving ~3-4 delta samples across the 30s aggregation
    # window - a large improvement over the original once-at-birth behavior (BUG #3).
    # New-process detection (the `is_new` branch below) still runs every cycle
    # regardless, since catching a new process promptly matters more than resampling
    # cadence for existing ones. See bugs_debugs.txt BUG #7 and BUG #9 (BUG #9 is why
    # this is every-4th-cycle and not every-other as BUG #7 originally set it, and why
    # directory_enumerated/open_files() is NOT part of this resampling loop at all -
    # see _get_directory_enumerated_proxy()'s docstring above).
    cycle_count = 0

    while True:

        try:

            cycle_count += 1
            resample_tracked = (cycle_count % 4 == 0)

            for proc in psutil.process_iter(
                [
                    "pid",
                    "name",
                    "cmdline",
                    "ppid",
                    "memory_percent"
                ]
            ):

                try:

                    pid = proc.info["pid"]
                    is_new = pid not in seen_processes

                    if is_new:
                        process_name = proc.info["name"] or "Unknown"
                        parent_pid = proc.info["ppid"]

                        command_line = " ".join(
                            proc.info["cmdline"]
                        ) if proc.info["cmdline"] else ""

                        arguments = (
                            len(proc.info["cmdline"]) - 1
                            if proc.info["cmdline"]
                            else 0
                        )

                        try:
                            parent_name = psutil.Process(parent_pid).name()
                        except:
                            parent_name = "Unknown"

                        try:
                            # Non-blocking: interval=0.1 previously stalled this loop for
                            # 100ms per newly-seen process (serially - a burst of 50 new
                            # processes added 5s of pure blocking per poll cycle). cpu_usage
                            # is informational only (event_aggregator.py never reads it - it
                            # is not one of the 12 trained features), so an instantaneous,
                            # occasionally-0.0 reading is an acceptable trade for removing
                            # that blocking cost. See bugs_debugs.txt BUG #7.
                            cpu_usage = proc.cpu_percent(interval=None)
                        except:
                            cpu_usage = 0

                        try:
                            memory_usage = proc.info["memory_percent"]
                        except:
                            memory_usage = 0

                        dll_loaded = _get_dll_loaded_count(proc)
                        apistats = _get_crypto_api_proxy(proc)
                        directory_enumerated = _get_directory_enumerated_proxy(proc)
                        tree_command_line = _get_tree_command_line_length(proc)

                        seen_processes[pid] = (dll_loaded, apistats, directory_enumerated)

                        collector.add_event(

                            source="process",

                            event="created",

                            process=process_name,

                            pid=pid,

                            details={

                                "parent_pid": parent_pid,

                                "parent_name": parent_name,

                                "command_line": command_line,

                                "arguments": arguments,

                                "cpu_usage": round(cpu_usage, 2),

                                "memory_usage": round(memory_usage, 2),

                                "dll_loaded": dll_loaded,

                                "apistats": apistats,

                                "directory_enumerated": directory_enumerated,

                                "tree_command_line": tree_command_line

                            }

                        )

                    elif resample_tracked:
                        # Already-tracked process: re-sample and emit only the increase since
                        # last cycle, so ongoing behavior (e.g. mass encryption in a single
                        # long-lived process) is actually captured over time.
                        #
                        # directory_enumerated is deliberately NOT resampled here - see BUG #9:
                        # its underlying open_files() call is ~8x more expensive than
                        # memory_maps() (benchmarked ~148ms vs ~18ms/process) and made a full
                        # resampling pass take 5-25x longer than the poll interval itself. It
                        # keeps its BUG #3 birth-time-only limitation for the process-level
                        # signal; a second, cost-free source (distinct directories touched,
                        # from file_monitor.py's watchdog events) supplements it - see
                        # file_monitor.py and event_aggregator.py.
                        prev_dll, prev_api, prev_dirs = seen_processes[pid]

                        dll_loaded = _get_dll_loaded_count(proc)
                        apistats = _get_crypto_api_proxy(proc)

                        delta_dll = max(0, dll_loaded - prev_dll)
                        delta_api = max(0, apistats - prev_api)

                        seen_processes[pid] = (dll_loaded, apistats, prev_dirs)

                        if delta_dll or delta_api:
                            process_name = proc.info["name"] or "Unknown"

                            collector.add_event(

                                source="process",

                                event="activity",

                                process=process_name,

                                pid=pid,

                                details={

                                    "dll_loaded": delta_dll,

                                    "apistats": delta_api

                                }

                            )

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess
                ):
                    continue

                except Exception:
                    continue

            # Drop tracking state for processes that have exited, so the dict doesn't grow
            # unbounded over a long monitoring session.
            current_pids = set(psutil.pids())
            for pid in list(seen_processes.keys()):
                if pid not in current_pids:
                    del seen_processes[pid]

            time.sleep(2)

        except KeyboardInterrupt:

            print("\nProcess Monitor Stopped.")

            break

        except Exception as e:

            print(f"Process Monitor Error: {e}")

            time.sleep(2)