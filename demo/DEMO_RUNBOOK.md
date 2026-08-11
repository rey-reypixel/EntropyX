# WatchRansom Live Demo Runbook

## Setup (do this BEFORE the audience arrives, ~2 min)

1. Open two terminal windows side by side, both `cd` into the project root:
   ```
   cd "F:\Summer Internship\gravity_2.0\gravity_2"
   ```
2. In Terminal A, reset the demo victim files:
   ```
   python demo\simulate_ransomware_behavior.py --setup
   ```
   This creates 400 dummy files across 8 subfolders in `demo\victim_files\` -
   entirely self-contained, nothing outside this folder is touched.

## Live demo sequence

1. **Terminal A**: start the monitor pointed at the demo folder:
   ```
   python main.py --folder demo\victim_files
   ```
   Narrate while it starts: this loads the trained XGBoost classifier and
   autoencoder, then starts 4 real-time monitors (file, process, network,
   registry). It begins a 30-second collection window.

2. Wait until you see `[SCANNING] ... remaining | Collected Events: ...`
   counting down - this confirms it's live and actually watching the folder.

3. **Terminal B**, a few seconds into the window, launch the simulated attack:
   ```
   python demo\simulate_ransomware_behavior.py --attack
   ```
   Narrate: this is a self-contained simulator - no real cryptography, no
   real malware, it only touches its own dummy files inside
   `demo\victim_files\`. It reads each file, overwrites it in place, then
   writes an "encrypted" `.locked` copy and deletes the original - genuinely
   exercising the same file operations real ransomware performs, safely.

4. Let the 30-second window finish. WatchRansom will print a scan result.
   Point out on screen:
   - The alert level (CRITICAL) and label
   - "Anomaly Score" - the autoencoder's independent zero-day detection path
   - `monitor\logs\alerts.csv` / `alerts.json` - the alert is logged, not
     just printed

5. When prompted `Do you want to run the scan again? (y/n)`, type `n` to end
   the demo cleanly.

## After the demo

Clean up the generated files:
```
python demo\simulate_ransomware_behavior.py --cleanup
```

## What WILL actually happen (tested 3x before the demo, consistent result)

Tested end-to-end before the demo. Here is exactly what happens, honestly:

- **The autoencoder (anomaly detector) reliably fires**: anomaly_score = 1.0
  in all 3 pre-demo test runs, which crosses the 0.9 CRITICAL threshold. You
  WILL see a `[CRITICAL] ZERO-DAY THREAT DETECTED` message on screen.
- **The XGBoost classifier says "Goodware" (label G), not "Ransomware"**, in
  all 3 test runs. Diagnosed why: process_monitor.py is system-wide, not
  scoped to the demo folder - it also observes the WatchRansom monitoring
  process itself (which loads TensorFlow/XGBoost/scikit-learn, pulling in
  thousands of shared library handles). This pushes dll_loaded to ~11,000+
  in the aggregated window - more than 100x anything in the training data
  (even the highest-signal ransomware training sample topped out around
  109). The classifier has never seen a value like that for ANY class, so
  its output on this specific feature is effectively undefined/out-of-
  distribution, not a real judgment about the file activity.

**This is the demo's actual talking point, not a problem to hide**: it's a
live illustration of exactly the kind of "offline accuracy isn't the same
as live correctness" finding this project's engineering work centered on
(see bugs_debugs.txt) - and it's a genuine, unscripted demonstration of why
having a SECOND, independent detection path (the autoencoder) matters: the
system still catches and alerts on the attack even when one of its two
models is confused by an unrelated noise source. Say this plainly rather
than being caught off guard by it live.

## Talking points while it's running

- The system combines a supervised classifier (XGBoost, trained on a public
  ransomware-behavior dataset, 98.10% 10-fold cross-validated accuracy) with
  an unsupervised autoencoder trained only on normal activity - so it can
  flag activity that "looks wrong" even when the classifier is confused or
  the pattern doesn't match a known ransomware family.
- Watch for the CRITICAL alert to come from the anomaly score, not
  necessarily the classifier label - see note above, this is expected and
  is itself the interesting story.
- The bulk of this project's actual engineering work was auditing and fixing
  the gap between "the model looks accurate offline" and "the system
  actually detects things correctly when running live" - 10 real defects
  were found and fixed this way (full writeup in bugs_debugs.txt), and this
  demo is live evidence of exactly that kind of gap in action.

## If something goes wrong live

- If NEITHER path fires after 30s: say so honestly ("this run didn't cross
  the alert threshold - here's the anomaly score it did produce") rather
  than pretending it worked. The 10-fold CV numbers (98.10% accuracy) are
  the real, validated result regardless of what one live demo run shows.
- If you need a second attempt, answer `y` at the prompt to run another
  30-second window, re-run `--setup` (the previous run consumed the victim
  files), then launch `--attack` again partway through the new window.
- Close other heavy applications (browsers with many tabs, IDEs) before the
  demo if possible - fewer background processes means less dll_loaded noise
  from unrelated sources, for a cleaner signal.
