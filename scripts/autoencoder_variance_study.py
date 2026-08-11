#!/usr/bin/env python3
"""
Autoencoder seed variance + held-out evaluation study (v2 audit task #9,
items 3-4 - addresses B2 and B10).

Two problems with the shipped autoencoder evaluation:

B2 - train/eval contamination: train_autoencoder.py fits on ALL goodware
rows in balanced_dataset.csv; evaluate_autoencoder.py then scores
ransomware_dataset.csv's goodware rows. results/data_quality.json shows
100% of the distinct goodware vectors in the "evaluation" set are also in
the training set - the reported 0.508 mean error / 0.541678 threshold /
0.6069 F1 are all in-sample.

B10 - unseeded variance: train_autoencoder.py never seeded numpy or
TensorFlow. F1 across retrains has been reported as 0.70 -> 0.62 -> 0.70
-> 0.61 -> 0.58 (see bugs_debugs.txt), each time attributed to a data
change, with no control for how much of that swing is just training
noise from a different random initialization.

This script fixes both in one pass: splits goodware ONCE (fixed split,
reused across all seeds so seeds are comparable) into train/calibration/
held-out-test, trains the autoencoder fresh for each of N_SEEDS seeds
using ONLY the train split, selects a threshold via percentile search on
the CALIBRATION split (never touched during training), and reports final
F1 on the HELD-OUT TEST split (never touched during training OR
threshold calibration) - then reports the F1 mean/std across all seeds.

KNOWN LIMITATION (stated here rather than hidden): the E/L (ransomware)
rows are used in BOTH the threshold-search step and the final-F1 step,
because there is only one ransomware dataset available - splitting it
further would leave too few samples for either step to be meaningful.
This mirrors the original evaluate_autoencoder.py's own methodology. The
fix here is specifically for the GOODWARE side of the leakage (B2); the
E/L reuse is a real, acknowledged, separate limitation, not silently
fixed by this script.

Run from the project root:  .venv/Scripts/python.exe scripts/autoencoder_variance_study.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF's own logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

N_SEEDS = 10
SPLIT_RANDOM_STATE = 123  # fixed once, so every seed is evaluated on the SAME held-out data
SELECTED_FEATURES = [
    "file_read", "file_created", "regkey_written", "regkey_read", "apistats",
    "command_line", "directory_enumerated", "dll_loaded", "tree_command_line",
    "arguments", "urls", "proc_pid",
]
PERCENTILES = [90, 95, 97, 99]

HISTORICAL_F1_SERIES = [0.70, 0.6216, 0.7036, 0.6069, 0.5811]  # from bugs_debugs.txt, for comparison


def build_autoencoder(input_dim):
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.models import Model

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(10, activation="relu")(input_layer)
    encoder = Dense(6, activation="relu")(encoder)
    encoder = Dense(3, activation="relu")(encoder)
    decoder = Dense(6, activation="relu")(encoder)
    decoder = Dense(10, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    model = Model(inputs=input_layer, outputs=decoder)
    model.compile(optimizer="adam", loss=MeanSquaredError())
    return model


def reconstruction_error(model, X_scaled):
    reconstructed = model.predict(X_scaled, verbose=0)
    return np.mean(np.square(X_scaled - reconstructed), axis=1)


def best_threshold_by_f1(calib_goodware_err, ransomware_err):
    actual = np.array(["Normal"] * len(calib_goodware_err) + ["Suspicious"] * len(ransomware_err))
    all_err = np.concatenate([calib_goodware_err, ransomware_err])

    best_f1, best_threshold, best_pct = -1.0, None, None
    per_percentile = {}
    for pct in PERCENTILES:
        threshold = float(np.percentile(calib_goodware_err, pct))
        predictions = np.where(all_err > threshold, "Suspicious", "Normal")
        f1 = f1_score(actual, predictions, pos_label="Suspicious")
        per_percentile[pct] = {"threshold": threshold, "f1": float(f1)}
        if f1 > best_f1:
            best_f1, best_threshold, best_pct = f1, threshold, pct

    return best_threshold, best_pct, best_f1, per_percentile


def evaluate_at_threshold(held_out_goodware_err, ransomware_err, threshold):
    actual = np.array(["Normal"] * len(held_out_goodware_err) + ["Suspicious"] * len(ransomware_err))
    all_err = np.concatenate([held_out_goodware_err, ransomware_err])
    predictions = np.where(all_err > threshold, "Suspicious", "Normal")
    report = classification_report(actual, predictions, output_dict=True, zero_division=0)
    return report["Suspicious"]["f1-score"], report


def main():
    print(f"\n{'=' * 70}\n AUTOENCODER SEED VARIANCE + HELD-OUT EVALUATION STUDY\n{'=' * 70}\n")

    df = pd.read_csv("datasets/balanced_dataset.csv")
    X_all = df[SELECTED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

    X_good = X_all[df["family"] == "G"]
    X_e = X_all[df["family"] == "E"].to_numpy(dtype=float)
    X_l = X_all[df["family"] == "L"].to_numpy(dtype=float)
    print(f"Goodware: {len(X_good)}   E: {len(X_e)}   L: {len(X_l)}")

    # ONE fixed split, reused by every seed: 60% train / 20% calibration / 20% held-out test.
    good_train, good_temp = train_test_split(
        X_good, test_size=0.40, random_state=SPLIT_RANDOM_STATE
    )
    good_calib, good_test = train_test_split(
        good_temp, test_size=0.50, random_state=SPLIT_RANDOM_STATE
    )
    print(f"Goodware split (fixed across all seeds): "
          f"train={len(good_train)}  calibration={len(good_calib)}  held-out-test={len(good_test)}")

    per_seed_results = []

    for seed in range(1, N_SEEDS + 1):
        print(f"\n--- seed {seed}/{N_SEEDS} ---")
        np.random.seed(seed)
        import tensorflow as tf
        tf.random.set_seed(seed)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(good_train)

        model = build_autoencoder(input_dim=train_scaled.shape[1])
        model.fit(train_scaled, train_scaled, epochs=50, batch_size=32,
                  validation_split=0.20, verbose=0)

        calib_err = reconstruction_error(model, scaler.transform(good_calib))
        test_err = reconstruction_error(model, scaler.transform(good_test))
        e_err = reconstruction_error(model, scaler.transform(X_e))
        l_err = reconstruction_error(model, scaler.transform(X_l))
        ransomware_err = np.concatenate([e_err, l_err])

        threshold, pct, calib_f1, per_percentile = best_threshold_by_f1(calib_err, ransomware_err)
        held_out_f1, held_out_report = evaluate_at_threshold(test_err, ransomware_err, threshold)

        result = {
            "seed": seed,
            "goodware_train_mean_error": float(np.mean(reconstruction_error(model, train_scaled))),
            "goodware_calibration_mean_error": float(np.mean(calib_err)),
            "goodware_held_out_test_mean_error": float(np.mean(test_err)),
            "E_mean_error": float(np.mean(e_err)),
            "L_mean_error": float(np.mean(l_err)),
            "chosen_percentile": pct,
            "chosen_threshold": threshold,
            "calibration_f1_at_chosen_threshold": calib_f1,
            "held_out_test_f1_at_chosen_threshold": held_out_f1,
            "percentile_search": per_percentile,
        }
        per_seed_results.append(result)
        print(f"  goodware(test) mean err={result['goodware_held_out_test_mean_error']:.4f}  "
              f"E mean err={result['E_mean_error']:.2f}  L mean err={result['L_mean_error']:.2f}")
        print(f"  chosen threshold={threshold:.6f} (p{pct})  "
              f"calib F1={calib_f1:.4f}  HELD-OUT F1={held_out_f1:.4f}")

    held_out_f1s = np.array([r["held_out_test_f1_at_chosen_threshold"] for r in per_seed_results])

    print(f"\n{'=' * 70}\n SUMMARY ACROSS {N_SEEDS} SEEDS (held-out F1)\n{'=' * 70}\n")
    print(f"Mean: {held_out_f1s.mean():.4f}   Std: {held_out_f1s.std():.4f}   "
          f"Min/Max: {held_out_f1s.min():.4f} / {held_out_f1s.max():.4f}")
    hist = np.array(HISTORICAL_F1_SERIES)
    print(f"\nFor comparison, historical single-run F1 series (bugs_debugs.txt): {HISTORICAL_F1_SERIES}")
    print(f"Historical range: {hist.max() - hist.min():.4f}   "
          f"This study's seed-only range: {held_out_f1s.max() - held_out_f1s.min():.4f}")

    result = {
        "generated_by": "scripts/autoencoder_variance_study.py",
        "n_seeds": N_SEEDS,
        "split_random_state": SPLIT_RANDOM_STATE,
        "goodware_split_sizes": {
            "train": len(good_train), "calibration": len(good_calib), "held_out_test": len(good_test),
        },
        "known_limitation": (
            "E/L (ransomware) rows are reused in both threshold-search and "
            "final-F1 steps - only one ransomware dataset is available. "
            "This fixes the GOODWARE side of the train/eval contamination "
            "(B2); E/L reuse is a separate, acknowledged limitation, not "
            "fixed here."
        ),
        "per_seed": per_seed_results,
        "held_out_f1_mean": float(held_out_f1s.mean()),
        "held_out_f1_std": float(held_out_f1s.std()),
        "held_out_f1_min": float(held_out_f1s.min()),
        "held_out_f1_max": float(held_out_f1s.max()),
        "historical_single_run_f1_series": HISTORICAL_F1_SERIES,
        "historical_range": float(hist.max() - hist.min()),
        "this_study_seed_only_range": float(held_out_f1s.max() - held_out_f1s.min()),
        "interpretation": (
            "If this study's seed-only range is comparable to the historical "
            "range, B10 is confirmed: the F1 fluctuation across retrains is "
            "substantially explained by unseeded training variance, not by "
            "the data changes each retrain was attributed to. If this "
            "study's range is much smaller than the historical range, the "
            "data changes likely do explain most of the historical swing, "
            "and training variance is a minor contributor."
        ),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/autoencoder_variance.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\n[SAVED] results/autoencoder_variance.json")


if __name__ == "__main__":
    main()
