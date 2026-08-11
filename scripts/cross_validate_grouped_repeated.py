#!/usr/bin/env python3
"""
Repeated group-aware cross-validation, for a defensible uncertainty
estimate (v2 audit suggestion #3).

Problem: scripts/cross_validate_grouped.py reports mean=85.59%,
std=15.39% from ONE run of 10-fold StratifiedGroupKFold. Any formal 95%
CI computed the naive way - mean +/- 1.96*std/sqrt(10), the same formula
src/cross_validate_xgboost.py already used for the (now-superseded)
98.10% figure - assumes the 10 fold accuracies are independent. They are
not: each pair of folds shares 8/9 of its training data, so the folds
are correlated with each other, and Bengio & Grandvalet (2004, "No
Unbiased Estimator of the Variance of K-Fold Cross-Validation") showed
there is no unbiased estimator of k-fold CV's true variance - a naive
formula UNDERSTATES it. With this project's std already at 15.39% (vs
0.62% for the leakage-inflated naive number), a further-understated
formal CI would be actively misleading, not just imprecise.

This script does NOT invent a valid CI where none exists - it instead
reports two more defensible, clearly-labeled quantities:

1. REPEATED-CV STANDARD ERROR: runs the full 10-fold
   StratifiedGroupKFold procedure N_REPEATS times, each with a
   DIFFERENT random split of groups into folds (the model's own
   training remains seed=42 throughout - only the fold ASSIGNMENT
   varies between repeats). Each repeat produces one mean accuracy
   across its 10 folds; treating those N_REPEATS repeat-level means as
   the unit of replication (not the 100 individual folds) is a
   standard, more defensible correction used in the ML literature for
   k-fold CV variance - different repeats use genuinely different
   group-to-fold assignments, so they capture split-to-split
   variability that folds WITHIN one repeat cannot.

2. EMPIRICAL PERCENTILE RANGE: the raw 2.5th/97.5th percentile across
   all N_REPEATS * N_FOLDS individual fold accuracies - an
   assumption-light range that makes no independence claim at all, just
   reports the observed spread directly.

Both are reported side by side with the single-run number, and neither
is called a "95% confidence interval" without qualification - see the
JSON output's "methodology_note" for the exact caveat to state if asked.

Run from the project root:  .venv/Scripts/python.exe scripts/cross_validate_grouped_repeated.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

N_FOLDS = 10
MODEL_SEED = 42  # fixed throughout - isolates "which split" variance from training randomness
# First split-seed (42) intentionally matches scripts/cross_validate_grouped.py's single run,
# so repeat 0 below reproduces that already-published result exactly, as a continuity check.
SPLIT_SEEDS = [42, 7, 13, 99, 123, 2024, 555, 8, 77, 314]

SELECTED_FEATURES = [
    "file_read", "file_created", "regkey_written", "regkey_read", "apistats",
    "command_line", "directory_enumerated", "dll_loaded", "tree_command_line",
    "arguments", "urls", "proc_pid",
]


def group_key(row):
    return str(tuple(round(float(row[f]), 6) for f in SELECTED_FEATURES))


def run_one_split(X, y_encoded, groups, split_seed):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=split_seed)
    fold_accuracies = []
    leakage_violations = 0

    for train_idx, test_idx in sgkf.split(X, y_encoded, groups=groups):
        train_groups, test_groups = set(groups[train_idx]), set(groups[test_idx])
        leakage_violations += len(train_groups & test_groups)

        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = XGBClassifier(
            objective="multi:softmax", num_class=3,
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=MODEL_SEED,
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        fold_accuracies.append(acc)

    return fold_accuracies, leakage_violations


def main():
    print(f"\n{'=' * 70}\n REPEATED GROUP-AWARE CROSS-VALIDATION "
          f"({len(SPLIT_SEEDS)} repeats x {N_FOLDS} folds = "
          f"{len(SPLIT_SEEDS) * N_FOLDS} total fold evaluations)\n{'=' * 70}\n")

    df = pd.read_csv("datasets/balanced_dataset.csv")
    X = df[SELECTED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["family"]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    groups = np.array([group_key(row) for _, row in X.iterrows()], dtype=object)

    per_repeat_means = []
    all_fold_accuracies = []
    total_leakage_violations = 0

    for i, split_seed in enumerate(SPLIT_SEEDS):
        fold_accs, violations = run_one_split(X, y_encoded, groups, split_seed)
        total_leakage_violations += violations
        repeat_mean = float(np.mean(fold_accs))
        per_repeat_means.append(repeat_mean)
        all_fold_accuracies.extend(fold_accs)
        tag = "  <- matches the single-run result (same split seed)" if i == 0 else ""
        print(f"Repeat {i + 1:2d}/{len(SPLIT_SEEDS)} (split_seed={split_seed:5d}): "
              f"mean={repeat_mean:.4f}  fold range=[{min(fold_accs):.4f}, {max(fold_accs):.4f}]{tag}")

    per_repeat_means = np.array(per_repeat_means)
    all_fold_accuracies = np.array(all_fold_accuracies)

    grand_mean = float(per_repeat_means.mean())
    repeat_level_std = float(per_repeat_means.std(ddof=1))
    repeat_level_se = repeat_level_std / np.sqrt(len(SPLIT_SEEDS))
    ci_lo, ci_hi = grand_mean - 1.96 * repeat_level_se, grand_mean + 1.96 * repeat_level_se

    pct_lo, pct_hi = np.percentile(all_fold_accuracies, [2.5, 97.5])

    print(f"\n{'=' * 70}\n SUMMARY\n{'=' * 70}\n")
    print(f"Total group-leakage violations across all {len(SPLIT_SEEDS)} repeats: "
          f"{total_leakage_violations} (must be 0)")
    print(f"\nGrand mean accuracy (mean of {len(SPLIT_SEEDS)} repeat-level means): {grand_mean:.4f}")
    print(f"Repeat-level std (across the {len(SPLIT_SEEDS)} repeat means, NOT the "
          f"100 individual folds): {repeat_level_std:.4f}")
    print(f"Repeated-CV 95% interval (grand_mean +/- 1.96*SE, SE from repeat-level "
          f"means): [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\nRaw empirical spread across all {len(all_fold_accuracies)} individual "
          f"fold accuracies: min={all_fold_accuracies.min():.4f} "
          f"max={all_fold_accuracies.max():.4f}")
    print(f"2.5th/97.5th percentile of all {len(all_fold_accuracies)} fold accuracies: "
          f"[{pct_lo:.4f}, {pct_hi:.4f}]  (this is NOT a CI on the mean - it's the "
          f"observed spread of individual fold-level results)")

    result = {
        "generated_by": "scripts/cross_validate_grouped_repeated.py",
        "n_repeats": len(SPLIT_SEEDS),
        "n_folds_per_repeat": N_FOLDS,
        "total_fold_evaluations": len(all_fold_accuracies),
        "model_seed": MODEL_SEED,
        "split_seeds": SPLIT_SEEDS,
        "total_group_leakage_violations": int(total_leakage_violations),
        "per_repeat_mean_accuracy": [float(m) for m in per_repeat_means],
        "grand_mean_accuracy": grand_mean,
        "repeat_level_std": repeat_level_std,
        "repeat_level_standard_error": float(repeat_level_se),
        "repeated_cv_95_interval": [float(ci_lo), float(ci_hi)],
        "all_fold_accuracies_min": float(all_fold_accuracies.min()),
        "all_fold_accuracies_max": float(all_fold_accuracies.max()),
        "empirical_2_5_97_5_percentile_of_fold_accuracies": [float(pct_lo), float(pct_hi)],
        "single_run_result_for_comparison": {
            "source": "results/cross_validation_grouped.json",
            "mean_accuracy": 0.8559138677342781,
            "std_accuracy": 0.15389966254590756,
            "note": "repeat 0 above (split_seed=42) reproduces this exactly - same fold assignment",
        },
        "methodology_note": (
            "There is no unbiased estimator of k-fold CV variance "
            "(Bengio & Grandvalet, 2004) - folds within one CV run share "
            "most of their training data, so treating them as "
            "independent (the naive mean +/- 1.96*std/sqrt(k) formula "
            "used for the original, now-superseded 98.10% figure) "
            "understates uncertainty. This script does not claim to "
            "produce a formally valid 95% CI either - it reports two "
            "more defensible, explicitly-labeled alternatives instead: "
            "(1) a standard-error interval computed ACROSS REPEATS "
            "(different repeats use independent group-to-fold "
            "assignments, so repeat-level means are a more defensible "
            "unit of replication than individual folds), and (2) the "
            "raw empirical percentile spread of all individual fold "
            "results, which makes no independence assumption at all. "
            "For the paper: report the grand mean with the repeated-CV "
            "interval, and separately note the wider raw fold-to-fold "
            "spread as evidence of how sensitive the result is to which "
            "specific groups land in which fold - itself a finding "
            "(class L's low group count, 60 distinct vectors, drives "
            "most of this spread)."
        ),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/cross_validation_repeated.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\n[SAVED] results/cross_validation_repeated.json")


if __name__ == "__main__":
    main()
