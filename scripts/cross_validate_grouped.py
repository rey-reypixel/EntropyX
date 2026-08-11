#!/usr/bin/env python3
"""
GroupKFold cross-validation for the practical XGBoost model (v2 audit
task #9, item 1 - fixes B1/B4).

src/cross_validate_xgboost.py uses StratifiedKFold(shuffle=True), which
treats every row as an independent sample. results/data_quality.json
shows that's wrong: 61.5% of balanced_dataset.csv's rows have at least
one byte-identical twin (same 12-feature vector) elsewhere in the
dataset - mostly because the source dataset is 20 ransomware + 20
goodware artifacts run 10x across 5 platforms, and projecting 50
sandbox-derived features down to the 12 this project can actually
collect live collapses most of what distinguished those runs. Under
shuffled k-fold, most of a duplicated row's twins land in the training
folds, so the held-out copy is memorised, not generalised to - the
98.10% figure in cross_validation_results.txt measures that.

This script groups by the exact 12-feature vector (the model's actual
input - not a hash of the 50-column source row, which would under-group:
two rows identical in the 12 features the model sees but different in
unused columns would still leak). StratifiedGroupKFold keeps every
occurrence of a given vector entirely in train or entirely in test,
never split, while still trying to preserve class balance across folds.

Also reports the old naive (shuffled StratifiedKFold) number alongside
this one, labeled explicitly as inflated by duplicate leakage - per the
v2 audit's recommendation, both numbers belong in the paper, not just
this corrected one.

Run from the project root:  .venv/Scripts/python.exe scripts/cross_validate_grouped.py
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
RANDOM_STATE = 42

SELECTED_FEATURES = [
    "file_read", "file_created", "regkey_written", "regkey_read", "apistats",
    "command_line", "directory_enumerated", "dll_loaded", "tree_command_line",
    "arguments", "urls", "proc_pid",
]


def group_key(row):
    return tuple(round(float(row[f]), 6) for f in SELECTED_FEATURES)


def main():
    print(f"\n{'=' * 70}\n GROUPED (LEAKAGE-SAFE) {N_FOLDS}-FOLD CROSS-VALIDATION\n{'=' * 70}\n")

    df = pd.read_csv("datasets/balanced_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['family'].value_counts()}\n")

    X = df[SELECTED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["family"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    classes = encoder.classes_
    print(f"Class mapping: {dict(zip(classes, range(len(classes))))}\n")

    # String-encode each 12-feature tuple into a single hashable label -
    # np.array() on a list of equal-length tuples silently produces a 2D
    # array (one column per tuple element), which GroupKFold/StratifiedGroupKFold
    # cannot use as group labels (they need one hashable value per row).
    groups = np.array([str(group_key(row)) for _, row in X.iterrows()], dtype=object)
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    print(f"Distinct 12-feature vectors (groups): {len(unique_groups)} "
          f"(vs {len(df)} rows - {len(df) - len(unique_groups)} rows share a vector with another row)")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies = []
    per_class_precision = {c: [] for c in classes}
    per_class_recall = {c: [] for c in classes}
    per_class_f1 = {c: [] for c in classes}
    aggregate_confusion = np.zeros((len(classes), len(classes)), dtype=int)
    leakage_violations = 0

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y_encoded, groups=groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            leakage_violations += len(overlap)
            print(f"  [ERROR] Fold {fold_idx}: {len(overlap)} group(s) appear in BOTH train and test!")

        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = XGBClassifier(
            objective="multi:softmax", num_class=3,
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=range(len(classes)), zero_division=0
        )
        for i, c in enumerate(classes):
            per_class_precision[c].append(precision[i])
            per_class_recall[c].append(recall[i])
            per_class_f1[c].append(f1[i])

        aggregate_confusion += confusion_matrix(y_test, y_pred, labels=range(len(classes)))

        print(f"Fold {fold_idx:2d}/{N_FOLDS}: accuracy = {acc:.4f}  "
              f"(train groups={len(train_groups)}, test groups={len(test_groups)})")

    fold_accuracies = np.array(fold_accuracies)

    print(f"\n{'=' * 70}\n SUMMARY\n{'=' * 70}\n")
    print(f"Group-leakage violations across all folds: {leakage_violations} (must be 0)")
    print(f"Mean accuracy: {fold_accuracies.mean():.4f}")
    print(f"Std deviation: {fold_accuracies.std():.4f}")
    print(f"Min / Max:     {fold_accuracies.min():.4f} / {fold_accuracies.max():.4f}")

    per_class = {}
    for c in classes:
        p, r, f = np.array(per_class_precision[c]), np.array(per_class_recall[c]), np.array(per_class_f1[c])
        per_class[c] = {
            "precision_mean": float(p.mean()), "precision_std": float(p.std()),
            "recall_mean": float(r.mean()), "recall_std": float(r.std()),
            "f1_mean": float(f.mean()), "f1_std": float(f.std()),
        }
        print(f"  {c}: precision={p.mean():.4f}+/-{p.std():.4f}  "
              f"recall={r.mean():.4f}+/-{r.std():.4f}  f1={f.mean():.4f}+/-{f.std():.4f}")

    print(f"\nAggregate confusion matrix (order={list(classes)}):")
    print(aggregate_confusion)

    # Pull in the old naive number for a direct side-by-side in the paper.
    naive = {}
    naive_path = "cross_validation_results.txt"
    if os.path.exists(naive_path):
        with open(naive_path, encoding="utf-8") as f:
            naive_text = f.read()
        naive = {"source_file": naive_path, "raw_text": naive_text}

    result = {
        "generated_by": "scripts/cross_validate_grouped.py",
        "method": "StratifiedGroupKFold, group = exact 12-feature vector "
                  "(the model's actual input) - see module docstring for why "
                  "this group key and not a hash of the 50-column source row",
        "n_folds": N_FOLDS,
        "random_state": RANDOM_STATE,
        "dataset_shape": list(df.shape),
        "n_rows": len(df),
        "n_distinct_groups": int(len(unique_groups)),
        "group_leakage_violations": int(leakage_violations),
        "fold_accuracies": [float(a) for a in fold_accuracies],
        "mean_accuracy": float(fold_accuracies.mean()),
        "std_accuracy": float(fold_accuracies.std()),
        "min_accuracy": float(fold_accuracies.min()),
        "max_accuracy": float(fold_accuracies.max()),
        "per_class": per_class,
        "aggregate_confusion_matrix": {
            "order": list(classes),
            "matrix": aggregate_confusion.tolist(),
        },
        "naive_ungrouped_baseline_for_comparison": naive,
        "interpretation": (
            "This is the number to report as headline accuracy. The naive "
            "(ungrouped, shuffled StratifiedKFold) number in "
            "cross_validation_results.txt should be reported alongside it, "
            "explicitly labeled as inflated by duplicate-vector leakage - "
            "not silently dropped, since the delta between the two numbers "
            "IS itself part of this project's finding (see "
            "results/data_quality.json)."
        ),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/cross_validation_grouped.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\n[SAVED] results/cross_validation_grouped.json")


if __name__ == "__main__":
    main()
