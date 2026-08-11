#!/usr/bin/env python3
"""
Stratified k-fold cross-validation for the practical XGBoost model.

The single 80/20 train_test_split used in train_practical_xgboost.py produces
accuracy estimates that vary with which rows happen to land in the test fold -
report.txt repeatedly flagged this across three retraining rounds (98.43% /
98.28% / 98.30%, with small confusion-matrix differences each time) as noise
that a single split cannot distinguish from a real effect. K-fold CV averages
over multiple splits for a more statistically defensible estimate, and lets us
report a standard deviation alongside the mean - matching the methodology of
the source dataset's own paper (Herrera-Silva & Hernandez-Alvarez, Sensors
2023, which reports 10-fold CV results for their own models).
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from xgboost import XGBClassifier

N_FOLDS = 10
RANDOM_STATE = 42

SELECTED_FEATURES = [
    "file_read",
    "file_created",
    "regkey_written",
    "regkey_read",
    "apistats",
    "command_line",
    "directory_enumerated",
    "dll_loaded",
    "tree_command_line",
    "arguments",
    "urls",
    "proc_pid"
]


def main():
    print(f"\n{'='*60}")
    print(f" {N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*60}\n")

    df = pd.read_csv("datasets/balanced_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['family'].value_counts()}\n")

    X = df[SELECTED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["family"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    classes = encoder.classes_
    print(f"Class mapping: {dict(zip(classes, range(len(classes))))}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies = []
    # per-class precision/recall/f1, collected per fold then averaged
    per_class_precision = {c: [] for c in classes}
    per_class_recall = {c: [] for c in classes}
    per_class_f1 = {c: [] for c in classes}
    aggregate_confusion = np.zeros((len(classes), len(classes)), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), start=1):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Fit scaler on this fold's training data only - avoids leaking
        # test-fold statistics into the scaler, which train_practical_xgboost.py's
        # single-split version does not have to worry about doing per-fold.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=range(len(classes)), zero_division=0
        )
        for i, c in enumerate(classes):
            per_class_precision[c].append(precision[i])
            per_class_recall[c].append(recall[i])
            per_class_f1[c].append(f1[i])

        aggregate_confusion += confusion_matrix(y_test, y_pred, labels=range(len(classes)))

        print(f"Fold {fold_idx:2d}/{N_FOLDS}: accuracy = {acc:.4f}")

    fold_accuracies = np.array(fold_accuracies)

    print(f"\n{'='*60}")
    print(" CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}\n")
    print(f"Mean accuracy: {fold_accuracies.mean():.4f}")
    print(f"Std deviation: {fold_accuracies.std():.4f}")
    print(f"Min / Max:     {fold_accuracies.min():.4f} / {fold_accuracies.max():.4f}")
    print(f"95% CI (approx, mean +/- 1.96*std/sqrt(n)): "
          f"{fold_accuracies.mean() - 1.96 * fold_accuracies.std() / np.sqrt(N_FOLDS):.4f} - "
          f"{fold_accuracies.mean() + 1.96 * fold_accuracies.std() / np.sqrt(N_FOLDS):.4f}")

    print(f"\nPer-class metrics (mean +/- std across {N_FOLDS} folds):")
    lines = []
    for c in classes:
        p = np.array(per_class_precision[c])
        r = np.array(per_class_recall[c])
        f = np.array(per_class_f1[c])
        line = (f"  {c}: precision={p.mean():.4f}+/-{p.std():.4f}  "
                f"recall={r.mean():.4f}+/-{r.std():.4f}  "
                f"f1={f.mean():.4f}+/-{f.std():.4f}")
        print(line)
        lines.append(line)

    print(f"\nAggregate confusion matrix (summed across all {N_FOLDS} folds, "
          f"rows=actual, cols=predicted, order={list(classes)}):")
    print(aggregate_confusion)

    # Save a summary to disk for the paper
    with open("cross_validation_results.txt", "w") as f:
        f.write(f"{N_FOLDS}-Fold Stratified Cross-Validation Results\n")
        f.write(f"Dataset: datasets/balanced_dataset.csv (shape {df.shape})\n")
        f.write(f"Class distribution:\n{df['family'].value_counts().to_string()}\n\n")
        f.write(f"Per-fold accuracy: {[round(a, 4) for a in fold_accuracies]}\n")
        f.write(f"Mean accuracy: {fold_accuracies.mean():.4f}\n")
        f.write(f"Std deviation: {fold_accuracies.std():.4f}\n")
        f.write(f"Min / Max: {fold_accuracies.min():.4f} / {fold_accuracies.max():.4f}\n\n")
        f.write("Per-class metrics (mean +/- std):\n")
        for line in lines:
            f.write(line + "\n")
        f.write(f"\nAggregate confusion matrix (order={list(classes)}):\n")
        f.write(str(aggregate_confusion) + "\n")

    print("\n[SAVED] cross_validation_results.txt")


if __name__ == "__main__":
    main()
