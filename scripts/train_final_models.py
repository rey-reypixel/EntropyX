#!/usr/bin/env python3
"""
Train and ship the FINAL production models (v2 audit suggestion #1).

Two things this fixes about the previously-deployed models/*.pkl,
models/*.keras (both dated 2026-07-22, unchanged since before this v2
session):

1. models/practical_xgboost.pkl was trained on only 80% of
   balanced_dataset.csv (src/train_practical_xgboost.py holds back a
   20% test split to print a spot-check accuracy). Now that
   scripts/cross_validate_grouped.py gives an honest, leakage-safe
   generalization estimate (85.59% mean, std 15.39%) from
   cross-validation, there is no remaining reason to withhold 20% of
   the data from the artifact that actually ships - standard practice
   is: use CV to ESTIMATE generalization, then train the final deployed
   model on ALL available data. This script does that: fits on all
   2999 rows, same architecture/hyperparameters/seed as before.

2. models/autoencoder_model.keras was trained without a seed
   (non-reproducible - see bugs_debugs.txt BUG #13's sibling finding,
   B10) and its calibration threshold in config.json can silently go
   stale after any retrain (BUG #6, recurred 3 times previously). This
   script trains with a fixed, documented seed (42, matching the
   XGBoost convention used everywhere else in this project), then
   IMMEDIATELY recalibrates and writes the new anomaly_threshold to
   config.json in the same run, closing the exact gap BUG #6 warns
   about ("any future re-training ... MUST be followed by re-running
   evaluate_autoencoder.py and updating config.json").

The honest, out-of-sample PERFORMANCE ESTIMATES for the paper still
come from the separate, independent evaluation studies
(results/cross_validation_grouped.json, results/autoencoder_variance.json)
- NOT from this script. This script's job is only to produce the best
final artifact to actually run, now that those separate studies have
told us the training procedure is trustworthy. Mixing the two purposes
(estimating generalization vs. producing the shipped artifact) in one
step is what let stale/held-back-data models ship silently before.

Run from the project root:  .venv/Scripts/python.exe scripts/train_final_models.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SEED = 42  # matches train_practical_xgboost.py / cross_validate_xgboost.py / cross_validate_grouped.py
SELECTED_FEATURES = [
    "file_read", "file_created", "regkey_written", "regkey_read", "apistats",
    "command_line", "directory_enumerated", "dll_loaded", "tree_command_line",
    "arguments", "urls", "proc_pid",
]
PERCENTILES = [90, 95, 97, 99]


def train_final_xgboost():
    print(f"\n{'=' * 70}\n FINAL XGBOOST - trained on 100% of balanced_dataset.csv\n{'=' * 70}\n")

    df = pd.read_csv("datasets/balanced_dataset.csv")
    X = df[SELECTED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["family"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        objective="multi:softmax", num_class=3,
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=SEED,
    )
    model.fit(X_scaled, y_encoded)

    # Training-set accuracy is NOT a generalization estimate (it's fit on
    # everything the model was trained on) - reported only as a sanity
    # check that training converged, never as a headline number.
    train_acc = accuracy_score(y_encoded, model.predict(X_scaled))
    print(f"Training-set fit accuracy (sanity check only, NOT a generalization "
          f"estimate - see results/cross_validation_grouped.json for that): {train_acc:.4f}")

    importance = dict(zip(SELECTED_FEATURES, [float(x) for x in model.feature_importances_]))
    top3 = sorted(importance.items(), key=lambda kv: -kv[1])[:3]
    print(f"Top 3 features: {top3}")

    joblib.dump(model, "models/practical_xgboost.pkl")
    joblib.dump(scaler, "models/practical_scaler.pkl")
    joblib.dump(encoder, "models/label_encoder.pkl")
    print("[SAVED] models/practical_xgboost.pkl, practical_scaler.pkl, label_encoder.pkl")

    return {
        "seed": SEED,
        "trained_on_rows": len(df),
        "trained_on_fraction_of_dataset": 1.0,
        "training_set_fit_accuracy_sanity_check_only": float(train_acc),
        "top_3_features": top3,
        "generalization_estimate_source": "results/cross_validation_grouped.json (85.59% mean, std 15.39%) - NOT measured by this script",
    }


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


def train_final_autoencoder():
    print(f"\n{'=' * 70}\n FINAL AUTOENCODER - seed={SEED}, trained on 100% of goodware\n{'=' * 70}\n")

    np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)

    df = pd.read_csv("datasets/balanced_dataset.csv")
    X_all = df[SELECTED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_good = X_all[df["family"] == "G"]
    print(f"Goodware samples: {X_good.shape}")

    scaler = StandardScaler()
    X_good_scaled = scaler.fit_transform(X_good)

    model = build_autoencoder(input_dim=X_good_scaled.shape[1])
    model.fit(X_good_scaled, X_good_scaled, epochs=50, batch_size=32, validation_split=0.20, verbose=1)

    joblib.dump(scaler, "models/autoencoder_scaler.pkl")
    model.save("models/autoencoder_model.keras")
    print("[SAVED] models/autoencoder_model.keras, autoencoder_scaler.pkl")

    # Recalibrate the threshold against the SOURCE dataset, same methodology
    # as evaluate_autoencoder.py - and update config.json in this same run
    # (BUG #6 discipline: never let the deployed threshold go stale relative
    # to whatever model is actually loaded).
    print("\nRecalibrating anomaly_threshold against datasets/ransomware_dataset.csv...")
    with open("datasets/ransomware_dataset.csv", newline="", encoding="utf-8-sig") as f:
        lines = f.readlines()
    import io
    src_df = pd.read_csv(io.StringIO("".join(lines[1:])))  # skip the "Table S1" pre-header line

    X_src = src_df[SELECTED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_src = src_df["family"]

    X_src_scaled = scaler.transform(X_src)
    reconstructed = model.predict(X_src_scaled, verbose=0)
    mse = np.mean(np.square(X_src_scaled - reconstructed), axis=1)

    results = pd.DataFrame({"family": y_src.values, "reconstruction_error": mse})
    goodware_errors = results[results["family"] == "G"]["reconstruction_error"]
    actual = np.where(results["family"] == "G", "Normal", "Suspicious")

    best_f1, best_threshold, best_pct = -1.0, None, None
    for pct in PERCENTILES:
        threshold = float(np.percentile(goodware_errors, pct))
        predictions = np.where(results["reconstruction_error"] > threshold, "Suspicious", "Normal")
        f1 = f1_score(actual, predictions, pos_label="Suspicious")
        print(f"  p{pct}: threshold={threshold:.6f}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_threshold, best_pct = f1, threshold, pct

    print(f"\nChosen: p{best_pct}, threshold={best_threshold:.6f}, F1={best_f1:.4f} "
          f"(this F1 is evaluated the SAME way evaluate_autoencoder.py always has - "
          f"on ransomware_dataset.csv goodware, which overlaps this model's training "
          f"data. NOT a held-out estimate. See results/autoencoder_variance.json for "
          f"the honest held-out F1 = 0.637 +/- 0.02.)")

    for cls in ["G", "E", "L"]:
        avg = results[results["family"] == cls]["reconstruction_error"].mean()
        print(f"  {cls} avg reconstruction error: {avg:.4f}")

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    old_threshold = config["ml"]["anomaly_threshold"]
    config["ml"]["anomaly_threshold"] = best_threshold
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"\n[UPDATED] config.json: ml.anomaly_threshold {old_threshold} -> {best_threshold}")

    return {
        "seed": SEED,
        "trained_on_goodware_rows": len(X_good),
        "trained_on_fraction_of_goodware": 1.0,
        "chosen_percentile": best_pct,
        "chosen_threshold": best_threshold,
        "in_sample_f1_at_chosen_threshold": float(best_f1),
        "goodware_avg_error": float(results[results["family"] == "G"]["reconstruction_error"].mean()),
        "E_avg_error": float(results[results["family"] == "E"]["reconstruction_error"].mean()),
        "L_avg_error": float(results[results["family"] == "L"]["reconstruction_error"].mean()),
        "old_config_threshold": old_threshold,
        "held_out_f1_estimate_source": "results/autoencoder_variance.json (0.6371 +/- 0.0201, 10 seeds) - NOT measured by this script",
    }


def main():
    xgb_summary = train_final_xgboost()
    ae_summary = train_final_autoencoder()

    summary = {
        "generated_by": "scripts/train_final_models.py",
        "purpose": (
            "Produce the final shipped model artifacts using the full "
            "dataset and a documented seed, now that separate CV/seed "
            "studies (results/cross_validation_grouped.json, "
            "results/autoencoder_variance.json) give trustworthy "
            "generalization estimates. Backup of the previous artifacts: "
            "models/backup_pre_v2_final_retrain_20260812/"
        ),
        "xgboost": xgb_summary,
        "autoencoder": ae_summary,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/final_model_training.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n[SAVED] results/final_model_training.json")


if __name__ == "__main__":
    main()
