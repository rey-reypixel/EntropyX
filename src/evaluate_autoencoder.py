import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("datasets/ransomware_dataset.csv")

df.columns = df.iloc[0]
df = df[1:]
df.reset_index(drop=True, inplace=True)

# ==========================================
# SELECT FEATURES
# ==========================================
selected_features = [
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

X = df[selected_features].apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)

y = df["family"]

# ==========================================
# LOAD SCALER
# ==========================================

print("Loading scaler...")

scaler = joblib.load(
    "models/autoencoder_scaler.pkl"
)

X_scaled = scaler.transform(X)

# ==========================================
# LOAD AUTOENCODER
# ==========================================

print("Loading autoencoder...")

autoencoder = load_model(
    "models/autoencoder_model.keras"
)

# ==========================================
# RECONSTRUCTION
# ==========================================

print("Calculating reconstruction errors...")

reconstructed = autoencoder.predict(
    X_scaled,
    verbose=0
)

# Mean Squared Error per sample

mse = np.mean(
    np.square(X_scaled - reconstructed),
    axis=1
)

# ==========================================
# ATTACH ERRORS
# ==========================================

results = pd.DataFrame({
    "family": y,
    "reconstruction_error": mse
})

# ==========================================
# CLASS-WISE ANALYSIS
# ==========================================

print("\nAverage Reconstruction Error")

for cls in ["G", "E", "L"]:

    avg_error = results[
        results["family"] == cls
    ]["reconstruction_error"].mean()

    print(f"{cls}: {avg_error:.6f}")

# ==========================================
# ERROR STATISTICS
# ==========================================

print("\nDetailed Statistics")

print(
    results.groupby("family")[
        "reconstruction_error"
    ].describe()
)

# ==========================================
# THRESHOLD OPTIMIZATION
# ==========================================

from sklearn.metrics import classification_report, f1_score

goodware_errors = results[
    results["family"] == "G"
]["reconstruction_error"]

thresholds = {
    "90th Percentile": np.percentile(goodware_errors, 90),
    "95th Percentile": np.percentile(goodware_errors, 95),
    "97th Percentile": np.percentile(goodware_errors, 97),
    "99th Percentile": np.percentile(goodware_errors, 99)
}

best_f1 = 0
best_threshold = None

actual = np.where(
    results["family"] == "G",
    "Normal",
    "Suspicious"
)

print("\n" + "="*60)
print("THRESHOLD EVALUATION")
print("="*60)

for name, threshold in thresholds.items():

    predictions = np.where(
        results["reconstruction_error"] > threshold,
        "Suspicious",
        "Normal"
    )

    report = classification_report(
        actual,
        predictions,
        output_dict=True
    )

    suspicious_f1 = report["Suspicious"]["f1-score"]

    print(f"\n{name}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Suspicious F1 Score: {suspicious_f1:.4f}")

    print(
        classification_report(
            actual,
            predictions
        )
    )

    if suspicious_f1 > best_f1:
        best_f1 = suspicious_f1
        best_threshold = threshold

print("\n" + "="*60)
print("BEST THRESHOLD")
print("="*60)
print(f"Threshold = {best_threshold:.6f}")
print(f"Best F1 Score = {best_f1:.4f}")

# ==========================================
# SAVE RESULTS
# ==========================================

results.to_csv(
    "autoencoder_results.csv",
    index=False
)

print(
    "\nResults saved to autoencoder_results.csv"
)