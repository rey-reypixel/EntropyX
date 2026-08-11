import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
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
# PRACTICAL FEATURES
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
# SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# TRAIN ONLY ON GOODWARE
# ==========================================

X_goodware = X_scaled[y == "G"]

print("Goodware Samples:", X_goodware.shape)

# ==========================================
# ISOLATION FOREST
# ==========================================

print("\nTraining Isolation Forest...")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
)

iso_forest.fit(X_goodware)

print("Training Complete.")

# ==========================================
# PREDICT ALL SAMPLES
# ==========================================

predictions = iso_forest.predict(X_scaled)

# sklearn:
#  1 = normal
# -1 = anomaly

predictions = np.where(
    predictions == -1,
    "Suspicious",
    "Normal"
)

# ==========================================
# GROUND TRUTH
# ==========================================

actual = np.where(
    y == "G",
    "Normal",
    "Suspicious"
)

# ==========================================
# REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        actual,
        predictions
    )
)

# ==========================================
# ANOMALY SCORES
# ==========================================

scores = iso_forest.decision_function(
    X_scaled
)

results = pd.DataFrame({
    "family": y,
    "anomaly_score": scores
})

print("\nAverage Anomaly Score")

for cls in ["G", "E", "L"]:

    avg_score = results[
        results["family"] == cls
    ]["anomaly_score"].mean()

    print(f"{cls}: {avg_score:.6f}")

# ==========================================
# SAVE RESULTS
# ==========================================

results.to_csv(
    "isolation_forest_results.csv",
    index=False
)

print(
    "\nResults saved to isolation_forest_results.csv"
)