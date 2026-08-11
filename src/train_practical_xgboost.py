import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading dataset...")

df = pd.read_csv("datasets/balanced_dataset.csv")

print("Dataset Shape:", df.shape)

# =====================================================
# SELECT PRACTICAL FEATURES
# =====================================================

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

print("\nSelected Features:")
for feature in selected_features:
    print(feature)

# =====================================================
# FEATURES & LABELS
# =====================================================

X = df[selected_features].copy()
y = df["family"]

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# =====================================================
# LABEL ENCODING
# =====================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nClass Mapping:")

for idx, label in enumerate(encoder.classes_):
    print(f"{label} -> {idx}")

# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =====================================================
# XGBOOST MODEL
# =====================================================

print("\nTraining Practical XGBoost...")

model = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

print("Training Complete.")

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(model, "models/practical_xgboost.pkl")
joblib.dump(scaler, "models/practical_scaler.pkl")

print("\nModel saved successfully.")