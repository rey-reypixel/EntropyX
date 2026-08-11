import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# =====================================================
# STEP 1: LOAD DATASET
# =====================================================

print("Loading dataset...")

df = pd.read_csv("datasets/ransomware_dataset.csv")

# First row contains actual column names
df.columns = df.iloc[0]

# Remove duplicated header row
df = df[1:]

# Reset index
df.reset_index(drop=True, inplace=True)

print(f"Dataset Shape: {df.shape}")

# =====================================================
# STEP 2: SEPARATE FEATURES AND LABELS
# =====================================================

y = df["family"]

X = df.drop("family", axis=1)

# =====================================================
# STEP 3: CONVERT FEATURES TO NUMERIC
# =====================================================

print("\nConverting features to numeric...")

X = X.apply(pd.to_numeric, errors="coerce")

# Check for NaN values generated during conversion
print("Missing Values After Conversion:", X.isnull().sum().sum())

# Replace NaN values with 0 if any exist
X = X.fillna(0)

# =====================================================
# STEP 4: ENCODE LABELS
# =====================================================

print("\nEncoding labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nLabel Mapping:")

for idx, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {idx}")

# Save label encoder
os.makedirs("models", exist_ok=True)

joblib.dump(label_encoder, "models/label_encoder.pkl")

# =====================================================
# STEP 5: FEATURE SCALING
# =====================================================

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# =====================================================
# STEP 6: TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain-Test Split Completed")

print(f"X_train Shape : {X_train.shape}")
print(f"X_test Shape  : {X_test.shape}")
print(f"y_train Shape : {y_train.shape}")
print(f"y_test Shape  : {y_test.shape}")

# =====================================================
# STEP 7: SAVE PROCESSED DATA
# =====================================================

joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")

joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("\nPreprocessed data saved successfully.")

print("\nPreprocessing Complete.")