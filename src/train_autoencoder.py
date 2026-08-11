import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError

# ==========================================
# SEEDING (v2 audit design principle: randomness must be seeded and
# variance measured, not assumed - see bugs_debugs.txt INVESTIGATION #11
# and scripts/autoencoder_variance_study.py, which runs this training
# procedure across 10 seeds to measure how much of the F1 fluctuation
# reported across retrains (0.70 -> 0.62 -> 0.70 -> 0.61 -> 0.58) is
# training-variance noise vs a real data effect)
# ==========================================

parser = argparse.ArgumentParser(description="Train the goodware-only autoencoder")
parser.add_argument("--seed", type=int, default=None,
                     help="Random seed for numpy/tensorflow (unset = nondeterministic, matches original behavior)")
args, _ = parser.parse_known_args()

if args.seed is not None:
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)
    print(f"[SEED] Using random seed {args.seed}")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("datasets/balanced_dataset.csv")

# ==========================================
# FEATURES (V2)
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
# GOODWARE ONLY
# ==========================================

X_goodware = X[y == "G"]

print("Goodware Samples:", X_goodware.shape)

# NOTE: burst-window filtering (excluding the top 5% of goodware windows by
# total feature activity before autoencoder training) was tried here and
# empirically made F1 WORSE (0.6069 -> 0.5811), not better - see
# bugs_debugs.txt INVESTIGATION #11 for the full experiment and numbers.
# The hypothesis (that a few extreme-activity windows widen the learned
# "normal" boundary and hurt separation from ransomware) was reasonable but
# not supported by the result. Deliberately NOT re-adding that filter here -
# if you're tempted to try it again, read INVESTIGATION #11 first.

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_goodware_scaled = scaler.fit_transform(X_goodware)

joblib.dump(
    scaler,
    "models/autoencoder_scaler.pkl"
)

# ==========================================
# AUTOENCODER
# ==========================================

input_dim = X_goodware_scaled.shape[1]

input_layer = Input(shape=(input_dim,))

# Encoder

encoder = Dense(10, activation="relu")(input_layer)
encoder = Dense(6, activation="relu")(encoder)
encoder = Dense(3, activation="relu")(encoder)

# Decoder

decoder = Dense(6, activation="relu")(encoder)
decoder = Dense(10, activation="relu")(decoder)
decoder = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(
    inputs=input_layer,
    outputs=decoder
)

autoencoder.compile(
    optimizer="adam",
    loss=MeanSquaredError()
)

autoencoder.summary()

# ==========================================
# TRAIN
# ==========================================

history = autoencoder.fit(
    X_goodware_scaled,
    X_goodware_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
    verbose=1
)

# ==========================================
# SAVE MODEL
# ==========================================

autoencoder.save(
    "models/autoencoder_model.keras"
)

print("\nAutoencoder Saved Successfully.")