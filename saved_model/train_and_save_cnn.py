"""
train_and_save_cnn.py
=====================
Trains the binary CNN from your notebook (cnn_binary_model)
and saves it as  cnn_binary_model.h5  in the SAME folder as this script.

Place this file at:
    C:\\Users\\PC\\Desktop\\MIT_2\\train_and_save_cnn.py

Run:
    cd C:\\Users\\PC\\Desktop\\MIT_2
    python train_and_save_cnn.py

Output:
    C:\\Users\\PC\\Desktop\\MIT_2\\cnn_binary_model.h5   ← what ecg_server.py loads
"""
import wfdb
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# ═══════════════════════════════════════════════════════════
# 1.  CONFIGURATION  — only change this block
# ═══════════════════════════════════════════════════════════

BASE_PATH = r"C:\Users\PC\Desktop\MIT_2\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"

# Exactly the records your notebook uses
RECORDS = [str(i) for i in range(100, 125)] + [str(i) for i in range(200, 235)]

WINDOW_BEFORE = 100   # samples before R-peak
WINDOW_AFTER  = 200   # samples after  R-peak  →  total window = 300
WINDOW_SIZE   = WINDOW_BEFORE + WINDOW_AFTER   # must equal 300

EPOCHS      = 20
BATCH_SIZE  = 64
TEST_SIZE   = 0.2

# Where to save — same folder as this script
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_binary_model.h5")

# ═══════════════════════════════════════════════════════════
# 2.  LOAD MIT-BIH DATA  (identical to your notebook)
# ═══════════════════════════════════════════════════════════

print("\n📂  Loading MIT-BIH Arrhythmia Database …")
print(f"    Path : {BASE_PATH}\n")

X, y, hr_list = [], [], []

for rec in RECORDS:
    try:
        path   = os.path.join(BASE_PATH, rec)
        record = wfdb.rdrecord(path)
        ann    = wfdb.rdann(path, "atr")
        signal = record.p_signal[:, 0]

        for i in range(1, len(ann.sample)):
            idx   = ann.sample[i]
            label = ann.symbol[i]

            if idx - WINDOW_BEFORE >= 0 and idx + WINDOW_AFTER < len(signal):
                beat = signal[idx - WINDOW_BEFORE : idx + WINDOW_AFTER]

                y.append(0 if label == "N" else 1)   # binary: Normal=0, Abnormal=1
                X.append(beat)

                rr = (ann.sample[i] - ann.sample[i - 1]) / record.fs
                hr_list.append(60 / rr)

        print(f"  ✅  Record {rec:>3}  — {len(ann.sample)} beats")

    except Exception as e:
        print(f"  ⚠   Record {rec:>3}  skipped: {e}")

X  = np.array(X,  dtype=np.float32)
y  = np.array(y,  dtype=np.int32)
hr = np.array(hr_list, dtype=np.float32)

# Clean unrealistic HR values (same as notebook)
hr[(hr < 30) | (hr > 200)] = np.nan
hr = pd.Series(hr).interpolate().bfill().ffill().values

min_len = min(len(X), len(hr))
X, y, hr = X[:min_len], y[:min_len], hr[:min_len]

print(f"\n📊  Dataset   X={X.shape}   y={y.shape}")
print(f"    Labels   Normal(0)={np.sum(y==0):,}   Abnormal(1)={np.sum(y==1):,}")

# ═══════════════════════════════════════════════════════════
# 3.  NORMALISE + RESHAPE  (min-max to [-1,1])
# ═══════════════════════════════════════════════════════════

def normalise(arr):
    mn, mx = arr.min(axis=1, keepdims=True), arr.max(axis=1, keepdims=True)
    denom  = np.where(mx - mn < 1e-6, 1.0, mx - mn)
    return ((arr - mn) / denom * 2 - 1).astype(np.float32)

X = normalise(X)
X = X.reshape(X.shape[0], WINDOW_SIZE, 1)   # (N, 300, 1)

# ═══════════════════════════════════════════════════════════
# 4.  TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

print(f"\n🔀  Train={len(X_train):,}   Test={len(X_test):,}")

# ═══════════════════════════════════════════════════════════
# 5.  BUILD CNN  (exact architecture from your notebook)
# ═══════════════════════════════════════════════════════════

print("\n🧠  Building CNN model …")

cnn_binary_model = Sequential([
    Conv1D(32, kernel_size=5, activation="relu", input_shape=(WINDOW_SIZE, 1)),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=5, activation="relu"),
    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(64, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid"),   # binary output: 0=Normal, 1=Abnormal
])

cnn_binary_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

cnn_binary_model.summary()

# ═══════════════════════════════════════════════════════════
# 6.  TRAIN
# ═══════════════════════════════════════════════════════════

print(f"\n🚀  Training for {EPOCHS} epochs …\n")

history = cnn_binary_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1,
)

# ═══════════════════════════════════════════════════════════
# 7.  EVALUATE
# ═══════════════════════════════════════════════════════════

loss, acc = cnn_binary_model.evaluate(X_test, y_test, verbose=0)
print(f"\n📈  Test accuracy : {acc:.4f}   |   Loss : {loss:.4f}")

# Quick F1
y_probs = cnn_binary_model.predict(X_test, verbose=0).flatten()
y_pred  = (y_probs > 0.5).astype(int)

precision_m = tf.keras.metrics.Precision()
recall_m    = tf.keras.metrics.Recall()
precision_m.update_state(y_test, y_pred)
recall_m.update_state(y_test, y_pred)
p = precision_m.result().numpy()
r = recall_m.result().numpy()
f1 = 2 * p * r / (p + r + 1e-8)
print(f"    Precision : {p:.4f}   Recall : {r:.4f}   F1 : {f1:.4f}")

# ═══════════════════════════════════════════════════════════
# 8.  SAVE  →  cnn_binary_model.h5
# ═══════════════════════════════════════════════════════════

cnn_binary_model.save(SAVE_PATH)
print(f"\n✅  Model saved  →  {SAVE_PATH}")
print("    Copy this file next to ecg_server.py and you're done!\n")
