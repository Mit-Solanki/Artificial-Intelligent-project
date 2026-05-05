# VitalSense — ECG Arrhythmia Detection & Health Monitoring Pipeline

##  Overview

VitalSense is an end-to-end biomedical AI pipeline for real-time ECG arrhythmia detection and patient health risk assessment. The system combines deep learning-based ECG classification with vitals-based risk scoring to produce a unified clinical decision output.

The pipeline classifies ECG beats into three categories:
- **Normal (0)** — Healthy sinus rhythm
- **Cardiac Arrhythmia (1)** — Irregular but non-immediately-dangerous rhythm
- **Dangerous Arrhythmia (2)** — High-risk rhythm requiring urgent attention

---

##  Datasets Used

| Dataset | Purpose |
|---|---|
| **MIT-BIH Arrhythmia Database** | Primary ECG training data (multi-class classification) |
| **VFDB (Ventricular Fibrillation)** | Dangerous arrhythmia class (Class 2) augmentation |
| **BIDMC PPG and Respiration Dataset** | Binary ECG pre-training (Normal vs Abnormal) |
| **VitalDB** | HR / SpO2 / Respiration rate vitals for LSTM training |

All ECG data loaded via `wfdb`. VitalDB vitals loaded via Parquet format.

---

##  Models in the Final Pipeline

The following three models are actively used in the prediction system. All other models trained during development were experimental and are not part of the final decision pipeline.

---

### 1. `cnn_arrhythmia_model` — 3-Class CNN (Primary ECG Classifier)

The core classification model. Trained on MIT-BIH + VFDB data combined, it classifies a 300-sample ECG beat segment into one of three arrhythmia classes.

**Architecture:**
```
Conv1D(32, 5) → BatchNorm → MaxPool
Conv1D(64, 5) → BatchNorm → MaxPool
Conv1D(128, 3) → MaxPool
Flatten → Dense(128) → Dropout(0.5) → Dense(3, softmax)
```

**Training:**
- Input shape: `(300, 1)`
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: 10, Batch: 64
- Labels: 0 = Normal, 1 = Arrhythmia, 2 = Dangerous

---

### 2. `lstm_vitals_model` — Stacked LSTM (Vitals Classifier)

Trained on VitalDB vitals sequences (HR, SpO2, Respiration Rate). Classifies a 20-timestep window of patient vitals into a risk category used alongside the ECG prediction.

**Architecture:**
```
LSTM(64, return_sequences=True) → Dropout(0.2)
LSTM(32) → Dropout(0.2)
Dense(16, relu) → Dense(3, softmax)
```

**Training:**
- Input shape: `(20, 3)` — [HR, SpO2, RR] per timestep
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Saved to: `saved_model/lstm_ecg_model.keras`

---

### 3. `get_vitals_pred(hr)` — Rule-Based Vitals Fallback

A deterministic HR threshold function used as a real-time substitute for `lstm_vitals_model` when live SpO2 and RR data are unavailable (e.g., during MIT-BIH test evaluation).

```python
def get_vitals_pred(hr):
    if hr < 50 or hr > 120:
        return 2   # Critical
    elif hr < 60 or hr > 100:
        return 1   # Warning
    else:
        return 0   # Normal
```

This is replaced by `lstm_vitals_model` when full vitals data is available.

---

##  Final Decision Pipeline

```
ECG beat (300 samples)
        │
        ▼
cnn_arrhythmia_model ──► ecg_pred (0 / 1 / 2)
                                        │
HR / SpO2 / RR                          ▼
        │                    final_decision()  ──► Clinical Alert
        ▼                         ▲
lstm_vitals_model ──► vitals_pred (0 / 1 / 2)
```

**Decision Logic (`final_decision`):**

| Condition | Output |
|---|---|
| `ecg_pred == 2` | 🚨 Dangerous Arrhythmia |
| `ecg_pred == 1` AND `vitals_pred == 2` | 🚨 High Risk Cardiac Condition |
| `ecg_pred == 1` | ⚠️ Cardiac Arrhythmia |
| `spo2 < 90` | 🚨 Hypoxia |
| `rr > 25` or `rr < 8` | ⚠️ Respiratory Distress |
| `vitals_pred == 2` | 🚨 Critical Health Condition |
| `vitals_pred == 1` | ⚠️ General Health Risk |
| `hr < 50` | ⚠️ Bradycardia |
| `hr > 120` | ⚠️ Tachycardia |
| All clear | ✅ Normal Condition |

---

##  Models Trained but NOT in Final Pipeline

These were trained during development and benchmarking but are superseded by `cnn_arrhythmia_model`:

| Model | Purpose | Why Not Used |
|---|---|---|
| `rf_model` (Random Forest) | Early baseline on MIT-BIH | Lower accuracy; no temporal feature extraction |
| `cnn_binary_model` (Binary CNN) | BIDMC Normal vs Abnormal | Superseded by 3-class `cnn_arrhythmia_model` |

---

##  Installation

```bash
pip install numpy matplotlib wfdb scikit-learn tensorflow pandas joblib pyarrow
```

---

##  Usage

### 1. Load and Segment ECG Data
```python
import wfdb
record     = wfdb.rdrecord(path)
annotation = wfdb.rdann(path, 'atr')

# Segment 300-sample windows around each R-peak annotation
for i in range(len(annotation.sample)):
    idx    = annotation.sample[i]
    symbol = str(annotation.symbol[i]).replace('(', '').strip()
    segment = signal[idx - 150 : idx + 150]
```

### 2. Run the Full Prediction Pipeline
```python
ecg_pred    = cnn_arrhythmia_model.predict(ecg_sample.reshape(1, 300, 1), verbose=0).argmax()
vitals_pred = get_vitals_pred(hr_value)          # or lstm_vitals_model.predict(...)
result      = final_decision(ecg_pred, vitals_pred, hr_value, spo2, rr)
print(result)
```

### 3. Load Saved Models
```python
import tensorflow as tf, joblib

cnn_arrhythmia_model = tf.keras.models.load_model("saved_model/cnn_arrhythmia_model.keras")
lstm_vitals_model    = tf.keras.models.load_model("saved_model/lstm_ecg_model.keras")
scaler               = joblib.load("saved_model/scaler.pkl")
```

---

##  Saved Artifacts

| File | Description |
|---|---|
| `saved_model/cnn_arrhythmia_model.keras` | 3-class CNN (final ECG classifier) |
| `saved_model/lstm_ecg_model.keras` | LSTM vitals model |
| `saved_model/scaler.pkl` | Fitted StandardScaler for vitals preprocessing |
| `saved_model/training_history.json` | LSTM training history for plotting |
| `comparison_data/X_train.npy` | Training ECG segments |
| `comparison_data/X_test.npy` | Test ECG segments |
| `comparison_data/y_train.npy` | Training labels |
| `comparison_data/y_test.npy` | Test labels |
| `comparison_data/hr_test.npy` | Heart rate values for test set |

---

##  Results

| Model | Accuracy | Notes |
|---|---|---|
| Random Forest | ~85% | Fast baseline, not in final pipeline |
| CNN Binary | ~98.9% | BIDMC binary, not in final pipeline |
| **CNN Arrhythmia (3-class)** | **~97%+** |  Final ECG classifier |
| LSTM Vitals | Varies |  Final vitals classifier |

---

##  Key Functions

| Function | Description |
|---|---|
| `wfdb.rdrecord()` | Load ECG signal |
| `wfdb.rdann()` | Load beat annotations |
| `build_lstm_model()` | Factory function for LSTM architecture |
| `get_vitals_pred(hr)` | Rule-based vitals risk scoring |
| `final_decision(ecg_pred, vitals_pred, hr, spo2, rr)` | Unified clinical decision output |

---

##  Notes

- ECG segments must be exactly 300 samples for `cnn_arrhythmia_model`
- VFDB annotations use `(AFIB` style symbols — strip the `(` before label lookup
- `get_vitals_pred()` uses HR only; replace with `lstm_vitals_model` for full vitals input
- Use GPU for training (`tf.config.list_physical_devices('GPU')` to verify)
- Dataset class imbalance between Normal / Arrhythmia / Dangerous may affect recall on minority classes

---

##  Future Improvements

- Replace `get_vitals_pred()` with live `lstm_vitals_model` inference using real SpO2 + RR
- Attention-based models (Transformer / SE-Net) for ECG feature weighting
- Real-time ECG streaming and inference
- Deployment on embedded systems (ESP32 / Raspberry Pi)
- Mobile app integration for patient monitoring

---

##  Author

Project developed as part of AI/ML and Biomedical Signal Processing exploration — **VitalSense**.
