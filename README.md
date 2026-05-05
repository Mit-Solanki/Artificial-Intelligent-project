    # Artificial-Intelligent-project 
    #VitalSence

# ECG Arrhythmia Detection Pipeline (CNN + ML + Hybrid Model)

## 📌 Overview

This project implements an end-to-end ECG signal processing and
classification pipeline using: - Traditional Machine Learning (Random
Forest) - Deep Learning- Hybrid Mode (CNN) l (CNN + LSTM)

The goal is to classify ECG beats into: - Normal (N) - Arrhythmia
(Abnormal)

------------------------------------------------------------------------

## 📂 Dataset

-   MIT-BIH Arrhythmia Dataset
-   Loaded using `wfdb`

------------------------------------------------------------------------

## ⚙️ Features

-   ECG signal visualization
-   R-peak based segmentation
-   Binary classification (Normal vs Arrhythmia)
-   GPU-enabled TensorFlow training
-   Model comparison

------------------------------------------------------------------------

## 🧠 Models Used

### 1. Random Forest

-   Input: Flattened ECG beats
-   Fast baseline model

### 2. CNN

-   Extracts spatial features from ECG waveform

### 3. CNN + LSTM (Hybrid)

-   CNN → feature extraction
-   LSTM → temporal dependencies

------------------------------------------------------------------------

## 🛠️ Installation

``` bash
pip install numpy matplotlib wfdb scikit-learn tensorflow pandas
```

------------------------------------------------------------------------

## 🚀 Usage

### 1. Load ECG Data

``` python
record = wfdb.rdrecord(path)
annotation = wfdb.rdann(path, 'atr')
```

### 2. Segment ECG Beats

-   Extract 200-sample window around R-peak

### 3. Train Models

-   Random Forest
-   CNN
-   CNN + LSTM

### 4. Evaluate

``` python
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)
```

------------------------------------------------------------------------

## 📊 Model Comparison

-   Accuracy
-   F1 Score
-   Visualization using matplotlib

------------------------------------------------------------------------

## 📈 Results

  Model           Accuracy   Notes
  --------------- ---------- ----------------------------
  Random Forest   Moderate   Fast baseline
  CNN             Good       Learns waveform features
  CNN + LSTM      Best       Captures temporal patterns

------------------------------------------------------------------------

## 🔧 Key Functions

### Data Processing

-   `wfdb.rdrecord()`
-   `wfdb.rdann()`

### ML Models

-   `RandomForestClassifier()`

### Deep Learning

-   `Conv1D()`
-   `MaxPooling1D()`
-   `LSTM()`
-   `Dense()`
-   `Dropout()`

------------------------------------------------------------------------

## ⚠️ Notes

-   Ensure proper ECG segmentation
-   Dataset imbalance can affect performance
-   Use GPU for faster training

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Multi-class classification
-   Attention-based models
-   Real-time ECG processing
-   Deployment on embedded systems (ESP32)

------------------------------------------------------------------------

## 👨‍💻 Author

Project developed as part of AI/ML and Biomedical Signal Processing
exploration.
