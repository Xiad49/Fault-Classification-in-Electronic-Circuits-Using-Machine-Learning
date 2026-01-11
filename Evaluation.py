"""
Evaluation.py

Evaluate trained ML models for fault classification in electronic circuits.

Compatible with:
- Data_Preprocessing.py output
- CSV datasets
- sklearn (.joblib) models
- Keras (.keras / .h5) models
"""

import os
import glob
import json
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import load_model as keras_load_model

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = r"C:\Project\Code"
DATASET_DIR = os.path.join(BASE_DIR, "Data-set's")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts")
PROCESSED_DIR = os.path.join(BASE_DIR, "data_processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------
# Dataset loading
# --------------------------------------------------
def find_csv_dataset() -> str:
    csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in {DATASET_DIR}")
    return sorted(csv_files)[0]

def load_dataset_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    print(f"[evaluation] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # Detect label column (non-numeric)
    non_numeric_cols = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
    if not non_numeric_cols:
        raise ValueError("No non-numeric column found for labels")
    label_column = non_numeric_cols[0]

    # Numeric feature columns
    feature_cols = [c for c in df.columns if c != label_column and np.issubdtype(df[c].dtype, np.number)]
    if not feature_cols:
        raise ValueError("No numeric feature columns found")
    X = df[feature_cols].values

    # Handle missing values in features
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Load label map (string -> int)
    label_map_path = os.path.join(PROCESSED_DIR, "label_map.json")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError("label_map.json not found. Run Data_Preprocessing.py first.")
    with open(label_map_path, "r") as f:
        str_label_map = json.load(f)

    # Safe conversion of CSV labels to integers
    y = np.array([str_label_map.get(str(lbl), -1) for lbl in df[label_column]], dtype=np.int32)

    # Filter out unknown labels for evaluation
    mask = y != -1
    unknown_count = np.sum(y == -1)
    if unknown_count > 0:
        print(f"[evaluation][WARNING] {unknown_count} sample(s) ignored due to unknown labels not in label_map")

    X = X[mask]
    y = y[mask]

    # If no labels match, return None to skip evaluation
    if len(y) == 0:
        print("[evaluation][WARNING] No labels in the CSV match the training label_map. Skipping evaluation.")
        return None, None, None

    # Inverse map for plotting / metrics
    inv_label_map = {v: k for k, v in str_label_map.items()}

    print(f"[evaluation] Features used : {len(feature_cols)}")
    print(f"[evaluation] Label column  : {label_column}")
    print(f"[evaluation] Classes       : {inv_label_map}")

    return X, y, inv_label_map

# --------------------------------------------------
# Model loading
# --------------------------------------------------
def load_latest_sklearn(pattern: str):
    matches = glob.glob(pattern)
    if not matches:
        return None
    path = sorted(matches)[-1]
    print(f"[evaluation] Loaded sklearn model: {os.path.basename(path)}")
    return load(path)

def load_latest_keras(pattern: str):
    matches = glob.glob(pattern)
    if not matches:
        return None
    path = sorted(matches)[-1]
    print(f"[evaluation] Loaded Keras model: {os.path.basename(path)}")
    return keras_load_model(path)

# --------------------------------------------------
# Plotting utilities
# --------------------------------------------------
def plot_class_distribution(y: np.ndarray, label_map: Dict[int, str], save_path: str):
    unique, counts = np.unique(y, return_counts=True)
    class_names = [label_map[i] for i in unique]

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, counts, color="skyblue")
    plt.xlabel("Class")
    plt.ylabel("Samples")
    plt.title("Class Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[saved] {save_path}")

def plot_confusion_matrix(cm: np.ndarray, labels: list, name: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[saved] {path}")

# --------------------------------------------------
# Evaluation logic
# --------------------------------------------------
def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray, label_map: Dict[int, str], is_keras: bool):
    print(f"\n[evaluation] Evaluating {name}")

    if is_keras:
        preds = model.predict(X, verbose=0)
        y_pred = np.argmax(preds, axis=1)
    else:
        y_pred = model.predict(X)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    labels = [label_map[i] for i in sorted(label_map)]

    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred, target_names=labels, zero_division=0))

    plot_confusion_matrix(cm, labels, name)

    metrics = {
        "model": name,
        "accuracy": float(acc),
        "samples": int(len(y)),
        "labels": label_map
    }
    metrics_path = os.path.join(RESULTS_DIR, f"{name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[saved] {metrics_path}")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("[evaluation] Starting evaluation pipeline")

    csv_path = find_csv_dataset()
    X, y, label_map = load_dataset_from_csv(csv_path)

    # Skip evaluation if no matching labels
    if X is None or y is None:
        print("[evaluation] No matching labels. Evaluation skipped.")
        return

    plot_class_distribution(
        y,
        label_map,
        os.path.join(RESULTS_DIR, "class_distribution.png")
    )

    # Load latest models
    models = {
        "svm": (load_latest_sklearn(os.path.join(MODELS_DIR, "svm_*.joblib")), False),
        "knn": (load_latest_sklearn(os.path.join(MODELS_DIR, "knn_*.joblib")), False),
        "shallow_nn": (load_latest_keras(os.path.join(MODELS_DIR, "shallow_nn_*.keras")), True)
    }

    for name, (model, is_keras) in models.items():
        if model is not None:
            evaluate_model(name, model, X, y, label_map, is_keras)

    print("\n[evaluation] Evaluation completed successfully")

# --------------------------------------------------
if __name__ == "__main__":
    main()
