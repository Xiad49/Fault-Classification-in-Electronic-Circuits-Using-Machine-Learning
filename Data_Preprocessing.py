"""
Data_Preprocessing.py

Auto-robust preprocessing pipeline for circuit fault classification
- Auto-detects label column
- Handles rare classes safely
- Train/val/test split
- Feature scaling
- Saves artifacts + label_map.json
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ---------------- Config ----------------

PROJECT_ROOT = r"C:\Project\Code"

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data_processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

TRAIN_NPY = os.path.join(PROCESSED_DATA_DIR, "X_train.npy")
VAL_NPY = os.path.join(PROCESSED_DATA_DIR, "X_val.npy")
TEST_NPY = os.path.join(PROCESSED_DATA_DIR, "X_test.npy")
Y_TRAIN_NPY = os.path.join(PROCESSED_DATA_DIR, "y_train.npy")
Y_VAL_NPY = os.path.join(PROCESSED_DATA_DIR, "y_val.npy")
Y_TEST_NPY = os.path.join(PROCESSED_DATA_DIR, "y_test.npy")
SCALER_NPY = os.path.join(PROCESSED_DATA_DIR, "scaler_params.npz")
CLASS_WEIGHTS_NPY = os.path.join(PROCESSED_DATA_DIR, "class_weights.npy")
LABEL_MAP_JSON = os.path.join(PROCESSED_DATA_DIR, "label_map.json")

RANDOM_STATE = 42

# ---------------- Label detection ----------------

def detect_label_column(df: pd.DataFrame) -> str:
    preferred = ["label", "Label", "fault_label", "class", "target"]
    for col in preferred:
        if col in df.columns:
            return col

    non_numeric = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    if not non_numeric:
        raise ValueError("No suitable label column found in dataset.")

    return non_numeric[0]

# ---------------- Feature extraction ----------------

def build_feature_label_arrays(df: pd.DataFrame, label_col: str):
    feature_cols = [
        c for c in df.columns
        if c != label_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int32)

    return X, y, feature_cols

# ---------------- Safe split ----------------

def safe_train_val_test_split(
    X, y,
    test_size=0.2,
    val_size=0.2
):
    class_counts = pd.Series(y).value_counts()
    min_count = class_counts.min()

    if min_count < 2:
        print(
            "[data_preprocessing][WARNING] "
            "Some classes have < 2 samples. "
            "Stratified split disabled."
        )
        stratify = None
    else:
        stratify = y

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    stratify_tmp = y_tmp if stratify is not None else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=stratify_tmp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# ---------------- Scaling ----------------

def scale_splits(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    return (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
        scaler
    )

# ---------------- Class weights ----------------

def compute_balancing_weights(y_train):
    classes = np.unique(y_train)

    if len(classes) < 2:
        return None

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    return {int(c): float(w) for c, w in zip(classes, weights)}

# ---------------- Save artifacts ----------------

def save_processed_artifacts(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    scaler, class_weights, label_map
):
    np.save(TRAIN_NPY, X_train)
    np.save(VAL_NPY, X_val)
    np.save(TEST_NPY, X_test)
    np.save(Y_TRAIN_NPY, y_train)
    np.save(Y_VAL_NPY, y_val)
    np.save(Y_TEST_NPY, y_test)

    np.savez(
        SCALER_NPY,
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        var_=scaler.var_,
        n_features_in_=scaler.n_features_in_
    )

    if class_weights:
        np.save(
            CLASS_WEIGHTS_NPY,
            np.array(list(class_weights.items()), dtype=object)
        )

    with open(LABEL_MAP_JSON, "w") as f:
        json.dump(label_map, f, indent=4)

# ---------------- Main pipeline ----------------

def handle_missing_values(df):
    # Replace inf values
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    # Fill numeric NaNs with column mean
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Drop rows that still contain NaN (safety)
    df = df.dropna()

    return df



def run_full_preprocessing_pipeline(
    external_csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    balance_classes: bool = True
):
    if not os.path.exists(external_csv_path):
        raise FileNotFoundError("CSV dataset not found.")

    df = pd.read_csv(external_csv_path)

    label_col = detect_label_column(df)
    print(f"[data_preprocessing] Label column detected: '{label_col}'")

    labels = sorted(df[label_col].unique())
    label_map = {lbl: i for i, lbl in enumerate(labels)}
    df[label_col] = df[label_col].map(label_map)

    X, y, feature_names = build_feature_label_arrays(df, label_col)

    X_train, X_val, X_test, y_train, y_val, y_test = safe_train_val_test_split(
        X, y, test_size, val_size
    )

    X_train, X_val, X_test, scaler = scale_splits(
        X_train, X_val, X_test
    )

    class_weights = (
        compute_balancing_weights(y_train)
        if balance_classes else None
    )

    save_processed_artifacts(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler, class_weights, label_map
    )

    print("[data_preprocessing] Preprocessing complete.")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "class_weights": class_weights,
        "label_map": label_map
    }

# ---------------- Run directly ----------------

# ---------------- Load processed data ----------------

def load_processed_data():
    if not os.path.exists(TRAIN_NPY):
        raise FileNotFoundError(
            "Processed data not found. Run preprocessing first."
        )

    X_train = np.load(TRAIN_NPY)
    X_val = np.load(VAL_NPY)
    X_test = np.load(TEST_NPY)

    y_train = np.load(Y_TRAIN_NPY)
    y_val = np.load(Y_VAL_NPY)
    y_test = np.load(Y_TEST_NPY)

    scaler_params = np.load(SCALER_NPY)
    scaler = StandardScaler()
    scaler.mean_ = scaler_params["mean_"]
    scaler.scale_ = scaler_params["scale_"]
    scaler.var_ = scaler_params["var_"]
    scaler.n_features_in_ = int(scaler_params["n_features_in_"])

    class_weights = None
    if os.path.exists(CLASS_WEIGHTS_NPY):
        cw_arr = np.load(CLASS_WEIGHTS_NPY, allow_pickle=True)
        class_weights = {int(k): float(v) for k, v in cw_arr}

    with open(LABEL_MAP_JSON, "r") as f:
        label_map = json.load(f)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "class_weights": class_weights,
        "label_map": label_map
    }


if __name__ == "__main__":
    run_full_preprocessing_pipeline(
        external_csv_path=r"C:\Project\Code\Data-set's\circuit_fault_dataset-1.csv"
    )
