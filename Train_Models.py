# Train_Models.py
import os
import json
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Sklearn
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import loguniform

# TensorFlow/Keras
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

# Preprocessing module
import Data_Preprocessing as dp

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
PROJECT_ROOT = r"C:\Project\Code"
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_JOBS = -1

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(obj, filename: str):
    path = os.path.join(ARTIFACT_DIR, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
    print(f"[INFO] Saved JSON report to: {path}")

def save_model(model, path):
    path = Path(path)
    if isinstance(model, (Sequential, Model)):
        path = path.with_suffix(".keras")
        model.save(path)
        print(f"[INFO] Saved Keras model to: {path}")
    else:
        path = path.with_suffix(".joblib")
        joblib.dump(model, path)
        print(f"[INFO] Saved model to: {path}")

def evaluate_and_report(name, model, X_test, y_test):
    # Ensure X_test has no NaNs
    X_test = SimpleImputer(strategy="mean").fit_transform(X_test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    result = {
        "model_name": name,
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    save_json(result, f"{name}_evaluation_{timestamp()}.json")

    print(f"\n=== {name} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(cm)
    print(classification_report(y_test, y_pred, zero_division=0))

    return result

# ---------------------------------------------------------------------
# CV safety check
# ---------------------------------------------------------------------
def safe_cv(y, desired_splits=5):
    min_class = np.min(np.bincount(y))
    if min_class < 2:
        return None
    return StratifiedKFold(
        n_splits=min(desired_splits, min_class),
        shuffle=True,
        random_state=RANDOM_STATE
    )

# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------
def build_svm_pipeline(y_train):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),   # ✅ Fix NaNs
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=RANDOM_STATE))
    ])

    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"]
    }

    cv = safe_cv(y_train)
    if cv is None:
        print("[WARNING] SVM CV disabled (too few samples per class)")
        return pipe

    return GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="accuracy", n_jobs=N_JOBS, refit=True, verbose=1
    )

def build_knn_pipeline(y_train):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),   # ✅ Fix NaNs
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])

    param_grid = {
        "clf__n_neighbors": [3, 5, 7],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"]
    }

    cv = safe_cv(y_train)
    if cv is None:
        print("[WARNING] k-NN CV disabled (too few samples per class)")
        return pipe

    return GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="accuracy", n_jobs=N_JOBS, refit=True, verbose=1
    )

# ---------------------------------------------------------------------
# Keras model
# ---------------------------------------------------------------------
def create_shallow_nn(input_dim, n_classes, hidden_units=64, lr=1e-3, dropout=0.2):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden_units, activation="relu"),
        Dropout(dropout),
        Dense(hidden_units // 2, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def keras_model_wrapper(hidden_units=64, lr=1e-3, dropout=0.2, input_dim=None, n_classes=None):
    return create_shallow_nn(input_dim, n_classes, hidden_units, lr, dropout)

def build_keras_classifier(X, y, input_dim, n_classes):
    cv = safe_cv(y, desired_splits=3)

    keras_clf = KerasClassifier(
        model=keras_model_wrapper,
        model__input_dim=input_dim,
        model__n_classes=n_classes,
        verbose=0
    )

    if cv is None:
        print("[WARNING] NN CV disabled (too few samples per class)")
        return keras_clf

    param_dist = {
        "model__hidden_units": [32, 64],
        "model__lr": loguniform(1e-4, 1e-2),
        "model__dropout": [0.0, 0.2],
        "batch_size": [32],
        "epochs": [20]
    }

    return RandomizedSearchCV(
        keras_clf,
        param_distributions=param_dist,
        n_iter=5,
        scoring="accuracy",
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=1
    )

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    data = dp.load_processed_data()
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    label_map = data["label_map"]

    save_json(label_map, "label_map.json")

    # Combine train + val
    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])

    # ---------------- Handle NaNs ----------------
    imputer = SimpleImputer(strategy="mean")
    X_full = imputer.fit_transform(X_full)
    X_test = imputer.transform(X_test)

    if len(np.unique(y_full)) < 2:
        raise RuntimeError("Only one class present. Training aborted.")

    print(f"[INFO] Data loaded: X={X_full.shape}, classes={len(np.unique(y_full))}")

    # ---------------- SVM ----------------
    print("\n[INFO] Training SVM...")
    svm_model = build_svm_pipeline(y_full)
    svm_model.fit(X_full, y_full)
    final_svm = svm_model.best_estimator_ if hasattr(svm_model, "best_estimator_") else svm_model
    evaluate_and_report("svm", final_svm, X_test, y_test)
    save_model(final_svm, f"{ARTIFACT_DIR}/svm_{timestamp()}")

    # ---------------- k-NN ----------------
    print("\n[INFO] Training k-NN...")
    knn_model = build_knn_pipeline(y_full)
    knn_model.fit(X_full, y_full)
    final_knn = knn_model.best_estimator_ if hasattr(knn_model, "best_estimator_") else knn_model
    evaluate_and_report("knn", final_knn, X_test, y_test)
    save_model(final_knn, f"{ARTIFACT_DIR}/knn_{timestamp()}")

    # ---------------- NN ----------------
    print("\n[INFO] Training shallow NN...")
    nn_search = build_keras_classifier(X_full, y_full, X_full.shape[1], len(label_map))
    nn_search.fit(X_full, y_full)
    final_nn = nn_search.best_estimator_ if hasattr(nn_search, "best_estimator_") else nn_search
    evaluate_and_report("shallow_nn", final_nn, X_test, y_test)
    save_model(final_nn.model_, f"{ARTIFACT_DIR}/shallow_nn_{timestamp()}")

    print("\n[INFO] All models trained successfully.")

if __name__ == "__main__":
    main()
