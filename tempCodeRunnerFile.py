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


# NGSPICE path if needed
os.environ["NGSPICE_EXECUTABLE"] = r"C:\ngspice\bin\ngspice.exe"

# Sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

# scikeras wrapper for TensorFlow/Keras
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

# Latest preprocessing module
from Data_Preprocessing import load_processed_data

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
    """
    Save a model to disk.
    - Keras models are saved in native .keras format
    - Other models (scikit-learn, etc.) are saved with joblib (.joblib)
    """
    path = Path(path)
    if isinstance(model, (Sequential, Model)):
        # Save Keras model as native .keras
        if path.suffix != ".keras":
            path = path.with_suffix(".keras")
        model.save(path)
        print(f"[INFO] Saved Keras model to: {path}")
    else:
        # Save other models using joblib
        if path.suffix != ".joblib":
            path = path.with_suffix(".joblib")
        joblib.dump(model, path)
        print(f"[INFO] Saved model to: {path}")

def evaluate_and_report(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    result = {
        "model_name": name,
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    save_json(result, f"{name}_evaluation_{timestamp()}.json")

    print(f"\n=== {name} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return result

# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------

def build_svm_pipeline():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=RANDOM_STATE))
    ])
    param_grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=N_JOBS, refit=True, verbose=1)

def build_knn_pipeline():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
    param_grid = {
        "clf__n_neighbors": [3,5,7,9],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan", "minkowski"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=N_JOBS, refit=True, verbose=1)

# ---------------------------------------------------------------------
# Keras model (top-level function)
# ---------------------------------------------------------------------
#@tf.function(reduce_retracing=True)
def create_shallow_nn(input_dim, n_classes, hidden_units=64, lr=1e-3, dropout=0.2):
    
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_units, activation="relu"))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(hidden_units // 2, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def keras_model_wrapper(hidden_units=64, lr=1e-3, dropout=0.2, input_dim=None, n_classes=None):
    return create_shallow_nn(input_dim=input_dim, n_classes=n_classes,
                             hidden_units=hidden_units, lr=lr, dropout=dropout)

def build_keras_classifier(input_dim, n_classes):
    keras_clf = KerasClassifier(model=keras_model_wrapper, verbose=0,
                                model__input_dim=input_dim, model__n_classes=n_classes)
    param_dist = {
        "model__hidden_units": [32, 64, 128],
        "model__lr": loguniform(1e-4, 1e-2),
        "model__dropout": [0.0, 0.2, 0.4],
        "batch_size": [32, 64],
        "epochs": [20, 40],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        keras_clf,
        param_distributions=param_dist,
        n_iter=10,
        scoring="accuracy",
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1
    )

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # Load preprocessed data
    data = load_processed_data()
    X = np.vstack([data["X_train"], data["X_val"], data["X_test"]])
    y = np.hstack([data["y_train"], data["y_val"], data["y_test"]])

    print(f"[INFO] Loaded data: X.shape={X.shape}, y.shape={y.shape}")

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # ----------------- Train SVM -----------------
    print("\n[INFO] Training SVM...")
    svm_search = build_svm_pipeline()
    svm_search.fit(X_train, y_train)
    print(f"[INFO] Best SVM params: {svm_search.best_params_}")
    svm_model = svm_search.best_estimator_
    evaluate_and_report("svm", svm_model, X_test, y_test)
    save_model(svm_model, f"{ARTIFACT_DIR}/svm_model_{timestamp()}.joblib")

    # ----------------- Train k-NN -----------------
    print("\n[INFO] Training k-NN...")
    knn_search = build_knn_pipeline()
    knn_search.fit(X_train, y_train)
    print(f"[INFO] Best k-NN params: {knn_search.best_params_}")
    knn_model = knn_search.best_estimator_
    evaluate_and_report("knn", knn_model, X_test, y_test)
    save_model(knn_model, f"{ARTIFACT_DIR}/knn_model_{timestamp()}.joblib")

    # ----------------- Train shallow NN -----------------
    print("\n[INFO] Training shallow neural network...")
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    keras_search = build_keras_classifier(input_dim, n_classes)
    keras_search.fit(X_train, y_train)
    print(f"[INFO] Best NN params: {keras_search.best_params_}")
    nn_model = keras_search.best_estimator_
    evaluate_and_report("shallow_nn", nn_model, X_test, y_test)
    save_model(nn_model.model_, f"{ARTIFACT_DIR}/shallow_nn_model_{timestamp()}")  # saves as .keras

    print("\n[INFO] All models trained and saved successfully.")

if __name__ == "__main__":
    main()
