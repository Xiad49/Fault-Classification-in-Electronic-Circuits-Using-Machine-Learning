"""
run_experiment.py

Master runner script for:
- Data preprocessing
- Model training
- Evaluation

Allows manual dataset selection and step-by-step control.
"""

import sys

# --------------------------------------------------
# USER INPUT SECTION (CHANGE ONLY THIS PART)
# --------------------------------------------------

# Path to dataset CSV
DATASET_PATH = r"C:\Project\Code\Data-set's\lite_dataset.csv"

# Control pipeline steps
RUN_PREPROCESSING = True
RUN_TRAINING = True
RUN_EVALUATION = True

# --------------------------------------------------
# PROJECT ROOT SETUP
# --------------------------------------------------

PROJECT_ROOT = r"C:\Project\Code"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print("\n[RUNNER] Starting full experiment pipeline")
print(f"[RUNNER] Dataset selected: {DATASET_PATH}")

# --------------------------------------------------
# STEP 1: DATA PREPROCESSING
# --------------------------------------------------

if RUN_PREPROCESSING:
    print("\n[RUNNER] Step 1: Data Preprocessing")

    import Data_Preprocessing as dp

    dp.run_full_preprocessing_pipeline(
        external_csv_path=DATASET_PATH,
        test_size=0.2,
        val_size=0.2,
        balance_classes=True
    )

    print("[RUNNER] Preprocessing completed")

# --------------------------------------------------
# STEP 2: MODEL TRAINING
# --------------------------------------------------

if RUN_TRAINING:
    print("\n[RUNNER] Step 2: Model Training")

    import Train_Models as tm
    tm.main()

    print("[RUNNER] Training completed")

# --------------------------------------------------
# STEP 3: MODEL EVALUATION
# --------------------------------------------------

if RUN_EVALUATION:
    print("\n[RUNNER] Step 3: Model Evaluation")

    import Evaluation as ev
    ev.main()

    print("[RUNNER] Evaluation completed")

print("\n[RUNNER] Experiment finished successfully")
