"""
benchmark_tabpfn.py

Benchmark runtime and scaling behaviour of TabPFN on the LYFO feature matrix.

Steps:
- Load WIDE_DATA, feature matrix, and selected features.
- Build train/test splits using held-out patient IDs.
- Add supplemental features and apply clipping.
- Convert to NumPy arrays.
- Define benchmarking functions for runtime and prediction scaling.
- Run a prediction scaling benchmark and print timings.
"""

import os
import time

import numpy as np
import pandas as pd
import shap  # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt  # noqa: F401

from tqdm import tqdm
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder  # noqa: F401
from sklearn.model_selection import train_test_split  # noqa: F401
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    auc,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    DetCurveDisplay,
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from imblearn.combine import SMOTETomek  # noqa: F401
from xgboost import XGBClassifier  # noqa: F401

from tabpfn import TabPFNClassifier
from tabpfn.model_loading import load_fitted_tabpfn_model  # noqa: F401

from lyfo_treatment_failure_prediction.helpers.constants import *  # noqa: F401,F403
from lyfo_treatment_failure_prediction.helpers.processing_helper import (
    clip_values,
    get_features_and_outcomes,
)
from lyfo_treatment_failure_prediction.helpers.sql_helper import *  # noqa: F401,F403

sns.set_context("paper")

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"
os.environ[
    "TABPFN_MODEL_CACHE_DIR"
] = "/ngc/tools/eb/v5.0/amd/software/causal_lymphoma/1.0.0/models/"

seed = 46
DLBCL_ONLY = False

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

# Exclude patients with missing age_diagnosis
wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]
feature_matrix = feature_matrix[
    ~feature_matrix["patientid"].isin(wrong_patientids)
].reset_index(drop=True)

# Selected features (from Lasso feature_selection)
features = pd.read_csv("results/feature_names_all.csv")["features"].values

test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)

if DLBCL_ONLY:
    train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Define outcome and predictors (same logic as feature_selection.py)
# ---------------------------------------------------------------------------
outcome_column = [col for col in feature_matrix.columns if "outc" in col]
outcome = outcome_column[-1]

col_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"]
col_to_leave.extend(outcome_column)

ipi_cols = [col for col in feature_matrix.columns if "NCCN_" in col]
col_to_leave.extend(ipi_cols)

predictor_columns = [col for col in train.columns if col not in col_to_leave]
predictor_columns = [
    col
    for col in predictor_columns
    if col not in ["pred_RKKP_subtype_fallback_-1", "pred_RKKP_hospital_fallback_-1"]
]

# Start with feature list from CSV
features = list(features)

# Ensure global supplemental_columns (from constants) are present
for col in supplemental_columns:
    if col not in features:
        features.append(col)

# Clip values per feature based on train distribution
for column in tqdm(features, desc="Clipping features"):
    clip_values(train, test, column)

# Local supplemental columns (kept for parity with original script)
supplemental_columns = [
    "pred_RKKP_hospital_fallback_-1",
    "pred_RKKP_subtype_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
]

features = list(features)
features.extend(supplemental_columns)

# ---------------------------------------------------------------------------
# Build final train/test arrays with selected features
# ---------------------------------------------------------------------------
(
    X_train_smtom,
    y_train_smtom,
    X_test,
    y_test,
    X_test_specific,
    y_test_specific,
    test_specific,
) = get_features_and_outcomes(
    train,
    test,
    WIDE_DATA,
    outcome,
    features,
)

X_train_numpy = X_train_smtom.to_numpy().astype("float32")
X_test_numpy = X_test.to_numpy().astype("float32")
y_train_numpy = y_train_smtom.to_numpy()
y_test_numpy = y_test.to_numpy()

# NCCN predictor columns (defined but unused, kept for parity)
NCCN_IPIS = [
    "pred_RKKP_age_diagnosis_fallback_-1",
    "pred_RKKP_LDH_diagnosis_fallback_-1",
    "pred_RKKP_AA_stage_diagnosis_fallback_-1",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
    "pred_RKKP_PS_diagnosis_fallback_-1",
]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def benchmark_runtime(n_samples: int = 500, n_features: int = 20, n_test: int = 50):
    """
    Benchmark prediction runtime for a fixed test size (n_test) with
    subsampled training data and a given number of features.
    """
    # Subsample training data
    X_sub, y_sub = resample(
        X_train_numpy, y_train_numpy, n_samples=n_samples, random_state=42
    )
    X_sub = X_sub[:, :n_features]
    X_test_sub = X_test_numpy[:n_test, :n_features]

    # Fit TabPFN
    clf = TabPFNClassifier(
        model_path=(
            "/ngc/tools/eb/v5.0/amd/software/causal_lymphoma/1.0.0/models/"
            "tabpfn-v2-classifier.ckpt"
        ),
        random_state=42,
        ignore_pretraining_limits=True,
        device="cpu",
    )
    clf.fit(X_sub, y_sub)

    # Time predictions
    start = time.time()
    clf.predict(X_test_sub)
    elapsed = time.time() - start

    return elapsed


def benchmark_pred_scaling(
    n_samples: int = 1000,
    n_features: int = 60,
    test_sizes: list[int] = None,
):
    """
    Benchmark prediction scaling across different test sizes
    for a fixed subsample of training data and a fixed number of features.
    """
    if test_sizes is None:
        test_sizes = [50, 500, 2000]

    # Subsample training data
    X_sub, y_sub = resample(
        X_train_numpy, y_train_numpy, n_samples=n_samples, random_state=42
    )
    X_sub = X_sub[:, :n_features]

    # Fit TabPFN
    clf = TabPFNClassifier(
        model_path=(
            "/ngc/tools/eb/v5.0/amd/software/causal_lymphoma/1.0.0/models/"
            "tabpfn-v2-classifier.ckpt"
        ),
        random_state=42,
        ignore_pretraining_limits=True,
        device="cpu",
    )
    clf.fit(X_sub, y_sub)

    results = {}
    for n_test in test_sizes:
        X_test_sub = X_test_numpy[:n_test, :n_features]
        start = time.time()
        clf.predict(X_test_sub)
        elapsed = time.time() - start
        results[n_test] = elapsed
        print(f"{n_test} test samples, {n_features} features -> {elapsed:.2f} seconds")

    return results


# ---------------------------------------------------------------------------
# Run the scaling benchmark (same as original script)
# ---------------------------------------------------------------------------

# Example run with 60 features
benchmark_pred_scaling(n_samples=1000, n_features=60, test_sizes=[50, 500, 2000])

# Scales for optional runtime benchmarking (left commented as in original)
scales = [
    (100, 2),
    (200, 5),   # tiny
    (500, 10),
    (1000, 20),
    (2000, 40),
    (5000, 60),
]

# for n_samples, n_features in scales:
#     t = benchmark_runtime(n_samples=n_samples, n_features=n_features, n_test=500)
#     print(f"{n_samples} samples, {n_features} features -> {t:.2f} seconds for 500 predictions")
