import numpy as np
import time
from sklearn.utils import resample
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os 
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"
os.environ["TABPFN_MODEL_CACHE_DIR"]="/ngc/tools/eb/v5.0/amd/software/causal_lymphoma/1.0.0/models/"

from tabpfn import TabPFNClassifier
from tabpfn.model_loading import load_fitted_tabpfn_model
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
)

from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    DetCurveDisplay,
)
import shap


from tqdm import tqdm
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from helpers.constants import *
from helpers.processing_helper import *
from helpers.sql_helper import *

sns.set_context("paper")

seed = 46

DLBCL_ONLY = False


WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]

feature_matrix = feature_matrix[~feature_matrix["patientid"].isin(wrong_patientids)].reset_index(drop = True)


features = pd.read_csv("results/feature_names_all.csv")["features"].values
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)

if DLBCL_ONLY:
    train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[-1]
col_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"]
col_to_leave.extend(outcome_column)

ipi_cols = [x for x in feature_matrix.columns if "NCCN_" in x]
col_to_leave.extend(ipi_cols)

predictor_columns = [x for x in train.columns if x not in col_to_leave]

predictor_columns = [
    x
    for x in predictor_columns
    if x not in ["pred_RKKP_subtype_fallback_-1", "pred_RKKP_hospital_fallback_-1"]
]

features = list(features)

for i in supplemental_columns:
    if i not in features:
        features.append(i)

for column in tqdm(features):
    clip_values(train, test, column)

supplemental_columns = [
    "pred_RKKP_hospital_fallback_-1",
    "pred_RKKP_subtype_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
]

features = list(features)
features.extend(supplemental_columns)

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

X_train_numpy = X_train_smtom.to_numpy()
X_test_numpy = X_test.to_numpy()
y_train_numpy = y_train_smtom.to_numpy()
y_test_numpy = y_test.to_numpy()

X_train_numpy = X_train_numpy.astype("float32")
X_test_numpy = X_test_numpy.astype("float32")


NCCN_IPIS = [
    "pred_RKKP_age_diagnosis_fallback_-1",
    "pred_RKKP_LDH_diagnosis_fallback_-1",
    "pred_RKKP_AA_stage_diagnosis_fallback_-1",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
    "pred_RKKP_PS_diagnosis_fallback_-1",
]


def benchmark_runtime(n_samples=500, n_features=20, n_test=50):
    # Subsample
    X_sub, y_sub = resample(X_train_numpy, y_train_numpy, n_samples=n_samples, random_state=42)
    X_sub = X_sub[:, :n_features]
    X_test_sub = X_test_numpy[:n_test, :n_features]

    # Fit model
    clf = TabPFNClassifier(
    model_path = "/ngc/tools/eb/v5.0/amd/software/causal_lymphoma/1.0.0/models/tabpfn-v2-classifier.ckpt",
    random_state=42,
    ignore_pretraining_limits=True,
    device="cpu")
    clf.fit(X_sub, y_sub)

    # Time predictions
    start = time.time()
    clf.predict(X_test_sub)
    elapsed = time.time() - start

    return elapsed

def benchmark_pred_scaling(n_samples=1000, n_features=60, test_sizes=[50, 500, 2000]):
    # Subsample training data
    X_sub, y_sub = resample(X_train_numpy, y_train_numpy, n_samples=n_samples, random_state=42)
    X_sub = X_sub[:, :n_features]

    # Fit model
    clf = TabPFNClassifier(
    model_path = "/ngc/tools/eb/v5.0/amd/software/causal_lymphoma/1.0.0/models/tabpfn-v2-classifier.ckpt",
    random_state=42,
    ignore_pretraining_limits=True,
    device="cpu")
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

# Example run with 60 features
benchmark_pred_scaling(n_samples=1000, n_features=60, test_sizes=[50, 500, 2000])

# Run benchmarks at different scales
scales = [
    (100, 2),
    (200, 5),   # tiny
    (500, 10),
    (1000, 20),
    (2000, 40),
    (5000, 60)
]



# for n_samples, n_features in scales:
#     t = benchmark_runtime(n_samples=n_samples, n_features=n_features, n_test=500)
#     print(f"{n_samples} samples, {n_features} features -> {t:.2f} seconds for 500 predictions")