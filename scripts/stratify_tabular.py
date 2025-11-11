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
import joblib


from tqdm import tqdm
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample


from helpers.constants import *
from helpers.processing_helper import *
from helpers.sql_helper import *

sns.set_context("paper")

import joblib

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


NCCN_IPIS = [
    "pred_RKKP_age_diagnosis_fallback_-1",
    "pred_RKKP_LDH_diagnosis_fallback_-1",
    "pred_RKKP_AA_stage_diagnosis_fallback_-1",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
    "pred_RKKP_PS_diagnosis_fallback_-1",
]

if __name__ == "__main__":

    X_train_numpy = X_train_smtom.to_numpy()
    X_test_numpy = X_test.to_numpy()
    y_train_numpy = y_train_smtom.to_numpy()
    y_test_numpy = y_test.to_numpy()

    X_train_numpy = X_train_numpy.astype("float32")
    X_test_numpy = X_test_numpy.astype("float32")

    bst = TabPFNClassifier(
    model_path = "/ngc/tools/eb/v5.0/amd/software/causal_lymphoma/1.0.0/models/tabpfn-v2-classifier.ckpt",
    random_state=seed,
    ignore_pretraining_limits=True,
    device="cpu")

    # --- Config ---
    n_train = 500    # subsample training rows
    n_features = 20  # subsample feature columns
    n_test = 50      # subsample test rows
    outfile = "tabpfn_predictions_subset.pkl"

    # --- Subsample ---
    X_train_sub, y_train_sub = resample(X_train_numpy, y_train_numpy, n_samples=n_train, random_state=42)
    X_train_sub = X_train_sub[:, :n_features]
    X_test_sub = X_test_numpy[:n_test, :n_features]
    y_test_sub = y_test_numpy[:n_test]
    test_ids_sub = test_ids[:n_test] if 'test_ids' in globals() else np.arange(n_test)

    # --- Train TabPFN ---
    bst = TabPFNClassifier(
    model_path = "/ngc/tools/eb/v5.0/amd/software/causal_lymphoma/1.0.0/models/tabpfn-v2-classifier.ckpt",
    random_state=seed,
    ignore_pretraining_limits=True,
    device="cpu")
    print("Fitting TabPFN on subset...")

    bst.fit(X_train_sub, y_train_sub)

    # --- Predict ---
    print("Predicting subset test set...")
    probs_test = bst.predict_proba(X_test_sub)[:, 1]

    # --- Save ---
    out = {
        "y_true": y_test_sub,
        "y_pred_proba": probs_test,
    }
    joblib.dump(out, outfile)

    print(f"Done. Saved subset predictions to {outfile}")
    
    print("Fitting TabPFN...")
    bst = TabPFNClassifier(
    model_path = "/ngc/tools/eb/v5.0/amd/software/causal_lymphoma/1.0.0/models/tabpfn-v2-classifier.ckpt",
    random_state=seed,
    ignore_pretraining_limits=True,
    device="cpu")
    print("Fitting TabPFN on subset...")

    bst.fit(X_train_numpy, y_train_numpy)

    # --- Predict probabilities on full test set ---
    print("Predicting test set...")
    probs_test = bst.predict_proba(X_test_numpy)[:, 1]  # positive class prob

    # --- Save predictions to disk ---
    out = {
        "y_true": y_test_numpy,
        "y_pred_proba": probs_test,
    }
    joblib.dump(out, "results/tabpfn_predictions.pkl")

    print("Done. Saved to tabpfn_predictions.pkl")
