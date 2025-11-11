from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
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

import joblib


sns.set_context("paper")

seed = 46

DLBCL_ONLY = False


WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]

feature_matrix = feature_matrix[~feature_matrix["patientid"].isin(wrong_patientids)].reset_index(drop = True)

feature_matrix.replace(-1, np.nan, inplace=True)

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
# outcome = outcome_column[0]
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

test_with_treatment = test.merge(
    WIDE_DATA[["patientid", "regime_1_chemo_type_1st_line"]]
)


predictions = joblib.load("results/tabpfn_predictions.pkl")
y_pred_proba = predictions["y_pred_proba"]

test["y_pred_proba"] = y_pred_proba

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
    specific_immunotherapy=False,
    none_chop_like=False,
)

bst = "model"

results, best_threshold = check_performance_across_thresholds(X_test, y_test, bst, y_pred_proba=predictions["y_pred_proba"])

f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test, y_test, bst, 0.5, predictions["y_pred_proba"]
)

y_pred = [1 if x > 0.5 else 0 for x in y_pred_proba]

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test.values, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test.values, y_pred)).plot()

y_pred_proba = test_specific["y_pred_proba"].values

results, best_threshold = check_performance_across_thresholds(
    X_test_specific, y_test_specific, bst, y_pred_proba
)
y_pred = [1 if x > 0.5 else 0 for x in y_pred_proba]

test_specific["model_highrisk"] = y_pred

from sklearn.utils import resample

def stratified_bootstrap_metrics(
    y_true, y_pred_proba, y_pred_label, n_bootstraps=1000, seed=42
):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred_label = np.array(y_pred_label)

    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]

    roc_aucs, pr_aucs = [], []
    precisions, specificities, recalls, mccs = [], [], [], []

    for _ in tqdm(range(n_bootstraps)):
        # Stratified resampling
        pos_sample = rng.choice(positive_indices, size=len(positive_indices), replace=True)
        neg_sample = rng.choice(negative_indices, size=len(negative_indices), replace=True)
        sample_indices = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(sample_indices)

        y_true_bs = y_true[sample_indices]
        y_pred_proba_bs = y_pred_proba[sample_indices]
        y_pred_label_bs = y_pred_label[sample_indices]

        try:
            roc_aucs.append(roc_auc_score(y_true_bs, y_pred_proba_bs))
            pr_aucs.append(average_precision_score(y_true_bs, y_pred_proba_bs))
            precisions.append(precision_score(y_true_bs, y_pred_label_bs, zero_division=0))
            recalls.append(recall_score(y_true_bs, y_pred_label_bs, zero_division=0))
            mccs.append(matthews_corrcoef(y_true_bs, y_pred_label_bs))
        
        # Specificity
            cm = confusion_matrix(y_true_bs, y_pred_label_bs)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)

        except:
            continue

    def summary_stats(metric_list):
        return {
            "mean": np.mean(metric_list),
            "ci_lower": np.percentile(metric_list, 2.5),
            "ci_upper": np.percentile(metric_list, 97.5)
        }

    return {
        "roc_auc": summary_stats(roc_aucs),
        "pr_auc": summary_stats(pr_aucs),
        "precision": summary_stats(precisions),
        "recall": summary_stats(recalls),
        "specificity": summary_stats(specificities),
        "mcc": summary_stats(mccs),
    }

## 
y_pred_proba = test["y_pred_proba"]  # Get probabilities for class 1
y_pred_label = (y_pred_proba >= 0.5).astype(int)  # Apply 0.5 threshold (or whatever you used)

results = stratified_bootstrap_metrics(y_test, y_pred_proba, y_pred_label)

for metric, stats in results.items():
    print(f"{metric}: {stats['mean']:.3f} (95% CI: {stats['ci_lower']:.3f}–{stats['ci_upper']:.3f})")

## 
y_pred_proba = test_specific["y_pred_proba"]  # Get probabilities for class 1
y_pred_label = (y_pred_proba >= 0.5).astype(int)  # Apply 0.5 threshold (or whatever you used)

results = stratified_bootstrap_metrics(y_test_specific, y_pred_proba, y_pred_label)

for metric, stats in results.items():
    print(f"{metric}: {stats['mean']:.3f} (95% CI: {stats['ci_lower']:.3f}–{stats['ci_upper']:.3f})")



f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test_specific, y_test_specific, bst, 0.5, y_pred_proba=test_specific["y_pred_proba"]
)

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test_specific.values, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test_specific.values, y_pred)).plot()

