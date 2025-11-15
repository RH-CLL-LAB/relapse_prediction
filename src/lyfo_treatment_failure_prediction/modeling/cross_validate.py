from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

from tqdm import tqdm
from xgboost import XGBClassifier

from helpers.constants import *
from helpers.processing_helper import *

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")

# ---------------------------------------------------------------------
# Data loading and basic setup
# ---------------------------------------------------------------------

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

# Drop patients with missing age at diagnosis
wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]

seed = 46
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

# Remove patients with missing age from the feature matrix
feature_matrix = feature_matrix[
    ~feature_matrix["patientid"].isin(wrong_patientids)
].reset_index(drop=True)

# Model features (as selected via LASSO / feature_selection.py)
features = pd.read_csv("results/feature_names_all.csv")["features"].values

# Split into (global) train / test by patient IDs (outer split)
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)

# Outcome: use the last "outc_" column
outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[-1]

# Columns that should never be used as predictors
col_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"]
col_to_leave.extend(outcome_column)

ipi_cols = [x for x in feature_matrix.columns if "NCCN_" in x]
col_to_leave.extend(ipi_cols)

# (Not actually used later, but kept for completeness)
predictor_columns = [x for x in train.columns if x not in col_to_leave]
predictor_columns = [
    x
    for x in predictor_columns
    if x not in ["pred_RKKP_subtype_fallback_-1", "pred_RKKP_hospital_fallback_-1"]
]

# ---------------------------------------------------------------------
# Compute CNS-IPI for baseline comparisons
# ---------------------------------------------------------------------

WIDE_DATA["CNS_IPI_diagnosis"] = WIDE_DATA.apply(
    lambda x: calculate_CNS_IPI(
        x["age_diagnosis"],
        x["LDH_diagnosis"],  # needs to be normalized
        x["AA_stage_diagnosis"],
        x["extranodal_disease_diagnosis"],
        x["PS_diagnosis"],
        x["kidneys_diagnosis"],
    ),
    axis=1,
)

# Inner CV (5-fold stratified on "group")
train_splitter = train.copy()
skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
skf.get_n_splits(X=train[features], y=train["group"])

# ---------------------------------------------------------------------
# 1) Baseline: cross-validation for NCCN / CNS risk scores
# ---------------------------------------------------------------------

results_dataframes = []
results_stratified = []

for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    # Fold-level train/test
    train_fold = train_splitter.iloc[train_index]
    test_fold = train_splitter.iloc[test_index]

    # We only need y_test_specific & test_specific from this helper
    (
        _X_train_smtom,
        _y_train_smtom,
        _X_test,
        _y_test,
        X_test_specific,
        y_test_specific,
        test_specific,
    ) = get_features_and_outcomes(
        train_fold, test_fold, WIDE_DATA, outcome, col_to_leave
    )

    # Attach CNS-IPI to the fold-specific subset
    test_specific = test_specific.merge(
        WIDE_DATA[["patientid", "CNS_IPI_diagnosis"]]
    ).reset_index(drop=True)

    # NCCN-based prediction: using pred_RKKP_NCCN_IPI_diagnosis_fallback_-1
    test_specific.loc[
        test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] == -1,
        "pred_RKKP_NCCN_IPI_diagnosis_fallback_-1",
    ] = None

    # Binary label from NCCN: 1 if NCCN ≥ 6, else 0
    y_pred = [
        1 if x >= 6 else 0
        for x in test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"]
    ]

    # "Probabilities" derived from raw scores (CNS-IPI / 7 and NCCN / 9)
    weird_probabilities = (test_specific["CNS_IPI_diagnosis"] / 7).values
    weird_probabilities = [
        (idx, x) for idx, x in enumerate(weird_probabilities) if pd.notnull(x)
    ]
    indexes = [x[0] for x in weird_probabilities]
    weird_probabilities = [x[1] for x in weird_probabilities]

    weird_probabilities_NCCN = (
        test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] / 9
    ).values
    weird_probabilities_NCCN = [
        (idx, x) for idx, x in enumerate(weird_probabilities_NCCN) if pd.notnull(x)
    ]
    indexes_NCCN = [x[0] for x in weird_probabilities_NCCN]
    weird_probabilities_NCCN = [x[1] for x in weird_probabilities_NCCN]

    # Metrics for NCCN prediction
    f1 = f1_score(y_test_specific.values, y_pred)
    roc_auc = roc_auc_score(
        y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
    )
    recall = recall_score(y_test_specific.values, y_pred)
    precision = precision_score(y_test_specific.values, y_pred, zero_division=1)
    specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
    pr_auc = average_precision_score(
        y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
    )
    mcc = matthews_corrcoef(y_test_specific.values, y_pred)
    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"Fold {i} — NCCN/CNS baseline")
    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(cm)

    results_stratified.append(
        {
            "threshold": 0.5,
            "f1": f1,
            "roc_auc": roc_auc,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "pr_auc": pr_auc,
            "mcc": mcc,
            "confusion_matrix": cm,
            "seed": i,
        }
    )

# ---------------------------------------------------------------------
# 2) CV for model using only NCCN predictors (fixed threshold = 0.5)
# ---------------------------------------------------------------------

results_dataframes = []
results_stratified = []
train_splitter = train.copy()
features_original = features.copy()

for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    train_fold = train_splitter.iloc[train_index]
    test_fold = train_splitter.iloc[test_index]
    features = features_original.copy()

    # Ensure supplemental columns are present in the feature list
    features = list(features)
    for j in supplemental_columns:
        if j not in features:
            features.append(j)

    # Overwrite with the 5 NCCN predictors (as in original script)
    features = [
        "pred_RKKP_age_diagnosis_fallback_-1",
        "pred_RKKP_LDH_diagnosis_fallback_-1",
        "pred_RKKP_AA_stage_diagnosis_fallback_-1",
        "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
        "pred_RKKP_PS_diagnosis_fallback_-1",
    ]

    for column in tqdm(features, desc=f"Clipping NCCN features, fold {i}"):
        clip_values(train_fold, test_fold, column)

    (
        X_train_smtom,
        y_train_smtom,
        X_test,
        y_test,
        X_test_specific,
        y_test_specific,
        test_specific,
    ) = get_features_and_outcomes(train_fold, test_fold, WIDE_DATA, outcome, features)

    bst = XGBClassifier(
        missing=-1,
        n_estimators=3000,  # was 2000 before
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        gamma=0,
        subsample=1,
        colsample_bytree=0.9,
        objective="binary:logistic",
        reg_alpha=10,
        nthread=10,
        random_state=i,
    )

    bst.fit(X_train_smtom, y_train_smtom)

    # In this script, we are effectively using a fixed threshold = 0.5
    threshold = 0.5
    f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
        X_test_specific, y_test_specific, bst, threshold, y_pred_proba=[]
    )

    y_pred_proba = bst.predict_proba(X_test_specific).astype(float)
    y_pred = [1 if x[1] > threshold else 0 for x in y_pred_proba]

    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"Fold {i} — XGB with NCCN predictors")
    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(f"Threshold: {threshold}")
    print(cm)

    results_stratified.append(
        {
            "threshold": threshold,
            "f1": f1,
            "roc_auc": roc_auc,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "pr_auc": pr_auc,
            "mcc": mcc,
            "confusion_matrix": cm,
            "seed": i,
        }
    )

# ---------------------------------------------------------------------
# 3) CV for full ML model with extended features (threshold = 0.2)
# ---------------------------------------------------------------------

results_dataframes = []
results_stratified = []
train_splitter = train.copy()
features_original = features.copy()
tested_threshold = 0.2

for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    train_fold = train_splitter.iloc[train_index]
    test_fold = train_splitter.iloc[test_index]
    features = features_original.copy()

    # Optionally filter to DLBCL only (original script had this commented out)
    # train_fold = train_fold[train_fold["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)
    # test_fold = test_fold[test_fold["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

    features = list(features)

    for j in supplemental_columns:
        if j not in features:
            features.append(j)

    for column in tqdm(features, desc=f"Clipping all features, fold {i}"):
        clip_values(train_fold, test_fold, column)

    # Add subtype / hospital / sex predictors
    features.extend(
        [
            "pred_RKKP_subtype_fallback_-1",
            "pred_RKKP_hospital_fallback_-1",
            "pred_RKKP_sex_fallback_-1",
        ]
    )

    (
        X_train_smtom,
        y_train_smtom,
        X_test,
        y_test,
        X_test_specific,
        y_test_specific,
        test_specific,
    ) = get_features_and_outcomes(train_fold, test_fold, WIDE_DATA, outcome, features)

    bst = XGBClassifier(
        missing=-1,
        n_estimators=3000,  # was 2000 before
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        gamma=0,
        subsample=1,
        colsample_bytree=0.9,
        objective="binary:logistic",
        reg_alpha=10,
        nthread=10,  # was 6 before
        random_state=i,
    )

    bst.fit(X_train_smtom, y_train_smtom)

    y_pred_proba = bst.predict_proba(X_test_specific).astype(float)
    y_pred = [1 if x[1] > tested_threshold else 0 for x in y_pred_proba]

    f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
        X_test_specific, y_test_specific, bst, tested_threshold
    )

    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"Fold {i} — Full XGB model (threshold={tested_threshold})")
    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(f"Threshold: {tested_threshold}")
    print(cm)

    results_stratified.append(
        {
            "threshold": tested_threshold,
            "f1": f1,
            "roc_auc": roc_auc,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "pr_auc": pr_auc,
            "mcc": mcc,
            "confusion_matrix": cm,
            "seed": i,
        }
    )

# ---------------------------------------------------------------------
# Aggregate results from the last CV experiment
# ---------------------------------------------------------------------

results_stratified_df = pd.DataFrame(results_stratified)

summary = (
    results_stratified_df[
        [c for c in results_stratified_df.columns if c not in ["confusion_matrix", "seed"]]
    ]
    .melt()
    .groupby("variable")
    .agg(mean=("value", "mean"), std=("value", "std"))
)

# `summary` now holds mean ± std across folds for the last experiment.
