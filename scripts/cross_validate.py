from sklearn.model_selection import train_test_split, StratifiedKFold
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
from tqdm import tqdm
from xgboost import XGBClassifier
from helpers.constants import *
from helpers.processing_helper import *


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")

seed = 46
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")
# feature_matrix = feature_matrix[feature_matrix["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop = True)
features = pd.read_csv("results/feature_names_all.csv")["features"].values

test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)

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


results_dataframes = []
results_stratified = []
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train_splitter = train.copy()

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
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

skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
skf.get_n_splits(X=train[features], y=train["group"])

for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    train = train_splitter.iloc[train_index]
    test = train_splitter.iloc[test_index]
    (
        X_train_smtom,
        y_train_smtom,
        X_test,
        y_test,
        X_test_specific,
        y_test_specific,
        test_specific,
    ) = get_features_and_outcomes(train, test, WIDE_DATA, outcome, col_to_leave)

    test_specific = test_specific.merge(
        WIDE_DATA[["patientid", "CNS_IPI_diagnosis"]]
    ).reset_index(drop=True)

    test_specific.loc[
        test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] == -1,
        "pred_RKKP_NCCN_IPI_diagnosis_fallback_-1",
    ] = None
    y_pred = [
        1 if x >= 6 else 0
        for x in test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"]
    ]

    # y_pred = [
    #     0 if (pd.isnull(x)) or x < 4 else 1 for x in test_specific["CNS_IPI_diagnosis"]
    # ]

    weird_probabilities = (test_specific["CNS_IPI_diagnosis"] / 7).values
    # fix nans - should we exclude them? probably yes
    weird_probabilities = [
        (i, x) for i, x in enumerate(weird_probabilities) if pd.notnull(x)
    ]
    indexes = [x[0] for x in weird_probabilities]
    weird_probabilities = [x[1] for x in weird_probabilities]

    weird_probabilities_NCCN = (
        test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] / 9
    ).values
    weird_probabilities_NCCN = [
        (i, x) for i, x in enumerate(weird_probabilities_NCCN) if pd.notnull(x)
    ]
    indexes_NCCN = [x[0] for x in weird_probabilities_NCCN]
    weird_probabilities_NCCN = [x[1] for x in weird_probabilities_NCCN]

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

results_dataframes = []
results_stratified = []
train_splitter = train.copy()
features_original = features.copy()

for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    train = train_splitter.iloc[train_index]
    test = train_splitter.iloc[test_index]
    features = features_original.copy()

    features = list(features)

    for j in supplemental_columns:
        if j not in features:
            features.append(j)

    from tqdm import tqdm

    features = [
        "pred_RKKP_age_diagnosis_fallback_-1",
        "pred_RKKP_LDH_diagnosis_fallback_-1",
        "pred_RKKP_AA_stage_diagnosis_fallback_-1",
        "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
        "pred_RKKP_PS_diagnosis_fallback_-1",
    ]

    for column in tqdm(features):
        clip_values(train, test, column)

    (
        X_train_smtom,
        y_train_smtom,
        X_test,
        y_test,
        X_test_specific,
        y_test_specific,
        test_specific,
    ) = get_features_and_outcomes(train, test, WIDE_DATA, outcome, col_to_leave)

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

    results, best_threshold = check_performance_across_thresholds(
        X_test_specific, y_test_specific, bst
    )
    f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
        X_test_specific, y_test_specific, bst, 0.3
    )
    y_pred = bst.predict_proba(X_test_specific).astype(float)
    y_pred = [1 if x[1] > tested_threshold else 0 for x in y_pred]

    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(f"Best Threshold: {best_threshold}")
    print(cm)

    results_stratified.append(
        {
            "threshold": best_threshold,
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

results_dataframes = []
results_stratified = []
train_splitter = train.copy()
features_original = features.copy()
for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    train = train_splitter.iloc[train_index]
    test = train_splitter.iloc[test_index]
    features = features_original.copy()
    # train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)
    # test = test[test["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

    features = list(features)

    for j in supplemental_columns:
        if j not in features:
            features.append(j)

    from tqdm import tqdm

    for column in tqdm(features):
        clip_values(train, test, column)

    (
        X_train_smtom,
        y_train_smtom,
        X_test,
        y_test,
        X_test_specific,
        y_test_specific,
        test_specific,
    ) = get_features_and_outcomes(train, test, WIDE_DATA, outcome, col_to_leave)

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
        nthread=10,  # was 6 before-could bumpt this for faster fitting probably?
        # scale_pos_weight=scale_pos_weight,
        random_state=i,
    )

    bst.fit(X_train_smtom, y_train_smtom)

    y_pred = bst.predict_proba(X_test_specific).astype(float)
    y_pred = [1 if x[1] > tested_threshold else 0 for x in y_pred]

    f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
        X_test_specific, y_test_specific, bst, 0.3
    )

    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(f"Best Threshold: {best_threshold}")
    print(cm)

    results_stratified.append(
        {
            "threshold": best_threshold,
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


results_dataframes = []
results_stratified = []
train_splitter = train.copy()
features_original = features.copy()
features_original = list(feature_matrix.columns)
features_original = [x for x in features_original if x not in col_to_leave]
supplemental_columns = [
    "pred_RKKP_hospital_fallback_-1",
    "pred_RKKP_subtype_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
]
features_original = [x for x in features_original if x not in supplemental_columns]
for i in range(5):
    features = features_original.copy()
    train, test = train_test_split(
        train_splitter, test_size=0.2, random_state=i, stratify=train_splitter["group"]
    )

    from tqdm import tqdm

    for column in tqdm(features):
        clip_values(train, test, column)

    features = list(features)
    features.extend(supplemental_columns)

    X_train_smtom = train[[x for x in train.columns if x in features]]
    y_train_smtom = train[outcome]
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
        nthread=10,  # was 6 before-could bumpt this for faster fitting probably?
        # scale_pos_weight=scale_pos_weight,
        random_state=i,
    )

    bst.fit(X_train_smtom, y_train_smtom)

    test_specific = test[test["pred_RKKP_subtype_fallback_-1"] == 0]

    WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

    test_specific = test_specific.merge(
        WIDE_DATA[["patientid", "regime_1_chemo_type_1st_line"]]
    )

    test_specific = test_specific[
        test_specific["regime_1_chemo_type_1st_line"].isin(included_treatments)
    ].reset_index(drop=True)

    test_specific = test_specific.drop(columns="regime_1_chemo_type_1st_line")

    X_test_specific = test_specific[[x for x in test_specific.columns if x in features]]
    y_test_specific = test_specific[outcome]

    X_test = test[[x for x in test.columns if x in features]]
    y_test = test[outcome]

    results, best_threshold = check_performance_across_thresholds(
        X_test_specific, y_test_specific, bst
    )
    tested_threshold = 0.5

    results_dataframes.append(results)

    y_pred = bst.predict_proba(X_test_specific).astype(float)
    y_pred = [1 if x[1] > tested_threshold else 0 for x in y_pred]

    f1 = f1_score(y_test_specific.values, y_pred)
    roc_auc = roc_auc_score(
        y_test_specific.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
    recall = recall_score(y_test_specific.values, y_pred)
    precision = precision_score(y_test_specific.values, y_pred, zero_division=1)
    specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
    pr_auc = average_precision_score(
        y_test_specific.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
    mcc = matthews_corrcoef(y_test_specific.values, y_pred)
    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(f"Best Threshold: {best_threshold}")
    print(cm)

    results_stratified.append(
        {
            "threshold": best_threshold,
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


results_stratified_df = pd.DataFrame(results_stratified)

results_stratified_df[
    [x for x in results_stratified_df.columns if x not in ["confusion_matrix", "seed"]]
].melt().groupby("variable").agg(mean=("value", "mean"), std=("value", "std"))
