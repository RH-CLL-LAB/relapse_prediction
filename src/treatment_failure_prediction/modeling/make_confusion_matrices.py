import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from helpers.constants import *
from helpers.processing_helper import *

sns.set_context("paper")

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
seed = 46
DLBCL_ONLY = False

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

# Filter out problematic patients
wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]

feature_matrix = (
    pd.read_pickle("results/feature_matrix_all.pkl")
    .query("patientid not in @wrong_patientids")
    .reset_index(drop=True)
)
feature_matrix.replace(-1, np.nan, inplace=True)

# Test/train split
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)

if DLBCL_ONLY:
    train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

# ---------------------------------------------------------------------
# Define outcome and features
# ---------------------------------------------------------------------
outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[-1]

col_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"]
col_to_leave.extend(outcome_column)
col_to_leave.extend([x for x in feature_matrix.columns if "NCCN_" in x])

# Add selected and supplemental features
features = list(pd.read_csv("results/feature_names_all.csv")["features"].values)
supplemental_columns = [
    "pred_RKKP_hospital_fallback_-1",
    "pred_RKKP_subtype_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
]
for col in supplemental_columns:
    if col not in features:
        features.append(col)

# Clip extreme values to avoid leakage
for col in tqdm(features, desc="Clipping features"):
    clip_values(train, test, col)

# ---------------------------------------------------------------------
# Prepare outcome-specific datasets
# ---------------------------------------------------------------------
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
    only_DLBCL_filter=False,
)

# ---------------------------------------------------------------------
# Train base XGBoost model
# ---------------------------------------------------------------------
bst = XGBClassifier(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=3,
    gamma=0,
    subsample=1,
    colsample_bytree=0.9,
    objective="binary:logistic",
    reg_alpha=10,
    nthread=10,
    random_state=seed,
)

bst.fit(X_train_smtom, y_train_smtom)

# ---------------------------------------------------------------------
# Overall confusion matrix
# ---------------------------------------------------------------------
for threshold in [0.3, 0.5]:
    y_pred_probs = bst.predict_proba(X_test_specific)[:, 1]
    y_pred_labels = (y_pred_probs > threshold).astype(int)

    cm = confusion_matrix(y_test_specific.values, y_pred_labels)
    plot_confusion_matrix(cm)
    plt.savefig(
        f"plots/cm_treatment_failure_2_years_ml_all_{threshold}.pdf",
        bbox_inches="tight",
    )
    plt.close()

# ---------------------------------------------------------------------
# Confusion matrices by lymphoma subtype
# ---------------------------------------------------------------------
subtypes = pd.Categorical(WIDE_DATA["subtype"]).categories

for subtype_number, subtype_name in enumerate(subtypes):
    for threshold in [0.3, 0.5]:
        # Subset test data for this subtype
        subtype_test = test[test["pred_RKKP_subtype_fallback_-1"] == subtype_number]
        if subtype_test.empty:
            continue

        # Recreate X/y for this subset
        (
            _,
            _,
            _,
            _,
            X_test_sub,
            y_test_sub,
            _,
        ) = get_features_and_outcomes(
            train,
            subtype_test,
            WIDE_DATA,
            outcome,
            features,
            specific_immunotherapy=False,
            none_chop_like=False,
            only_DLBCL_filter=False,
        )

        y_pred_probs = bst.predict_proba(X_test_sub)[:, 1]
        y_pred_labels = (y_pred_probs > threshold).astype(int)

        cm = confusion_matrix(y_test_sub.values, y_pred_labels)
        plot_confusion_matrix(cm)
        plt.savefig(
            f"plots/cm_treatment_failure_2_years_ml_all_{threshold}_{subtype_name}.pdf",
            bbox_inches="tight",
        )
        plt.close()

# ---------------------------------------------------------------------
# Confusion matrices for "Other Lymphomas" (non-DLBCL)
# ---------------------------------------------------------------------
for threshold in [0.3, 0.5]:
    ol_test = test[test["pred_RKKP_subtype_fallback_-1"] != 0].reset_index(drop=True)
    if ol_test.empty:
        continue

    (
        _,
        _,
        _,
        _,
        X_test_ol,
        y_test_ol,
        _,
    ) = get_features_and_outcomes(
        train,
        ol_test,
        WIDE_DATA,
        outcome,
        features,
        specific_immunotherapy=False,
        none_chop_like=False,
        only_DLBCL_filter=False,
    )

    y_pred_probs = bst.predict_proba(X_test_ol)[:, 1]
    y_pred_labels = (y_pred_probs > threshold).astype(int)

    cm = confusion_matrix(y_test_ol.values, y_pred_labels)
    plot_confusion_matrix(cm)
    plt.savefig(
        f"plots/cm_treatment_failure_2_years_ml_all_{threshold}_OL.pdf",
        bbox_inches="tight",
    )
    plt.close()
