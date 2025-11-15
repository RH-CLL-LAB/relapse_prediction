import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier

from helpers.constants import *
from helpers.processing_helper import clip_values
# sql_helper imported in original but unused here

import seaborn as sns
sns.set_context("paper")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
seed = 46
DLBCL_ONLY = False
OUTPUT_DIR = "data/individual_outcomes"

# ---------------------------------------------------------------------
# IO / data prep
# ---------------------------------------------------------------------
WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

# Remove wrong patient ids (as in other scripts)
wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]
feature_matrix = (
    feature_matrix[~feature_matrix["patientid"].isin(wrong_patientids)]
    .reset_index(drop=True)
)

# Match other scripts: treat -1 as missing when modeling
feature_matrix = feature_matrix.replace(-1, np.nan)

# Split train / test on held-out patient ids
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)

if DLBCL_ONLY:
    train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

# Identify outcome columns (all with "outc" in name)
outcome_columns = [c for c in feature_matrix.columns if "outc" in c]

# Exclude leakage and non-predictor columns from modeling
col_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"]
col_to_leave.extend(outcome_columns)
col_to_leave.extend([c for c in feature_matrix.columns if "NCCN_" in c])

# Base features from file
features = list(pd.read_csv("results/feature_names_all.csv")["features"].values)

# Define supplemental columns (⚠️ must be defined BEFORE use)
supplemental_columns = [
    "pred_RKKP_hospital_fallback_-1",
    "pred_RKKP_subtype_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
]

# Ensure supplemental columns are included (without duplicates)
for col in supplemental_columns:
    if col not in features:
        features.append(col)

# Clip extreme values (train-driven) consistently across train/test
for col in tqdm(features, desc="Clipping feature values"):
    if col in train.columns:
        clip_values(train, test, col)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Helper: build matrices per outcome using your original helper
# ---------------------------------------------------------------------
def get_features_and_outcomes_for_outcome(outcome_name: str):
    from helpers.processing_helper import get_features_and_outcomes
    return get_features_and_outcomes(
        train=train,
        test=test,
        WIDE_DATA=WIDE_DATA,
        outcome=outcome_name,
        feature_list=features,
        specific_immunotherapy=False,
        none_chop_like=False,
        only_DLBCL_filter=False,
    )

# ---------------------------------------------------------------------
# Model template (kept identical to preserve behavior)
# ---------------------------------------------------------------------
def make_model(random_state: int):
    return XGBClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        gamma=0,
        subsample=1,
        colsample_bytree=0.9,
        objective="binary:logistic",
        reg_alpha=10,
        nthread=10,           # keep for backward-compat; newer xgboost uses n_jobs
        random_state=random_state,
    )

# ---------------------------------------------------------------------
# Train one model per outcome and save test probabilities
# ---------------------------------------------------------------------
for outcome in outcome_columns:
    (
        X_train_smtom,
        y_train_smtom,
        X_test,
        y_test,
        X_test_specific,
        y_test_specific,
        test_specific,
    ) = get_features_and_outcomes_for_outcome(outcome)

    model = make_model(seed)
    model.fit(X_train_smtom, y_train_smtom)

    # Positive-class probability
    proba = model.predict_proba(X_test_specific).astype(float)[:, 1]
    test_specific[f"ml_{outcome}"] = proba

    out_path = os.path.join(OUTPUT_DIR, f"test_specific_ml_{outcome}.csv")
    test_specific.to_csv(out_path, index=False)
