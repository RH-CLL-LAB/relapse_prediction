import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
import joblib
from helpers.processing_helper import get_features_and_outcomes, clip_values

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

seed = 46
np.random.seed(seed)

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

wrong_ids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]
feature_matrix = feature_matrix[~feature_matrix["patientid"].isin(wrong_ids)].reset_index(drop=True)
feature_matrix.replace(-1, np.nan, inplace=True)

# Core IPI predictors
features = [
    "pred_RKKP_age_diagnosis_fallback_-1",
    "pred_RKKP_LDH_diagnosis_fallback_-1",
    "pred_RKKP_AA_stage_diagnosis_fallback_-1",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
    "pred_RKKP_PS_diagnosis_fallback_-1",
]

test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)

outcome_columns = [c for c in feature_matrix.columns if "outc" in c]
outcome = outcome_columns[-1]

for col in tqdm(features, desc="Clipping feature values"):
    clip_values(train, test, col)

# Get stratified features/outcomes
(
    X_train_smtom,
    y_train_smtom,
    X_test,
    y_test,
    X_test_specific,
    y_test_specific,
    test_specific,
) = get_features_and_outcomes(
    train, test, WIDE_DATA, outcome, features,
    specific_immunotherapy=False, none_chop_like=False,
)

# Train IPI-only model
bst = XGBClassifier(
    missing=-1,
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

# Save predictions + model
test_specific["y_pred_proba_ml_ipi"] = bst.predict_proba(X_test_specific)[:, 1]
os.makedirs("plots", exist_ok=True)

test_specific.to_csv("data/test_specific_ml_ipi_and_comparators.csv", index=False)
bst.save_model("results/model_ipi_only.json")
