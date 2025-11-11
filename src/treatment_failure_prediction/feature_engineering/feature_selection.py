"""
feature_selection.py

Performs simple L1-based feature selection using Lasso:

- Loads the full feature matrix and test patient IDs.
- Splits into train/test based on patientid.
- Uses the *last* outcome column containing 'outc' as the target.
- Excludes patient IDs, timestamps, outcome columns, and NCCN/IPI columns.
- Fits a Lasso model (alpha = 0.01).
- Selects all predictors with |coef| > 0.001.
- Writes the selected feature names to: results/feature_names_all.csv

Behaviour matches the original script exactly.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

# ----------------------------------------------------------------------
# Configuration and data loading
# ----------------------------------------------------------------------
seed = 46  # kept for parity with other scripts, though not used in Lasso

test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)

# ----------------------------------------------------------------------
# Define outcome and predictor columns
# ----------------------------------------------------------------------
outcome_columns = [col for col in feature_matrix.columns if "outc" in col]
# Use the last outcome column (as in original code)
outcome = outcome_columns[-1]

cols_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"]
cols_to_leave.extend(outcome_columns)

ipi_cols = [col for col in feature_matrix.columns if "NCCN_" in col]
cols_to_leave.extend(ipi_cols)

predictor_columns = [col for col in train.columns if col not in cols_to_leave]

# Exclude these specific predictors (unchanged from original)
predictor_columns = [
    col
    for col in predictor_columns
    if col not in ["pred_RKKP_subtype_fallback_-1", "pred_RKKP_hospital_fallback_-1"]
]

# ----------------------------------------------------------------------
# Lasso feature selection
# ----------------------------------------------------------------------
lasso = Lasso(alpha=0.01)
lasso.fit(X=train[predictor_columns], y=train[outcome])

lasso_coef = np.abs(lasso.coef_)
selected_idx = [i for i, coef in enumerate(lasso_coef) if coef > 0.001]
feature_column_names = train[predictor_columns].columns[selected_idx]

# ----------------------------------------------------------------------
# Save selected features
# ----------------------------------------------------------------------
pd.DataFrame(feature_column_names, columns=["features"]).to_csv(
    "results/feature_names_all.csv",
    index=False,
)
