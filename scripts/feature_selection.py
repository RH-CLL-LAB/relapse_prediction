import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold

seed = 46
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")


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

lasso = Lasso(alpha=0.01)
lasso.fit(X=train[predictor_columns], y=train[outcome])

lasso_coef = np.abs(lasso.coef_)
features = [i for i, x in enumerate(lasso_coef) if x > 0.001]
feature_column_names = train[predictor_columns].columns[features]

pd.DataFrame(feature_column_names, columns=["features"]).to_csv(
    "results/feature_names_all.csv", index=False
)
