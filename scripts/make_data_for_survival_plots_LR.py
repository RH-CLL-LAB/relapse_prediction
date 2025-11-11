# TP FP TN FN
from datetime import timedelta
import pandas as pd
import xgboost
import numpy as np
import joblib

bst = xgboost.XGBClassifier()

bst.load_model("results/models/model_all.json")

X_test = pd.read_csv("results/X_test.csv")
X_test_specific = pd.read_csv("results/X_test_specific.csv")

test = pd.read_csv("results/test.csv")
test_specific = pd.read_csv("results/test_specific.csv")

lr_probs = joblib.load("results/lr_predictions.pkl")

test["lr_probs"] = lr_probs["y_pred_proba"]

test_specific = test_specific.merge(test[["patientid", "lr_probs"]])

def make_prediction_categorical(y_prob):
    if pd.isnull(y_prob):
        return None
    if y_prob < 0.1:
        return "Low"
    if y_prob < 0.3:
        return "Low-Intermediate"
    if y_prob < 0.65:
        return "Intermediate-High"
    if y_prob >= 0.65:
        return "High"


y_pred = bst.predict_proba(X_test_specific)
y_prob_categorical = [make_prediction_categorical(x[1]) for x in y_pred]

adjusted_y_pred_binary = np.argmax(y_pred, axis=1)
adjusted_y_pred_binary = [1 if x[1] > 0.2 else 0 for x in y_pred]

test_specific["y_pred"] = adjusted_y_pred_binary

test_specific = test_specific.reset_index(drop=True)

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_specific["lr_categorical"] = test_specific["lr_probs"].apply(
    make_prediction_categorical
)

test_specific_plotting = test_specific[["patientid", "y_pred", "lr_probs", "lr_categorical"]].merge(WIDE_DATA)
test_specific_plotting["date_event"] = test_specific_plotting["relapse_date"]
test_specific_plotting.loc[test_specific_plotting["date_event"].notna(), "event"] = 1

test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = test_specific_plotting[test_specific_plotting["relapse_date"].isna()]["date_death"]
test_specific_plotting.loc[
    (test_specific_plotting["date_event"].notna())
    & (test_specific_plotting["event"].isna()),
    "event",
] = 2
test_specific_plotting.loc[test_specific_plotting["date_event"].isna(), "event"] = 0
# test_specific_plotting.loc[test_specific_plotting["date_event"].isna(), "date_event"] = test_specific_plotting[test_specific_plotting["date_event"].isna()]["date_treatment_1st_line"].apply(lambda x: x + timedelta(days = 730))
test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = pd.to_datetime("2024-01-01")
test_specific_plotting["days_to_event"] = (
    test_specific_plotting["date_event"]
    - test_specific_plotting["date_treatment_1st_line"]
).dt.days
test_specific_plotting["group"] = test_specific_plotting["y_pred"].apply(
    lambda x: 1 if x > 0 else 0
)
test_specific_plotting["risk_prediction"] = y_prob_categorical
# test_specific_plotting = test_specific_plotting[
#     test_specific_plotting["age_at_tx"] < 75
# ].reset_index(drop=True)

test_specific_plotting[
    [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "lr_probs",
        "lr_categorical",
        "risk_prediction",
        "age_at_tx",
    ]
].to_csv("results/km_data_lyfo_FCR_LR.csv", index=False)


test_specific_plotting[test_specific_plotting["age_at_tx"] < 75].reset_index(drop=True)[
    [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "lr_probs",
        "lr_categorical",
        "risk_prediction",
        "age_at_tx",
    ]
].to_csv("results/km_data_lyfo_FCR_under_75_LR.csv", index=False)

y_pred = bst.predict_proba(X_test)
y_prob_categorical = [make_prediction_categorical(x[1]) for x in y_pred]
adjusted_y_pred_binary = [1 if x[1] > 0.2 else 0 for x in y_pred]
test["y_pred"] = adjusted_y_pred_binary

test = test.reset_index(drop=True)

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
test["lr_categorical"] = test["lr_probs"].apply(
    make_prediction_categorical
)


test_specific_plotting = test[["patientid", "y_pred", "lr_probs", "lr_categorical"]].merge(WIDE_DATA)
test_specific_plotting["date_event"] = test_specific_plotting["relapse_date"]
test_specific_plotting.loc[test_specific_plotting["date_event"].notna(), "event"] = 1

test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = test_specific_plotting[test_specific_plotting["relapse_date"].isna()]["date_death"]
test_specific_plotting.loc[
    (test_specific_plotting["date_event"].notna())
    & (test_specific_plotting["event"].isna()),
    "event",
] = 2

test_specific_plotting.loc[test_specific_plotting["date_event"].isna(), "event"] = 0
test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = pd.to_datetime("2024-01-01")
test_specific_plotting["days_to_event"] = (
    test_specific_plotting["date_event"]
    - test_specific_plotting["date_treatment_1st_line"]
).dt.days
test_specific_plotting["risk_prediction"] = y_prob_categorical

test_specific_plotting["group"] = test_specific_plotting["y_pred"].apply(
    lambda x: 1 if x > 0 else 0
)

# test_specific_plotting = test_specific_plotting[
#     test_specific_plotting["age_at_tx"] < 75
# ].reset_index(drop=True)

test_specific_plotting[
    [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "lr_probs",
        "lr_categorical",
        "risk_prediction",
        "age_at_tx",
    ]
].to_csv("results/km_data_lyfo_FCR_all_LR.csv", index=False)

test_specific_plotting[test_specific_plotting["age_at_tx"] < 75].reset_index(drop=True)[
    [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "lr_probs",
        "lr_categorical",
        "risk_prediction",
        "age_at_tx",
    ]
].to_csv("results/km_data_lyfo_FCR_all_under_75_LR.csv", index=False)
