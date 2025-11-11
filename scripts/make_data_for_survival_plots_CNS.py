# TP FP TN FN
import math
from datetime import timedelta

import pandas as pd
import xgboost
import numpy as np


bst = xgboost.XGBClassifier()

bst.load_model("results/models/model_all.json")

X_test = pd.read_csv("results/X_test.csv")
X_test_specific = pd.read_csv("results/X_test_specific.csv")

test = pd.read_csv("results/test.csv")
test_specific = pd.read_csv("results/test_specific.csv")
WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")


def calculate_CNS_IPI(age, ldh, aa_stage, extranodal, ps, kidneys_diagnosis):
    # NOTE NOW RETURNING NANS FOR PATIENTS WITH MISSING VALUES

    if any(
        [
            math.isnan(age),
            math.isnan(ldh),
            math.isnan(aa_stage),
            # math.isnan(extranodal),
            math.isnan(ps),
            math.isnan(kidneys_diagnosis),
        ]
    ):
        return pd.NA
    total_score = 0
    if age > 60:
        total_score += 1
    if age < 70:
        upper_limit = 205
    else:
        upper_limit = 255

    if ldh > upper_limit:
        total_score += 1

    if ps > 1:
        total_score += 1

    if aa_stage > 2:
        total_score += 1
    if extranodal == 1:
        total_score += 1
    if kidneys_diagnosis:
        total_score += 1

    return total_score


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


def make_CNS_categorical(cns):
    if pd.isnull(cns):
        return None
    if cns < 2:
        return "Low"
    if cns < 4:
        return "Intermediate"
    else:
        return "High"


def make_prediction_categorical(y_prob):
    if pd.isnull(y_prob):
        return None
    if y_prob < 0.3:
        return "Low"
    if y_prob < 0.7:
        return "Intermediate"
    if y_prob >= 0.7:
        return "High"


def make_prediction_categorical(y_prob):
    if pd.isnull(y_prob):
        return None
    if y_prob < 0.2:
        return "Low"
    if y_prob < 0.5:
        return "Intermediate"
    if y_prob >= 0.5:
        return "High"


y_pred = bst.predict_proba(X_test_specific)
y_prob_categorical = [make_prediction_categorical(x[1]) for x in y_pred]

adjusted_y_pred_binary = np.argmax(y_pred, axis=1)
adjusted_y_pred_binary = [1 if x[1] > 0.2 else 0 for x in y_pred]

test_specific["y_pred"] = adjusted_y_pred_binary

test_specific = test_specific.reset_index(drop=True)

WIDE_DATA["CNS_categorical"] = WIDE_DATA["CNS_IPI_diagnosis"].apply(
    make_CNS_categorical
)

test_specific_plotting = test_specific[["patientid", "y_pred"]].merge(WIDE_DATA)
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

test_specific_plotting["group"] = test_specific_plotting["y_pred"].apply(
    lambda x: 1 if x > 0 else 0
)
test_specific_plotting["risk_prediction"] = y_prob_categorical
# test_specific_plotting = test_specific_plotting[test_specific_plotting["age_at_tx"] < 75].reset_index(drop = True)

test_specific_plotting[
    [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "CNS_IPI_diagnosis",
        "CNS_categorical",
        "risk_prediction",
        "age_at_tx",
    ]
].to_csv("results/cns_km_data_lyfo_FCR.csv", index=False)


test_specific_plotting[test_specific_plotting["age_at_tx"] < 75].reset_index(drop=True)[
    [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "CNS_IPI_diagnosis",
        "CNS_categorical",
        "risk_prediction",
        "age_at_tx",
    ]
].to_csv("results/cns_km_data_lyfo_FCR_under_75.csv", index=False)


# WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

# WIDE_DATA["CNS_categorical"] = WIDE_DATA["CNS_IPI_diagnosis"].apply(
#     make_CNS_categorical
# )

y_pred = bst.predict_proba(X_test)
y_prob_categorical = [make_prediction_categorical(x[1]) for x in y_pred]
adjusted_y_pred_binary = [1 if x[1] > 0.2 else 0 for x in y_pred]
test["y_pred"] = adjusted_y_pred_binary

test = test.reset_index(drop=True)

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
WIDE_DATA["CNS_categorical"] = WIDE_DATA["CNS_IPI_diagnosis"].apply(
    make_CNS_categorical
)


test_specific_plotting = test[["patientid", "y_pred"]].merge(WIDE_DATA)
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
# test_specific_plotting = test_specific_plotting[test_specific_plotting["age_at_tx"] < 75].reset_index(drop = True)

test_specific_plotting[
    [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "NCCN_IPI_diagnosis",
        "CNS_categorical",
        "risk_prediction",
        "age_at_tx",
    ]
].to_csv("results/cns_km_data_lyfo_FCR_all.csv", index=False)


test_specific_plotting[test_specific_plotting["age_at_tx"] < 75].reset_index(drop=True)[
    [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "NCCN_IPI_diagnosis",
        "CNS_categorical",
        "risk_prediction",
        "age_at_tx",
    ]
].to_csv("results/cns_km_data_lyfo_FCR_all_under_75.csv", index=False)
