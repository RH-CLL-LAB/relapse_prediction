from datetime import timedelta
import pandas as pd
import numpy as np
import xgboost
import joblib

# ---------------------------------------------------------------------
# Load models and data
# ---------------------------------------------------------------------
bst = xgboost.XGBClassifier()
bst.load_model("results/models/model_all.json")

X_test = pd.read_csv("results/X_test.csv")
X_test_specific = pd.read_csv("results/X_test_specific.csv")
test = pd.read_csv("results/test.csv")
test_specific = pd.read_csv("results/test_specific.csv")

# Logistic regression probabilities
lr_probs = joblib.load("results/lr_predictions.pkl")
test["lr_probs"] = lr_probs["y_pred_proba"]
test_specific = test_specific.merge(test[["patientid", "lr_probs"]], on="patientid", how="left")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def make_prediction_categorical(y_prob: float) -> str | None:
    """Map probability into 4 risk bins."""
    if pd.isnull(y_prob):
        return None
    if y_prob < 0.1:
        return "Low"
    if y_prob < 0.3:
        return "Low-Intermediate"
    if y_prob < 0.65:
        return "Intermediate-High"
    return "High"


def prepare_survival_data_LR(test_df: pd.DataFrame, X: pd.DataFrame, output_prefix: str):
    """
    Generate KM datasets using relapse endpoint (FCR).

    Writes:
        - results/km_data_{output_prefix}_LR.csv
        - results/km_data_{output_prefix}_under_75_LR.csv
    """
    # XGBoost predictions
    y_pred = bst.predict_proba(X)
    y_prob = [p[1] for p in y_pred]
    y_prob_cat = [make_prediction_categorical(p) for p in y_prob]

    # Binary prediction with same 0.2 cutoff
    y_pred_binary = [1 if p > 0.2 else 0 for p in y_prob]
    test_df = test_df.reset_index(drop=True).copy()
    test_df["y_pred"] = y_pred_binary
    test_df["risk_prediction"] = y_prob_cat
    test_df["lr_categorical"] = test_df["lr_probs"].apply(make_prediction_categorical)

    wide_data = pd.read_pickle("data/WIDE_DATA.pkl")

    merged = test_df[["patientid", "y_pred", "lr_probs", "lr_categorical", "risk_prediction"]].merge(
        wide_data, on="patientid", how="left"
    )

    # Define event and timing (relapse → 1, death → 2, censored → 0)
    merged["date_event"] = merged["relapse_date"]
    merged.loc[merged["date_event"].notna(), "event"] = 1

    merged.loc[merged["date_event"].isna(), "date_event"] = merged.loc[
        merged["relapse_date"].isna(), "date_death"
    ]
    merged.loc[
        merged["date_event"].notna() & merged["event"].isna(),
        "event",
    ] = 2
    merged.loc[merged["date_event"].isna(), "event"] = 0
    merged.loc[merged["date_event"].isna(), "date_event"] = pd.to_datetime("2024-01-01")

    merged["days_to_event"] = (merged["date_event"] - merged["date_treatment_1st_line"]).dt.days
    merged["group"] = merged["y_pred"].apply(lambda x: 1 if x > 0 else 0)

    km_cols = [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "lr_probs",
        "lr_categorical",
        "risk_prediction",
        "age_at_tx",
    ]

    merged[km_cols].to_csv(f"results/km_data_{output_prefix}_LR.csv", index=False)

    merged_under_75 = merged[merged["age_at_tx"] < 75].reset_index(drop=True)
    merged_under_75[km_cols].to_csv(f"results/km_data_{output_prefix}_under_75_LR.csv", index=False)

    return merged


# ---------------------------------------------------------------------
# Generate all datasets
# ---------------------------------------------------------------------
prepare_survival_data_LR(test_specific, X_test_specific, "lyfo_FCR")
prepare_survival_data_LR(test, X_test, "lyfo_FCR_all")
