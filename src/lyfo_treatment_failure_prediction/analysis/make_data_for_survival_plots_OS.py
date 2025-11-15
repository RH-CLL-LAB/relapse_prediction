"""
make_data_for_survival_plots_OS.py

Generates Kaplan–Meier–ready CSVs for overall survival (OS) analysis.
- Loads model and test data.
- Computes predicted probabilities and categorical risk groups.
- Merges with WIDE_DATA clinical features.
- Outputs survival data for full and under-75 cohorts.

All logic matches the original implementation exactly.
"""

from datetime import timedelta
import pandas as pd
import numpy as np
import xgboost

# ---------------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------------
bst = xgboost.XGBClassifier()
bst.load_model("results/models/model_all.json")

X_test = pd.read_csv("results/X_test.csv")
X_test_specific = pd.read_csv("results/X_test_specific.csv")
test = pd.read_csv("results/test.csv")
test_specific = pd.read_csv("results/test_specific.csv")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def make_NCCN_categorical(nccn: float) -> str | None:
    """Convert NCCN-IPI numerical scores to categorical labels."""
    if pd.isnull(nccn) or nccn == -1:
        return None
    if nccn < 2:
        return "Low"
    if nccn < 4:
        return "Low-Intermediate"
    if nccn < 6:
        return "Intermediate-High"
    return "High"


def make_prediction_categorical(y_prob: float) -> str | None:
    """Map prediction probability to risk category (final calibrated thresholds)."""
    if pd.isnull(y_prob):
        return None
    if y_prob < 0.1:
        return "Low"
    if y_prob < 0.3:
        return "Low-Intermediate"
    if y_prob < 0.65:
        return "Intermediate-High"
    return "High"


def prepare_survival_data_OS(test_df: pd.DataFrame, X: pd.DataFrame, output_prefix: str):
    """
    Generate OS Kaplan–Meier–ready data (date_death endpoint).
    Writes:
        - results/km_data_{output_prefix}_OS.csv
        - results/km_data_{output_prefix}_under_75_OS.csv
    """
    # Compute predicted probabilities and categories
    y_pred = bst.predict_proba(X)
    y_prob = [p[1] for p in y_pred]
    y_prob_cat = [make_prediction_categorical(p) for p in y_prob]

    # Binary predictions (threshold = 0.2)
    y_pred_binary = [1 if p > 0.2 else 0 for p in y_prob]
    test_df = test_df.reset_index(drop=True).copy()
    test_df["y_pred"] = y_pred_binary

    # Load WIDE_DATA
    wide_data = pd.read_pickle("data/WIDE_DATA.pkl")
    wide_data["NCCN_categorical"] = wide_data["NCCN_IPI_diagnosis"].apply(make_NCCN_categorical)

    # Merge predictions with patient metadata
    merged = test_df[["patientid", "y_pred"]].merge(wide_data, on="patientid", how="left")

    # Define survival event: death or censoring
    merged["date_event"] = merged["date_death"]
    merged.loc[merged["date_event"].notna(), "event"] = 1
    merged.loc[merged["date_event"].isna(), "event"] = 0
    merged.loc[merged["date_event"].isna(), "date_event"] = pd.to_datetime("2024-01-01")

    # Time to event
    merged["days_to_event"] = (merged["date_event"] - merged["date_treatment_1st_line"]).dt.days
    merged["risk_prediction"] = y_prob_cat
    merged["y_pred_prob"] = y_prob
    merged["group"] = merged["y_pred"].apply(lambda x: 1 if x > 0 else 0)

    # Columns for output
    km_cols = [
        "patientid",
        "days_to_event",
        "event",
        "group",
        "NCCN_IPI_diagnosis",
        "NCCN_categorical",
        "risk_prediction",
        "age_at_tx",
        "y_pred_prob",
    ]

    # Export for all ages
    merged[km_cols].to_csv(f"results/km_data_{output_prefix}_OS.csv", index=False)

    # Export for under-75 subgroup
    merged_under_75 = merged[merged["age_at_tx"] < 75].reset_index(drop=True)
    merged_under_75[[c for c in km_cols if c != "y_pred_prob"]].to_csv(
        f"results/km_data_{output_prefix}_under_75_OS.csv", index=False
    )

    return merged


# ---------------------------------------------------------------------
# Generate both survival datasets
# ---------------------------------------------------------------------
prepare_survival_data_OS(test_specific, X_test_specific, "lyfo")
prepare_survival_data_OS(test, X_test, "lyfo_all")
