"""
get_patient_characteristics.py

Utility script to compute descriptive statistics for the DLBCL cohort:
- Recomputes CNS-IPI and NCCN-IPI scores from WIDE_DATA.
- Filters to non-test DLBCL patients.
- Produces counts and percentages for key clinical variables.
"""

import math
from datetime import timedelta
import pickle as pkl  # kept for parity with original, though unused
import pandas as pd

from lyfo_treatment_failure_prediction.helpers.processing_helper import (
    calculate_CNS_IPI,
    calculate_NCCN_IPI,
)


def get_table_from_data(filtered: pd.DataFrame):
    """Return counts and percentages for key baseline characteristics."""
    # Age groups at treatment
    age_at_treatment_under_40 = len(filtered[filtered["age_at_tx"] <= 40])
    age_at_treatment_under_60 = len(
        filtered[(filtered["age_at_tx"] >= 41) & (filtered["age_at_tx"] <= 60)]
    )
    age_at_treatment_under_75 = len(
        filtered[(filtered["age_at_tx"] >= 61) & (filtered["age_at_tx"] <= 75)]
    )
    age_at_treatment_over_75 = len(filtered[filtered["age_at_tx"] > 75])

    # Sex
    males = filtered["sex"].value_counts()

    # Ann Arbor stage
    aa_stage = filtered["AA_stage_diagnosis"].value_counts()

    # Performance status
    ps = filtered["PS_diagnosis"].value_counts()

    # B symptoms
    b_symptoms = filtered["b_symptoms_diagnosis"].value_counts()

    # LDH categories (age-dependent ULN)
    ldh_uln = len(
        filtered[(filtered["LDH_diagnosis"] <= 205) & (filtered["age_diagnosis"] < 70)]
    )
    ldh_under_three_uln = len(
        filtered[
            (filtered["LDH_diagnosis"] > 205)
            & (filtered["LDH_diagnosis"] < 205 * 3)
            & (filtered["age_diagnosis"] < 70)
        ]
    )
    ldh_over_three_uln = len(
        filtered[
            (filtered["LDH_diagnosis"] >= 205 * 3)
            & (filtered["age_diagnosis"] < 70)
        ]
    )

    ldh_uln += len(
        filtered[(filtered["LDH_diagnosis"] <= 255) & (filtered["age_diagnosis"] >= 70)]
    )
    ldh_under_three_uln += len(
        filtered[
            (filtered["LDH_diagnosis"] > 255)
            & (filtered["LDH_diagnosis"] < 255 * 3)
            & (filtered["age_diagnosis"] >= 70)
        ]
    )
    ldh_over_three_uln += len(
        filtered[
            (filtered["LDH_diagnosis"] >= 255 * 3)
            & (filtered["age_diagnosis"] >= 70)
        ]
    )

    # Other lab / disease features
    alc = len(filtered[filtered["ALC_diagnosis"] <= 0.84])
    alb = len(filtered[filtered["ALB_diagnosis"] <= 35])
    hgb = len(filtered[filtered["HB_diagnosis"] < 6.2])
    bulky = len(filtered[filtered["tumor_diameter_diagnosis"] >= 10])

    ipi = filtered["IPI_score_diagnosis"].value_counts()
    nccn_ipi = filtered["NCCN_IPI_diagnosis"].value_counts()
    cns_ipi = filtered["CNS_IPI_diagnosis"].value_counts()

    not_percent = {
        "age_at_treatment_under_40": age_at_treatment_under_40,
        "age_at_treatment_under_60": age_at_treatment_under_60,
        "age_at_treatment_under_75": age_at_treatment_under_75,
        "age_at_treatment_over_75": age_at_treatment_over_75,
        "males": males,
        "aa_stage": aa_stage,
        "ps": ps,
        "b_symptoms": b_symptoms,
        "ldh_uln": ldh_uln,
        "ldh_under_three_uln": ldh_under_three_uln,
        "ldh_over_three_uln": ldh_over_three_uln,
        "alc": alc,
        "alb": alb,
        "hgb": hgb,
        "bulky": bulky,
        "ipi": ipi,
        "nccn_ipi": nccn_ipi,
        "cns_ipi": cns_ipi,
    }

    percent = {k: round((v / len(filtered)) * 100, 2) for k, v in not_percent.items()}

    return not_percent, percent


def get_table_from_data_na(filtered: pd.DataFrame):
    """Return counts and percentages of missing values for key variables."""
    age_at_treatment_under_40 = len(filtered[filtered["age_at_tx"].isna()])
    males = filtered["sex"].isna().sum()
    aa_stage = filtered["AA_stage_diagnosis"].isna().sum()
    ps = filtered["PS_diagnosis"].isna().sum()
    b_symptoms = filtered["b_symptoms_diagnosis"].isna().sum()
    ldh_uln = filtered["LDH_diagnosis"].isna().sum()
    alc = filtered["ALC_diagnosis"].isna().sum()
    alb = filtered["ALB_diagnosis"].isna().sum()
    hgb = filtered["HB_diagnosis"].isna().sum()
    bulky = filtered["tumor_diameter_diagnosis"].isna().sum()
    ipi = filtered["IPI_score_diagnosis"].isna().sum()
    nccn_ipi = filtered["NCCN_IPI_diagnosis"].isna().sum()
    cns_ipi = filtered["CNS_IPI_diagnosis"].isna().sum()

    not_percent = {
        "age_at_treatment_under_40": age_at_treatment_under_40,
        "males": males,
        "aa_stage": aa_stage,
        "ps": ps,
        "b_symptoms": b_symptoms,
        "ldh_uln": ldh_uln,
        "alc": alc,
        "alb": alb,
        "hgb": hgb,
        "bulky": bulky,
        "ipi": ipi,
        "nccn_ipi": nccn_ipi,
        "cns_ipi": cns_ipi,
    }

    percent = {k: round((v / len(filtered)) * 100, 2) for k, v in not_percent.items()}

    return not_percent, percent


# ----------------------------------------------------------------------
# Main script logic (kept at top level for identical behaviour)
# ----------------------------------------------------------------------

# Load WIDE_DATA from preprocessed pickle
WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

# Compute CNS-IPI and NCCN-IPI at diagnosis
WIDE_DATA["CNS_IPI_diagnosis"] = WIDE_DATA.apply(
    lambda x: calculate_CNS_IPI(
        x["age_diagnosis"],
        x["LDH_diagnosis"],
        x["AA_stage_diagnosis"],
        x["extranodal_disease_diagnosis"],
        x["PS_diagnosis"],
        x["kidneys_diagnosis"],
    ),
    axis=1,
)

WIDE_DATA["NCCN_IPI_diagnosis"] = WIDE_DATA.apply(
    lambda x: calculate_NCCN_IPI(
        x["age_diagnosis"],
        x["LDH_diagnosis"],
        x["AA_stage_diagnosis"],
        x["extranodal_disease_diagnosis"],
        x["PS_diagnosis"],
    ),
    axis=1,
)

# Remove temporary IDs / incomplete rows
WIDE_DATA = WIDE_DATA[WIDE_DATA["age_diagnosis"].notna()].reset_index(drop=True)

# Exclude test-patients
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
filtered = WIDE_DATA[~WIDE_DATA["patientid"].isin(test_patientids)]
filtered = filtered[filtered["subtype"] == "DLBCL"]

# Compute descriptive tables
not_percent, percent = get_table_from_data(filtered)
na_not_percent, na_percent = get_table_from_data_na(filtered)

# These variables are now available in the namespace if you run this script
# in an interactive session / notebook-style workflow.
__all__ = [
    "get_table_from_data",
    "get_table_from_data_na",
    "WIDE_DATA",
    "filtered",
    "not_percent",
    "percent",
    "na_not_percent",
    "na_percent",
]
