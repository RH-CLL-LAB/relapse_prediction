"""
wide_data.py — Constructs the WIDE_DATA dataframe used throughout the project.

All transformations and filtering are identical to the original script.
"""

from datetime import timedelta
import math
from pathlib import Path

import pandas as pd

# Import helpers via the package namespace (no logic change)
from lyfo_treatment_failure_prediction.helpers.sql_helper import (
    get_cohort_string_from_data,
    load_data_from_table,
)
from lyfo_treatment_failure_prediction.helpers.processing_helper import *  # noqa: F403

from lyfo_treatment_failure_prediction.utils.config import PROJECT_ROOT

# === Data Loading ===

rkkp_path = Path("/ngc/projects2/dalyca_r/clean_r/RKKP_LYFO_CLEAN.csv")
extra_path = Path("/ngc/projects2/dalyca_r/mikwer_r/RKKP_LYFO_EXTRA_RELAPS_psAnon.csv")

# Load primary dataset
rkkp_df = pd.read_csv(rkkp_path)

# Filter patients with available treatment dates
dlbcl_rkkp_df = rkkp_df[rkkp_df["date_treatment_1st_line"].notna()].reset_index(
    drop=True
)

# Convert relevant columns to datetime format
date_columns = ["date_treatment_1st_line", "date_diagnosis"]
dlbcl_rkkp_df[date_columns] = dlbcl_rkkp_df[date_columns].apply(pd.to_datetime)

# Calculate days between diagnosis and treatment
dlbcl_rkkp_df["days_between_diagnosis_and_treatment"] = (
    dlbcl_rkkp_df["date_treatment_1st_line"] - dlbcl_rkkp_df["date_diagnosis"]
).dt.days

# Load additional dataset and merge with primary dataset
lyfo_extra = pd.read_csv(extra_path)
WIDE_DATA = dlbcl_rkkp_df.merge(lyfo_extra, how="left").reset_index(drop=True)

# Convert additional date columns to datetime format
additional_dates = ["relaps_pato_dt", "date_death"]
WIDE_DATA[additional_dates] = WIDE_DATA[additional_dates].apply(pd.to_datetime)

# Filter patients with treatment dates before 2022-01-01
WIDE_DATA = WIDE_DATA[WIDE_DATA["date_treatment_1st_line"] < "2022-01-01"].reset_index(
    drop=True
)

# === Relapse labels ===
WIDE_DATA["relapse_date"] = WIDE_DATA["date_relapse_confirmed_2nd_line"]
WIDE_DATA.loc[WIDE_DATA["relapse_date"].notna(), "relapse_label"] = 1
WIDE_DATA["relapse_label"] = WIDE_DATA["LYFO_15_002=relapsskema"]
WIDE_DATA.loc[
    WIDE_DATA["relaps_pato_dt"].notna(), ["relapse_label", "relapse_date"]
] = (1, WIDE_DATA["relaps_pato_dt"])
WIDE_DATA.loc[WIDE_DATA["relapse_date"].isna(), "relapse_date"] = WIDE_DATA[
    "relaps_lpr_dt"
].fillna(WIDE_DATA["date_treatment_2nd_line"])

# Compute days until relapse and death
WIDE_DATA["days_to_death"] = (
    WIDE_DATA["date_death"] - WIDE_DATA["date_treatment_1st_line"]
).dt.days
WIDE_DATA["days_to_relapse"] = (
    WIDE_DATA["relapse_date"] - WIDE_DATA["date_treatment_1st_line"]
).dt.days
WIDE_DATA["relapse_date"] = pd.to_datetime(WIDE_DATA["relapse_date"])
WIDE_DATA.loc[WIDE_DATA["date_death"].notna(), "dead_label"] = 1

# Filter out uncertain relapse cases
WIDE_DATA = WIDE_DATA[WIDE_DATA["relapse_label"] != 0].reset_index(drop=True)

WIDE_DATA = WIDE_DATA[
    (WIDE_DATA["days_to_death"] >= 0) | (WIDE_DATA["days_to_death"].isna())
]
WIDE_DATA = WIDE_DATA[
    (WIDE_DATA["days_to_relapse"] >= 0) | (WIDE_DATA["days_to_relapse"].isna())
]

# Extract treatment year
WIDE_DATA["year_treat"] = WIDE_DATA["date_treatment_1st_line"].dt.year

# Fill NA with pandas NA
WIDE_DATA = WIDE_DATA.fillna(pd.NA)

# === Cohort definition ===
lyfo_cohort = get_cohort_string_from_data(WIDE_DATA)
lyfo_cohort_strings = lyfo_cohort.replace("(", "('").replace(")", "')").replace(
    ", ", "', '"
)

# === Merge with cause-of-death info ===
death = load_data_from_table("SDS_t_dodsaarsag_2", cohort=lyfo_cohort)
WIDE_DATA = WIDE_DATA.merge(death[["c_dodsmaade", "patientid"]], how="left")

WIDE_DATA.loc[WIDE_DATA["c_dodsmaade"] < 3, "dead_label"] = pd.NA
WIDE_DATA = WIDE_DATA[[x for x in WIDE_DATA.columns if x != "c_dodsmaade"]]

# === Merge demographic info ===
patient = load_data_from_table(
    "patient", subset_columns=["patientid", "sex", "date_birth"]
)
patient["sex_from_patient_table"] = patient["sex"].replace(
    {"F": "Female", "M": "Male"}
)
patient["date_birth"] = pd.to_datetime(patient["date_birth"])
patient = patient.drop("sex", axis=1)
WIDE_DATA = WIDE_DATA.merge(patient, on="patientid", how="left")

# Fill missing demographic data
WIDE_DATA.loc[WIDE_DATA["sex"].isna(), "sex"] = WIDE_DATA["sex_from_patient_table"]
WIDE_DATA.loc[WIDE_DATA["age_diagnosis"].isna(), "age_diagnosis"] = round(
    (WIDE_DATA["date_diagnosis"] - WIDE_DATA["date_birth"]).dt.days / 365.5
)

# Compute additional time-based variables
WIDE_DATA["days_from_diagnosis_to_tx"] = (
    WIDE_DATA["date_treatment_1st_line"] - WIDE_DATA["date_diagnosis"]
).dt.days
WIDE_DATA["age_at_tx"] = round(
    (WIDE_DATA["date_treatment_1st_line"] - WIDE_DATA["date_birth"]).dt.days / 365.5
)

# Remove unnecessary columns
WIDE_DATA.drop(columns=["sex_from_patient_table", "date_birth"], inplace=True)

# === Derived clinical variables ===
def calculate_NCCN_IPI(age, ldh, aa_stage, extranodal, ps):
    """Compute NCCN IPI score (unchanged)."""
    if any(map(math.isnan, [age, ldh, aa_stage, ps])):
        return pd.NA

    total_score = sum(
        [
            3 if age > 75 else 2 if age > 60 else 1 if age > 40 else 0,
            2
            if ldh / (255 if age >= 70 else 205) > 3
            else 1
            if ldh / (255 if age >= 70 else 205) > 1
            else 0,
            1 if aa_stage > 2 else 0,
            1 if extranodal == 1 else 0,
            1 if ps >= 2 else 0,
        ]
    )
    return total_score


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


def calculate_nodality_of_disease(extranodal, nodal):
    """Compute nodality category (-1, 1, 2)."""
    return 2 if extranodal == 1 and nodal == 1 else 1 if nodal == 1 else -1


WIDE_DATA["nodality_disease_diagnosis"] = WIDE_DATA.apply(
    lambda x: calculate_nodality_of_disease(
        x["extranodal_disease_diagnosis"], x["nodal_disease_diagnosis"]
    ),
    axis=1,
)

non_date_columns = [x for x in WIDE_DATA.columns if "date" not in x]
WIDE_DATA[non_date_columns] = WIDE_DATA[non_date_columns].fillna(-1)

# Binary value normalization
candidates = [
    c
    for c in WIDE_DATA.columns
    if set(WIDE_DATA[c]) == set([-1, 0, 1, 2]) and "IPI" not in c
]
for column in candidates:
    WIDE_DATA[column] = WIDE_DATA[column].apply(lambda x: -1 if x == 2 else x)

# Convert certain missing lab values
WIDE_DATA.loc[WIDE_DATA["ALB_diagnosis"] == -1, "ALB_diagnosis"] = (
    WIDE_DATA.loc[WIDE_DATA["ALB_diagnosis"] == -1, "ALB_uM_diagnosis"] * 15.05
)
WIDE_DATA.loc[WIDE_DATA["KREA_diagnosis"] == -1, "KREA_diagnosis"] = (
    WIDE_DATA.loc[WIDE_DATA["KREA_diagnosis"] == -1, "KREA_mM_diagnosis"] * 100
)
WIDE_DATA.loc[WIDE_DATA["B2M_diagnosis"] == -1, "B2M_diagnosis"] = (
    WIDE_DATA.loc[WIDE_DATA["B2M_diagnosis"] == -1, "B2M_nmL_diagnosis"] * 100
)

# --- Final public objects (exported) ---
__all__ = ["WIDE_DATA", "lyfo_cohort", "lyfo_cohort_strings"]