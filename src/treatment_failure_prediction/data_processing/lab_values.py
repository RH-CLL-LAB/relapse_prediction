"""
lab_values.py — Small set of lab-related features.

Creates:
- lab_data["LAB_IGHVIMGT"]: IGHV mutational status encoded as 0/1.
- lab_data["LAB_BIOBANK_SAMPLES"]: biobank sample presence flags.
- lab_data["LAB_Flowcytometry"]: flow cytometry presence flags.

Behaviour is identical to the original script.
"""

import pandas as pd
from tqdm import tqdm

from lyfo_treatment_failure_prediction.helpers.sql_helper import download_and_rename_data
from lyfo_treatment_failure_prediction.helpers.processing_helper import *  # noqa: F403
from lyfo_treatment_failure_prediction.data_processing.wide_data import lyfo_cohort
from lyfo_treatment_failure_prediction.data_processing.lookup_tables import (
    ATC_LOOKUP_TABLE,  # noqa: F401  (import kept for parity with original)
)

tqdm.pandas()

# ---------------------------------------------------------------------------
# Configuration for lab-related tables
# ---------------------------------------------------------------------------

lab_dict = {
    "LAB_IGHVIMGT": {
        "date_sample": "timestamp",
        "patientid": "patientid",
        "ighv": "value",
    },
    "LAB_BIOBANK_SAMPLES": {
        "patientid": "patientid",
        "date_samplecollection": "timestamp",
        "Type": "variable_code",
    },
    "LAB_Flowcytometry": {
        "patientid": "patientid",
        "DATE": "timestamp",
        "MATERIAL": "variable_code",
    },
}

# ---------------------------------------------------------------------------
# Load data for each table
# ---------------------------------------------------------------------------

lab_data = {
    table_name: download_and_rename_data(
        table_name=table_name,
        config_dict=lab_dict,
        cohort=lyfo_cohort,
    )
    for table_name in lab_dict
}

# ---------------------------------------------------------------------------
# Transformations (unchanged logic)
# ---------------------------------------------------------------------------

# IGHV: categorical to ordered codes (Unmutated / Mutated)
lab_data["LAB_IGHVIMGT"].loc[:, "variable_code"] = "IGHV"
lab_data["LAB_IGHVIMGT"].loc[:, "value"] = pd.Categorical(
    lab_data["LAB_IGHVIMGT"]["value"],
    ordered=True,
    categories=["Unmutated", "Mutated"],
).codes

# Biobank samples: presence flag = 1
lab_data["LAB_BIOBANK_SAMPLES"].loc[:, "value"] = 1

# Flow cytometry: presence flag = 1
lab_data["LAB_Flowcytometry"].loc[:, "value"] = 1

# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------
__all__ = ["lab_data"]
