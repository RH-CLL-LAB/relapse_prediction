import pandas as pd
from tqdm import tqdm

from lyfo_treatment_failure_prediction.helpers.sql_helper import download_and_rename_data
from lyfo_treatment_failure_prediction.data_processing.wide_data import lyfo_cohort

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
