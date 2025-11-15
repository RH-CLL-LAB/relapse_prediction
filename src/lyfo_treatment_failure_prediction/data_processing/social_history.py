"""
social_history.py — Load and clean social history data for the LYFO cohort.

This version is behaviour-preserving: all logic and transformations remain
identical to the original script. Only import paths and readability improved.
"""

from datetime import timedelta
import pandas as pd

# Package-style imports (no logic change)
from lyfo_treatment_failure_prediction.helpers.sql_helper import (
    download_and_rename_data,
)
from lyfo_treatment_failure_prediction.helpers.processing_helper import *  # noqa: F403
from lyfo_treatment_failure_prediction.data_processing.wide_data import lyfo_cohort


# ---------------------------------------------------------------------------
# Download and reshape social history data
# ---------------------------------------------------------------------------

social_history_dict = {
    "SP_Social_Hx": {
        "patientid": "patientid",
        "registringsdato": "timestamp",
        "ryger": "smoking_cat",
        "pakkeromdagen": "smoking_num",
        "drikker": "drinking_cat",
        "damper": "vape_cat",
    }
}

# Download and rename using the provided cohort
social_history_data = download_and_rename_data(
    "SP_Social_Hx", social_history_dict, cohort=lyfo_cohort
)

# Ensure numeric format for smoking quantity
social_history_data["smoking_num"] = pd.to_numeric(
    social_history_data["smoking_num"], errors="coerce"
)

# Melt wide -> long
social_history_data = social_history_data.melt(
    id_vars=["patientid", "timestamp", "data_source"], var_name="variable_code"
)

# ---------------------------------------------------------------------------
# Correct known timestamp bug for dates before 2000
# ---------------------------------------------------------------------------

mistake = social_history_data[
    social_history_data["timestamp"] < pd.to_datetime("2000-01-01")
].reset_index(drop=True)

mistake["seconds"] = (
    mistake["timestamp"] - pd.to_datetime("1970-01-01")
).dt.seconds
mistake["timestamp"] = pd.to_datetime(mistake["timestamp"])
mistake["date"] = mistake.apply(
    lambda x: x["timestamp"] + timedelta(days=x["seconds"]), axis=1
)

# Merge corrected timestamps back
social_history_data = social_history_data.merge(mistake, how="left")

# Replace timestamp where correction available
social_history_data.loc[
    social_history_data["date"].notna(), "timestamp"
] = social_history_data.loc[social_history_data["date"].notna(), "date"]

# Keep consistent output columns
social_history_data = social_history_data[
    ["patientid", "timestamp", "data_source", "variable_code", "value"]
]

# Explicit public symbol (matches old import behaviour)
__all__ = ["social_history_data"]