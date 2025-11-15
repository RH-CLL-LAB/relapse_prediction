"""
persimune.py — Load and process PERSIMUNE microbiology and biochemistry data.

Behaviour:
- Loads microbiology analysis and culture tables.
- Converts categorical microbiology results to numeric values.
- Groups variable names into clinically meaningful categories (e.g. “covid”, “herpes”).
- Adds leukocyte-related biochemistry results.
- Produces a single dictionary `persimune_dict` with cleaned data sources.

All logic, naming, and outputs are identical to the original code.
"""

import pandas as pd
from lyfo_treatment_failure_prediction.helpers.sql_helper import download_and_rename_data
from lyfo_treatment_failure_prediction.helpers.processing_helper import *  # noqa: F403
from lyfo_treatment_failure_prediction.data_processing.wide_data import lyfo_cohort_strings

# ---------------------------------------------------------------------------
# Mapping configuration
# ---------------------------------------------------------------------------
PERSIMUNE_MAPPING = {
    "PERSIMUNE_microbiology_analysis": {
        "patientid": "patientid",
        "samplingdatetime": "timestamp",
        "analysisshortname": "variable_code",
        "c_categoricalresult": "value",
    },
    "PERSIMUNE_microbiology_culture": {
        "patientid": "patientid",
        "samplingdatetime": "timestamp",
        "c_domain": "variable_code",
        "c_pm_categoricalresult": "value",
    },
}

# ---------------------------------------------------------------------------
# Download microbiology data
# ---------------------------------------------------------------------------
persimune_dict = {
    table_name: download_and_rename_data(
        table_name, PERSIMUNE_MAPPING, cohort=lyfo_cohort_strings
    )
    for table_name in PERSIMUNE_MAPPING
}

# ---------------------------------------------------------------------------
# Convert categorical results to binary indicators
# ---------------------------------------------------------------------------
value_conversion_dict = {
    "Negative": 0,
    "NULL": 0,
    "Positive": 1,
    "Possible Cont.": 1,
    "Not analyzed": 0,
}

# Culture table: map categorical values directly to 0/1
persimune_dict["PERSIMUNE_microbiology_culture"]["value"] = (
    persimune_dict["PERSIMUNE_microbiology_culture"]["value"]
    .apply(lambda x: value_conversion_dict.get(x, 0))
)

# Analysis table: mark missing as "Missing"
analysis_df = persimune_dict["PERSIMUNE_microbiology_analysis"]
analysis_df.loc[analysis_df["value"].isna(), "value"] = "Missing"

# Convert "Positive"/"High" etc. to binary 1/0
positive_terms = ["Positive", "High"]
analysis_df["value"] = analysis_df["value"].apply(
    lambda x: 1 if any(term in str(x) for term in positive_terms) else 0
)

# ---------------------------------------------------------------------------
# Clean variable names and remove nulls
# ---------------------------------------------------------------------------
valid_mask = (
    analysis_df["variable_code"].notna()
    & (analysis_df["variable_code"] != "NULL")
    & (analysis_df["variable_code"] != "_")
)
analysis_df = analysis_df.loc[valid_mask].reset_index(drop=True)
persimune_dict["PERSIMUNE_microbiology_analysis"] = analysis_df

# ---------------------------------------------------------------------------
# Group variables into broader diagnostic categories
# ---------------------------------------------------------------------------
variable_groups = {
    "covid": ["SARS-CoV-2", "Coronavirus", "Corona", "coronavirus", "SARS-Cov"],
    "herpes": ["Herpes"],
    "cmv": ["CMV"],
    "influenza": ["Influenza", "Inf."],
    "parainfluenza": ["Parainfluenza"],
    "epstein": ["Epstein", "EBV", "EBNA IgG"],
    "cytomegalovirus": ["Cytomegalovirus"],
    "chlamydophila": ["Chlamydophila", "Chlamy."],
    "chlamydia": ["Chlamydia"],
    "adenovirus": ["Adeno"],
    "rotavirus": ["Rota"],
    "bocavirus": ["Boca"],
    "sapovirus": ["Sapo"],
    "noro": ["Norovirus"],
    "astro": ["Astro"],
    "entero": ["Entero"],
    "rhino": ["Rhino"],
    "toxoplasma": ["Toxoplasma", "Toxoplasmose", "Toxo"],
    "bartonella": ["Bartonella"],
    "hepatitis": ["Hepatitis"],
    "pneumokok": ["Pneumokok"],
    "legionella": ["Legionella"],
    "aspergillus": ["Aspergillus"],
    "varicella": ["Varicella"],
}

def _apply_variable_grouping(value: str) -> str:
    """Map variable names to broader group names (if applicable)."""
    for group, terms in variable_groups.items():
        if any(term in str(value) for term in terms):
            return group
    return value

analysis_df["variable_code"] = analysis_df["variable_code"].apply(_apply_variable_grouping)

# ---------------------------------------------------------------------------
# Add PERSIMUNE leukocyte data
# ---------------------------------------------------------------------------
persimune_dict["PERSIMUNE_leukocytes"] = download_and_rename_data(
    "PERSIMUNE_biochemistry",
    {
        "PERSIMUNE_biochemistry": {
            "patientid": "patientid",
            "samplingdatetime": "timestamp",
            "analysiscode": "variable_code",
            "c_associatedleukocytevalue": "value",
        }
    },
    cohort=lyfo_cohort_strings,
)

persimune_dict["PERSIMUNE_leukocytes"]["data_source"] = "PERSIMUNE_leukocytes"
persimune_dict["PERSIMUNE_leukocytes"]["variable_code"] = "all"

# ---------------------------------------------------------------------------
# Final cleanup: ensure types and timestamps are consistent
# ---------------------------------------------------------------------------
for dataset_name, df in persimune_dict.items():
    df["patientid"] = df["patientid"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    persimune_dict[dataset_name] = df.reset_index(drop=True)

__all__ = ["persimune_dict"]