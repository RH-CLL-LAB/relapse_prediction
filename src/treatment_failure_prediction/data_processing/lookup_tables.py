"""
lookup_tables.py — Centralised loading of shared code/lookup tables.

This module exposes:
- DIAG_LOOKUP_TABLE
- ATC_LOOKUP_TABLE
- NPU_LOOKUP_TABLE
- SNOMED_LOOKUP_TABLE

Behaviour is identical to the original script:
- Same SQL tables
- Same selected columns
- Same deduplication and lowercasing
"""

from lyfo_treatment_failure_prediction.helpers.sql_helper import load_data_from_table
from lyfo_treatment_failure_prediction.helpers.processing_helper import *  # noqa: F403


# ---------------------------------------------------------------------------
# Diagnosis codes (ICD / DST)
# ---------------------------------------------------------------------------
DIAG_LOOKUP_TABLE = load_data_from_table(
    "Codes_DST_DIAG_CODES",
    subset_columns=["Kode", "Tekst"],
)

# ---------------------------------------------------------------------------
# ATC codes (medication)
# ---------------------------------------------------------------------------
ATC_LOOKUP_TABLE = load_data_from_table(
    "Codes_ATC",
    subset_columns=["class_code", "class_name"],
)

# ---------------------------------------------------------------------------
# NPU codes (lab measurements)
# ---------------------------------------------------------------------------
NPU_LOOKUP_TABLE = load_data_from_table(
    "Codes_NPU",
    subset_columns=["NPU code", "Component"],
)

# ---------------------------------------------------------------------------
# SNOMED codes (pathology)
# ---------------------------------------------------------------------------
SNOMED_LOOKUP_TABLE = (
    load_data_from_table(
        "CODES_SNOMED",
        subset_columns=["SKSkode", "Kodetekst"],
    )
    .drop_duplicates(subset="SKSkode")
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Post-processing (unchanged from original)
# ---------------------------------------------------------------------------

# NOTE: NPU_aggregation also removes a lot of information!
# Is this a feature or a bug? Should probably include both!

ATC_LOOKUP_TABLE = ATC_LOOKUP_TABLE.drop_duplicates()
ATC_LOOKUP_TABLE["class_name"] = ATC_LOOKUP_TABLE["class_name"].str.lower()

__all__ = [
    "DIAG_LOOKUP_TABLE",
    "ATC_LOOKUP_TABLE",
    "NPU_LOOKUP_TABLE",
    "SNOMED_LOOKUP_TABLE",
]
