from lyfo_treatment_failure_prediction.helpers.sql_helper import load_data_from_table


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
