"""
sks_opr.py — Loads and processes SKS procedure and hospital referral data
for the LYFO cohort.

All transformations, merges, and filters are identical to the original version.
Only imports, structure, and readability were improved.
"""

import pandas as pd

# Package imports — logic unchanged
from lyfo_treatment_failure_prediction.helpers.sql_helper import (
    load_data_from_table,
    download_and_rename_data,
)
from lyfo_treatment_failure_prediction.helpers.processing_helper import *  # noqa: F403
from lyfo_treatment_failure_prediction.data_processing.wide_data import (
    lyfo_cohort,
    WIDE_DATA,
)

# ---------------------------------------------------------------------------
# Load SDS codes
# ---------------------------------------------------------------------------
sds_codes = (
    load_data_from_table("SDS_koder")
    .query("kode_tekst != 'Ikke klassificeret i perioden'")
    .reset_index(drop=True)
)

# Select relevant columns
sds_codes = sds_codes[
    [
        "kode",
        "kode_tekst",
        "niveau1_tekst",
        "niveau2_tekst",
        "niveau3_tekst",
        "niveau4_tekst",
        "niveau5_tekst",
        "niveau6_tekst",
        "niveau7_tekst",
        "niveau8_tekst",
        "niveau9_tekst",
    ]
]

# Remove duplicates, prioritizing rows with fewer NaN values
sds_codes["count"] = sds_codes.isnull().sum(axis=1)
sds_codes = (
    sds_codes.sort_values("count")
    .drop_duplicates("kode")
    .drop(columns="count")
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Load and clean procedure data
# ---------------------------------------------------------------------------
procedure_tables = ["SDS_procedurer_kirurgi", "SDS_procedurer_andre"]

procedure_data = {
    table: load_data_from_table(
        table,
        subset_columns=[
            "dw_ek_forloeb",
            "dw_ek_kontakt",
            "procedurekode",
            "procedurekode_parent",
            "tidspunkt_start",
        ],
    )
    for table in procedure_tables
}

# Remove duplicates and invalid rows
for key, df in procedure_data.items():
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.query("dw_ek_forloeb.notna() or dw_ek_kontakt.notna()")
    df = df[df["procedurekode_parent"].notna()].drop(
        columns=["procedurekode_parent", "tidspunkt_start"]
    )
    df = df.query("dw_ek_forloeb != 'dw_ek_forloeb'")
    df["dw_ek_kontakt"] = pd.to_numeric(df["dw_ek_kontakt"], errors="coerce")
    df["dw_ek_forloeb"] = pd.to_numeric(df["dw_ek_forloeb"], errors="coerce")
    procedure_data[key] = df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Load contact and episode data
# ---------------------------------------------------------------------------
SDS_kontakter = load_data_from_table(
    "SDS_kontakter",
    subset_columns=[
        "dw_ek_kontakt",
        "dw_ek_forloeb",
        "patientid",
        "dato_start",
        "dato_slut",
    ],
    cohort=lyfo_cohort,
)

SDS_forloeb = SDS_kontakter[["dw_ek_forloeb", "patientid", "dato_start", "dato_slut"]]
SDS_contacts = SDS_kontakter[["dw_ek_kontakt", "patientid", "dato_start", "dato_slut"]]

# ---------------------------------------------------------------------------
# Merge procedures with patient contacts
# ---------------------------------------------------------------------------
procedure_results = {}
for key, df in procedure_data.items():
    merged_data = pd.concat([SDS_forloeb.merge(df), SDS_contacts.merge(df)])
    merged_data = merged_data[
        ["patientid", "dato_start", "dato_slut", "procedurekode"]
    ].reset_index(drop=True)
    merged_data["value"] = (
        merged_data["dato_slut"] - merged_data["dato_start"]
    ).dt.days + 1
    merged_data.rename(
        columns={"dato_start": "timestamp", "procedurekode": "variable_code"},
        inplace=True,
    )
    procedure_results[key] = merged_data

# ---------------------------------------------------------------------------
# Download and rename additional hospital data
# ---------------------------------------------------------------------------
hospital_data_sources = {
    "sks_opr_at_the_hospital": "view_sds_t_adm_t_sksopr",
    "sks_opr_not_at_the_hospital": "view_sds_t_adm_t_sksopr",
    "sks_ube_at_the_hospital": "view_sds_t_adm_t_sksube",
    "sks_ube_not_at_the_hospital": "view_sds_t_adm_t_sksube",
}

hospital_data = {
    key: download_and_rename_data(
        source,
        {
            source: {
                "c_opr": "variable_code",
                "v_behdage": "value",
                "patientid": "patientid",
                "d_inddto" if "at_the_hospital" in key else "d_hendto": "timestamp",
            }
        },
        cohort=lyfo_cohort,
    )
    for key, source in hospital_data_sources.items()
}

# ---------------------------------------------------------------------------
# Combine referrals and clean data
# ---------------------------------------------------------------------------
sks_referals = pd.concat(
    [
        hospital_data["sks_opr_not_at_the_hospital"],
        hospital_data["sks_ube_not_at_the_hospital"],
        procedure_results["SDS_procedurer_kirurgi"],
        procedure_results["SDS_procedurer_andre"],
    ]
).reset_index(drop=True)

sks_referals["data_source"] = "sks_referals"
sks_referals["value"].fillna(1, inplace=True)
sks_referals = sks_referals.merge(sds_codes, left_on="variable_code", right_on="kode")
sks_referals = sks_referals[~sks_referals["variable_code"].str.startswith("V")]
sks_referals = sks_referals[~sks_referals["variable_code"].str.contains("CD20|CHOP")]

# Melt (reshape)
sks_referals = sks_referals.melt(
    id_vars=["patientid", "value", "timestamp", "variable_code", "data_source", "kode"],
    value_name="variable_names",
)
sks_referals = sks_referals[sks_referals["variable_names"] != ""].reset_index(drop=True)

# Merge with treatment dates and filter by timestamp
sks_referals_unique = sks_referals.merge(
    WIDE_DATA[["patientid", "date_treatment_1st_line"]]
)
sks_referals_unique = sks_referals_unique[
    sks_referals_unique["timestamp"] < sks_referals_unique["date_treatment_1st_line"]
]

# Aggregate data
sks_referals_unique = (
    sks_referals_unique.groupby(["patientid", "variable", "variable_names"])
    .agg(timestamp=("timestamp", "max"))
    .reset_index()
)
sks_referals_unique.rename(
    columns={"variable": "variable_code", "variable_names": "value"}, inplace=True
)
sks_referals_unique["value"] = 1
sks_referals_unique["data_source"] = "sks_referals_unique"

# ---------------------------------------------------------------------------
# At-the-hospital data
# ---------------------------------------------------------------------------
sks_at_the_hospital = pd.concat(
    [hospital_data["sks_opr_at_the_hospital"], hospital_data["sks_ube_at_the_hospital"]]
).reset_index(drop=True)
sks_at_the_hospital["data_source"] = "sks_at_the_hospital"
sks_at_the_hospital["value"].fillna(1, inplace=True)
sks_at_the_hospital = sks_at_the_hospital.merge(
    sds_codes, left_on="variable_code", right_on="kode"
)
sks_at_the_hospital = sks_at_the_hospital[
    ~sks_at_the_hospital["variable_code"].str.startswith("V")
]
sks_at_the_hospital = sks_at_the_hospital[
    ~sks_at_the_hospital["variable_code"].str.contains("CD20|CHOP")
]

sks_at_the_hospital = sks_at_the_hospital.melt(
    id_vars=["patientid", "value", "timestamp", "variable_code", "data_source", "kode"],
    value_name="variable_names",
)
sks_at_the_hospital = sks_at_the_hospital[
    sks_at_the_hospital["variable_names"] != ""
].reset_index(drop=True)

# Merge with treatment dates and filter by timestamp
sks_at_the_hospital_unique = sks_at_the_hospital.merge(
    WIDE_DATA[["patientid", "date_treatment_1st_line"]]
)
sks_at_the_hospital_unique = sks_at_the_hospital_unique[
    sks_at_the_hospital_unique["timestamp"]
    < sks_at_the_hospital_unique["date_treatment_1st_line"]
]

# Aggregate data
sks_at_the_hospital_unique = (
    sks_at_the_hospital_unique.groupby(["patientid", "variable", "variable_names"])
    .agg(timestamp=("timestamp", "max"))
    .reset_index()
)
sks_at_the_hospital_unique.rename(
    columns={"variable": "variable_code", "variable_names": "value"}, inplace=True
)
sks_at_the_hospital_unique["value"] = 1
sks_at_the_hospital_unique["data_source"] = "sks_at_the_hospital_unique"

# ---------------------------------------------------------------------------
# Public exports (to match original usage)
# ---------------------------------------------------------------------------
__all__ = [
    "sks_referals",
    "sks_referals_unique",
    "sks_at_the_hospital",
    "sks_at_the_hospital_unique",
]
