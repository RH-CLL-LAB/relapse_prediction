"""
medicine.py — Process medication data from multiple sources.

This module:
- Loads administered and prescribed medicine data.
- Harmonizes ATC codes.
- Computes days of medication and cumulative dosage pre-treatment.
- Calculates polypharmacy scores.
- Derives medicine-to-treatment timing features.

All logic matches the original implementation exactly.
"""

import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from lyfo_treatment_failure_prediction.helpers.sql_helper import download_and_rename_data
from lyfo_treatment_failure_prediction.helpers.processing_helper import *  # noqa: F403
from lyfo_treatment_failure_prediction.data_processing.wide_data import lyfo_cohort, WIDE_DATA
from lyfo_treatment_failure_prediction.data_processing.lookup_tables import ATC_LOOKUP_TABLE

tqdm.pandas()

# -----------------------------------------------------------------------------
# 1. Table configurations
# -----------------------------------------------------------------------------
adm_medicine_dict = {
    "adm_medicine": {
        "patientid": "patientid",
        "d_adm_date": "timestamp",
        "d_ord_start_date": "start_date",
        "d_ord_slut_date": "end_date",
        "c_atc": "variable_code",
        "v_adm_dosis": "value",
        "v_adm_dosis_enhed": "unit",
    },
    "SDS_indberetningmedpris": {
        "c_atc": "variable_code",
        "patientid": "patientid",
        "d_adm": "timestamp",
        "d_ord_start": "start_date",
        "d_ord_slut": "end_date",
        "v_styrke_num": "value",
        "v_styrke_enhed": "unit",
    },
}

ord_medicine_dict = {
    "SP_OrdineretMedicin": {
        "patientid": "patientid",
        "order_start_time": "timestamp",
        "order_end_time": "end_date",
        "atc": "variable_code",
        "hv_discrete_dose": "value",
    },
    "SDS_epikur": {
        "atc": "variable_code",
        "eksd": "timestamp",
        "doso": "value",
        "patientid": "patientid",
    },
}

# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------
def melt_data(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape ATC levels into long format for consistent merging."""
    df = df.drop(columns=["variable_code"], errors="ignore").reset_index(drop=True)
    melted = df.melt(
        id_vars=["patientid", "timestamp", "data_source", "value"],
        var_name="level",
        value_name="variable_code",
    )
    return melted[["patientid", "timestamp", "data_source", "value", "variable_code"]]


def convert_to_atc_codes(data: pd.DataFrame) -> pd.DataFrame:
    """Expand ATC variable codes into levels 1–5 and merge lookup names."""
    data = data[data["variable_code"].notna()].reset_index(drop=True)

    for level, end in zip(range(1, 6), [1, 3, 4, 5, None]):
        col = f"atc_level_{level}"
        data[col] = data["variable_code"].apply(lambda x: x[:end] if end else x)

    melted = melt_data(data)
    merged = melted.merge(
        ATC_LOOKUP_TABLE, left_on="variable_code", right_on="class_code", how="left"
    )[["patientid", "timestamp", "data_source", "value", "class_name"]]
    return merged.rename(columns={"class_name": "variable_code"}).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 3. Calculate medication days and cumulative dosage
# -----------------------------------------------------------------------------
def calculate_days_from_data(data_dict: dict, data_source_str: str):
    """Calculate 90-, 365-, and 1095-day medication windows before first-line treatment."""

    medicine_days = {
        t: download_and_rename_data(t, data_dict, cohort=lyfo_cohort)
        for t in data_dict
    }
    combined = pd.concat(medicine_days.values(), ignore_index=True).drop_duplicates()
    combined["data_source"] = data_source_str

    for col in ["start_date", "end_date"]:
        if col not in combined.columns:
            combined[col] = pd.NaT

    combined.loc[combined["start_date"].isna(), "start_date"] = combined["timestamp"]
    combined.loc[combined["end_date"].isna(), "end_date"] = combined["start_date"]
    combined.loc[combined["end_date"] < combined["start_date"], "end_date"] = combined[
        "start_date"
    ]

    combined = combined.merge(WIDE_DATA[["patientid", "date_treatment_1st_line"]])
    combined = combined[combined["start_date"] < combined["date_treatment_1st_line"]]
    combined = combined[combined["variable_code"] != ""].drop_duplicates(
        subset=["patientid", "timestamp", "value", "variable_code"]
    )

    combined["value"] = pd.to_numeric(combined["value"], errors="coerce").fillna(1)
    combined["date_before_treatment"] = combined.progress_apply(
        lambda x: min(x["end_date"], x["date_treatment_1st_line"]), axis=1
    )

    # predefine time windows
    for d in [90, 365, 1095]:
        combined[f"{d}_days_before_treatment"] = combined[
            "date_treatment_1st_line"
        ] - datetime.timedelta(days=d)

    def _calc_days(data, days: int):
        data[f"date_after_{days}_days"] = data.progress_apply(
            lambda x: max(x["start_date"], x[f"{days}_days_before_treatment"]), axis=1
        )
        data[f"days_of_medication_{days}_days"] = (
            data["date_before_treatment"] - data[f"date_after_{days}_days"]
        ).dt.days.clip(lower=1)
        data = data[data[f"days_of_medication_{days}_days"] > 0]
        data[f"cumulative_dosage_{days}_days"] = (
            data["value"] * data[f"days_of_medication_{days}_days"]
        )

        days_df = data.rename(
            columns={
                "start_date": "timestamp",
                f"days_of_medication_{days}_days": "value",
            }
        )[["patientid", "timestamp", "variable_code", "data_source", "value"]]
        days_df["data_source"] = f"{data_source_str}_{days}_days_count"

        cumulative_df = data.rename(
            columns={
                "start_date": "timestamp",
                f"cumulative_dosage_{days}_days": "value",
            }
        )[["patientid", "timestamp", "variable_code", "data_source", "value"]]
        cumulative_df["data_source"] = f"{data_source_str}_{days}_days_cumulative"

        return convert_to_atc_codes(days_df), cumulative_df

    return (
        *_calc_days(combined, 90),
        *_calc_days(combined, 365),
        *_calc_days(combined, 1095),
    )[::2]  # only count DataFrames (not cumulative)


# -----------------------------------------------------------------------------
# 4. Polypharmacy scores
# -----------------------------------------------------------------------------
def get_poly_pharmacy_scores(data: pd.DataFrame, data_source: str):
    """Compute polypharmacy statistics pre- and post-diagnosis."""
    df = data.drop_duplicates(
        subset=["patientid", "variable_code", "timestamp", "value"]
    ).dropna(subset=["variable_code"])

    df = df.merge(WIDE_DATA[["patientid", "date_treatment_1st_line", "date_diagnosis"]])
    before_treatment = df[df["timestamp"] <= df["date_treatment_1st_line"]]
    after_diagnosis = before_treatment[
        before_treatment["timestamp"] >= before_treatment["date_diagnosis"]
    ]

    def _expand_levels(subdf: pd.DataFrame, suffix: str):
        for lvl, end in zip(range(1, 6), [1, 3, 4, 5, None]):
            subdf[f"atc_level_{lvl}"] = subdf["variable_code"].apply(
                lambda x: x[:end] if end else x
            )
        melted = subdf.melt(id_vars=["timestamp", "patientid"], var_name="variable_code")
        mapped = melted.merge(
            ATC_LOOKUP_TABLE, left_on="value", right_on="class_code"
        )[["patientid", "timestamp", "variable_code", "class_name"]]
        mapped.rename(columns={"class_name": "value"}, inplace=True)
        mapped = (
            mapped.groupby(["patientid", "variable_code", "value"])
            .agg(timestamp=("timestamp", "max"))
            .reset_index()
        )
        mapped["value"] = 1
        mapped["data_source"] = f"{data_source}_{suffix}"
        return mapped

    return (
        _expand_levels(before_treatment.copy(), "poly_pharmacy"),
        _expand_levels(after_diagnosis.copy(), "poly_pharmacy_since_diagnosis"),
    )


# -----------------------------------------------------------------------------
# 5. Days from treatment calculations
# -----------------------------------------------------------------------------
def get_days_from_treatment(data: pd.DataFrame) -> pd.DataFrame:
    """Compute day differences between medication and first-line treatment."""
    merged = data.merge(WIDE_DATA[["patientid", "date_treatment_1st_line"]])
    merged = merged[merged["date_treatment_1st_line"] > merged["timestamp"]].copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged["days_from_treatment"] = (
        merged["date_treatment_1st_line"] - merged["timestamp"]
    ).dt.days

    grouped = (
        merged.groupby(["patientid", "variable_code"])
        .agg(max_days=("days_from_treatment", "max"), min_days=("days_from_treatment", "min"))
        .reset_index()
    )
    grouped["days_between_max_and_min"] = grouped["max_days"] - grouped["min_days"]

    test_ids = pd.read_csv("data/test_patientids.csv")["patientid"]
    grouped = grouped[~grouped["patientid"].isin(test_ids)]

    # filter only drugs given to >10% of patients
    threshold = 0.10 * len(WIDE_DATA[~WIDE_DATA["patientid"].isin(test_ids)])
    drug_counts = grouped.groupby("variable_code")["patientid"].nunique().reset_index()
    valid_drugs = drug_counts[drug_counts["patientid"] > threshold]["variable_code"]
    filtered = grouped[grouped["variable_code"].isin(valid_drugs)]

    melted = filtered.melt(
        id_vars=["patientid", "variable_code"],
        value_vars=["max_days", "min_days", "days_between_max_and_min"],
    )
    pivot = melted.pivot(
        index="patientid", columns=["variable_code", "variable"], values="value"
    ).reset_index()
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot.rename(columns={"patientid_": "patientid"}, inplace=True)
    pivot = pivot.fillna(-1)
    return pivot


# -----------------------------------------------------------------------------
# 6. Execute processing and build medicine_dict
# -----------------------------------------------------------------------------
adm_medicine_days = calculate_days_from_data(adm_medicine_dict, "adm_medicine_days")
ord_medicine_days = calculate_days_from_data(ord_medicine_dict, "ord_medicine_days")

adm_medicine_data = {
    t: download_and_rename_data(t, adm_medicine_dict, cohort=lyfo_cohort)
    for t in adm_medicine_dict
}
ord_medicine_data = {
    t: download_and_rename_data(t, ord_medicine_dict, cohort=lyfo_cohort)
    for t in ord_medicine_dict
}

ord_medicine_data["SDS_epikur"]["value"].fillna(1, inplace=True)
ord_medicine_data["SP_OrdineretMedicin"].loc[
    ord_medicine_data["SP_OrdineretMedicin"]["value"] == "-", "value"
] = 1
mask = ord_medicine_data["SP_OrdineretMedicin"]["value"].astype(str).str.contains("-", na=False)
ord_medicine_data["SP_OrdineretMedicin"].loc[mask, "value"] = (
    ord_medicine_data["SP_OrdineretMedicin"].loc[mask, "value"]
    .apply(lambda x: sum(float(v) for v in x.split("-")) / len(x.split("-")))
)

adm_concat = pd.concat(adm_medicine_data.values(), ignore_index=True)[
    ["patientid", "timestamp", "variable_code", "value", "data_source"]
]
ord_concat = pd.concat(ord_medicine_data.values(), ignore_index=True)[
    ["patientid", "timestamp", "variable_code", "value", "data_source"]
]

adm_poly, adm_poly_diag = get_poly_pharmacy_scores(adm_concat, "adm_medicine")
ord_poly, ord_poly_diag = get_poly_pharmacy_scores(ord_concat, "ord_medicine")

adm_concat = convert_to_atc_codes(adm_concat)
ord_concat = convert_to_atc_codes(ord_concat)
adm_concat["data_source"] = "administered_medicine"
ord_concat["data_source"] = "ordered_medicine"

adm_days_treat = get_days_from_treatment(adm_concat).add_prefix("administered_")
ord_days_treat = get_days_from_treatment(ord_concat).add_prefix("ordered_")

# -----------------------------------------------------------------------------
# Export dictionary
# -----------------------------------------------------------------------------
medicine_dict = {
    "adm_medicine_90_days_count": adm_medicine_days[0],
    "adm_medicine_365_days_count": adm_medicine_days[1],
    "adm_medicine_1095_days_count": adm_medicine_days[2],
    "ord_medicine_90_days_count": ord_medicine_days[0],
    "ord_medicine_365_days_count": ord_medicine_days[1],
    "ord_medicine_1095_days_count": ord_medicine_days[2],
    "adm_medicine_poly_pharmacy": adm_poly,
    "adm_medicine_poly_pharmacy_since_diagnosis": adm_poly_diag,
    "ord_medicine_poly_pharmacy": ord_poly,
    "ord_medicine_poly_pharmacy_since_diagnosis": ord_poly_diag,
    "administered_medicine": adm_concat,
    "ordered_medicine": ord_concat,
}

__all__ = ["medicine_dict"]
