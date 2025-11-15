"""
laboratory_measurements.py — General lab measurements and PERSIMUNE biochemistry.

Produces:
- lab_measurements_data: cleaned numeric lab measurements with daily timestamps.
- lab_measurements_data_all: per-patient lab-available flag before treatment.

Behaviour is identical to the original script.
"""

import pandas as pd
from tqdm import tqdm

from lyfo_treatment_failure_prediction.helpers.sql_helper import download_and_rename_data
from lyfo_treatment_failure_prediction.data_processing.wide_data import (
    lyfo_cohort,
    lyfo_cohort_strings,
    WIDE_DATA,
)
from lyfo_treatment_failure_prediction.data_processing.lookup_tables import (
    NPU_LOOKUP_TABLE,  # kept for parity with original, even though unused
)

tqdm.pandas()

# ---------------------------------------------------------------------------
# 1. Load laboratorymeasurements + PERSIMUNE biochemistry
# ---------------------------------------------------------------------------

lab_dict = {
    "laboratorymeasurements": {
        "patientid": "patientid",
        "samplingdate": "timestamp",
        "analysisgroup": "variable_code",
        "c_value": "value",
        "unit": "unit",
    },
}

lab_measurements = {
    table_name: download_and_rename_data(
        table_name=table_name,
        config_dict=lab_dict,
        cohort=lyfo_cohort,
    )
    for table_name in lab_dict
}

persimune = download_and_rename_data(
    "PERSIMUNE_biochemistry",
    {
        "PERSIMUNE_biochemistry": {
            "patientid": "patientid",
            "c_samplingdatetime": "timestamp",
            "c_analysisgroup": "variable_code",
            "c_resultvaluenumeric": "value",
            "c_resultunit": "unit",
        },
    },
    cohort=lyfo_cohort_strings,
)

# Treat NPU codes that are not in analysis groups as their own thing
persimune["patientid"] = persimune["patientid"].astype(int)
persimune["timestamp"] = pd.to_datetime(persimune["timestamp"], errors="coerce", utc=True)
persimune["timestamp"] = persimune["timestamp"].dt.tz_localize(None)

lab_measurements["PERSIMUNE_biochemistry"] = persimune

lab_measurements_data = pd.concat(lab_measurements).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Basic cleaning: non-null variable codes & remove non-result values
# ---------------------------------------------------------------------------

lab_measurements_data = lab_measurements_data[
    lab_measurements_data["variable_code"].notna()
].reset_index(drop=True)

list_of_non_values = [
    "Aflyst",
    "cancelled",
    "Erstattet",
    "Afbestilt",
    "Ej målt",
    "Dublet",
    "Mislykket",
    "Ikke taget",
    "Ej modtaget",
    "Ej mulig",
    "For gammel",
    "Annulleret",
    "Inkonklusiv",
    "Makuleret",
    "Ej målbar",
    "Ej beregnet",
    "Ej udført",
    "For lidt",
    "Fejlglas",
    "Fejlfyldt",
    "Ej tilstede",
    "ANNUL",
    "Ikke udført",
    "ej oplyst",
    "Bortkommet",
    "For gammelt",
    "Ugyldig",
    "Udføres ikke",
    "Barkode fejl",
    "Ej oplyst",
    "Ej på is",
]

lab_measurements_data = lab_measurements_data[
    ~lab_measurements_data["value"].isin(list_of_non_values)
].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. Timestamp normalization, deduplication, and unit cleaning
# ---------------------------------------------------------------------------

# Convert timestamp to date (drop time-of-day)
lab_measurements_data["timestamp"] = pd.to_datetime(
    pd.to_datetime(lab_measurements_data["timestamp"]).dt.date
)

lab_measurements_data = lab_measurements_data.drop_duplicates(
    subset=["patientid", "timestamp", "value", "variable_code"]
).reset_index(drop=True)

lab_measurements_data["unit"] = lab_measurements_data["unit"].str.lower()

# ---------------------------------------------------------------------------
# 4. Convert values to numeric
# ---------------------------------------------------------------------------

# Ensure string type, then normalize decimal separators and remove prefixes like "<", ">" etc.
lab_measurements_data["value"] = lab_measurements_data["value"].astype(str)
lab_measurements_data["value"] = lab_measurements_data["value"].str.replace(",", ".")
lab_measurements_data["value"] = lab_measurements_data["value"].str.replace(
    r"([<>=+])|(Ca )", "", regex=True
)

lab_measurements_data["value_numeric"] = pd.to_numeric(
    lab_measurements_data["value"], errors="coerce"
)

# Some specific corrections (unchanged from original)
negatives = [
    "Ubeskyttet",
    "Ej KBA RGH",
    "Tvivl",
    "Neg",
    "Ej KB",
    "NODIF",
    "neg",
    "Negativ",
    "NEG",
    "NEGATIV",
    "nan",
    "Ikke påvist",
    "-",
    "Ej KBA",
    "Ej RGH KBA",
    "IKKE PÅVIST",
    "negativ",
    "Umærket glas",
]
positives = [
    "Udført",
    "Se note",
    "Spor",
    "HVH KMA",
    "HER KMA",
    "Koaguleret",
    "RH KMA",
    "Sendt",
    "Påvist",
    "Gruppesvar",
    "Gruppe",
    "Positiv",
    "KOMM",
    "Se tidl svar",
    "Se kommentar",
    "Visuel Diff.",
    "Hæmolyse",
    "Spor ery",
    "POS",
    "Spor hgb",
    "Lipædi",
    "Se bilag",
    "Se patoweb",
    "Taget",
    "Er taget",
    "LISTE",
    "RH VTL",
    "RH Vækst",
    "KMA HER",
    "KMA",
    "LAVALB",
    "Lipæmi",
    "BOH KMA",
    "Se blodinfo",
    "Se INR",
    "Ikterus",
    "Oligokloni",
    "Formsvar",
    "SSI",
    "TRIG4",
    "LEU",
    "L1.0",
    "Sendt RHM",
    "Icterisk",
    "POSITIV",
    "HH KMA",
    "FLEU10",
    "Oligo",
    "BFLEU10",
    "Se patobank",
    "pos",
]

lab_measurements_data.loc[
    lab_measurements_data["value"] == "LEU0.5", "value_numeric"
] = 0.5

# ---------------------------------------------------------------------------
# 5. Final lab_measurements_data (per-measurement, numeric)
# ---------------------------------------------------------------------------

lab_measurements_data = (
    lab_measurements_data[["patientid", "timestamp", "variable_code", "value_numeric"]]
    .rename(columns={"value_numeric": "value"})
    .reset_index(drop=True)
)
lab_measurements_data["data_source"] = "labmeasurements"

# ---------------------------------------------------------------------------
# 6. lab_measurements_data_all (per-patient flag for any lab before treatment)
# ---------------------------------------------------------------------------

lab_measurements_data_all = lab_measurements_data.copy()

lab_measurements_data_all = lab_measurements_data_all.merge(
    WIDE_DATA[["patientid", "date_treatment_1st_line", "date_diagnosis"]]
)

lab_measurements_data_all = lab_measurements_data_all[
    lab_measurements_data_all["timestamp"]
    <= lab_measurements_data_all["date_treatment_1st_line"]
].reset_index(drop=True)

lab_measurements_data_all = (
    lab_measurements_data_all.groupby(["patientid", "variable_code"])
    .agg(timestamp=("timestamp", "max"))
    .reset_index()
)

lab_measurements_data_all["variable_code"] = "all"
lab_measurements_data_all["value"] = 1
lab_measurements_data_all["data_source"] = "lab_measurements_data_all"

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = ["lab_measurements_data", "lab_measurements_data_all"]
