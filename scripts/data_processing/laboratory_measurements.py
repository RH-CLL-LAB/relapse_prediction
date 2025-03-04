from helpers.sql_helper import *
from helpers.processing_helper import *
from data_processing.wide_data import lyfo_cohort, lyfo_cohort_strings
from data_processing.lookup_tables import NPU_LOOKUP_TABLE
from wide_data import WIDE_DATA
from tqdm import tqdm


tqdm.pandas()

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
        table_name=table_name, config_dict=lab_dict, cohort=lyfo_cohort
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

# treat NPU codes that are not in analysis groups as their own thing

persimune["patientid"] = persimune["patientid"].astype(int)
persimune["timestamp"] = pd.to_datetime(
    persimune["timestamp"], errors="coerce", utc=True
)

persimune["timestamp"] = persimune["timestamp"].dt.tz_localize(None)

lab_measurements["PERSIMUNE_biochemistry"] = persimune

lab_measurements_data = pd.concat(lab_measurements).reset_index(drop=True)

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

# okay new approach - first drop all duplicates
lab_measurements_data["timestamp"] = pd.to_datetime(
    pd.to_datetime(lab_measurements_data["timestamp"]).dt.date
)

lab_measurements_data = lab_measurements_data.drop_duplicates(
    subset=["patientid", "timestamp", "value", "variable_code"]
).reset_index(drop=True)

lab_measurements_data["unit"] = lab_measurements_data["unit"].str.lower()


# change to strings to fix string values

lab_measurements_data.loc[:, "value"] = lab_measurements_data["value"].astype(str)

lab_measurements_data["value"] = lab_measurements_data["value"].str.replace(",", ".")

lab_measurements_data["value"] = lab_measurements_data["value"].str.replace(
    r"([<>=+])|(Ca )", "", regex=True
)

lab_measurements_data["value_numeric"] = pd.to_numeric(
    lab_measurements_data["value"], errors="coerce"
)

# make negative and positive findings string lists
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


# for blood types

lab_measurements_data = (
    lab_measurements_data[["patientid", "timestamp", "variable_code", "value_numeric"]]
    .rename(columns={"value_numeric": "value"})
    .reset_index(drop=True)
)
lab_measurements_data["data_source"] = "labmeasurements"

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
