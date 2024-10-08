from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort, lyfo_cohort_strings
from data_processing.lookup_tables import NPU_LOOKUP_TABLE
from tqdm import tqdm

tqdm.pandas()

# laboratorymeasurements - unit in unit
# SDS_lab_forsker - unit in unit
# PERSIMUNE_biochemistry - unit in c_resultunit
# SP_AlleProvesvar - no unit

lab_dict = {
    "laboratorymeasurements": {
        "patientid": "patientid",
        "samplingdate": "timestamp",
        "analysiscode": "variable_code",
        "value": "value",
        "unit": "unit",
    },
    "SDS_lab_forsker": {
        "patientid": "patientid",
        "samplingdate": "timestamp",
        "analysiscode": "variable_code",
        "value": "value",
        "unit": "unit",
    },
    "SP_AlleProvesvar": {
        "patientid": "patientid",
        "specimn_taken_time": "timestamp",
        "component": "variable_code",
        "ord_value": "value",
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
            "analysiscode": "variable_code",
            "c_resultvaluenumeric": "value",
            "c_resultunit": "unit",
        },
    },
    cohort=lyfo_cohort_strings,
)

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

lab_measurements_data = (
    lab_measurements_data.merge(
        NPU_LOOKUP_TABLE, left_on="variable_code", right_on="NPU code"
    )[["patientid", "timestamp", "data_source", "value", "unit", "Component"]]
    .rename(columns={"Component": "variable_code"})
    .reset_index(drop=True)
)

units = (
    lab_measurements_data.groupby(["variable_code", "unit"])
    .agg(count=("value", "count"))
    .reset_index()
)

units_counts = units["variable_code"].value_counts().reset_index()

# okay, so first we fix all the string stuff

# first stupid fix for having value-column
# be unedited
lab_measurements_data.loc[:, "value"] = lab_measurements_data["value"].astype(str)

lab_measurements_data["value"] = lab_measurements_data["value"].str.replace(",", ".")

lab_measurements_data["value"] = lab_measurements_data["value"].str.replace(
    r"([<>=+])|(Ca )", "", regex=True
)

lab_measurements_data["value_numeric"] = pd.to_numeric(
    lab_measurements_data["value"], errors="coerce"
)


# yeah, I think it still sort of makes sense to just
# say that text responses are in here with the
# rest of them
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


translation_dict = {
    "0 RhD POS": 0,
    "0 RhD NEG": 5,
    "0 RhD neg": 5,
    "A RhD pos": 1,
    "A RhD POS": 1,
    "B RhD POS": 2,
    "B RhD pos": 2,
    "B RhD NEG": 6,
    "B RhD neg": 6,
    "0 RhD pos": 0,
    "A RhD NEG": 3,
    "A RhD neg": 3,
    "O RhD pos": 0,
    "AB RhD POS": 4,
    "AB RhD pos": 4,
    "AB RhD NEG": 7,
    "AB RhD neg": 7,
}

lab_measurements_data.loc[
    lab_measurements_data["variable_code"] == "Erythrocyte antigen", "value_numeric"
] = lab_measurements_data[
    lab_measurements_data["variable_code"] == "Erythrocyte antigen"
][
    "value"
].progress_apply(
    lambda x: translation_dict.get(x)
)

lab_measurements_data.loc[lab_measurements_data["value"] == "Ca 15", "value"] = 15

lab_measurements_data.loc[
    (lab_measurements_data["value_numeric"].isna())
    & (lab_measurements_data["value"].isin(positives)),
    "value_numeric",
] = 1


lab_measurements_data.loc[
    lab_measurements_data["value_numeric"].isna(), "value_numeric"
] = 0


# lets just ignore all the categorical stuff for now -
# we can hope that some atc codes are only categorical

# ohhh so we need to put in as a place holder
# that no unit is also a unit... omg

lab_measurements_data.loc[lab_measurements_data["unit"].isnull(), "unit"] = "[NO_UNIT]"

lab_measurements_data["unit"] = lab_measurements_data["unit"].str.replace(
    r"((ï¿½)|(× )|(� )|(</sup>)|(</sup)|(<d7> ))", "", regex=True
)
lab_measurements_data["unit"] = lab_measurements_data["unit"].str.replace(
    r"((<sup>)|(\^))", "e", regex=True
)
lab_measurements_data["unit"] = lab_measurements_data["unit"].str.replace(
    r"<b5>", "µ", regex=True
)
lab_measurements_data["unit"] = lab_measurements_data["unit"].str.replace(
    r" ", "", regex=True
)

# uuhhh okay.
# what about just fucking cutting
# a lot of the units that have less than some threshold of
# of rows? that's at least important to keep in mind for later

# yep, that's what we're doing
unit_counts = (
    lab_measurements_data.groupby(["variable_code", "unit"])
    .agg(size_per_unit=("value", "count"))
    .reset_index()
)

variable_counts = (
    lab_measurements_data.groupby(["variable_code"])
    .agg(count_per_variable=("value", "count"), unique_ids=("patientid", "nunique"))
    .reset_index()
)

units = unit_counts.merge(variable_counts)

units["ratio"] = units["size_per_unit"] / units["count_per_variable"]

from data_processing.wide_data import WIDE_DATA

# NOTE: count the actual unique patientids per variable instead of this proxy
units = units[units["unique_ids"] > 0.05 * len(WIDE_DATA)].reset_index(drop=True)

units = units[units["ratio"] > 0.05].reset_index(drop=True)

lab_measurements_data = units[["variable_code", "unit"]].merge(lab_measurements_data)

# okay I think finally we are getting somewhere

# wait I can't do that here...

# unit_overall_counts = lab_measurements_data["unit"].value_counts().reset_index()

# grouper = lab_measurements_data[["variable_code", "unit", "value_numeric"]].groupby(["variable_code", "unit"])
# q1, q3 = grouper.quantile(0.25), grouper.quantile(0.75)
# q1 = q1.rename(columns={"value_numeric": "lower_quartile"})
# q3 = q3.rename(columns={"value_numeric": "higher_quartile"})
# q1.merge(q3)

unit_counts = units["variable_code"].value_counts().reset_index()


def normalize_units(data, variable_code, unit, factor):
    data.loc[
        (data["variable_code"] == variable_code) & (data["unit"] == unit),
        "value_numeric",
    ] = (
        data.loc[(data["variable_code"] == variable_code) & (data["unit"] == unit)][
            "value_numeric"
        ]
        * factor
    )


specific_units = unit_counts[unit_counts["variable_code"] == 2]
# specific_units


normalize_units(lab_measurements_data, "Iodide peroxidase antibody", "10e3int.enh/l", 5)
normalize_units(lab_measurements_data, "Iodide peroxidase antibody", "kiu/l", 5)

normalize_units(lab_measurements_data, "Erythrocytes", "fl", 0.001)
normalize_units(lab_measurements_data, "Erythrocytes", "10e12/l", 1 / 10)

normalize_units(lab_measurements_data, "Reticulocytes", "10e-3", 4)

lab_measurements_data = lab_measurements_data[
    ~(
        (lab_measurements_data["variable_code"] == "Coagulation, tissue factor-induced")
        & (lab_measurements_data["unit"] == "%")
    )
].reset_index(drop=True)
lab_measurements_data = lab_measurements_data[
    ~(
        (lab_measurements_data["variable_code"] == "Haemoglobin A1c")
        & (lab_measurements_data["unit"] == "%")
    )
].reset_index(drop=True)

normalize_units(lab_measurements_data, "Oxygen", "kpa", 1 / 10)

# NOTE: some of the no units are coded wrong and are in the same
# unit space as g/l

normalize_units(lab_measurements_data, "Iron", "", 55)
normalize_units(lab_measurements_data, "Iron", "[NO_UNIT]", 55)

normalize_units(lab_measurements_data, "Estradiol", "pmol/l", 1 / 1000)

normalize_units(lab_measurements_data, "Urate", "µmol/l", 1 / 1000)

normalize_units(lab_measurements_data, "M-component", "g/l", 1 / 10)

normalize_units(lab_measurements_data, "Urine sampling", "min", 1 / 60)

normalize_units(lab_measurements_data, "Albumin", "mg/l", 1 / 1000)

normalize_units(lab_measurements_data, "Haemoglobin", "fmol", 10)

normalize_units(lab_measurements_data, "Carbon dioxide", "kpa", 5)

normalize_units(lab_measurements_data, "Thyroxine", "nmol/l", 1 / 6)

normalize_units(lab_measurements_data, "Methotrexate", "µmol/l", 1000)

normalize_units(lab_measurements_data, "Triiodothyronine", "pmol/l", 1 / 3)

normalize_units(lab_measurements_data, "beta-2-Microglobulin", "nmol/l", 1 / 100)
lab_measurements_data = lab_measurements_data[["patientid", "timestamp", "variable_code", "value_numeric"]].rename(columns = {"value_numeric": "value"}).reset_index(drop = True)
lab_measurements_data["data_source"] = "labmeasurements"