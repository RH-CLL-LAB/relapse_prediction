from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from tqdm import tqdm
from data_processing.medicine import *
from data_processing.persimune import persimune_dict
from data_processing.lookup_tables import *
from data_processing.social_history import *
from data_processing.blood_tests import *
from data_processing.lyfo_aki import *
from data_processing.wide_data import *
from data_processing.sks_opr import *
from data_processing.picture_diagnostics import *

# performance status and IPIs for the patients

print("load IPIs and WIDE DATA")

IPIs = pd.read_csv(
    "../../../../../projects2/dalyca_r/clean_r/shared_projects/end_of_project_scripts_to_gihub/DALYCARE_methods/output/IPI_2.csv",
    sep=";",
)

diseases = IPIs["Disease"].unique()

for disease in diseases:
    IPIs.loc[IPIs["Disease"] == disease, f"{disease}_PS"] = IPIs[
        IPIs["Disease"] == disease
    ]["PS"]
    IPIs.loc[IPIs["Disease"] == disease, f"{disease}_IPI"] = IPIs[
        IPIs["Disease"] == disease
    ]["IPI"]

IPIs_ffill = (
    IPIs[[x for x in IPIs.columns if x not in ["Sex", "PS", "IPI", "Disease"]]]
    .groupby("patientid")
    .fillna(method="ffill")
)
IPIs_ffill["patientid"] = IPIs["patientid"]
IPIs_ffill = IPIs_ffill.groupby("patientid").agg("last").reset_index()

IPIs_ffill = IPIs_ffill.fillna(-1)

WIDE_DATA = WIDE_DATA.merge(IPIs_ffill, how="left")

TABLE_TO_LONG_FORMAT_MAPPING = {
    "SDS_pato": {
        "d_rekvdato": "timestamp",
        "patientid": "patientid",
        "c_snomedkode": "variable_code",
    },
    "SP_AlleProvesvar": {
        "patientid": "patientid",
        "component": "variable_code",
        "specimn_taken_time": "timestamp",
        "ord_value": "value",
    },
    "SDS_lab_forsker": {
        "samplingdate": "timestamp",
        "analysiscode": "variable_code",
        "value": "value",
        "patientid": "patientid",
    },
    # we need for administered medicine to have the days
    # of how long they were on the drug and
    # and cumulative dosis
    "laboratorymeasurements": {
        "patientid": "patientid",
        "samplingdate": "timestamp",
        "analysiscode": "variable_code",
        "c_value": "value",
    },
    "view_laboratorymeasuments_c_groups": {
        "patientid": "patientid",
        "samplingdate": "timestamp",
        "analysisgroup": "variable_code",
        "value": "value",
    },
    "diagnoses_all": {
        "patientid": "patientid",
        "date_diagnosis": "timestamp",
        "diagnosis": "variable_code",
    },
    # NOTE: some of these variables are
    # categorical (bevidsthedsniveau)
    "SP_VitaleVaerdier": {
        "patientid": "patientid",
        "recorded_time": "timestamp",
        "displayname": "variable_code",
        "meas_value_clean": "value",
    },
}

print("load data from data dict")

data_dict = {
    table_name: (
        download_and_rename_data(
            table_name,
            TABLE_TO_LONG_FORMAT_MAPPING,
            cohort=lyfo_cohort,
            cohort_column="patientid",
        )
    )
    for table_name in tqdm(TABLE_TO_LONG_FORMAT_MAPPING)
}

# there are duplicates in diagnosis_all in

# this wont work - if they had the diagnosis after
# treatment time, then we wont include them
data_dict["SP_AlleProvesvar"]["value"] = pd.to_numeric(
    data_dict["SP_AlleProvesvar"]["value"].str.replace("[><]", "", regex=True),
    errors="coerce",
)

diagnosis_all_filtered = data_dict["diagnoses_all"].merge(
    WIDE_DATA[["patientid", "date_treatment_1st_line"]], how="left"
)
diagnosis_all_filtered = diagnosis_all_filtered[
    diagnosis_all_filtered["timestamp"]
    < diagnosis_all_filtered["date_treatment_1st_line"]
].reset_index(drop=True)[["patientid", "timestamp", "variable_code", "data_source"]]

diagnosis_all_comorbidity = (
    diagnosis_all_filtered.groupby(["patientid", "variable_code"])
    .agg(timestamp=("timestamp", "max"))
    .reset_index()
)
diagnosis_all_comorbidity = diagnosis_all_comorbidity.rename(
    columns={"variable_code": "value"}
).reset_index(drop=True)
diagnosis_all_comorbidity["value"] = "all"
# data_dict["adm_medicine_days"] = adm_medicine_days

data_dict["sks_at_the_hospital"] = sks_at_the_hospital
data_dict["sks_referals"] = sks_referals

for dataset in persimune_dict:
    data_dict[dataset] = persimune_dict[dataset]

for dataset in medicine_data:
    data_dict[dataset] = medicine_data[dataset]
# social history

del persimune_dict
del medicine_data

data_dict["diagnoses_all"] = (
    data_dict["diagnoses_all"]
    .merge(DIAG_LOOKUP_TABLE, left_on="variable_code", right_on="Kode")[
        ["patientid", "timestamp", "data_source", "Tekst"]
    ]
    .rename(columns={"Tekst": "variable_code"})
    .reset_index(drop=True)
)

# merging NPU codes

data_dict["laboratorymeasurements_concat"] = (
    data_dict["laboratorymeasurements"]
    .merge(NPU_LOOKUP_TABLE, left_on="variable_code", right_on="NPU code")[
        ["patientid", "timestamp", "data_source", "value", "Component"]
    ]
    .rename(columns={"Component": "variable_code"})
    .reset_index(drop=True)
)
data_dict["laboratorymeasurements_concat"][
    "data_source"
] = "laboratorymeasurements_concat"


data_dict["PERSIMUNE_biochemistry"] = (
    data_dict["PERSIMUNE_biochemistry"]
    .merge(NPU_LOOKUP_TABLE, left_on="variable_code", right_on="NPU code")[
        ["patientid", "timestamp", "data_source", "value", "Component"]
    ]
    .rename(columns={"Component": "variable_code"})
    .reset_index(drop=True)
)

data_dict["SDS_pato"] = (
    data_dict["SDS_pato"]
    .merge(SNOMED_LOOKUP_TABLE, left_on="variable_code", right_on="SKSkode")[
        ["patientid", "timestamp", "data_source", "Kodetekst"]
    ]
    .rename(columns={"Kodetekst": "variable_code"})
    .reset_index(drop=True)
)


data_dict["SP_SocialHX"] = social_history_data
data_dict["SP_Bloddyrkning_Del1"] = blood_tests
data_dict["LYFO_AKI"] = LYFO_AKI
data_dict["SP_BilleddiagnostiskeUndersøgelser_Del1"] = picture_diagnostics

LONG_DATA = pd.concat(data_dict).reset_index(drop=True)

LONG_DATA = LONG_DATA[LONG_DATA["timestamp"] != "NULL"].reset_index(drop=True)

LONG_DATA["timestamp"] = pd.to_datetime(LONG_DATA["timestamp"])

# stupid fix for getting variables to work with value for pato and diagnoses

LONG_DATA.loc[LONG_DATA["data_source"].isin(["SDS_pato", "diagnoses_all"]), "value"] = (
    1  # doesn't matter what value this is
)

translation_dict = {"Negative": 0, "Positive": 1}

LONG_DATA.loc[
    LONG_DATA["data_source"].isin(
        ["PERSIMUNE_microbiology_analysis", "PERSIMUNE_microbiology_culture"]
    ),
    "value",
] = LONG_DATA[
    LONG_DATA["data_source"].isin(
        ["PERSIMUNE_microbiology_analysis", "PERSIMUNE_microbiology_culture"]
    )
][
    "value"
].apply(
    lambda x: translation_dict.get(x, -1)
)

LONG_DATA.loc[LONG_DATA["data_source"] == "LYFO_AKI", "variable_code"] = "n_aki"

LONG_DATA.loc[LONG_DATA["value"].isna(), "value"] = 1

# convert to fit diagnosis to treatment format

LONG_DATA["patientid"] = LONG_DATA["patientid"].astype(int)

print("dropping duplicates")
# NOTE: This scales terribly - should probably do it for each datasource instead
print(len(LONG_DATA))
LONG_DATA = LONG_DATA.drop_duplicates().reset_index(drop=True)
print(len(LONG_DATA))
# 263524163
# 154054307

# GASp - HALF ARE DUPLICATES WHAT THE
# That's so insane

WIDE_DATA.to_pickle("data/WIDE_DATA.pkl")

LONG_DATA.to_pickle("data/LONG_DATA.pkl")
