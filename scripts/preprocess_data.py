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
from data_processing.lab_values import *
from data_processing.laboratory_measurements import (
    lab_measurements_data,
    lab_measurements_data_all,
)
from data_processing.pathology_specifics import pathology_genes, concatenated_pathology

# performance status and IPIs for the patients


from datetime import timedelta

print("load IPIs and WIDE DATA")

IPIs = pd.read_csv(
    "../../../../../projects2/dalyca_r/clean_r/shared_projects/end_of_project_scripts_to_gihub/DALYCARE_methods/output/IPI_2.csv",
    sep=";",
)


IPIs["IPI"] = pd.Categorical(
    IPIs["IPI"], ordered=True, categories=["Low", "Intermediate", "High", "Very high"]
).codes

IPIs_concat = (
    IPIs.groupby(["patientid"])
    .agg(PS=("PS", "mean"), IPI=("IPI", "mean"))
    .reset_index()
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

IPIs_ffill = IPIs_ffill.merge(IPIs_concat, how="left")

WIDE_DATA = WIDE_DATA.merge(IPIs_ffill, how="left")

TABLE_TO_LONG_FORMAT_MAPPING = {
    "SDS_pato": {
        "d_rekvdato": "timestamp",
        "patientid": "patientid",
        "c_snomedkode": "variable_code",
    },
    "diagnoses_all": {
        "patientid": "patientid",
        "date_diagnosis": "timestamp",
        "diagnosis": "variable_code",
    },
    "SP_VitaleVaerdier": {
        "patientid": "patientid",
        "recorded_time": "timestamp",
        "displayname": "variable_code",
        "meas_value_clean": "value",
    },
    "PERSIMUNE_radiology": {
        "patientid": "patientid",
        "bookingdatetime": "timestamp",
        "resultcode": "variable_code",
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


data_dict["sks_at_the_hospital"] = sks_at_the_hospital
data_dict["sks_referals"] = sks_referals
data_dict["sks_at_the_hospital_unique"] = sks_at_the_hospital_unique
data_dict["sks_referals_unique"] = sks_referals_unique

for dataset in persimune_dict:
    data_dict[dataset] = persimune_dict[dataset]

for dataset in medicine_dict:
    data_dict[dataset] = medicine_dict[dataset]

for dataset in lab_data:
    data_dict[dataset] = lab_data[dataset]

# delete for memory
del persimune_dict
del medicine_dict

data_dict["diagnoses_all"] = (
    data_dict["diagnoses_all"]
    .merge(DIAG_LOOKUP_TABLE, left_on="variable_code", right_on="Kode")[
        ["patientid", "timestamp", "data_source", "Tekst"]
    ]
    .rename(columns={"Tekst": "variable_code"})
    .reset_index(drop=True)
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
diagnosis_all_comorbidity["variable_code"] = "all"

diagnosis_all_comorbidity["value"] = 1

diagnosis_all_comorbidity["data_source"] = "diagnoses_all_comorbidity"

data_dict["diagnoses_all_comorbidity"] = diagnosis_all_comorbidity

data_dict["SDS_pato"] = (
    data_dict["SDS_pato"]
    .merge(SNOMED_LOOKUP_TABLE, left_on="variable_code", right_on="SKSkode")[
        ["patientid", "timestamp", "data_source", "Kodetekst"]
    ]
    .rename(columns={"Kodetekst": "variable_code"})
    .reset_index(drop=True)
)

data_dict["labmeasurements"] = lab_measurements_data
data_dict["lab_measurements_data_all"] = lab_measurements_data_all
data_dict["SP_SocialHX"] = social_history_data
data_dict["SP_Bloddyrkning_Del1"] = blood_tests
data_dict["blood_tests_all"] = blood_tests_all
data_dict["gene_alterations"] = pathology_genes
data_dict["pathology_concat"] = concatenated_pathology


LYFO_AKI["timestamp"] = pd.to_datetime(LYFO_AKI["timestamp"])

data_dict["LYFO_AKI"] = LYFO_AKI
data_dict["SP_BilleddiagnostiskeUndersøgelser_Del1"] = picture_diagnostics


LONG_DATA = pd.concat(data_dict).reset_index(drop=True)

LONG_DATA = LONG_DATA[LONG_DATA["timestamp"] != "NULL"].reset_index(drop=True)

LONG_DATA["timestamp"] = pd.to_datetime(LONG_DATA["timestamp"])

LONG_DATA.loc[
    LONG_DATA["data_source"].isin(["SDS_pato", "diagnoses_all", "PERSIMUNE_radiology"]),
    "value",
] = 1  # doesn't matter what value this is


LONG_DATA.loc[LONG_DATA["data_source"] == "LYFO_AKI", "variable_code"] = "n_aki"

LONG_DATA.loc[LONG_DATA["value"].isna(), "value"] = 1

# convert to fit diagnosis to treatment format

LONG_DATA["patientid"] = LONG_DATA["patientid"].astype(int)

for column in WIDE_DATA.columns:
    if "date" in column:
        WIDE_DATA[column] = pd.to_datetime(WIDE_DATA[column], errors="coerce")

LONG_DATA = LONG_DATA.merge(WIDE_DATA[["patientid", "date_treatment_1st_line"]])


# filtering so we only have data from before treatment AND after 3 year before treatment
LONG_DATA = LONG_DATA[
    (LONG_DATA["timestamp"] <= LONG_DATA["date_treatment_1st_line"])
    & (
        LONG_DATA["timestamp"]
        >= LONG_DATA["date_treatment_1st_line"] - datetime.timedelta(days=365 * 3)
    )
].reset_index(drop=True)


LONG_DATA = LONG_DATA[
    [x for x in LONG_DATA.columns if x != "date_treatment_1st_line"]
].reset_index(drop=True)

WIDE_DATA.to_pickle("data/WIDE_DATA.pkl")

LONG_DATA.to_pickle("data/LONG_DATA.pkl")
