from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from tqdm import tqdm
from data_processing.adm_medicine_days import *
from data_processing.persimune import persimune_dict
from data_processing.lookup_tables import *
from data_processing.social_history import *
from data_processing.blood_tests import *
from data_processing.lyfo_aki import *
from data_processing.wide_data import *

TABLE_TO_LONG_FORMAT_MAPPING = {
    "SDS_pato": {
        "d_rekvdato": "timestamp",
        "patientid": "patientid",
        "c_snomedkode": "variable_code",
    },
    "RECEPTDATA_CLEAN_pseudo": {
        "atc_kode": "variable_code",
        "expdato": "timestamp",
        "PATIENTID": "patientid",
        "styrke": "value",
    },
    "adm_medicine": {
        "patientid": "patientid",
        "d_ord_start_date": "timestamp",
        "c_atc": "variable_code",
        "v_styrke_num": "value",
        # "v_adm_dosis": "value",
    },
    # we need for administered medicine to have the days
    # of how long they were on the drug and
    # and cumulative dosis
    "laboratorymeasurements": {
        "patientid": "patientid",
        "samplingdate": "timestamp",
        "analysiscode": "variable_code",
        "value": "value",
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
}


data_dict = {
    table_name: (
        download_and_rename_data(
            table_name,
            TABLE_TO_LONG_FORMAT_MAPPING,
            cohort=lyfo_cohort,
            cohort_column="PATIENTID",
        )
        if table_name == "RECEPTDATA_CLEAN_pseudo"
        else download_and_rename_data(
            table_name,
            TABLE_TO_LONG_FORMAT_MAPPING,
            cohort=lyfo_cohort,
            cohort_column="patientid",
        )
    )
    for table_name in TABLE_TO_LONG_FORMAT_MAPPING
}

data_dict["adm_medicine_days"] = adm_medicine_days

for dataset in persimune_dict:
    data_dict[dataset] = persimune_dict[dataset]
# social history

for dataset in ["RECEPTDATA_CLEAN_pseudo", "adm_medicine", "adm_medicine_days"]:
    data_dict[dataset]["atc_level_1"] = data_dict[dataset]["variable_code"].apply(
        lambda x: x[0:1]
    )

    data_dict[dataset]["atc_level_2"] = data_dict[dataset]["variable_code"].apply(
        lambda x: x[0:3]
    )

    data_dict[dataset]["atc_level_3"] = data_dict[dataset]["variable_code"].apply(
        lambda x: x[0:4]
    )

    data_dict[dataset]["atc_level_4"] = data_dict[dataset]["variable_code"].apply(
        lambda x: x[0:5]
    )

    data_dict[dataset]["atc_level_5"] = data_dict[dataset]["variable_code"]


def melt_data(df: pd.DataFrame):
    df = df[[x for x in df.columns if x != "variable_code"]].reset_index(drop=True)
    return df.melt(
        id_vars=["patientid", "timestamp", "data_source", "value"],
        var_name="level",
        value_name="variable_code",
    ).reset_index(drop=True)[
        ["patientid", "timestamp", "data_source", "value", "variable_code"]
    ]


data_dict["RECEPTDATA_CLEAN_pseudo"] = melt_data(data_dict["RECEPTDATA_CLEAN_pseudo"])
data_dict["adm_medicine"] = melt_data(data_dict["adm_medicine"])
data_dict["adm_medicine_days"] = melt_data(data_dict["adm_medicine_days"])
data_dict["adm_medicine"]["value"] = pd.to_numeric(data_dict["adm_medicine"]["value"])

data_dict["RECEPTDATA_CLEAN_pseudo"] = (
    data_dict["RECEPTDATA_CLEAN_pseudo"]
    .merge(ATC_LOOKUP_TABLE, left_on="variable_code", right_on="class_code")[
        ["patientid", "timestamp", "data_source", "value", "class_name"]
    ]
    .rename(columns={"class_name": "variable_code"})
    .reset_index(drop=True)
)
data_dict["adm_medicine"] = (
    data_dict["adm_medicine"]
    .merge(ATC_LOOKUP_TABLE, left_on="variable_code", right_on="class_code")[
        ["patientid", "timestamp", "data_source", "value", "class_name"]
    ]
    .rename(columns={"class_name": "variable_code"})
    .reset_index(drop=True)
)

data_dict["adm_medicine_days"] = (
    data_dict["adm_medicine_days"]
    .merge(ATC_LOOKUP_TABLE, left_on="variable_code", right_on="class_code")[
        ["patientid", "timestamp", "data_source", "value", "class_name"]
    ]
    .rename(columns={"class_name": "variable_code"})
    .reset_index(drop=True)
)

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


data_dict["SP_SocialHx"] = social_history_data
data_dict["view_sp_bloddyrkning_del1"] = blood_tests
data_dict["LYFO_AKI"] = LYFO_AKI

LONG_DATA = pd.concat(data_dict).reset_index(drop=True)

timestamp_dates = [
    x for x in DATE_CONVERTER if DATE_CONVERTER.get(x) == date_from_timestamp
]
unix_dates = [
    x for x in DATE_CONVERTER if DATE_CONVERTER.get(x) == date_from_origin_unix
]

for date_list in tqdm([timestamp_dates, unix_dates]):
    if date_list == timestamp_dates:
        LONG_DATA.loc[LONG_DATA["data_source"].isin(date_list), "timestamp"] = (
            pd.to_datetime(
                LONG_DATA[LONG_DATA["data_source"].isin(date_list)]["timestamp"],
                unit="s",
                errors="coerce",
            ).dt.date
        )
    elif date_list == unix_dates:
        LONG_DATA.loc[LONG_DATA["data_source"].isin(date_list), "timestamp"] = (
            pd.to_datetime(
                LONG_DATA[LONG_DATA["data_source"].isin(date_list)]["timestamp"],
                origin="unix",
                unit="d",
                errors="coerce",
            ).dt.date
        )

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

LONG_DATA.loc[LONG_DATA["value"].isna(), "value"] = 1

WIDE_DATA.to_pickle("data/relapse_data/WIDE_DATA.pkl")

LONG_DATA.to_pickle("data/relapse_data/LONG_DATA.pkl")
