from helpers.sql_helper import *
from helpers.preprocessing_helper import *
import seaborn as sns
from tqdm import tqdm
import numpy as np
import os


DIAG_LOOKUP_TABLE = load_data_from_table(
    "Codes_DST_DIAG_CODES", subset_columns=["Kode", "Tekst"]
)

ATC_LOOKUP_TABLE = load_data_from_table(
    "Codes_ATC", subset_columns=["class_code", "class_name"]
)
NPU_LOOKUP_TABLE = load_data_from_table(
    "Codes_NPU", subset_columns=["NPU code", "Component"]
)

# NOTE: NPU_aggregation also removes a lot of information!
# Is this a feature or a bug? Should probably include both!


ATC_LOOKUP_TABLE = ATC_LOOKUP_TABLE.drop_duplicates()

rkkp_df = pd.read_csv("/ngc/projects2/dalyca_r/clean_r/RKKP_LYFO_CLEAN.csv")

dlbcl_rkkp_df = rkkp_df[rkkp_df["date_treatment_1st_line"].notna()].reset_index(
    drop=True
)
dlbcl_rkkp_df["date_treatment_1st_line"] = pd.to_datetime(
    dlbcl_rkkp_df["date_treatment_1st_line"]
)

# dlbcl_rkkp_df = dlbcl_rkkp_df[dlbcl_rkkp_df["subtype"] == "DLBCL"].reset_index(drop=True)

lyfo_extra = pd.read_csv(
    "/ngc/projects2/dalyca_r/mikwer_r/RKKP_LYFO_EXTRA_RELAPS_psAnon.csv"
)

# merge extra information
WIDE_DATA = dlbcl_rkkp_df.merge(lyfo_extra).reset_index(drop=True)


WIDE_DATA["relaps_pato_dt"] = pd.to_datetime(WIDE_DATA["relaps_pato_dt"])

WIDE_DATA["date_diagnosis"] = pd.to_datetime(WIDE_DATA["date_diagnosis"])
WIDE_DATA["date_death"] = pd.to_datetime(WIDE_DATA["date_death"])

# relapse label - is this true?

WIDE_DATA["relapse_label"] = WIDE_DATA["LYFO_15_002=relapsskema"]

WIDE_DATA["relapse_date"] = WIDE_DATA["date_relapse_confirmed_2nd_line"]

WIDE_DATA.loc[WIDE_DATA["relaps_pato_dt"].notna(), "relapse_label"] = 1

WIDE_DATA.loc[WIDE_DATA["relaps_pato_dt"].notna(), "relapse_date"] = WIDE_DATA[
    WIDE_DATA["relaps_pato_dt"].notna()
]["relaps_pato_dt"]

WIDE_DATA["days_to_death"] = (
    WIDE_DATA["date_death"] - WIDE_DATA["date_treatment_1st_line"]
).dt.days

WIDE_DATA["relapse_date"] = pd.to_datetime(WIDE_DATA["relapse_date"])

WIDE_DATA["days_to_relapse"] = (
    WIDE_DATA["relapse_date"] - WIDE_DATA["date_treatment_1st_line"]
).dt.days

WIDE_DATA.loc[WIDE_DATA["days_to_death"] < 730, "relapse_label"] = 1
WIDE_DATA.loc[
    (WIDE_DATA["days_to_death"] < 730) & (WIDE_DATA["relapse_date"].isna()),
    "relapse_date",
] = WIDE_DATA[(WIDE_DATA["date_death"].notna()) & ((WIDE_DATA["relapse_date"].isna()))][
    "date_death"
] 
WIDE_DATA.loc[
    (WIDE_DATA["days_to_death"] < 730)
    & ((WIDE_DATA["days_to_relapse"] > 730) | (WIDE_DATA["days_to_relapse"].isna())),
    "proxy_death",
] = 1

# remove uncertain patients from the cohort
WIDE_DATA = WIDE_DATA[WIDE_DATA["relapse_label"] != 0].reset_index(drop=True)

lyfo_cohort = get_cohort_string_from_data(WIDE_DATA)
lyfo_cohort_strings = lyfo_cohort
lyfo_cohort_strings = lyfo_cohort_strings.replace("(", "('")
lyfo_cohort_strings = lyfo_cohort_strings.replace(")", "')")
lyfo_cohort_strings = lyfo_cohort_strings.replace(", ", "', '")

# Handled by fallback instead
# WIDE_DATA = WIDE_DATA.replace(np.NaN, -1)

adm_medicine_days = download_and_rename_data(
    "adm_medicine",
    {
        "adm_medicine": {
            "patientid": "patientid",
            "d_ord_start_date": "timestamp",
            "d_ord_slut_date": "end_date",
            "c_atc": "variable_code",
        }
    },
    cohort=lyfo_cohort,
)

adm_medicine_days["timestamp"] = pd.to_datetime(adm_medicine_days["timestamp"])
adm_medicine_days["end_date"] = pd.to_datetime(adm_medicine_days["end_date"])
adm_medicine_days["value"] = (
    adm_medicine_days["end_date"] - adm_medicine_days["timestamp"]
).dt.days

adm_medicine_days = adm_medicine_days[
    ["patientid", "timestamp", "variable_code", "value"]
]

adm_medicine_days["data_source"] = "adm_medicine_days"

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
        "v_styrke_num": "value"
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
    table_name: download_and_rename_data(
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
    for table_name in TABLE_TO_LONG_FORMAT_MAPPING
}

data_dict["adm_medicine_days"] = adm_medicine_days

# social history

social_history_dict = {
    "SP_SocialHx": {
        "patientid": "patientid",
        "registringsdato": "timestamp",
        "ryger": "smoking_cat",
        "pakkeromdagen": "smoking_num",
        "drikker": "drinking_cat",
        "damper": "vape_cat",
    }
}

social_history_data = download_and_rename_data(
    "SP_SocialHx",
    social_history_dict,
    cohort=lyfo_cohort,
)

social_history_data["smoking_num"] = pd.to_numeric(
    social_history_data["smoking_num"], errors="coerce"
)

social_history_data = social_history_data.melt(
    id_vars=["patientid", "timestamp", "data_source"], var_name="variable_code"
)

PERSIMUNE_MAPPING = {
    "PERSIMUNE_biochemistry": {
        "patientid": "patientid",
        "samplingdatetime": "timestamp",
        "analysiscode": "variable_code",
        "c_resultvaluenumeric": "value",
    },
    "PERSIMUNE_microbiology_analysis": {
        "patientid": "patientid",
        "samplingdatetime": "timestamp",
        "analysisshortname": "variable_code",
        "c_categoricalresult": "value",
    },
    "PERSIMUNE_microbiology_culture": {
        "patientid": "patientid",
        "samplingdatetime": "timestamp",
        "c_domain": "variable_code",
        "c_pm_categoricalresult": "value",
    },
}

# persimune patientid is in some weird object format :'()

persimune_dict = {
    table_name: download_and_rename_data(
        table_name, PERSIMUNE_MAPPING, cohort=lyfo_cohort_strings
    )
    for table_name in PERSIMUNE_MAPPING
}

for dataset in persimune_dict:
    persimune_dict[dataset]["patientid"] = persimune_dict[dataset]["patientid"].astype(
        int
    )
    persimune_dict[dataset]["timestamp"] = pd.to_datetime(
        persimune_dict[dataset]["timestamp"], errors="coerce", utc=True
    )

    persimune_dict[dataset]["timestamp"] = persimune_dict[dataset][
        "timestamp"
    ].dt.tz_localize(None)
    persimune_dict[dataset] = persimune_dict[dataset][
        persimune_dict[dataset]["patientid"].isin(WIDE_DATA["patientid"])
    ].reset_index(drop=True)
    data_dict[dataset] = persimune_dict[dataset]

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


blood_tests = download_and_rename_data(
    "view_sp_bloddyrkning_del1",
    {
        "view_sp_bloddyrkning_del1": {
            "patientid": "patientid",
            "komponentnavn": "variable_code",
            "pr_vetagningstidspunkt_dato": "timestamp",
        }
    },
    cohort=lyfo_cohort,
)


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

# we need to merge the different atc codes to the common ones
# and then make a misc category

# this is not the setup we want - we want to have both the
# polypharmacy (just n_unique on values of variable_codes)
# and then the counts of individual levels
# current, we're just getting how many counts were for individual level atc codes and the latest value.

# fix dates

LYFO_AKI = pd.read_csv("/ngc/projects2/dalyca_r/clean_r/LYFO_AKI.csv")

LYFO_AKI["data_source"] = "LYFO_AKI"

LYFO_AKI["variable_code"] = 1

LYFO_AKI = (
    LYFO_AKI[["patientid", "sampledate", "n.AKI", "data_source"]]
    .rename(columns={"n.AKI": "value", "sampledate": "timestamp"})
    .reset_index(drop=True)
)

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
        LONG_DATA.loc[
            LONG_DATA["data_source"].isin(date_list), "timestamp"
        ] = pd.to_datetime(
            LONG_DATA[LONG_DATA["data_source"].isin(date_list)]["timestamp"],
            unit="s",
            errors="coerce",
        ).dt.date
    elif date_list == unix_dates:
        LONG_DATA.loc[
            LONG_DATA["data_source"].isin(date_list), "timestamp"
        ] = pd.to_datetime(
            LONG_DATA[LONG_DATA["data_source"].isin(date_list)]["timestamp"],
            origin="unix",
            unit="d",
            errors="coerce",
        ).dt.date

LONG_DATA["timestamp"] = pd.to_datetime(LONG_DATA["timestamp"])

# stupid fix for getting variables to work with value for pato and diagnoses

LONG_DATA.loc[
    LONG_DATA["data_source"].isin(["SDS_pato", "diagnoses_all"]), "value"
] = 1  # doesn't matter what value this is

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
