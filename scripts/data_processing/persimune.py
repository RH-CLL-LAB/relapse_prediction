from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from wide_data import lyfo_cohort_strings

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
