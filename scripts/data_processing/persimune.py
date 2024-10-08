from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort_strings

PERSIMUNE_MAPPING = {
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

# we don't really know what's up with this
# we need to group NPU codes

# we need to check for slopes if we can also get an intercept
# and not just the coefficient
# check with some dummy examples and make predictions before hand

persimune_dict["PERSIMUNE_leukocytes"] = download_and_rename_data(
    "PERSIMUNE_biochemistry",
    {
        "PERSIMUNE_biochemistry": {
            "patientid": "patientid",
            "samplingdatetime": "timestamp",
            "analysiscode": "variable_code",
            "c_associatedleukocytevalue": "value",
        }
    },
    cohort=lyfo_cohort_strings,
)
persimune_dict["PERSIMUNE_leukocytes"]["data_source"] = "PERSIMUNE_leukocytes"
# overall leukocytes
persimune_dict["PERSIMUNE_leukocytes"]["variable_code"] = "all"

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
