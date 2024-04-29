from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from wide_data import lyfo_cohort

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
