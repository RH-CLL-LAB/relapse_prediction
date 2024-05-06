from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort

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
