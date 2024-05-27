from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort

blood_tests = download_and_rename_data(
    "SP_Bloddyrkning_Del1",
    {
        "SP_Bloddyrkning_Del1": {
            "patientid": "patientid",
            "komponentnavn": "variable_code",
            "prøvetagningstidspunkt": "timestamp",
            "prøveresultat": "value",
        }
    },
    cohort=lyfo_cohort,
)
