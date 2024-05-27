from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort

picture_diagnostics = download_and_rename_data(
    "SP_BilleddiagnostiskeUndersøgelser_Del1",
    {
        "SP_BilleddiagnostiskeUndersøgelser_Del1": {
            "patientid": "patientid",
            "bestillingsnavn": "variable_code",
            "bestillingstidspunkt": "timestamp",
            "status": "value",
        }
    },
    cohort=lyfo_cohort,
)
