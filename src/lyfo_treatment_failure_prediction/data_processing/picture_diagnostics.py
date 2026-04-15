from lyfo_treatment_failure_prediction.helpers.sql_helper import (
    download_and_rename_data,
)
from lyfo_treatment_failure_prediction.data_processing.wide_data import lyfo_cohort


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

__all__ = ["picture_diagnostics"]
