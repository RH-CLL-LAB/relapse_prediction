from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from wide_data import lyfo_cohort

# fix dates

LYFO_AKI = pd.read_csv("/ngc/projects2/dalyca_r/clean_r/LYFO_AKI.csv")

LYFO_AKI["data_source"] = "LYFO_AKI"

LYFO_AKI["variable_code"] = 1

LYFO_AKI = (
    LYFO_AKI[["patientid", "sampledate", "n.AKI", "data_source"]]
    .rename(columns={"n.AKI": "value", "sampledate": "timestamp"})
    .reset_index(drop=True)
)
