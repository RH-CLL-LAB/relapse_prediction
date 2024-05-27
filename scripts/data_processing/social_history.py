from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort

social_history_dict = {
    "SP_SocialHX": {
        "patientid": "patientid",
        "registringsdato": "timestamp",
        "ryger": "smoking_cat",
        "pakkeromdagen": "smoking_num",
        "drikker": "drinking_cat",
        "damper": "vape_cat",
    }
}

[x for x in IMPORT_PUBLIC if "SP_Soci" in x]

social_history_data = download_and_rename_data(
    "SP_SocialHX",
    social_history_dict,
    cohort=lyfo_cohort,
)

social_history_data["smoking_num"] = pd.to_numeric(
    social_history_data["smoking_num"], errors="coerce"
)

social_history_data = social_history_data.melt(
    id_vars=["patientid", "timestamp", "data_source"], var_name="variable_code"
)
