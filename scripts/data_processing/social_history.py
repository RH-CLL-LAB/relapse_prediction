from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort
from datetime import timedelta

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


mistake = social_history_data[
    social_history_data["timestamp"] < pd.to_datetime("2000-01-01")
].reset_index(drop=True)

mistake["seconds"] = (mistake["timestamp"] - pd.to_datetime("1970-01-01")).dt.seconds

mistake["timestamp"] = pd.to_datetime(mistake["timestamp"])

mistake["date"] = mistake.apply(
    lambda x: x["timestamp"] + timedelta(days=x["seconds"]), axis=1
)

social_history_data = social_history_data.merge(mistake, how="left")

social_history_data.loc[
    social_history_data["date"].notna(), "timestamp"
] = social_history_data[social_history_data["date"].notna()]["date"]

social_history_data = social_history_data[
    ["patientid", "timestamp", "data_source", "variable_code", "value"]
]
