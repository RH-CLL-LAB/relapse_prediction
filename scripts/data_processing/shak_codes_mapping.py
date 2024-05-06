from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from helpers.constants import LPR_THREE_TABLES


table_name = "Codes_shak"  # insert name of table to load
limit = False  # insert limit of how many rows to include
subset_columns = []  # insert subset columns needed for the table
data_codes = load_data_from_table(
    table_name=table_name, subset_columns=subset_columns, limit=limit
)

data_codes = data_codes[data_codes["a"].str.contains("sgh")].reset_index(drop=True)
data_codes["a"] = data_codes["a"].str.replace("sgh", "")
data_codes["b"] = data_codes["b"].apply(lambda x: x[24:])

# NOTE: Consider also including all non-"sghs"
# there might be weird ways of writing


data_codes["a"] = data_codes["a"].astype(int)
data_codes.loc[data_codes["a"] < 2512, "region"] = "Region Hovedstaden"
data_codes.loc[
    (2512 <= data_codes["a"]) & (4200 > data_codes["a"]), "region"
] = "Region Sjælland"

data_codes.loc[
    (4200 <= data_codes["a"]) & (6000 > data_codes["a"]), "region"
] = "Region Syddanmark"


data_codes.loc[
    (6000 <= data_codes["a"]) & (8000 > data_codes["a"]), "region"
] = "Region Midtjylland"


data_codes.loc[
    (8000 <= data_codes["a"]) & (9001 > data_codes["a"]), "region"
] = "Region Nordjylland"

data_codes.loc[data_codes["a"] == 4001, "region"] = "Region Hovedstaden"

data_codes.loc[data_codes["a"] == 6008, "region"] = "Region Syddanmark"

# data_codes.loc[data_codes["a"] == 6006, "region"] = "Region Midtjylland"

SHAK_CODES_MERGE_TABLE = (
    data_codes[data_codes["region"].notna()][["a", "region"]]
    .rename(columns={"a": "hospital_id"})
    .reset_index(drop=True)
)

# SHAK_TO_REGION_MAPPING = data_codes.set_index("a").to_dict()["region"]
