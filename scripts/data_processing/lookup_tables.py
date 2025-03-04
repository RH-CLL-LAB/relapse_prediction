from helpers.sql_helper import *
from helpers.processing_helper import *


DIAG_LOOKUP_TABLE = load_data_from_table(
    "Codes_DST_DIAG_CODES", subset_columns=["Kode", "Tekst"]
)

ATC_LOOKUP_TABLE = load_data_from_table(
    "Codes_ATC", subset_columns=["class_code", "class_name"]
)
NPU_LOOKUP_TABLE = load_data_from_table(
    "Codes_NPU", subset_columns=["NPU code", "Component"]
)

SNOMED_LOOKUP_TABLE = (
    load_data_from_table("CODES_SNOMED", subset_columns=["SKSkode", "Kodetekst"])
    .drop_duplicates(subset="SKSkode")
    .reset_index(drop=True)
)

# NOTE: NPU_aggregation also removes a lot of information!
# Is this a feature or a bug? Should probably include both!


ATC_LOOKUP_TABLE = ATC_LOOKUP_TABLE.drop_duplicates()

ATC_LOOKUP_TABLE["class_name"] = ATC_LOOKUP_TABLE["class_name"].str.lower()
