from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from helpers.constants import *

SDS_LAB_MAP = {
    "SDS_lab_forsker": {
        "patientid": "id",
        "samplingdate": "date",
        "laboratorium_idcode": "idcode",
    }
}

sds_lab = download_and_rename_data("SDS_lab_forsker", SDS_LAB_MAP)

codes = load_data_from_table("SDS_lab_labidcodes", subset_columns=["idcode", "region"])
sds_lab = sds_lab.merge(codes)[["id", "date", "region", "data_source"]]
