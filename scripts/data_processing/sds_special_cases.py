from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from helpers.constants import *

# individual loads for merging purposes:

data_udtil = load_data_from_table("SDS_t_udtilsgh", subset_columns=["v_recnum"])
sds_t_adm = load_data_from_table(
    "SDS_t_adm", subset_columns=["d_inddto", "k_recnum", "patientid", "c_sgh"]
).rename(columns={"c_sgh": "hospital_id"})

data_udtil = data_udtil.rename(columns={"v_recnum": "k_recnum"})
sds_data_udtil = data_udtil.merge(sds_t_adm).reset_index(drop=True)
sds_data_udtil = sds_data_udtil.rename(columns={"d_inddto": "date", "patientid": "id"})[
    ["date", "id", "hospital_id"]
].reset_index(drop=True)
sds_t_adm = sds_t_adm.rename(columns={"d_inddto": "date", "patientid": "id"})[
    ["date", "id", "hospital_id"]
].reset_index(drop=True)
sds_t_adm["data_source"] = "SDS_t_adm"
sds_data_udtil["data_source"] = "SDS_t_udtilsgh"

subset_of_sds_data = [sds_t_adm, sds_data_udtil]
