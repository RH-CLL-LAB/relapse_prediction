from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort

sks_opr_at_the_hospital = download_and_rename_data(
    "view_sds_t_adm_t_sksopr",
    {
        "view_sds_t_adm_t_sksopr": {
            "c_opr": "variable_code",
            "v_behdage": "value",
            "patientid": "patientid",
            "d_inddto": "timestamp",
        }
    },
    cohort=lyfo_cohort,
)


sks_opr_not_at_the_hospital = download_and_rename_data(
    "view_sds_t_adm_t_sksopr",
    {
        "view_sds_t_adm_t_sksopr": {
            "c_opr": "variable_code",
            "v_behdage": "value",
            "patientid": "patientid",
            "d_hendto": "timestamp",
        }
    },
    cohort=lyfo_cohort,
)

sks_ube_at_the_hospital = download_and_rename_data(
    "view_sds_t_adm_t_sksube",
    {
        "view_sds_t_adm_t_sksube": {
            "c_opr": "variable_code",
            "v_behdage": "value",
            "patientid": "patientid",
            "d_inddto": "timestamp",
        }
    },
    cohort=lyfo_cohort,
)


sks_ube_not_at_the_hospital = download_and_rename_data(
    "view_sds_t_adm_t_sksube",
    {
        "view_sds_t_adm_t_sksube": {
            "c_opr": "variable_code",
            "v_behdage": "value",
            "patientid": "patientid",
            "d_hendto": "timestamp",
        }
    },
    cohort=lyfo_cohort,
)

sks_referals = pd.concat(
    [sks_opr_not_at_the_hospital, sks_ube_not_at_the_hospital]
).reset_index(drop=True)
sks_referals["data_source"] = "sks_referals"
sks_at_the_hospital = pd.concat(
    [sks_opr_at_the_hospital, sks_ube_at_the_hospital]
).reset_index(drop=True)
sks_at_the_hospital["data_source"] = "sks_at_the_hospital"
