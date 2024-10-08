from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort

# NOTE: merge the new stuff and create things

# SDS_procedurer
# SDS_procedurer_andre
SDS_procedurer_kirurgi = load_data_from_table(
    "SDS_procedurer_kirurgi",
    subset_columns=["dw_ek_forloeb", "dw_ek_kontakt", "procedurekode"],
)
SDS_procedurer_andre = load_data_from_table(
    "SDS_procedurer_andre",
    subset_columns=["dw_ek_forloeb", "dw_ek_kontakt", "procedurekode"],
)
SDS_procedurer_andre = SDS_procedurer_andre[
    (SDS_procedurer_andre["dw_ek_forloeb"].notna())
    | (SDS_procedurer_andre["dw_ek_kontakt"].notna())
].reset_index(drop=True)
SDS_procedurer_kirurgi = SDS_procedurer_kirurgi[
    (SDS_procedurer_kirurgi["dw_ek_forloeb"].notna())
    | (SDS_procedurer_kirurgi["dw_ek_kontakt"].notna())
].reset_index(drop=True)

SDS_kontakter = load_data_from_table(
    "SDS_kontakter",
    subset_columns=[
        "dw_ek_kontakt",
        "dw_ek_forloeb",
        "patientid",
        "dato_start",
        "dato_slut",
    ],
    cohort=lyfo_cohort,
)

SDS_forloeb = SDS_kontakter[["dw_ek_forloeb", "patientid", "dato_start", "dato_slut"]]

SDS_contacts = SDS_kontakter[["dw_ek_kontakt", "patientid", "dato_start", "dato_slut"]]

SDS_procedurer_kirurgi["dw_ek_kontakt"] = pd.to_numeric(
    SDS_procedurer_kirurgi["dw_ek_kontakt"], errors="coerce"
)
SDS_procedurer_kirurgi["dw_ek_forloeb"] = pd.to_numeric(
    SDS_procedurer_kirurgi["dw_ek_forloeb"], errors="coerce"
)
SDS_procedures_surgery = pd.concat(
    [
        SDS_forloeb.merge(SDS_procedurer_kirurgi),
        SDS_contacts.merge(SDS_procedurer_kirurgi),
    ]
).reset_index(drop=True)[["patientid", "dato_start", "dato_slut", "procedurekode"]]

SDS_procedurer_andre["dw_ek_kontakt"] = pd.to_numeric(
    SDS_procedurer_andre["dw_ek_kontakt"], errors="coerce"
)
SDS_procedurer_andre["dw_ek_forloeb"] = pd.to_numeric(
    SDS_procedurer_andre["dw_ek_forloeb"], errors="coerce"
)
SDS_procedures_other = pd.concat(
    [SDS_forloeb.merge(SDS_procedurer_andre), SDS_contacts.merge(SDS_procedurer_andre)]
).reset_index(drop=True)[["patientid", "dato_start", "dato_slut", "procedurekode"]]

SDS_procedures_other["value"] = (
    SDS_procedures_other["dato_slut"] - SDS_procedures_other["dato_start"]
).dt.days + 1
SDS_procedures_surgery["value"] = (
    SDS_procedures_surgery["dato_slut"] - SDS_procedures_surgery["dato_start"]
).dt.days + 1

SDS_procedures_other = (
    SDS_procedures_other[["patientid", "dato_start", "procedurekode", "value"]]
    .rename(columns={"dato_start": "timestamp", "procedurekode": "variable_code"})
    .reset_index(drop=True)
)
SDS_procedures_surgery = (
    SDS_procedures_surgery[["patientid", "dato_start", "procedurekode", "value"]]
    .rename(columns={"dato_start": "timestamp", "procedurekode": "variable_code"})
    .reset_index(drop=True)
)

SDS_procedures_other = SDS_procedures_other.drop_duplicates().reset_index(drop=True)

SDS_procedures_surgery = SDS_procedures_surgery.drop_duplicates().reset_index(drop=True)

# SDS_kontakter = load_data_from_table("SDS_forloeb", cohort=lyfo_cohort)


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

# this has to be fixed - need to exclude NAs for behdage I guess

sks_referals = pd.concat(
    [
        sks_opr_not_at_the_hospital,
        sks_ube_not_at_the_hospital,
        SDS_procedures_surgery,
        SDS_procedures_other,
    ]
).reset_index(drop=True)
sks_referals["data_source"] = "sks_referals"

sks_referals.loc[sks_referals["value"].isna(), "value"] = 1

sks_at_the_hospital = pd.concat(
    [sks_opr_at_the_hospital, sks_ube_at_the_hospital]
).reset_index(drop=True)
sks_at_the_hospital["data_source"] = "sks_at_the_hospital"
sks_at_the_hospital.loc[sks_at_the_hospital["value"].isna(), "value"] = 1
