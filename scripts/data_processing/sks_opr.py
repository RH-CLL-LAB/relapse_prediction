from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort, WIDE_DATA

# NOTE: merge the new stuff and create things
sds_codes = load_data_from_table("SDS_koder")
sds_codes = sds_codes[
    sds_codes["kode_tekst"] != "Ikke klassificeret i perioden"
].reset_index(drop=True)
sds_codes = sds_codes[
    [
        "kode",
        "kode_tekst",
        "niveau1_tekst",
        "niveau2_tekst",
        "niveau3_tekst",
        "niveau4_tekst",
        "niveau5_tekst",
        "niveau6_tekst",
        "niveau7_tekst",
        "niveau8_tekst",
        "niveau9_tekst",
    ]
]

sds_codes["count"] = pd.isnull(sds_codes).sum(1)
sds_codes = (
    sds_codes.sort_values(["count"])
    .drop_duplicates(subset=["kode"], keep="first")
    .drop("count", 1)
)

# SDS_procedurer
# SDS_procedurer_andre
SDS_procedurer_kirurgi = load_data_from_table(
    "SDS_procedurer_kirurgi",
    subset_columns=[
        "dw_ek_forloeb",
        "dw_ek_kontakt",
        "procedurekode",
        "procedurekode_parent",
        "tidspunkt_start",
    ],
)
SDS_procedurer_andre = load_data_from_table(
    "SDS_procedurer_andre",
    subset_columns=[
        "dw_ek_forloeb",
        "dw_ek_kontakt",
        "procedurekode",
        "procedurekode_parent",
        "tidspunkt_start",
    ],
)

SDS_procedurer_kirurgi = SDS_procedurer_kirurgi.drop_duplicates().reset_index(drop=True)
SDS_procedurer_andre = SDS_procedurer_andre.drop_duplicates().reset_index(drop=True)

SDS_procedurer_andre = SDS_procedurer_andre[
    (SDS_procedurer_andre["dw_ek_forloeb"].notna())
    | (SDS_procedurer_andre["dw_ek_kontakt"].notna())
].reset_index(drop=True)
SDS_procedurer_kirurgi = SDS_procedurer_kirurgi[
    (SDS_procedurer_kirurgi["dw_ek_forloeb"].notna())
    | (SDS_procedurer_kirurgi["dw_ek_kontakt"].notna())
].reset_index(drop=True)

# fix duplicates from SDS

SDS_procedurer_kirurgi = (
    SDS_procedurer_kirurgi[SDS_procedurer_kirurgi["procedurekode_parent"].notna()]
    .reset_index(drop=True)
    .drop(columns=["procedurekode_parent", "tidspunkt_start"])
)

SDS_procedurer_andre = (
    SDS_procedurer_andre[SDS_procedurer_andre["procedurekode_parent"].notna()]
    .reset_index(drop=True)
    .drop(columns=["procedurekode_parent", "tidspunkt_start"])
)

# remove weird row containing column names

SDS_procedurer_kirurgi = SDS_procedurer_kirurgi[
    SDS_procedurer_kirurgi["dw_ek_forloeb"] != "dw_ek_forloeb"
].reset_index(drop=True)

SDS_procedurer_andre = SDS_procedurer_andre[
    SDS_procedurer_andre["dw_ek_forloeb"] != "dw_ek_forloeb"
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
    [
        sks_opr_not_at_the_hospital,
        sks_ube_not_at_the_hospital,
        SDS_procedures_surgery,
        SDS_procedures_other,
    ]
).reset_index(drop=True)
sks_referals["data_source"] = "sks_referals"

sks_referals.loc[sks_referals["value"].isna(), "value"] = 1

sks_referals = sks_referals.merge(sds_codes, left_on="variable_code", right_on="kode")
sks_referals = sks_referals[
    ~sks_referals["variable_code"].str.startswith("V")
].reset_index(drop=True)

# remove possible data leakage features
sks_referals = sks_referals[
    ~sks_referals["variable_code"].str.contains("CD20")
].reset_index(drop=True)

sks_referals = sks_referals[
    ~sks_referals["variable_code"].str.contains("CHOP")
].reset_index(drop=True)


sks_referals = sks_referals.melt(
    id_vars=["patientid", "value", "timestamp", "variable_code", "data_source", "kode"],
    value_name="variable_names",
)

sks_referals = sks_referals[sks_referals["variable_names"] != ""].reset_index(drop=True)

sks_referals_unique = sks_referals.merge(
    WIDE_DATA[["patientid", "date_treatment_1st_line"]]
)

sks_referals_unique = sks_referals_unique[
    sks_referals_unique["timestamp"] < sks_referals_unique["date_treatment_1st_line"]
].reset_index(drop=True)

sks_referals_unique = (
    sks_referals_unique.groupby(["patientid", "variable", "variable_names"])
    .agg(timestamp=("timestamp", "max"))
    .reset_index()
)

sks_referals_unique = sks_referals_unique.rename(
    columns={"variable": "variable_code", "variable_names": "value"}
).reset_index(drop=True)

sks_referals_unique["value"] = 1
sks_referals_unique["data_source"] = "sks_referals_unique"

sks_referals = sks_referals[
    ["patientid", "value", "timestamp", "variable_names", "data_source"]
].rename(columns={"variable_names": "variable_code"})

sks_at_the_hospital = pd.concat(
    [sks_opr_at_the_hospital, sks_ube_at_the_hospital]
).reset_index(drop=True)
sks_at_the_hospital["data_source"] = "sks_at_the_hospital"
sks_at_the_hospital.loc[sks_at_the_hospital["value"].isna(), "value"] = 1
# sks_at_the_hospital = sks_at_the_hospital.drop_duplicates().reset_index(drop=True)

sks_at_the_hospital = sks_at_the_hospital.merge(
    sds_codes, left_on="variable_code", right_on="kode"
)
# get all value codes out (what are these??)
sks_at_the_hospital = sks_at_the_hospital[
    ~sks_at_the_hospital["variable_code"].str.startswith("V")
].reset_index(drop=True)
# lets make sense of it

sks_at_the_hospital = sks_at_the_hospital[
    ~sks_at_the_hospital["variable_code"].str.contains("CD20")
].reset_index(drop=True)

sks_at_the_hospital = sks_at_the_hospital[
    ~sks_at_the_hospital["variable_code"].str.contains("CHOP")
].reset_index(drop=True)


sks_at_the_hospital = sks_at_the_hospital.melt(
    id_vars=["patientid", "value", "timestamp", "variable_code", "data_source", "kode"],
    value_name="variable_names",
)

sks_at_the_hospital = sks_at_the_hospital[
    sks_at_the_hospital["variable_names"] != ""
].reset_index(drop=True)

sks_at_the_hospital_unique = sks_at_the_hospital.merge(
    WIDE_DATA[["patientid", "date_treatment_1st_line"]]
)

sks_at_the_hospital_unique = sks_at_the_hospital_unique[
    sks_at_the_hospital_unique["timestamp"]
    < sks_at_the_hospital_unique["date_treatment_1st_line"]
].reset_index(drop=True)

sks_at_the_hospital_unique = (
    sks_at_the_hospital_unique.groupby(["patientid", "variable", "variable_names"])
    .agg(timestamp=("timestamp", "max"))
    .reset_index()
)

sks_at_the_hospital_unique = sks_at_the_hospital_unique.rename(
    columns={"variable": "variable_code", "variable_names": "value"}
).reset_index(drop=True)

sks_at_the_hospital_unique["value"] = 1
sks_at_the_hospital_unique["data_source"] = "sks_at_the_hospital_unique"

sks_at_the_hospital = sks_at_the_hospital[
    ["patientid", "value", "timestamp", "variable_names", "data_source"]
].rename(columns={"variable_names": "variable_code"})
