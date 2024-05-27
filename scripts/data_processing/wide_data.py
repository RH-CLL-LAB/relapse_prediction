from helpers.sql_helper import *
from helpers.preprocessing_helper import *

rkkp_df = pd.read_csv("/ngc/projects2/dalyca_r/clean_r/RKKP_LYFO_CLEAN.csv")

dlbcl_rkkp_df = rkkp_df[rkkp_df["date_treatment_1st_line"].notna()].reset_index(
    drop=True
)
dlbcl_rkkp_df["date_treatment_1st_line"] = pd.to_datetime(
    dlbcl_rkkp_df["date_treatment_1st_line"]
)
dlbcl_rkkp_df["date_diagnosis"] = pd.to_datetime(dlbcl_rkkp_df["date_diagnosis"])

dlbcl_rkkp_df["days_between_diagnosis_and_treatment"] = (
    dlbcl_rkkp_df["date_treatment_1st_line"] - dlbcl_rkkp_df["date_diagnosis"]
).dt.days

# dlbcl_rkkp_df = dlbcl_rkkp_df[dlbcl_rkkp_df["subtype"] == "DLBCL"].reset_index(drop=True)

lyfo_extra = pd.read_csv(
    "/ngc/projects2/dalyca_r/mikwer_r/RKKP_LYFO_EXTRA_RELAPS_psAnon.csv"
)

# merge extra information
WIDE_DATA = dlbcl_rkkp_df.merge(lyfo_extra).reset_index(drop=True)


WIDE_DATA["relaps_pato_dt"] = pd.to_datetime(WIDE_DATA["relaps_pato_dt"])

WIDE_DATA["date_diagnosis"] = pd.to_datetime(WIDE_DATA["date_diagnosis"])
WIDE_DATA["date_death"] = pd.to_datetime(WIDE_DATA["date_death"])

# relapse label - is this true?

WIDE_DATA["relapse_label"] = WIDE_DATA["LYFO_15_002=relapsskema"]

WIDE_DATA["relapse_date"] = WIDE_DATA["date_relapse_confirmed_2nd_line"]

WIDE_DATA.loc[WIDE_DATA["relaps_pato_dt"].notna(), "relapse_label"] = 1

WIDE_DATA.loc[WIDE_DATA["relaps_pato_dt"].notna(), "relapse_date"] = WIDE_DATA[
    WIDE_DATA["relaps_pato_dt"].notna()
]["relaps_pato_dt"]

WIDE_DATA["days_to_death"] = (
    WIDE_DATA["date_death"] - WIDE_DATA["date_treatment_1st_line"]
).dt.days

WIDE_DATA["relapse_date"] = pd.to_datetime(WIDE_DATA["relapse_date"])

WIDE_DATA["days_to_relapse"] = (
    WIDE_DATA["relapse_date"] - WIDE_DATA["date_treatment_1st_line"]
).dt.days

WIDE_DATA.loc[WIDE_DATA["days_to_death"] < 730, "relapse_label"] = 1
WIDE_DATA.loc[
    (WIDE_DATA["days_to_death"] < 730) & (WIDE_DATA["relapse_date"].isna()),
    "relapse_date",
] = WIDE_DATA[(WIDE_DATA["date_death"].notna()) & ((WIDE_DATA["relapse_date"].isna()))][
    "date_death"
]
WIDE_DATA.loc[
    (WIDE_DATA["days_to_death"] < 730)
    & ((WIDE_DATA["days_to_relapse"] > 730) | (WIDE_DATA["days_to_relapse"].isna())),
    "proxy_death",
] = 1

# remove uncertain patients from the cohort
WIDE_DATA = WIDE_DATA[WIDE_DATA["relapse_label"] != 0].reset_index(drop=True)

WIDE_DATA = WIDE_DATA.fillna(pd.NA)

lyfo_cohort = get_cohort_string_from_data(WIDE_DATA)
lyfo_cohort_strings = lyfo_cohort
lyfo_cohort_strings = lyfo_cohort_strings.replace("(", "('")
lyfo_cohort_strings = lyfo_cohort_strings.replace(")", "')")
lyfo_cohort_strings = lyfo_cohort_strings.replace(", ", "', '")


death = load_data_from_table("SDS_t_dodsaarsag_2", cohort=lyfo_cohort)
WIDE_DATA = WIDE_DATA.merge(death[["c_dodsmaade", "patientid"]], how="left")

WIDE_DATA.loc[WIDE_DATA["c_dodsmaade"] < 3, "proxy_death"] = pd.NA

WIDE_DATA = WIDE_DATA[[x for x in WIDE_DATA.columns if x != "c_dodsmaade"]]


WIDE_DATA.loc[:, "year_treat"] = WIDE_DATA["date_treatment_1st_line"].dt.year

# hmm strange - there's 680 patients that have no age or sex in RKKP

# NOTE: Some patients are not in
# patient table ... right now, we're just ignoring that fact.
# That also means that some patients will not have age at diagnosis
# or age at treatment. Some of these can probably be found in
# RKKP, but it's a very small number

# load_data_from_table("patient", limit = 10)

# load patientdata to fill in the gaps
patient = load_data_from_table(
    "patient", subset_columns=["patientid", "sex", "date_birth"]
)

patient["sex_from_patient_table"] = (
    patient["sex"].str.replace("F", "Female").str.replace("M", "Male")
)

patient = patient[["patientid", "sex_from_patient_table", "date_birth"]]

patient["date_birth"] = pd.to_datetime(patient["date_birth"])

WIDE_DATA = WIDE_DATA.merge(patient, on="patientid", how="left")

WIDE_DATA.loc[WIDE_DATA["sex"].isna(), "sex"] = WIDE_DATA[WIDE_DATA["sex"].isna()][
    "sex_from_patient_table"
]

WIDE_DATA["age_at_diagnosis"] = round(
    (WIDE_DATA["date_diagnosis"] - WIDE_DATA["date_birth"]).dt.days / 365.5
)

# could be that this should be days from diagnosis to treatment

WIDE_DATA["days_from_diagnosis_to_tx"] = (
    WIDE_DATA["date_treatment_1st_line"] - WIDE_DATA["date_diagnosis"]
).dt.days

WIDE_DATA["age_at_tx"] = round(
    (WIDE_DATA["date_treatment_1st_line"] - WIDE_DATA["date_birth"]).dt.days / 365.5
)

WIDE_DATA = WIDE_DATA[
    [x for x in WIDE_DATA.columns if x not in ["sex_from_patient_table", "date_birth"]]
]
