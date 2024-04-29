from helpers.sql_helper import *
from helpers.preprocessing_helper import *

rkkp_df = pd.read_csv("/ngc/projects2/dalyca_r/clean_r/RKKP_LYFO_CLEAN.csv")

dlbcl_rkkp_df = rkkp_df[rkkp_df["date_treatment_1st_line"].notna()].reset_index(
    drop=True
)
dlbcl_rkkp_df["date_treatment_1st_line"] = pd.to_datetime(
    dlbcl_rkkp_df["date_treatment_1st_line"]
)

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

lyfo_cohort = get_cohort_string_from_data(WIDE_DATA)
lyfo_cohort_strings = lyfo_cohort
lyfo_cohort_strings = lyfo_cohort_strings.replace("(", "('")
lyfo_cohort_strings = lyfo_cohort_strings.replace(")", "')")
lyfo_cohort_strings = lyfo_cohort_strings.replace(", ", "', '")
