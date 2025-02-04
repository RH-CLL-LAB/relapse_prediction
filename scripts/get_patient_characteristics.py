from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from datetime import timedelta

rkkp_df = pd.read_csv("/ngc/projects2/dalyca_r/clean_r/RKKP_LYFO_CLEAN.csv")
# check in with Carsten or Peter Brown

included_treatments = ["chop", "choep", "maxichop"]  # "cop", "minichop"

dlbcl_rkkp_df = rkkp_df[rkkp_df["date_treatment_1st_line"].notna()].reset_index(
    drop=True
)


# # included treatments

# dlbcl_rkkp_df = dlbcl_rkkp_df[
#     dlbcl_rkkp_df["regime_1_chemo_type_1st_line"].isin(included_treatments)
# ].reset_index(drop=True)

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
WIDE_DATA = dlbcl_rkkp_df.merge(lyfo_extra, how="left").reset_index(drop=True)

WIDE_DATA["relaps_pato_dt"] = pd.to_datetime(WIDE_DATA["relaps_pato_dt"])
WIDE_DATA["relaps_lpr_dt"] = pd.to_datetime(WIDE_DATA["relaps_lpr_dt"])

WIDE_DATA["date_diagnosis"] = pd.to_datetime(WIDE_DATA["date_diagnosis"])
WIDE_DATA["date_death"] = pd.to_datetime(WIDE_DATA["date_death"])


# last_death = WIDE_DATA["date_death"].max()
# last_treatment = last_death - timedelta(days=730)
# # keep everyone instead, and just annotate who is probably not
# # with full followup
# # people can relapse within 2 years where we actually
# # have a full followup

# WIDE_DATA = WIDE_DATA[
#     WIDE_DATA["date_treatment_1st_line"] < last_treatment
# ].reset_index(drop=True)

# all patients up until 2022 have a full 2 year followup

WIDE_DATA = WIDE_DATA[
    WIDE_DATA["date_treatment_1st_line"] < pd.to_datetime("2022-01-01")
].reset_index(drop=True)

# relapse label - is this true?
# PETER BROWN LABELS

WIDE_DATA["relapse_date"] = WIDE_DATA["date_relapse_confirmed_2nd_line"]

WIDE_DATA.loc[WIDE_DATA["relapse_date"].notna(), "relapse_label"] = 1

WIDE_DATA["relapse_label"] = WIDE_DATA["LYFO_15_002=relapsskema"]

WIDE_DATA.loc[WIDE_DATA["relaps_pato_dt"].notna(), "relapse_label"] = 1

WIDE_DATA.loc[WIDE_DATA["relaps_pato_dt"].notna(), "relapse_date"] = WIDE_DATA[
    WIDE_DATA["relaps_pato_dt"].notna()
]["relaps_pato_dt"]

# WIDE_DATA.loc[WIDE_DATA["relaps_lpr_dt"].notna(), "relapse_label"] = 1

# WIDE_DATA.loc[WIDE_DATA["relapse_date"].isna(), "relapse_date"] = WIDE_DATA[
#     WIDE_DATA["relapse_date"].isna()
# ]["relaps_lpr_dt"]

WIDE_DATA.loc[WIDE_DATA["relapse_date"].isna(), "relapse_date"] = WIDE_DATA[
    WIDE_DATA["relapse_date"].isna()
]["date_treatment_2nd_line"]

WIDE_DATA.loc[WIDE_DATA["relapse_date"].notna(), "relapse_label"] = 1

WIDE_DATA["days_to_death"] = (
    WIDE_DATA["date_death"] - WIDE_DATA["date_treatment_1st_line"]
).dt.days

WIDE_DATA["relapse_date"] = pd.to_datetime(WIDE_DATA["relapse_date"])

WIDE_DATA["days_to_relapse"] = (
    WIDE_DATA["relapse_date"] - WIDE_DATA["date_treatment_1st_line"]
).dt.days

WIDE_DATA.loc[WIDE_DATA["date_death"].notna(), "dead_label"] = 1

# remove uncertain patients from the cohort
# make uncertain patients be relapses
WIDE_DATA = WIDE_DATA[WIDE_DATA["relapse_label"] != 0].reset_index(drop=True)

WIDE_DATA = WIDE_DATA[
    (WIDE_DATA["days_to_death"] >= 0) | (WIDE_DATA["days_to_death"].isna())
]
WIDE_DATA = WIDE_DATA[
    (WIDE_DATA["days_to_relapse"] >= 0) | (WIDE_DATA["days_to_relapse"].isna())
]

# WIDE_DATA.loc[WIDE_DATA["relapse_label"] != 0, "relapse_label"] = 1

WIDE_DATA = WIDE_DATA.fillna(pd.NA)

lyfo_cohort = get_cohort_string_from_data(WIDE_DATA)
lyfo_cohort_strings = lyfo_cohort
lyfo_cohort_strings = lyfo_cohort_strings.replace("(", "('")
lyfo_cohort_strings = lyfo_cohort_strings.replace(")", "')")
lyfo_cohort_strings = lyfo_cohort_strings.replace(", ", "', '")


death = load_data_from_table("SDS_t_dodsaarsag_2", cohort=lyfo_cohort)

WIDE_DATA = WIDE_DATA.merge(death[["c_dodsmaade", "patientid"]], how="left")

WIDE_DATA.loc[WIDE_DATA["c_dodsmaade"] < 3, "dead_label"] = pd.NA
# probably also should change the date to NA so that we don't get into weird timeseriesflattener problems

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

WIDE_DATA.loc[WIDE_DATA["age_diagnosis"].isna(), "age_diagnosis"] = round(
    (
        WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["date_diagnosis"]
        - WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["date_birth"]
    ).dt.days
    / 365.5
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
import math
import pickle as pkl

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")


def calculate_NCCN_IPI(age, ldh, aa_stage, extranodal, ps):
    # NOTE NOW RETURNING NANS FOR PATIENTS WITH MISSING VALUES

    if any(
        [
            math.isnan(age),
            math.isnan(ldh),
            math.isnan(aa_stage),
            # math.isnan(extranodal),
            math.isnan(ps),
        ]
    ):
        return pd.NA
    total_score = 0
    if age > 75:
        total_score += 3
    elif age > 60:
        total_score += 2
    elif age > 40:
        total_score += 1

    if age < 70:
        upper_limit = 205
    else:
        upper_limit = 255

    ldh_normalized = ldh / upper_limit

    if ldh_normalized > 3:
        total_score += 2
    elif ldh_normalized > 1:
        total_score += 1

    if aa_stage > 2:
        total_score += 1
    if extranodal == 1:
        total_score += 1
    if ps >= 2:
        total_score += 1

    return total_score


def calculate_CNS_IPI(age, ldh, aa_stage, extranodal, ps, kidneys_diagnosis):
    # NOTE NOW RETURNING NANS FOR PATIENTS WITH MISSING VALUES

    if any(
        [
            math.isnan(age),
            math.isnan(ldh),
            math.isnan(aa_stage),
            # math.isnan(extranodal),
            math.isnan(ps),
            math.isnan(kidneys_diagnosis),
        ]
    ):
        return pd.NA
    total_score = 0
    if age > 60:
        total_score += 1
    if age < 70:
        upper_limit = 205
    else:
        upper_limit = 255

    if ldh > upper_limit:
        total_score += 1

    if ps > 1:
        total_score += 1

    if aa_stage > 2:
        total_score += 1
    if extranodal == 1:
        total_score += 1
    if kidneys_diagnosis:
        total_score += 1

    return total_score


WIDE_DATA["CNS_IPI_diagnosis"] = WIDE_DATA.apply(
    lambda x: calculate_CNS_IPI(
        x["age_diagnosis"],
        x["LDH_diagnosis"],  # needs to be normalized
        x["AA_stage_diagnosis"],
        x["extranodal_disease_diagnosis"],
        x["PS_diagnosis"],
        x["kidneys_diagnosis"],
    ),
    axis=1,
)

WIDE_DATA["NCCN_IPI_diagnosis"] = WIDE_DATA.apply(
    lambda x: calculate_NCCN_IPI(
        x["age_diagnosis"],
        x["LDH_diagnosis"],  # needs to be normalized
        x["AA_stage_diagnosis"],
        x["extranodal_disease_diagnosis"],
        x["PS_diagnosis"],
    ),
    axis=1,
)


def get_table_from_data(filtered):
    age_at_treatment_under_40 = len(filtered[filtered["age_at_tx"] <= 40])
    age_at_treatment_under_60 = len(
        filtered[(filtered["age_at_tx"] >= 41) & (filtered["age_at_tx"] <= 60)]
    )
    age_at_treatment_under_75 = len(
        filtered[(filtered["age_at_tx"] >= 61) & (filtered["age_at_tx"] <= 75)]
    )
    age_at_treatment_over_75 = len(filtered[filtered["age_at_tx"] > 75])

    # gender
    males = filtered["sex"].value_counts()

    # AA
    aa_stage = filtered["AA_stage_diagnosis"].value_counts()

    # PS
    ps = filtered["PS_diagnosis"].value_counts()

    # b symptoms
    b_symptoms = filtered["b_symptoms_diagnosis"].value_counts()

    # LDH

    ldh_uln = len(
        filtered[(filtered["LDH_diagnosis"] < 205) & (filtered["age_diagnosis"] < 70)]
    )

    ldh_under_three_uln = len(
        filtered[
            (filtered["LDH_diagnosis"] > 205)
            & (filtered["LDH_diagnosis"] < 205 * 3)
            & (filtered["age_diagnosis"] < 70)
        ]
    )

    ldh_over_three_uln = len(
        filtered[
            (filtered["LDH_diagnosis"] >= 205 * 3) & (filtered["age_diagnosis"] < 70)
        ]
    )

    ldh_uln += len(
        filtered[(filtered["LDH_diagnosis"] < 255) & (filtered["age_diagnosis"] >= 70)]
    )

    ldh_under_three_uln += len(
        filtered[
            (filtered["LDH_diagnosis"] > 255)
            & (filtered["LDH_diagnosis"] < 255 * 3)
            & (filtered["age_diagnosis"] >= 70)
        ]
    )

    ldh_over_three_uln += len(
        filtered[
            (filtered["LDH_diagnosis"] >= 255 * 3) & (filtered["age_diagnosis"] >= 70)
        ]
    )

    alc = len(filtered[filtered["ALC_diagnosis"] <= 0.84])

    alb = len(filtered[filtered["ALB_diagnosis"] <= 35])

    hgb = len(filtered[filtered["HB_diagnosis"] < 6.2])

    bulky = len(filtered[filtered["tumor_diameter_diagnosis"] >= 10])

    ipi = filtered["IPI_score_diagnosis"].value_counts()

    nccn_ipi = filtered["NCCN_IPI_diagnosis"].value_counts()

    cns_ipi = filtered["CNS_IPI_diagnosis"].value_counts()

    not_percent = {
        "age_at_treatment_under_40": age_at_treatment_under_40,
        "age_at_treatment_under_60": age_at_treatment_under_60,
        "age_at_treatment_under_75": age_at_treatment_under_75,
        "age_at_treatment_over_75": age_at_treatment_over_75,
        # gender
        "males": males,
        # AA
        "aa_stage": aa_stage,
        # PS
        "ps": ps,
        # b symptoms
        "b_symptoms": b_symptoms,
        # LDH
        "ldh_uln": ldh_uln,
        "ldh_under_three_uln": ldh_under_three_uln,
        "ldh_over_three_uln": ldh_over_three_uln,
        "alc": alc,
        "alb": alb,
        "hgb": hgb,
        "bulky": bulky,
        "ipi": ipi,
        "nccn_ipi": nccn_ipi,
        "cns_ipi": cns_ipi,
    }

    percent = {i: round((e / len(filtered)) * 100, 2) for i, e in not_percent.items()}

    return not_percent, percent


test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]

filtered = WIDE_DATA[
    (WIDE_DATA["subtype"] == "DLBCL")
    & (WIDE_DATA["regime_1_chemo_type_1st_line"].isin(included_treatments))
    & (WIDE_DATA["patientid"].isin(test_patientids))
]
filtered

filtered = WIDE_DATA[
    (WIDE_DATA["subtype"] != "DLBCL") & (WIDE_DATA["patientid"].isin(test_patientids))
]


filtered = WIDE_DATA[~WIDE_DATA["patientid"].isin(test_patientids)]

filtered["subtype"].value_counts()

filtered = filtered[filtered["subtype"] == "DLBCL"]

filtered[filtered["regime_1_chemo_type_1st_line"].isin(included_treatments)]

[x for x in WIDE_DATA.columns]

not_percent, percent = get_table_from_data(filtered)

not_percent
not_percent["alc"]
not_percent["alb"]
not_percent["hgb"]
not_percent["bulky"]
not_percent["ipi"]
not_percent["nccn_ipi"]
not_percent["cns_ipi"]

percent

percent["alc"]
percent["alb"]
percent["hgb"]
percent["bulky"]
percent["ipi"]
percent["nccn_ipi"]
percent["cns_ipi"]


def get_table_from_data_na(filtered):
    age_at_treatment_under_40 = len(filtered[filtered["age_at_tx"].isna()])

    # gender
    males = filtered["sex"].isna().sum()

    # AA
    aa_stage = filtered["AA_stage_diagnosis"].isna().sum()

    # PS
    ps = filtered["PS_diagnosis"].isna().sum()

    # b symptoms
    b_symptoms = filtered["b_symptoms_diagnosis"].isna().sum()

    # LDH

    ldh_uln = len(filtered[filtered["LDH_diagnosis"].isna()])

    alc = len(filtered[filtered["ALC_diagnosis"].isna()])

    alb = len(filtered[filtered["ALB_diagnosis"].isna()])

    hgb = len(filtered[filtered["HB_diagnosis"].isna()])

    bulky = len(filtered[filtered["tumor_diameter_diagnosis"].isna()])

    ipi = filtered["IPI_score_diagnosis"].isna().sum()

    nccn_ipi = filtered["NCCN_IPI_diagnosis"].isna().sum()

    cns_ipi = filtered["CNS_IPI_diagnosis"].isna().sum()

    not_percent = {
        "age_at_treatment_under_40": age_at_treatment_under_40,
        # "age_at_treatment_under_60": age_at_treatment_under_60,
        # "age_at_treatment_under_75": age_at_treatment_under_75,
        # "age_at_treatment_over_75": age_at_treatment_over_75,
        # gender
        "males": males,
        # AA
        "aa_stage": aa_stage,
        # PS
        "ps": ps,
        # b symptoms
        "b_symptoms": b_symptoms,
        # LDH
        "ldh_uln": ldh_uln,
        # "ldh_under_three_uln": ldh_under_three_uln,
        # "ldh_over_three_uln": ldh_over_three_uln,
        "alc": alc,
        "alb": alb,
        "hgb": hgb,
        "bulky": bulky,
        "ipi": ipi,
        "nccn_ipi": nccn_ipi,
        "cns_ipi": cns_ipi,
    }

    percent = {i: round((e / len(filtered)) * 100, 2) for i, e in not_percent.items()}

    return not_percent, percent


get_table_from_data_na(filtered)
