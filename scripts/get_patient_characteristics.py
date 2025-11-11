from helpers.sql_helper import *
from helpers.processing_helper import *
from datetime import timedelta

import math
import pickle as pkl

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")


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
        filtered[(filtered["LDH_diagnosis"] <= 205) & (filtered["age_diagnosis"] < 70)]
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
        filtered[(filtered["LDH_diagnosis"] <= 255) & (filtered["age_diagnosis"] >= 70)]
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

# remove temporary IDs 
WIDE_DATA = WIDE_DATA[WIDE_DATA["age_diagnosis"].notna()].reset_index(drop = True)


test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]

filtered = WIDE_DATA[
    (WIDE_DATA["subtype"] == "DLBCL")
    & ~(WIDE_DATA["regime_1_chemo_type_1st_line"].isin(included_treatments))
    & ~(WIDE_DATA["patientid"].isin(test_patientids))
]

filtered = WIDE_DATA[
    (WIDE_DATA["subtype"] != "DLBCL") & (WIDE_DATA["patientid"].isin(test_patientids))
]


filtered = WIDE_DATA[
    (WIDE_DATA["subtype"] != "DLBCL") & (~WIDE_DATA["patientid"].isin(test_patientids))
]


filtered = WIDE_DATA[~WIDE_DATA["patientid"].isin(test_patientids)]

filtered = filtered[filtered["subtype"] == "DLBCL"]

not_percent, percent = get_table_from_data(filtered)


len(filtered)

# they are under 70

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
        "males": males,
        # AA
        "aa_stage": aa_stage,
        # PS
        "ps": ps,
        # b symptoms
        "b_symptoms": b_symptoms,
        # LDH
        "ldh_uln": ldh_uln,
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
