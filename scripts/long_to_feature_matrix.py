import pandas as pd
import numpy as np
import datetime as dt
from timeseriesflattener.aggregators import (
    MaxAggregator,
    MinAggregator,
)

from feature_specification import feature_specs
from feature_maker.scripts.feature_maker import FeatureMaker
import datetime
from data_processing.wide_data import WIDE_DATA

test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]

CACHED_DATA = True
INCLUDE_PROXIES = False
SINGLE_DISEASE = False

if CACHED_DATA:
    WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
    LONG_DATA = pd.read_pickle("data/LONG_DATA.pkl")
else:
    from preprocess_data import WIDE_DATA, LONG_DATA

if SINGLE_DISEASE:
    WIDE_DATA = WIDE_DATA[WIDE_DATA["subtype"] == "DLBCL"].reset_index(drop=True)


non_date_columns = [x for x in WIDE_DATA.columns if "date" not in x]

WIDE_DATA[non_date_columns] = WIDE_DATA[non_date_columns].fillna(-1)


def calculate_nodality_of_disease(extranodal, nodal):
    if extranodal == 1 and nodal == 1:
        return 2
    elif nodal == 1:
        return 1
    else:
        return -1


WIDE_DATA["nodality_disease_diagnosis"] = WIDE_DATA.apply(
    lambda x: calculate_nodality_of_disease(
        x["extranodal_disease_diagnosis"], x["nodal_disease_diagnosis"]
    ),
    axis=1,
)

c = [
    c
    for c in WIDE_DATA.columns
    if set(WIDE_DATA[c]) == set([-1, 0, 1, 2]) and "IPI" not in c
]


for column in c:
    WIDE_DATA[column] = WIDE_DATA[column].apply(lambda x: -1 if x == 2 else x)

WIDE_DATA.loc[WIDE_DATA["ALB_diagnosis"] == -1, "ALB_diagnosis"] = (
    WIDE_DATA[WIDE_DATA["ALB_diagnosis"] == -1]["ALB_uM_diagnosis"] * 15.05
)

WIDE_DATA.loc[WIDE_DATA["KREA_diagnosis"] == -1, "KREA_diagnosis"] = (
    WIDE_DATA[WIDE_DATA["KREA_diagnosis"] == -1]["KREA_mM_diagnosis"] * 100
)

WIDE_DATA.loc[WIDE_DATA["B2M_diagnosis"] == -1, "B2M_diagnosis"] = (
    WIDE_DATA[WIDE_DATA["B2M_diagnosis"] == -1]["B2M_nmL_diagnosis"] * 100
)

# should probably be something like this
ca_uncorrected = -1 * (
    0.02
    * (
        39
        - WIDE_DATA[WIDE_DATA["CA_albumin_corrected_diagnosis"] != -1]["ALB_diagnosis"]
    )
    - WIDE_DATA[WIDE_DATA["CA_albumin_corrected_diagnosis"] != -1][
        "CA_albumin_corrected_diagnosis"
    ]
)

LONG_DATA["patientid"] = LONG_DATA["patientid"].astype(int)

LONG_DATA = LONG_DATA[~LONG_DATA["variable_code"].str.contains("CD20")]

LONG_DATA = LONG_DATA[~LONG_DATA["variable_code"].str.contains("CHOP")]

persimune = pd.read_csv("data/persimune.csv")

labmeasurements = pd.read_csv("data/labmeasurements.csv")
labmeasurements_all = pd.read_csv("data/labmeasurements_all.csv")

LONG_DATA = LONG_DATA[
    ~LONG_DATA["data_source"].isin(
        [
            "PERSIMUNE_leukocytes",
            "PERSIMUNE_microbiology_analysis",
            "PERSIMUNE_micriobiology_culture",
            "labmeasurements",
            "labmeasurements_all",
        ]
    )
].reset_index(drop=True)

persimune["timestamp"] = pd.to_datetime(persimune["timestamp"])
labmeasurements["timestamp"] = pd.to_datetime(labmeasurements["timestamp"])
labmeasurements_all["timestamp"] = pd.to_datetime(labmeasurements_all["timestamp"])

LONG_DATA = pd.concat(
    [LONG_DATA, persimune, labmeasurements, labmeasurements_all]
).reset_index(drop=True)

LONG_DATA = LONG_DATA.merge(WIDE_DATA[["patientid", "date_treatment_1st_line"]])


# filtering so we only have data from before treatment AND after 3 year before treatment
LONG_DATA = LONG_DATA[
    (LONG_DATA["timestamp"] <= LONG_DATA["date_treatment_1st_line"])
    & (
        LONG_DATA["timestamp"]
        >= LONG_DATA["date_treatment_1st_line"] - datetime.timedelta(days=365 * 3)
    )
].reset_index(drop=True)


LONG_DATA = LONG_DATA[
    [x for x in LONG_DATA.columns if x != "date_treatment_1st_line"]
].reset_index(drop=True)


feature_maker = FeatureMaker(
    long_data=LONG_DATA,
    wide_data=WIDE_DATA,
    test_patientids=test_patientids,
    feature_processing_order=[
        "labmeasurements",
        "lab_measurements_data_all",
        "pathology_concat",
        "ord_medicine_poly_pharmacy",
        "ord_medicine_poly_pharmacy_since_diagnosis",
        "diagnoses_all_comorbidity",
        "sks_referals",
        "sks_referals_unique",
        "sks_at_the_hospital",
        "sks_at_the_hospital_unique",
        "SDS_pato",
        "blood_tests_all",
        "ordered_medicine",
        "PERSIMUNE_leukocytes",
        "PERSIMUNE_microbiology_analysis",
        "PERSIMUNE_microbiology_culture",
        "diagnoses_all",
    ],
)

feature_maker._reset_all_features()
feature_maker.specify_prediction_time_from_wide_format("date_treatment_1st_line")


# Remove response evaluation - leakage
list_of_exclusion_terms = [
    "date",
    "2nd_line",
    "relapse",
    "OS",
    "LYFO",
    "report_submitted",
    "dead",
    "FU",
    "patientid",
    "1st_line",
    "death",
    "treatment",
    # "age",
    "proxy",
    "dato",
    "_dt",
]

static_predictors = [
    x
    for x in feature_maker.wide_data.columns
    if not any([exclusion_term in x for exclusion_term in list_of_exclusion_terms])
]


for static_predictor in static_predictors:
    static_predictor_specification = {
        "data_source": "RKKP_LYFO",
        "value_column": static_predictor,
        "fallback": -1,
        "feature_base_name": f"RKKP_LYFO_{static_predictor}",
    }
    feature_maker.add_static_feature(static_predictor_specification)

from tqdm import tqdm

for feature_spec in tqdm(feature_specs):
    lookbacks = [dt.timedelta(90), dt.timedelta(365), dt.timedelta(365 * 3)]
    if "90" in feature_spec.get("data_source"):
        lookbacks = [dt.timedelta(90)]
    if "365" in feature_spec.get("data_source"):
        lookbacks = [dt.timedelta(365)]
    if "1095" in feature_spec.get("data_source"):
        lookbacks = [dt.timedelta(365 * 3)]

    feature_maker.add_features_given_ratio(
        data_source=feature_spec.get("data_source"),
        agg_funcs=feature_spec.get("agg_funcs"),
        lookbacks=lookbacks,
        proportion=feature_spec.get("proportion"),
        fallback=-1,
        collapse_rare_conditions_to_feature=True,
    )

translation_dict = {9: 0, 1: 1}  # 0: np.NAN

feature_maker.wide_data["relapse"] = feature_maker.wide_data["relapse_label"].apply(
    lambda x: translation_dict.get(x)
)


def define_succesful_treatment(date_death, date_relapse):
    if pd.isnull(date_death) and pd.isnull(date_relapse):
        succesful_treatment_date = pd.NaT
    elif pd.isnull(date_death) and pd.notnull(date_relapse):
        succesful_treatment_date = date_relapse
    elif pd.notnull(date_death) and pd.isnull(date_relapse):
        succesful_treatment_date = date_death
    elif pd.notnull(date_death) and pd.notnull(date_relapse):
        succesful_treatment_date = min((date_death, date_relapse))
    if pd.isnull(succesful_treatment_date):
        succesful_treatment_label = 0
    else:
        succesful_treatment_label = 1
    return succesful_treatment_date, succesful_treatment_label


(
    feature_maker.wide_data["succesful_treatment_date"],
    feature_maker.wide_data["succesful_treatment_label"],
) = zip(
    *feature_maker.wide_data.apply(
        lambda x: define_succesful_treatment(x["date_death"], x["relapse_date"]), axis=1
    )
)

feature_maker.add_outcome_from_wide_format(
    "date_death",
    "dead_label",
    [dt.timedelta(730), dt.timedelta(365 * 5)],
    [MaxAggregator()],
)

feature_maker.add_outcome_from_wide_format(
    "relapse_date",
    "relapse",
    [dt.timedelta(730), dt.timedelta(365 * 5)],
    [MaxAggregator()],
)

feature_maker.add_outcome_from_wide_format(
    "succesful_treatment_date",
    "succesful_treatment_label",
    [dt.timedelta(730)],
    [MaxAggregator()],
)

if __name__ == "__main__":
    feature_maker.make_all_features()

    feature_matrix = feature_maker.create_feature_matrix(None)

    # sum and counts default missing is 0 despite fallback = -1

    sum_columns = [
        x
        for x in feature_matrix.columns
        if "sum_fallback_-1" in x or "count_fallback_-1" in x
    ]

    for column in sum_columns:
        feature_matrix.loc[feature_matrix[column] == 0, column] = -1

    feature_matrix.loc[
        feature_matrix["outc_dead_label_within_0_to_730_days_max_fallback_0"] == -1,
        "outc_dead_label_within_0_to_730_days_max_fallback_0",
    ] = 0

    feature_matrix = feature_matrix.replace(np.NAN, -1)

    feature_matrix.loc[
        feature_matrix["outc_dead_label_within_0_to_730_days_max_fallback_0"] == -1,
        "outc_dead_label_within_0_to_730_days_max_fallback_0",
    ] = 0

    feature_matrix.loc[
        feature_matrix["outc_dead_label_within_0_to_1825_days_max_fallback_0"] == -1,
        "outc_dead_label_within_0_to_1825_days_max_fallback_0",
    ] = 0

    feature_matrix.columns = feature_matrix.columns.str.replace("<", "less_than")
    feature_matrix.columns = feature_matrix.columns.str.replace(",", "_comma_")
    feature_matrix.columns = feature_matrix.columns.str.replace(">", "more_than")
    feature_matrix.columns = feature_matrix.columns.str.replace("[", "left_bracket")
    feature_matrix.columns = feature_matrix.columns.str.replace("]", "right_bracket")

    feature_matrix["group"] = feature_matrix.apply(
        lambda x: f'relapse_{str(x["outc_relapse_within_0_to_730_days_max_fallback_0"])}_death_{str(x["outc_dead_label_within_0_to_730_days_max_fallback_0"])}_disease_{str(x["pred_RKKP_subtype_fallback_-1"])}',
        axis=1,
    )

    feature_matrix["proportion_of_missing"] = feature_matrix.eq(-1).sum(axis=1) / len(
        feature_matrix.columns
    )

    feature_matrix.to_pickle("results/feature_matrix_all.pkl")
