"""
long_to_feature_matrix.py

Main feature engineering pipeline:
- Loads wide and long data.
- Uses FeatureMaker to construct time-dependent features from multiple data sources.
- Adds static predictors and outcome definitions.
- Creates the final feature matrix and saves to 'results/feature_matrix_all.pkl'.
"""

import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm

from timeseriesflattener.aggregators import MaxAggregator, MinAggregator
from lyfo_treatment_failure_prediction.feature_engineering.feature_specification import (
    feature_specs,
)
from lyfo_treatment_failure_prediction.feature_maker.scripts.feature_maker import (
    FeatureMaker,
)
from lyfo_treatment_failure_prediction.data_processing.wide_data import WIDE_DATA

# ---------------------------------------------------------------------------
# Configuration flags
# ---------------------------------------------------------------------------
CACHED_DATA = True
INCLUDE_PROXIES = False  # not used in current script
SINGLE_DISEASE = False   # optional filter for subtype == DLBCL

# ---------------------------------------------------------------------------
# Load test patients and data
# ---------------------------------------------------------------------------
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]

if CACHED_DATA:
    WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
    LONG_DATA = pd.read_pickle("data/LONG_DATA.pkl")
else:
    from lyfo_treatment_failure_prediction.data_processing.preprocess_data import (
        WIDE_DATA,
        LONG_DATA,
    )

if SINGLE_DISEASE:
    WIDE_DATA = WIDE_DATA[WIDE_DATA["subtype"] == "DLBCL"].reset_index(drop=True)

LONG_DATA["patientid"] = LONG_DATA["patientid"].astype(int)

# ---------------------------------------------------------------------------
# Initialize FeatureMaker
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Add static features (non-temporal predictors)
# ---------------------------------------------------------------------------
# Exclude columns likely to leak or irrelevant for prediction
exclusion_terms = [
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
    "proxy",
    "dato",
    "_dt",
]

static_predictors = [
    col
    for col in feature_maker.wide_data.columns
    if not any(term in col for term in exclusion_terms)
]

for static_predictor in static_predictors:
    feature_spec = {
        "data_source": "RKKP_LYFO",
        "value_column": static_predictor,
        "fallback": -1,
        "feature_base_name": f"RKKP_LYFO_{static_predictor}",
    }
    feature_maker.add_static_feature(feature_spec)

# ---------------------------------------------------------------------------
# Add time-dependent features using pre-defined feature specs
# ---------------------------------------------------------------------------
for feature_spec in tqdm(feature_specs, desc="Adding time-dependent features"):
    lookbacks = [dt.timedelta(90), dt.timedelta(365), dt.timedelta(365 * 3)]
    src = feature_spec.get("data_source")

    if "90" in src:
        lookbacks = [dt.timedelta(90)]
    elif "365" in src:
        lookbacks = [dt.timedelta(365)]
    elif "1095" in src:
        lookbacks = [dt.timedelta(365 * 3)]

    feature_maker.add_features_given_ratio(
        data_source=src,
        agg_funcs=feature_spec.get("agg_funcs"),
        lookbacks=lookbacks,
        proportion=feature_spec.get("proportion"),
        fallback=-1,
        collapse_rare_conditions_to_feature=True,
    )

# ---------------------------------------------------------------------------
# Define outcome variables
# ---------------------------------------------------------------------------
translation_dict = {9: 0, 1: 1}  # 0 = NaN/unknown

feature_maker.wide_data["relapse"] = feature_maker.wide_data["relapse_label"].apply(
    lambda x: translation_dict.get(x)
)


def define_succesful_treatment(date_death, date_relapse):
    """Determine the earliest event (death or relapse)."""
    if pd.isnull(date_death) and pd.isnull(date_relapse):
        event_date = pd.NaT
    elif pd.isnull(date_death):
        event_date = date_relapse
    elif pd.isnull(date_relapse):
        event_date = date_death
    else:
        event_date = min(date_death, date_relapse)
    label = 0 if pd.isnull(event_date) else 1
    return event_date, label


(
    feature_maker.wide_data["succesful_treatment_date"],
    feature_maker.wide_data["succesful_treatment_label"],
) = zip(
    *feature_maker.wide_data.apply(
        lambda x: define_succesful_treatment(x["date_death"], x["relapse_date"]),
        axis=1,
    )
)

# Add outcomes: death, relapse, and combined treatment success
feature_maker.add_outcome_from_wide_format(
    "date_death", "dead_label", [dt.timedelta(730), dt.timedelta(365 * 5)], [MaxAggregator()]
)
feature_maker.add_outcome_from_wide_format(
    "relapse_date", "relapse", [dt.timedelta(730), dt.timedelta(365 * 5)], [MaxAggregator()]
)
feature_maker.add_outcome_from_wide_format(
    "succesful_treatment_date", "succesful_treatment_label", [dt.timedelta(730)], [MaxAggregator()]
)

# ---------------------------------------------------------------------------
# Main script execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    feature_maker.make_all_features()

    feature_matrix = feature_maker.create_feature_matrix(None)

    # Replace missing (0) counts/sums with -1 for clarity
    sum_columns = [
        c for c in feature_matrix.columns if "sum_fallback_-1" in c or "count_fallback_-1" in c
    ]
    for c in sum_columns:
        feature_matrix.loc[feature_matrix[c] == 0, c] = -1

    # Replace -1 fallback in outcome columns with 0 (not dead/relapsed)
    for col in [
        "outc_dead_label_within_0_to_730_days_max_fallback_0",
        "outc_dead_label_within_0_to_1825_days_max_fallback_0",
    ]:
        feature_matrix.loc[feature_matrix[col] == -1, col] = 0

    feature_matrix = feature_matrix.replace(np.NAN, -1)

    # Sanitize column names (remove symbols not allowed in ML models)
    feature_matrix.columns = (
        feature_matrix.columns.str.replace("<", "less_than")
        .str.replace(",", "_comma_")
        .str.replace(">", "more_than")
        .str.replace("[", "left_bracket")
        .str.replace("]", "right_bracket")
    )

    # Define grouping column
    feature_matrix["group"] = feature_matrix.apply(
        lambda x: f"relapse_{x['outc_relapse_within_0_to_730_days_max_fallback_0']}_"
                  f"death_{x['outc_dead_label_within_0_to_730_days_max_fallback_0']}_"
                  f"disease_{x['pred_RKKP_subtype_fallback_-1']}",
        axis=1,
    )

    # Track missingness
    feature_matrix["proportion_of_missing"] = (
        feature_matrix.eq(-1).sum(axis=1) / len(feature_matrix.columns)
    )

    feature_matrix.to_pickle("results/feature_matrix_all.pkl")
