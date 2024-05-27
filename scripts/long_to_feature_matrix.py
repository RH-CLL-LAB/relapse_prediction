import pandas as pd
import polars as pl
import numpy as np
import datetime as dt
from timeseriesflattener.aggregators import (
    MaxAggregator,
    MinAggregator,
    MeanAggregator,
    CountAggregator,
    SumAggregator,
    VarianceAggregator,
    HasValuesAggregator,
    SlopeAggregator,
    LatestAggregator,
    EarliestAggregator,
)

# NOTE: Consider precalculating this
from make_text_features import *
from feature_specification import feature_specs

# from load_to_long_format import *
# NOTE: make the relevant path to the feature maker
from feature_maker.scripts.feature_maker import FeatureMaker

CACHED_DATA = True
EXCLUDE_PROXIES = False

if CACHED_DATA:
    WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
    LONG_DATA = pd.read_pickle("data/LONG_DATA.pkl")
else:
    from preprocess_data import WIDE_DATA, LONG_DATA

# get proxies out
if EXCLUDE_PROXIES:
    WIDE_DATA.loc[WIDE_DATA["proxy_death"] == 1, "relapse_label"] = 9
    WIDE_DATA.loc[WIDE_DATA["proxy_death"] == 1, "relapse_date"] = pd.NaT

for column in WIDE_DATA.columns:
    if "date" in column:
        WIDE_DATA[column] = pd.to_datetime(WIDE_DATA[column])

LONG_DATA["patientid"] = LONG_DATA["patientid"].astype(int)

feature_maker = FeatureMaker(
    long_data=LONG_DATA,
    wide_data=WIDE_DATA,
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


for feature_spec in feature_specs:
    # okay this takes a shit ton of time now for some reason
    feature_maker.add_features_given_ratio(
        data_source=feature_spec.get("data_source"),
        agg_funcs=feature_spec.get("agg_funcs"),
        lookbacks=[dt.timedelta(90), dt.timedelta(365), dt.timedelta(365 * 5)],
        proportion=feature_spec.get("proportion"),
        fallback=-1,
        collapse_rare_conditions_to_feature=True,
    )
translation_dict = {9: 0, 1: 1}  # 0: np.NAN

feature_maker.wide_data["relapse"] = feature_maker.wide_data["relapse_label"].apply(
    lambda x: translation_dict.get(x)
)


feature_maker.wide_data.loc[
    feature_maker.wide_data["relapse_date"] == -1, "relapse_date"
] = pd.NaT

feature_maker.add_outcome_from_wide_format(
    "date_death", "proxy_death", [dt.timedelta(730)], [MaxAggregator()]
)

feature_maker.add_outcome_from_wide_format(
    "relapse_date", "relapse", [dt.timedelta(730)], [MaxAggregator()]
)

feature_maker.add_outcome_from_long_format(
    "LYFO_AKI", "acute_kidney_injury", [dt.timedelta(730)], [MinAggregator()]
)

# NOTE: Text features are only made within 1 year and 5 year lookbacks

if __name__ == "__main__":
    feature_maker.make_all_features()
    # adding features from text
    feature_maker.add_feature_from_polars_dataframe(
        embedded_text_with_metadata,
        aggregators=[MeanAggregator()],
        lookbehind_distances=[dt.timedelta(days=365), dt.timedelta(days=365 * 5)],
        fallback=-1,
        column_prefix="pred_tfidf",
    )

    feature_matrix = feature_maker.create_feature_matrix(None)
    feature_matrix.to_pickle("results/feature_matrix.pkl")
