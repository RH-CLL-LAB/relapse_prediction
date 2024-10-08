import pandas as pd
import polars as pl
import numpy as np
import datetime as dt
from timeseriesflattener.aggregators import (
    MaxAggregator,
    MinAggregator,
)

from feature_specification import feature_specs
from feature_maker.scripts.feature_maker import FeatureMaker

CACHED_DATA = True
INCLUDE_PROXIES = False

if CACHED_DATA:
    WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
    LONG_DATA = pd.read_pickle("data/LONG_DATA.pkl")
else:
    from preprocess_data import WIDE_DATA, LONG_DATA

# get proxies out
if INCLUDE_PROXIES:
    WIDE_DATA.loc[WIDE_DATA["proxy_death"] == 1, "relapse_label"] = 1
    WIDE_DATA.loc[WIDE_DATA["proxy_death"] == 1, "relapse_date"] = WIDE_DATA[
        WIDE_DATA["proxy_death"] == 1
    ]["date_death"]

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

from tqdm import tqdm

for feature_spec in tqdm(feature_specs):
    lookbacks = [dt.timedelta(90), dt.timedelta(365), dt.timedelta(365 * 5)]
    if "90" in feature_spec.get("data_source"):
        lookbacks = [dt.timedelta(90)]
    if "365" in feature_spec.get("data_source"):
        lookbacks = [dt.timedelta(365)]
    if "1825" in feature_spec.get("data_source"):
        lookbacks = [dt.timedelta(1825)]

    # okay this takes a shit ton of time now for some reason
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

feature_maker.wide_data["succesful_treatment_date"] = feature_maker.wide_data.apply(
    lambda x: min(x["date_death"], x["relapse_date"]), axis=1
)

feature_maker.wide_data["succesful_treatment_label"] = feature_maker.wide_data.apply(
    lambda x: max(x["dead_label"], x["relapse"]), axis=1
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

feature_maker.add_outcome_from_long_format(
    "LYFO_AKI", "acute_kidney_injury", [dt.timedelta(730)], [MinAggregator()]
)

feature_maker.add_outcome_from_wide_format(
    "succesful_treatment_date",
    "succesful_treatment_label",
    [dt.timedelta(730)],
    [MaxAggregator()],
)

if __name__ == "__main__":
    feature_maker.make_all_features()
    # adding features from text
    # feature_maker.add_feature_from_polars_dataframe(
    #     embedded_text_with_metadata,
    #     aggregators=[MeanAggregator()],
    #     lookbehind_distances=[dt.timedelta(days=365), dt.timedelta(days=365 * 5)],
    #     fallback=-1,
    #     column_prefix="pred_tfidf",
    # )

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

    # calculate correlated features
    features = feature_matrix.drop(columns=["patientid", "timestamp", "pred_time_uuid"])
    feature_names = features.columns

    def calculate_correlations(df, feature_name):
        data = df.corrwith(df[feature_name]).reset_index()
        data["target_column"] = feature_name
        return data

    def calculate_all_correlations(df):
        feature_names = df.columns
        list_of_correlations = []
        for feature in tqdm(feature_names):
            correlation = calculate_correlations(df, feature)
            list_of_correlations.append(correlation)
            df = df.drop(columns=feature)
        return list_of_correlations

    list_of_correlations = calculate_all_correlations(features)

    concatenated_features = pd.concat(list_of_correlations)

    concatenated_features = concatenated_features[
        concatenated_features["index"] != concatenated_features["target_column"]
    ].reset_index(drop=True)

    concatenated_features = concatenated_features[
        ~(
            (concatenated_features["index"].str.startswith("outc"))
            | (concatenated_features["target_column"].str.startswith("outc"))
        )
    ].reset_index(drop=True)

    concatenated_features = concatenated_features[
        concatenated_features[0].notna()
    ].reset_index(drop=True)

    number_of_missing = (
        (features == -1).sum().reset_index().rename(columns={0: "n_missing"})
    )

    concatenated_features = concatenated_features.merge(
        number_of_missing.rename(columns={"n_missing": "n_missing_index"})
    )

    concatenated_features = concatenated_features.merge(
        number_of_missing.rename(
            columns={"index": "target_column", "n_missing": "n_missing_target"}
        )
    )

    redundancy_dict = {}

    def find_redundant_features(index, target, index_missing, target_missing):
        if redundancy_dict.get(target) == None:
            if index_missing >= target_missing:
                redundancy_dict[index] = target
            else:
                redundancy_dict[target] = index

    # threshold set at correlation < 0.7 - can be tweaked to be more or less conservative

    concatenated_features[concatenated_features[0] > 0.70].apply(
        lambda x: find_redundant_features(
            x["index"], x["target_column"], x["n_missing_index"], x["n_missing_target"]
        ),
        axis=1,
    )

    feature_matrix = feature_matrix[
        [x for x in feature_matrix.columns if x not in redundancy_dict]
    ]

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

    feature_matrix.to_pickle("results/feature_matrix.pkl")
