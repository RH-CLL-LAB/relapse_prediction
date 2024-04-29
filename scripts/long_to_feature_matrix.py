import pandas as pd
import polars as pl
import numpy as np
import datetime as dt
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    auc,
    average_precision_score,
    matthews_corrcoef,
)

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

# from load_to_long_format import *
# NOTE: make the relevant path to the feature maker
from feature_maker.scripts.feature_maker import FeatureMaker

CACHED_DATA = True
EXCLUDE_PROXIES = False

if CACHED_DATA:
    WIDE_DATA = pd.read_pickle("../data/WIDE_DATA.pkl")
    LONG_DATA = pd.read_pickle("../data/LONG_DATA.pkl")
    LONG_DATA["patientid"] = LONG_DATA["patientid"].astype(int)
else:
    from scripts.preprocess_data import WIDE_DATA, LONG_DATA

# get proxies out
if EXCLUDE_PROXIES:
    WIDE_DATA.loc[WIDE_DATA["proxy_death"] == 1, "relapse_label"] = 9
    WIDE_DATA.loc[WIDE_DATA["proxy_death"] == 1, "relapse_date"] = pd.NaT

for column in WIDE_DATA.columns:
    if "date" in column:
        WIDE_DATA[column] = pd.to_datetime(WIDE_DATA[column])

feature_maker = FeatureMaker(
    long_format_data_mapping={},  # TABLE_TO_LONG_FORMAT_MAPPING,
    wide_format_data_mapping={},  # TABLE_TO_WIDE_FORMAT_MAPPING,
    premade_wide_format=WIDE_DATA,
    premade_long_format=LONG_DATA,
)

feature_maker._reset_all_features()
feature_maker.specify_prediction_time_from_wide_format("date_treatment_1st_line")


# Remove response evaluation - leakage
list_of_exclusion_terms = [
    "date",
    "2nd_line",
    "relapse",  # wait why doesn't that catch it
    "OS",
    "LYFO",
    "report_submitted",
    "dead",
    "FU",
    "patientid",
    "1st_line",
    "death",
    "treatment",
    "age",
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

feature_specs = [
    {
        "data_source": "adm_medicine",
        "agg_funcs": [CountAggregator()],
        "proportion": 0.05,
    },
    {
        "data_source": "laboratorymeasurements",
        "agg_funcs": [
            LatestAggregator(timestamp_col_name="timestamp"),
            CountAggregator(),
            MeanAggregator(),
            MaxAggregator(),
            MinAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
            SumAggregator(),
        ],
        "proportion": 0.9,
    },
    {
        "data_source": "laboratorymeasurements_concat",  # does this actually make sense?
        "agg_funcs": [
            LatestAggregator(timestamp_col_name="timestamp"),
            CountAggregator(),
            MeanAggregator(),
            MaxAggregator(),
            MinAggregator(),
        ],
        "proportion": 0.5,
    },
    {
        "data_source": "RECEPTDATA_CLEAN_pseudo",
        "agg_funcs": [CountAggregator(), SumAggregator()],
        "proportion": 0.05,
    },
    {"data_source": "SDS_pato", "agg_funcs": [CountAggregator()], "proportion": 0.05},
    {
        "data_source": "diagnoses_all",
        "agg_funcs": [CountAggregator()],
        "proportion": 0.05,
    },
    {
        "data_source": "SP_SocialHx",
        "agg_funcs": [LatestAggregator(timestamp_col_name="timestamp")],
        "proportion": 0.05,
    },
    {
        "data_source": "view_sp_bloddyrkning_del1",
        "agg_funcs": [CountAggregator()],
        "proportion": 0.05,
    },
    {
        "data_source": "LYFO_AKI",
        "agg_functions": [MaxAggregator()],
        "proportion": 0.05,
    },
    {
        "data_source": "PERSIMUNE_biochemistry",
        "agg_funcs": [
            CountAggregator(),
            MeanAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
            LatestAggregator(timestamp_col_name="timestamp"),
            MaxAggregator(),
            MinAggregator(),
        ],
        "proportion": 0.05,
    },
    {
        "data_source": "PERSIMUNE_microbiology_analysis",
        "agg_funcs": [CountAggregator(), MaxAggregator()],
        "proportion": 0.05,
    },
    {
        "data_source": "PERSIMUNE_microbiology_culture",
        "agg_funcs": [CountAggregator(), MaxAggregator()],
        "proportion": 0.01,
    },
    {
        "data_source": "view_laboratorymeasuments_c_groups",
        "agg_funcs": [
            CountAggregator(),
            MaxAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
            MeanAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
        ],
        "proportion": 0.3,
    },
    {
        "data_source": "adm_medicine_days",
        "agg_funcs": [CountAggregator(), MeanAggregator(), MaxAggregator()],
        "proportion": 0.4,
    },
]


for feature_spec in feature_specs:
    feature_maker.add_features_given_ratio(
        data_source=feature_spec.get("data_source"),
        agg_funcs=feature_spec.get("agg_funcs"),
        lookbacks=[dt.timedelta(90), dt.timedelta(365), dt.timedelta(365 * 5)],
        proportion=feature_spec.get("proportion"),
        fallback=-1,
    )


translation_dict = {9: 0, 1: 1}  # 0: np.NAN

feature_maker.wide_data["relapse"] = feature_maker.wide_data["relapse_label"].apply(
    lambda x: translation_dict.get(x)
)


# feature_maker.wide_data.loc[
#     feature_maker.wide_data["relapse_date"] == -1, "relapse_date"
# ] = feature_maker.wide_data[feature_maker.wide_data["relapse_date"] == -1][
#     "date_treatment_1st_line"
# ].apply(
#     lambda x: x + pd.Timedelta(730, "day")
# )

feature_maker.wide_data.loc[
    feature_maker.wide_data["relapse_date"] == -1, "relapse_date"
] = pd.NaT

feature_maker.add_outcome_from_wide_format(
    "date_death", "proxy_death", [dt.timedelta(730)], [MaxAggregator()]
)

feature_maker.add_outcome_from_wide_format(
    "relapse_date", "relapse", [dt.timedelta(730)], [MaxAggregator()]
)

if __name__ == "__main__":
    feature_maker.make_all_features()
    test = feature_maker.create_feature_matrix(None)
    # feature_maker.make_categorical_features_integers()
    feature_matrix = feature_maker.feature_matrix.collect()
    print(feature_matrix)


# # model the outcomes
# from sklearn.model_selection import train_test_split

# outcome_column = [x for x in feature_matrix if "outc" in x]

# outcome = outcome_column[0]
# col_to_leave = ["patientid", "timestamp", "prediction_time_uuid", outcome]

# # non_important_cols = score_df[score_df["score"] == 1]["index"].values
# # col_to_leave.extend(non_important_cols)

# # feature_matrix = pd.get_dummies(feature_maker.feature_matrix)

# X = feature_maker.feature_matrix[
#     [x for x in feature_maker.feature_matrix.columns if x not in col_to_leave]
# ]

# y = feature_maker.feature_matrix[outcome]

# seed = 42
# np.random.seed(seed)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=seed
# )

# proxy_deaths = feature_maker.wide_data[feature_maker.wide_data["proxy_death"] == 1][
#     "patientid"
# ].values

# dummy_X_train, dummy_X_test, dummy_y_train, dummy_y_test = train_test_split(
#     feature_matrix, feature_matrix["patientid"], test_size=0.20, random_state=seed
# )

# indexer = ~dummy_y_test.isin(proxy_deaths)

# from sklearn.pipeline import Pipeline
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.linear_model import RidgeClassifier
# from sklearn.compose import ColumnTransformer


# bst = XGBClassifier(
#     missing=-1,
#     n_estimators=1000,
#     learning_rate=0.30,
#     max_depth=3,
#     min_child_weight=5,
#     gamma=0,
#     subsample=0.4,
#     colsample_bytree=0.9,
#     objective="binary:logistic",
#     nthread=6,
#     scale_pos_weight=1,
# )

# # ct = ColumnTransformer([])

# # pipe = Pipeline([("scaler", Standar)])

# # bst = RandomForestClassifier(class_weight="balanced")
# # bst = SVC(class_weight="balanced")
# # bst = RidgeClassifier(class_weight="balanced")
# # bst = DecisionTreeClassifier(class_weight="balanced")
# # fit model

# bst.fit(X_train, y_train)

# # NOTE: only for LYFO patients, but we need to see if it just predicts death
# WIDE_DATA

# X_test_specific = X_test[
#     (X_test["pred_RKKP_LYFO_subtype"] == 0) & (indexer)
# ].reset_index()  # need to exclude proxy stuff here


# index = X_test_specific["index"].values
# X_test_specific = X_test_specific[
#     [x for x in X_test_specific.columns if "index" not in x]
# ]

# y_test_specific = y_test[index]

# for x, y in [(X_test, y_test), (X_test_specific, y_test_specific)]:
#     y_pred = bst.predict(x).astype(float)

#     f1 = f1_score(y.values, y_pred)
#     roc_auc = roc_auc_score(y.values, y_pred)
#     recall = recall_score(y.values, y_pred)
#     precision = precision_score(y.values, y_pred)
#     pr_auc = average_precision_score(y.values, y_pred)
#     mcc = matthews_corrcoef(y.values, y_pred)
#     print(f"F1: {f1}")
#     print(f"ROC-AUC: {roc_auc}")
#     print(f"Recall: {recall}")
#     print(f"Precision: {precision}")
#     print(f"PR-AUC: {pr_auc}")
#     print(f"MCC: {mcc}")

#     print(confusion_matrix(y.values, y_pred))


# parameters = bst.get_booster().get_score(importance_type="weight")

# score_df = pd.DataFrame(parameters.values(), index=parameters.keys(), columns=["score"])

# score_df = score_df.reset_index()

# score_df.loc[score_df["index"].str.contains("laboratory"), "data_modality"] = "lab"
# score_df.loc[score_df["index"].str.contains("RKKP"), "data_modality"] = "RKKP"
# score_df.loc[score_df["index"].str.contains("RECEPT"), "data_modality"] = "medicine"
# score_df.loc[
#     score_df["index"].str.contains("adm_medicine"), "data_modality"
# ] = "medicine"
# score_df.loc[score_df["index"].str.contains("pato"), "data_modality"] = "pato"
# score_df.loc[score_df["index"].str.contains("pato"), "data_modality"] = "pato"
# score_df.loc[
#     score_df["index"].str.contains("diagnoses_all"), "data_modality"
# ] = "diagnosis"
# score_df.loc[score_df["index"].str.contains("PERSIMUNE"), "data_modality"] = "persimune"

# score_df.loc[score_df["index"].str.contains("_365_"), "days"] = "365"
# score_df.loc[score_df["index"].str.contains("_1825_"), "days"] = "1825"
# score_df.loc[score_df["index"].str.contains("_90_"), "days"] = "90"


# sns.barplot(data=score_df, y="score", x="days")


# sns.barplot(data=score_df, y="score", x="data_modality")


# sns.barplot(data=score_df, y="score", x="data_modality", estimator="max")

# score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)

# score_df

# feature_matrix

# import seaborn as sns

# order = score_df["index"].head(30)

# score_df.head(20)

# sns.barplot(data=score_df.head(30), x="score", y="index", order=order)
