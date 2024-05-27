# model the outcomes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    auc,
    average_precision_score,
    matthews_corrcoef,
)

feature_matrix = pd.read_pickle("results/feature_matrix.pkl")

feature_matrix  # not that many more columns surprisingly


NOT_FROM_DIAGNOSIS = True
NOT_FROM_TEXT = False
AKI_PROXY = False

# replace strings because XGBoost doesn't like punctuation

feature_matrix.columns = feature_matrix.columns.str.replace("<", "less_than")

feature_matrix.columns = feature_matrix.columns.str.replace(",", "_comma_")

feature_matrix.columns = feature_matrix.columns.str.replace(">", "more_than")
feature_matrix.columns = feature_matrix.columns.str.replace("[", "left_bracket")
feature_matrix.columns = feature_matrix.columns.str.replace("]", "right_bracket")

if AKI_PROXY:  # makes f1 way better, MCC way worse
    feature_matrix.loc[
        (feature_matrix["outc_relapse_within_0_to_730_days_max_fallback_0"] == 0)
        & (
            feature_matrix[
                "outc_acute_kidney_injury_within_0_to_730_days_min_fallback_0"
            ]
            == 1
        ),
        "outc_relapse_within_0_to_730_days_max_fallback_0",
    ] = 1

outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[1]

col_to_leave = [
    "patientid",
    "timestamp",
    "pred_time_uuid",
]
col_to_leave.extend(outcome_column)

if NOT_FROM_DIAGNOSIS:
    diagnosis_predictors = [x for x in feature_matrix.columns if "_7300_" in x]
    col_to_leave.extend(diagnosis_predictors)


if NOT_FROM_TEXT:
    text_predictors = [x for x in feature_matrix.columns if "_tfidf_" in x]
    col_to_leave.extend(text_predictors)
# NOTE: non_important_columns can actually be far worse!
# we need to remove the ones not even in score_df!

# non_important_cols = score_df[score_df["score"] == 1]["index"].values
# col_to_leave.extend(non_important_cols)

# feature_matrix = pd.get_dummies(feature_maker.feature_matrix)


from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.compose import ColumnTransformer

# because of latest, there are still some funky ass stuff
feature_matrix = feature_matrix.replace(np.nan, -1)

X = feature_matrix[[x for x in feature_matrix.columns if x not in col_to_leave]]
y = feature_matrix[outcome]
seed = 46

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=seed
)

(
    dummy_X_train,
    dummy_X_test,
    dummy_y_train,
    dummy_y_test,
) = train_test_split(
    feature_matrix,
    feature_matrix["outc_proxy_death_within_0_to_730_days_max_fallback_0"],
    test_size=0.20,
    random_state=seed,
)


indexer = dummy_y_test == 0

bst = XGBClassifier(
    missing=-1,
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=5,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    nthread=10,  # was 6 before-could bumpt this for faster fitting probably?
    scale_pos_weight=1,
)

# bst = RidgeClassifier()

# ct = ColumnTransformer([])

# pipe = Pipeline([("scaler", Standar)])

# bst = RandomForestClassifier()
# bst = SVC(class_weight="balanced")
# bst = RidgeClassifier(class_weight="balanced")
# bst = DecisionTreeClassifier(class_weight="balanced")
# fit model

bst.fit(X_train, y_train)

X_test_specific = X_test[
    (X_test["pred_RKKP_subtype_fallback_-1"] == 0) & (indexer)
].reset_index()


X_test_specific = X_test[
    X_test["pred_RKKP_subtype_fallback_-1"] == 0
].reset_index()  # need to exclude proxy stuff here

index = X_test_specific["index"].values
X_test_specific = X_test_specific[
    [x for x in X_test_specific.columns if "index" not in x]
]

y_test_specific = y_test[index]
for x_, y_ in [(X_test, y_test), (X_test_specific, y_test_specific)]:
    y_pred = bst.predict(x_).astype(float)

    f1 = f1_score(y_.values, y_pred)
    roc_auc = roc_auc_score(y_.values, y_pred)
    recall = recall_score(y_.values, y_pred)
    precision = precision_score(y_.values, y_pred)
    pr_auc = average_precision_score(y_.values, y_pred)
    mcc = matthews_corrcoef(y_.values, y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(confusion_matrix(y_.values, y_pred))

# explain model using SHAP

import shap

# compute SHAP values
explainer = shap.TreeExplainer(bst)
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values, max_display=20)

shap.plots.scatter(shap_values[:, "pred_RKKP_sex_fallback_-1"])

shap.plots.scatter(shap_values[:, "pred_RKKP_days_from_diagnosis_to_tx_fallback_-1"])

shap.plots.scatter(
    shap_values[
        :,
        "pred_laboratorymeasurements_concat_Creatininium_within_0_to_1825_days_latest_fallback_-1",
    ]
)

shap.plots.scatter(shap_values[:, "pred_RKKP_B2M_nmL_diagnosis_fallback_-1"])


shap.plots.scatter(shap_values[:, "pred_RKKP_year_treat_fallback_-1"])


shap.plots.bar(shap_values.abs.mean(0))

shap.plots.beeswarm(shap_values, max_display=10)

# consider having

# pd.DataFrame(results).to_pickle("results/initial_tests.pkl")

parameters = bst.get_booster().get_score(importance_type="weight")

# NOTE: Checking feature importance here - should probably be in another script

# parameters = bst.get_booster().get_score(importance_type="total_gain")

score_df = pd.DataFrame(parameters.values(), index=parameters.keys(), columns=["score"])

score_df = score_df.reset_index()

score_df.loc[score_df["index"].str.contains("laboratory"), "data_modality"] = "lab"
score_df.loc[score_df["index"].str.contains("RKKP"), "data_modality"] = "RKKP"
score_df.loc[score_df["index"].str.contains("RECEPT"), "data_modality"] = "medicine"
score_df.loc[score_df["index"].str.contains("adm_medicine"), "data_modality"] = (
    "medicine"
)
score_df.loc[score_df["index"].str.contains("sks_"), "data_modality"] = "sks"
score_df.loc[score_df["index"].str.contains("pato"), "data_modality"] = "pato"
score_df.loc[score_df["index"].str.contains("diagnoses_all"), "data_modality"] = (
    "diagnosis"
)
score_df.loc[score_df["index"].str.contains("PERSIMUNE"), "data_modality"] = "persimune"
score_df.loc[score_df["index"].str.contains("epikur"), "data_modality"] = "medicine"
score_df.loc[score_df["index"].str.contains("_365_"), "days"] = "365"
score_df.loc[score_df["index"].str.contains("_1825_"), "days"] = "1825"
score_df.loc[score_df["index"].str.contains("_90_"), "days"] = "90"
score_df.loc[score_df["index"].str.contains("AKI"), "data_modality"] = "LYFO_AKI"
score_df.loc[score_df["index"].str.contains("Vitale"), "data_modality"] = "vitals"
score_df.loc[score_df["index"].str.contains("tfidf"), "data_modality"] = "text"


import seaborn as sns

sns.barplot(data=score_df, y="score", x="days")

sns.barplot(data=score_df, y="score", x="data_modality")

sns.barplot(data=score_df, y="score", x="data_modality", estimator="max")

score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)

order = score_df["index"].head(30)

score_df.head(20)

sns.barplot(data=score_df.head(30), x="score", y="index", order=order)

persimune_df = score_df[score_df["data_modality"] == "persimune"].head(30)

order = persimune_df["index"].head(30)
sns.barplot(data=persimune_df, x="score", y="index", order=order)


medicine_df = score_df[score_df["data_modality"] == "medicine"].head(30)

order = medicine_df["index"].head(30)
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 8))
sns.barplot(data=medicine_df, x="score", y="index", order=order)

diagnosis_df = score_df[score_df["data_modality"] == "diagnosis"].head(30)

order = diagnosis_df["index"].head(30)

plt.figure(figsize=(4, 8))
sns.barplot(data=diagnosis_df, x="score", y="index", order=order)


pato_df = score_df[score_df["data_modality"] == "pato"].head(30)

order = pato_df["index"].head(30)

plt.figure(figsize=(4, 8))
sns.barplot(data=pato_df, x="score", y="index", order=order)


lab_df = score_df[score_df["data_modality"] == "lab"].head(30)

order = lab_df["index"].head(30)

plt.figure(figsize=(4, 8))
sns.barplot(data=lab_df, x="score", y="index", order=order)


vitals_df = score_df[score_df["data_modality"] == "vitals"].head(30)

order = vitals_df["index"].head(30)

plt.figure(figsize=(4, 8))
sns.barplot(data=vitals_df, x="score", y="index", order=order)


text_df = score_df[score_df["data_modality"] == "text"].head(30)

order = text_df["index"].head(30)

plt.figure(figsize=(4, 8))
sns.barplot(data=text_df, x="score", y="index", order=order)


rkkp_df = score_df[score_df["data_modality"] == "RKKP"].head(30)

order = rkkp_df["index"].head(30)
plt.figure(figsize=(4, 8))
sns.barplot(data=rkkp_df, x="score", y="index", order=order)
