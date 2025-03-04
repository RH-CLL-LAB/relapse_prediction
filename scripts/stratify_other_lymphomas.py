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
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
)

from helpers.constants import *
from helpers.processing_helper import *

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

sns.set_context("paper")

seed = 46
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

features = pd.read_csv("results/feature_names_all.csv")["features"].values
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)

pd.Categorical(WIDE_DATA["subtype"]).categories

subtype_number = 3

# train = train[train["pred_RKKP_subtype_fallback_-1"] == subtype_number].reset_index(
#     drop=True
# )

outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[-1]
col_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"]
col_to_leave.extend(outcome_column)

ipi_cols = [x for x in feature_matrix.columns if "NCCN_" in x]
col_to_leave.extend(ipi_cols)

predictor_columns = [x for x in train.columns if x not in col_to_leave]

predictor_columns = [
    x
    for x in predictor_columns
    if x not in ["pred_RKKP_subtype_fallback_-1", "pred_RKKP_hospital_fallback_-1"]
]


features = list(features)

for i in supplemental_columns:
    if i not in features:
        features.append(i)

for column in tqdm(features):
    clip_values(train, test, column)

supplemental_columns = [
    "pred_RKKP_hospital_fallback_-1",
    "pred_RKKP_subtype_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
]

features = list(features)
features.extend(supplemental_columns)

X_train_smtom, y_train_smtom = (
    train[[x for x in train.columns if x in features]],
    train[outcome],
)

test_specific = test[
    test["pred_RKKP_subtype_fallback_-1"] == subtype_number
].reset_index(drop=True)

X_test_specific = test_specific[[x for x in test_specific.columns if x in features]]
y_test_specific = test_specific[outcome]

X_test = test[[x for x in test.columns if x in features]]
y_test = test[outcome]

bst = XGBClassifier(
    missing=-1,
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=3,
    gamma=0,
    subsample=1,
    colsample_bytree=0.9,
    objective="binary:logistic",
    reg_alpha=10,
    nthread=10,
    random_state=seed,
)

bst.fit(X_train_smtom, y_train_smtom)


def check_performance(X, y, threshold=0.5):
    y_pred = bst.predict_proba(X).astype(float)
    y_pred = [1 if x[1] > threshold else 0 for x in y_pred]

    f1 = f1_score(y.values, y_pred)
    roc_auc = roc_auc_score(y.values, bst.predict_proba(X).astype(float)[:, 1])
    recall = recall_score(y.values, y_pred)
    specificity = recall_score(y.values, y_pred, pos_label=0)
    precision = precision_score(y.values, y_pred, zero_division=1)
    pr_auc = average_precision_score(y.values, bst.predict_proba(X).astype(float)[:, 1])
    mcc = matthews_corrcoef(y.values, y_pred)
    return f1, roc_auc, recall, specificity, precision, pr_auc, mcc


def check_performance_across_thresholds(X, y):
    list_of_results = []
    for threshold in np.linspace(0, 1, num=100):
        f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
            X, y, threshold=threshold
        )
        list_of_results.append(
            {
                "threshold": threshold,
                "f1": f1,
                "roc_auc": roc_auc,
                "recall": recall,
                "specificity": specificity,
                "precision": precision,
                "pr_auc": pr_auc,
                "mcc": mcc,
            }
        )

    results = pd.DataFrame(list_of_results)

    best_threshold = results[results["mcc"] == results["mcc"].max()][
        "threshold"
    ].values[-1]

    results = results.melt(id_vars="threshold")
    sns.lineplot(data=results, x="threshold", y="value", hue="variable")
    plt.show()
    return results, best_threshold


results, best_threshold = check_performance_across_thresholds(X_test, y_test)

f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test, y_test, 0.5
)
y_pred = bst.predict_proba(X_test).astype(float)
y_pred = [1 if x[1] > 0.5 else 0 for x in y_pred]

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test.values, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test.values, y_pred)).plot()

results, best_threshold = check_performance_across_thresholds(
    X_test_specific, y_test_specific
)

y_pred = bst.predict_proba(X_test_specific).astype(float)
y_pred = [1 if x[1] > 0.3 else 0 for x in y_pred]

# y_pred = [1 if x >= 6 else 0 for x in test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"]]

f1 = f1_score(y_test_specific.values, y_pred)
roc_auc = roc_auc_score(
    y_test_specific.values,
    bst.predict_proba(X_test_specific).astype(float)[:, 1],
)
recall = recall_score(y_test_specific.values, y_pred)
specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
precision = precision_score(y_test_specific.values, y_pred)
pr_auc = average_precision_score(
    y_test_specific.values,
    bst.predict_proba(X_test_specific).astype(float)[:, 1],
)
mcc = matthews_corrcoef(y_test_specific.values, y_pred)

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test_specific.values, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test_specific.values, y_pred)).plot()


test_specific["y_pred"] = y_pred

outcomes = [x for x in test_specific.columns if "outc" in x]

all_outcomes_confusion_matrix = (
    test_specific.groupby(
        ["y_pred", outcomes[0], outcomes[1], outcomes[2], outcomes[3]]
    )
    .agg(n=("patientid", "count"))
    .reset_index()
)


all_outcomes_confusion_matrix = (
    test_specific.groupby(["y_pred", outcomes[0], outcomes[2]])
    .agg(n=("patientid", "count"))
    .reset_index()
)

all_outcomes_confusion_matrix = (
    test_specific.groupby(["y_pred", outcomes[2], outcomes[3], outcomes[1]])
    .agg(n=("patientid", "count"))
    .reset_index()
)

test_specific["outc_succesful_treatment_label_within_0_to_1825_days_max_fallback_0"] = (
    test_specific[outcomes[1]] + test_specific[outcomes[3]]
).apply(lambda x: min(x, 1))

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

test_specific = test_specific.merge(
    WIDE_DATA[["patientid", "CNS_IPI_diagnosis"]]
).reset_index(drop=True)


y_pred = [
    1 if x >= 6 else 0
    for x in test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"]
]

import math

y_pred = [
    0 if (pd.isnull(x)) or x < 4 else 1 for x in test_specific["CNS_IPI_diagnosis"]
]


y_pred = bst.predict_proba(X_test_specific).astype(float)
y_pred = [1 if x[1] > 0.2 else 0 for x in y_pred]

y_test_specific = test_specific[outcomes[-1]]

f1 = f1_score(y_test_specific.values, y_pred)
roc_auc = roc_auc_score(
    y_test_specific.values,
    bst.predict_proba(X_test_specific).astype(float)[:, 1],
)
recall = recall_score(y_test_specific.values, y_pred)
specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
precision = precision_score(y_test_specific.values, y_pred)
pr_auc = average_precision_score(
    y_test_specific.values,
    bst.predict_proba(X_test_specific).astype(float)[:, 1],
)
mcc = matthews_corrcoef(y_test_specific.values, y_pred)

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test_specific.values, y_pred))

import shap

# compute SHAP values
explainer = shap.TreeExplainer(bst)
shap_values = explainer(X_test)

y_pred = bst.predict_proba(X_test).astype(float)

indexes = [i for i, x in enumerate(y_pred) if x[1] > 0.75]
shap.plots.waterfall(shap_values[2])

shap_values[indexes]

shap.decision_plot(
    explainer.expected_value,
    explainer.shap_values(X_test)[2],
    X_test.columns,
    link="logit",
)


shap.summary_plot(
    shap_values[indexes],
    X_test.iloc[indexes],
    feature_names=X_test.columns,
)


feature_names = [
    "Performance status (diagnosis)",
    "Age (diagnosis)",
    "LDH (diagnosis)",
    "Count of CT-scans of cerebrum (90 days)",
    "Count of MR-scans of cerebrum (365 days)",
    "Count of treatments with blood or blood products (1095 days)",
    "Count of treatments with relation to blood, hematopoietic organs lymphatic tissue (90 days)",
    "Sex",
    "TRC (diagnosis)",
    "Count of prednisolone prescriptions (1095 dage)",
    "Count of days of hospitalization due to minor surgical procedures (365 days)",
    "Maximum of beta-2-microglubolin (1095 days)",
    "Maximum of neutrophilocytes (90 days)",
    "Age-adjusted IPI (diagnosis)",
    "Number of regions with leukemia (diagnosis)",
    "Count of normal cell findings from pathology (1095 days)",
    "Count of hospitalizations categorized as outpatient (90 days)",
    "Count of hospitalizations categorized as written communication (365 days)",
    "Count of hospitalizations categorized as treatment or care (90 days)",
    "Count of sulfonamides prescriptions (1095 days)",
    "Year of Treatment",
    "Hospital",
    "Count of Pathology tests resulting in unusable results (1095 days)",
    "Unique count of prescriptions (ATC-level = 5) since diagnosis (90 days)",
    "Count of X-ray scans of the thorax (365 days)",
    "Count of Epstein Virus microbiology findings (90 days)",
]

[x for x in X_test_specific.columns if "pred_ord_medicine" in x]

feature_names_original = [
    "pred_RKKP_PS_diagnosis_fallback_-1",
    "pred_RKKP_age_diagnosis_fallback_-1",
    "pred_RKKP_LDH_diagnosis_fallback_-1",
    "pred_sks_referals_CT-skanning af cerebrum_within_0_to_90_days_count_fallback_-1",
    "pred_sks_referals_MR-skanning af cerebrum_within_0_to_365_days_count_fallback_-1",
    "pred_sks_at_the_hospital_Behandling med blod og blodprodukter_within_0_to_1825_days_count_fallback_-1",
    "pred_sks_referals_Med beh m relation t blod_comma_ bloddan. organer og lymfatisk væv_within_0_to_90_days_count_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
    "pred_RKKP_TRC_diagnosis_fallback_-1",
    "pred_ordered_medicine_prednisolone_within_0_to_1825_days_count_fallback_-1",
    "pred_sks_at_the_hospital_Mindre kirurgiske procedurer_within_0_to_365_days_sum_fallback_-1",
    "pred_labmeasurements_B2M_within_0_to_1825_days_max_fallback_-1",
    "pred_labmeasurements_NEU_within_0_to_90_days_max_fallback_-1",
    "pred_RKKP_AAIPI_score_diagnosis_fallback_-1",
    "pred_RKKP_n_regions_diagnosis_fallback_-1",
    "pred_pathology_concat_normal_cells_within_0_to_1825_days_count_fallback_-1",
    "pred_sks_at_the_hospital_Ambulant_within_0_to_90_days_count_fallback_-1",
    "pred_sks_at_the_hospital_Skriftlig kommunikation_within_0_to_365_days_count_fallback_-1",
    "pred_sks_at_the_hospital_Behandlings- og plejeklassifikation_within_0_to_90_days_count_fallback_-1",
    "pred_ordered_medicine_sulfonamides_comma_ plain_within_0_to_1825_days_count_fallback_-1",
    "pred_RKKP_year_treat_fallback_-1",
    "pred_RKKP_hospital_fallback_-1",
    "pred_pathology_concat_unusable_within_0_to_1825_days_count_fallback_-1",
    "pred_ord_medicine_poly_pharmacy_since_diagnosis_atc_level_5_within_0_to_90_days_count_fallback_-1",
    "pred_sks_referals_Røntgenundersøgelse af thorax_within_0_to_365_days_count_fallback_-1",
    "pred_PERSIMUNE_microbiology_analysis_epstein_within_0_to_90_days_count_fallback_-1",
]


rename_dict = {
    feature_names_original[i]: feature_names[i] for i in range(len(feature_names))
}

X_test_specific_renamed = X_test_specific.rename(columns=rename_dict)

# compute SHAP values
explainer = shap.TreeExplainer(bst)
shap_values = explainer(X_test_specific_renamed)

figure = shap.summary_plot(
    shap_values,
    X_test_specific_renamed,
    # feature_names=feature_names,
    max_display=20,
    show=False,
)


# plt.savefig("plots/shap_values_dlbcl_only.png", dpi=300, bbox_inches="tight")
# plt.savefig("plots/shap_values_dlbcl_only.pdf", bbox_inches="tight")

plt.savefig("plots/shap_values.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/shap_values.pdf", bbox_inches="tight")


# compute SHAP values
explainer = shap.TreeExplainer(bst)
shap_values = explainer(X_test_specific_renamed)

figure = shap.summary_plot(
    shap_values,
    X_test_specific_renamed,
    # feature_names=feature_names,
    max_display=20,
    show=False,
)

bst.save_model("results/models/model_all.json")
test_specific.to_csv("results/test_specific.csv", index=False)
test.to_csv("results/test.csv", index=False)
X_test_specific.to_csv("results/X_test_specific.csv", index=False)
X_test.to_csv("results/X_test.csv", index=False)
