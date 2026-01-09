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

import xgboost


from sklearn.calibration import calibration_curve, CalibrationDisplay

from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    DetCurveDisplay,
)
import joblib

import matplotlib.patches as mpatches

from helpers.constants import *
from helpers.processing_helper import *


# make all math text regular
params = {"mathtext.default": "regular"}
plt.rcParams.update(params)


wide_data = pd.read_pickle("data/WIDE_DATA.pkl")
WIDE_DATA = wide_data.copy()

sns.set_context("paper")

seed = 46
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

features = pd.read_csv("results/feature_names_all.csv")["features"].values

wrong_patientids = wide_data[wide_data["age_diagnosis"].isna()]["patientid"]

feature_matrix = feature_matrix[~feature_matrix["patientid"].isin(wrong_patientids)].reset_index(drop = True)

feature_matrix.replace(-1, np.nan, inplace=True)


features = [
    "pred_RKKP_age_diagnosis_fallback_-1",
    "pred_RKKP_LDH_diagnosis_fallback_-1",  # needs to be normalized
    "pred_RKKP_AA_stage_diagnosis_fallback_-1",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
    "pred_RKKP_PS_diagnosis_fallback_-1",
]
# lab_measurement_features = [x for x in feature_matrix.columns if "labmeasurements" in x]

test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)

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

for column in tqdm(features):
    # this should be the way clip values is done
    clip_values(train, test, column)


(
    X_train_smtom,
    y_train_smtom,
    X_test,
    y_test,
    X_test_specific,
    y_test_specific,
    test_specific,
) = get_features_and_outcomes(
    train,
    test,
    WIDE_DATA,
    outcome,
    features,
    specific_immunotherapy=False,
    none_chop_like=False,
)

## load predictions from LR and TabPFN
lr_probs = joblib.load("results/lr_predictions.pkl")
tabpfn_probs = joblib.load("results/tabpfn_predictions.pkl")

## merging predictions onto test_specific
test["lr_probs"] = lr_probs["y_pred_proba"]
test["tabpfn_probs"] = tabpfn_probs["y_pred_proba"]

test_specific = test_specific.merge(test[["patientid", "lr_probs", "tabpfn_probs"]])


bst = XGBClassifier(
    missing=-1,
    n_estimators=3000,  # was 2000 before
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


disp = CalibrationDisplay.from_estimator(
    bst, X_test, y_test, n_bins=10, name="ML$_{\: All}$"
)

plt.savefig("plots/all_variables_all_lymphomas_calibration.png", dpi=300)
plt.savefig("plots/all_variables_all_lymphomas_calibration.pdf")

disp = CalibrationDisplay.from_estimator(
    bst, X_test_specific, y_test_specific, n_bins=10, name="ML$_{\: All}$"
)
plt.savefig("plots/all_variables_calibration.png", dpi=300)
plt.savefig("plots/all_variables_calibration.pdf")


brier_score_loss(y_test_specific, bst.predict_proba(X_test_specific)[:, 1])
brier_score_loss(y_test, bst.predict_proba(X_test)[:, 1])

results, best_threshold = check_performance_across_thresholds(X_test, y_test, bst, y_pred_proba=[])

f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test, y_test, bst, 0.2, y_pred_proba=[]
)
y_pred = bst.predict_proba(X_test).astype(float)
y_pred = [1 if x[1] > 0.2 else 0 for x in y_pred]

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test.values, y_pred))

results, best_threshold = check_performance_across_thresholds(
    X_test_specific, y_test_specific, bst, y_pred_proba=[]
)

y_pred = bst.predict_proba(X_test_specific).astype(float)
test_specific["y_pred_proba_ml_ipi"] = [x[1] for x in y_pred]
test_specific.to_csv("data/test_specific_ml_ipi_and_comparators.csv", index = False)
y_pred = [1 if x[1] > 0.5 else 0 for x in y_pred]

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


def stratified_bootstrap_metrics(
    y_true, y_pred_proba, y_pred_label, n_bootstraps=1000, seed=42, return_raw_values = False
):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred_label = np.array(y_pred_label)

    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]

    roc_aucs, pr_aucs = [], []
    precisions, specificities, recalls, mccs = [], [], [], []

    for _ in tqdm(range(n_bootstraps)):
        # Stratified resampling
        pos_sample = rng.choice(positive_indices, size=len(positive_indices), replace=True)
        neg_sample = rng.choice(negative_indices, size=len(negative_indices), replace=True)
        sample_indices = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(sample_indices)

        y_true_bs = y_true[sample_indices]
        y_pred_proba_bs = y_pred_proba[sample_indices]
        y_pred_label_bs = y_pred_label[sample_indices]

        try:
            roc_aucs.append(roc_auc_score(y_true_bs, y_pred_proba_bs))
            pr_aucs.append(average_precision_score(y_true_bs, y_pred_proba_bs))
            precisions.append(precision_score(y_true_bs, y_pred_label_bs, zero_division=0))
            recalls.append(recall_score(y_true_bs, y_pred_label_bs, zero_division=0))
            mccs.append(matthews_corrcoef(y_true_bs, y_pred_label_bs))
        
        # Specificity
            cm = confusion_matrix(y_true_bs, y_pred_label_bs)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)

        except:
            continue

    if return_raw_values:
        return {"roc_auc": roc_aucs,
        "pc_auc": pr_aucs,
        "precision": precisions,
        "recall": recalls,
        "specificity": specificities,
        "mcc": mccs}

    def summary_stats(metric_list):
        return {
            "mean": np.mean(metric_list),
            "ci_lower": np.percentile(metric_list, 2.5),
            "ci_upper": np.percentile(metric_list, 97.5)
        }

    return {
        "roc_auc": summary_stats(roc_aucs),
        "pr_auc": summary_stats(pr_aucs),
        "precision": summary_stats(precisions),
        "recall": summary_stats(recalls),
        "specificity": summary_stats(specificities),
        "mcc": summary_stats(mccs),
    }

## 
y_pred_proba = bst.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
y_pred_label = (y_pred_proba >= 0.3).astype(int)  # Apply 0.5 threshold (or whatever you used)

results = stratified_bootstrap_metrics(y_test, y_pred_proba, y_pred_label)

for metric, stats in results.items():
    print(f"{metric}: {stats['mean']:.3f} (95% CI: {stats['ci_lower']:.3f}–{stats['ci_upper']:.3f})")

## 
y_pred_proba = bst.predict_proba(X_test_specific)[:, 1]  # Get probabilities for class 1
y_pred_label = (y_pred_proba >= 0.5).astype(int)  # Apply 0.5 threshold (or whatever you used)

results = stratified_bootstrap_metrics(y_test_specific, y_pred_proba, y_pred_label)

for metric, stats in results.items():
    print(f"{metric}: {stats['mean']:.3f} (95% CI: {stats['ci_lower']:.3f}–{stats['ci_upper']:.3f})")


test_specific["y_pred"] = y_pred

outcomes = [x for x in test_specific.columns if "outc" in x]


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

# NOTE: Missing IPI has been encoded as -1
# which produces funky stuff here.

# TP FP TN FN


def make_NCCN_categorical(nccn):
    if pd.isnull(nccn):
        return None
    if nccn == -1:
        return None
    if nccn < 2:
        return "Low"
    if nccn < 4:
        return "Low-Intermediate"
    if nccn < 6:
        return "Intermediate-High"
    if nccn >= 6:
        return "High"


def make_prediction_categorical(y_prob):
    if pd.isnull(y_prob):
        return None
    if y_prob < 0.1:
        return "Low"
    if y_prob < 0.3:
        return "Low-Intermediate"
    if y_prob < 0.65:
        return "Intermediate-High"
    if y_prob >= 0.65:
        return "High"


test_specific["NCCN_categorical"] = pd.Categorical(
    test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"].apply(
        make_NCCN_categorical
    ),
    ordered=True,
    categories=["Low", "Low-Intermediate", "Intermediate-High", "High"],
)

y_pred = bst.predict_proba(X_test_specific)

test_specific["estimated_probability"] = y_pred[:, 1]


test_specific["ML_risk_group"] = pd.Categorical(
    test_specific["estimated_probability"].apply(make_prediction_categorical),
    ordered=True,
    categories=["Low", "Low-Intermediate", "Intermediate-High", "High"],
)

low_patch = mpatches.Patch(color=colors[0], label="Low", alpha=0.4)
intermediate_low_patch = mpatches.Patch(
    color=colors[1], label="Low-Intermediate", alpha=0.4
)
intermediate_high_patch = mpatches.Patch(
    color=colors[2], label="Intermediate-High", alpha=0.4
)
high_patch = mpatches.Patch(color=colors[3], label="High", alpha=0.4)


sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})
sns.set_style("white")
figure_axis = sns.stripplot(
    data=test_specific,
    y="NCCN_categorical",
    x="estimated_probability",
    hue=outcome,
    hue_order=[1, 0],
    jitter=0.35,
    palette=sns.color_palette("Set1"),
    dodge=True,
    size=4,
)
plt.fill_between([0, 0.1], -0.4, 3.4, alpha=0.4, color=colors[0], label="Low")
plt.fill_between(
    [0.1, 0.3], -0.4, 3.4, alpha=0.4, color=colors[1], label="Low-Intermediate"
)
plt.fill_between(
    [0.3, 0.65], -0.4, 3.4, alpha=0.4, color=colors[2], label="Intermediate-High"
)
plt.fill_between([0.65, 1], -0.4, 3.4, alpha=0.4, color=colors[3], label="High")
plt.hlines([0.5, 1.5, 2.5], 0, 1, colors="black", linestyles="dotted")
figure_axis.set_xlim(-0.05, 1.05)
first_legend = figure_axis.legend(
    ["Treatment Failure", "Treatment Success"],
    title="Outcome within 2 years",
    bbox_to_anchor=(1.235, 0.65),
    fontsize=11,
    title_fontsize=11,
    handletextpad=0.3
)
#for legend_handle in first_legend.legend_handles:
#    legend_handle.set_sizes([30])

another_legend = plt.legend(
    handles=[low_patch, intermediate_low_patch, intermediate_high_patch, high_patch],
    title="ML$_{\:All}$ Risk Groups",
    bbox_to_anchor=(1.005, 0.5),
    fontsize=11,
    title_fontsize=11,
)

figure_axis.add_artist(first_legend)
plt.xlabel("Estimated probability of treatment failure by the ML$_{\:All}$ model")
plt.ylabel("NCCN IPI Risk Groups")

ax2 = figure_axis.twiny()
ax2.set_xlim(-0.05, 1.05)
ax2.set_xlabel("ML$_{\:All}$ Risk Groups")
ax2.set_xticks([0.05, 0.2, 0.475, 0.825])
ax2.set_xticklabels(["Low", "Low-Intermediate", "Intermediate-High", "High"])
for item in (
    [figure_axis.title, figure_axis.xaxis.label, figure_axis.yaxis.label]
    + figure_axis.get_xticklabels()
    + figure_axis.get_yticklabels()
):
    item.set_fontsize(15)
figure_axis.yaxis.label.set_fontsize(17)
for item in (
    [ax2.title, ax2.xaxis.label, ax2.yaxis.label]
    + ax2.get_xticklabels()
    + ax2.get_yticklabels()
):
    item.set_fontsize(15)
ax2.xaxis.label.set_fontsize(17)
#plt.subplots_adjust(right = 0.8)
# figure_axis.savefig("plots/test.png", dpi=300, bbox_inches="tight")

plt.savefig("plots/ml_compared_to_nccn_stripplot.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/ml_compared_to_nccn_stripplot.pdf", bbox_inches="tight")
plt.savefig("plots/ml_compared_to_nccn_stripplot.svg", bbox_inches="tight")


test_specific.loc[
    test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] == -1,
    "pred_RKKP_NCCN_IPI_diagnosis_fallback_-1",
] = None
y_pred = [
    1 if x >= 6 else 0
    for x in test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"]
]

import math

y_pred = [
    0 if (pd.isnull(x)) or x < 4 else 1 for x in test_specific["CNS_IPI_diagnosis"]
]


dictionary = {}

weird_probabilities = (test_specific["CNS_IPI_diagnosis"] / 7).values
# fix nans - should we exclude them? probably yes
weird_probabilities = [
    (i, x) for i, x in enumerate(weird_probabilities) if pd.notnull(x)
]
indexes = [x[0] for x in weird_probabilities]
weird_probabilities = [x[1] for x in weird_probabilities]

weird_probabilities_NCCN = (
    test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] / 9
).values
weird_probabilities_NCCN = [
    (i, x) for i, x in enumerate(weird_probabilities_NCCN) if pd.notnull(x)
]
indexes_NCCN = [x[0] for x in weird_probabilities_NCCN]
weird_probabilities_NCCN = [x[1] for x in weird_probabilities_NCCN]


from sklearn.metrics import precision_recall_curve, roc_curve

y_pred = bst.predict_proba(X_test_specific).astype(float)

average_precision_score_ipi = average_precision_score(
    y_test_specific.values, [x[1] for x in y_pred]
)

precision_ipi, recall_ipi, _ = precision_recall_curve(
    y_test_specific.values, [x[1] for x in y_pred]
)

fpr_ipi, tpr_ipi, _ = roc_curve(y_test_specific.values, [x[1] for x in y_pred])

roc_auc_score_ipi = roc_auc_score(y_test_specific.values, [x[1] for x in y_pred])

y_pred = bst.predict_proba(X_test_specific).astype(float)

average_precision_score_dlbcl = average_precision_score(
    y_test_specific.values, [x[1] for x in y_pred]
)

precision_dlbcl, recall_dlbcl, _ = precision_recall_curve(
    y_test_specific.values, [x[1] for x in y_pred]
)

fpr_dlbcl, tpr_dlbcl, _ = roc_curve(y_test_specific.values, [x[1] for x in y_pred])

roc_auc_score_dlbcl = roc_auc_score(y_test_specific.values, [x[1] for x in y_pred])

bst = xgboost.XGBClassifier()

bst.load_model("results/models/model_all.json")


y_pred = bst.predict_proba(X_test_specific)


sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})
sns.set_style("white")

precision, recall, _ = precision_recall_curve(
    y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
)

plt.rcParams["figure.figsize"] = (11.7, 8.27)
fig, ax = plt.subplots()

display_NCCN = PrecisionRecallDisplay(
    precision,
    recall,
    average_precision=average_precision_score(
        y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
    ),
    estimator_name="NCCN IPI",
    #ax=ax
)
#fig = display_NCCN.plot()
display_NCCN.plot(ax=ax)
ax.set_aspect('auto')

precision, recall, _ = precision_recall_curve(
    y_test_specific.values, [x[1] for x in y_pred]
)
display_ML_model = PrecisionRecallDisplay(
    precision,
    recall,
    average_precision=average_precision_score(
        y_test_specific.values, [x[1] for x in y_pred]
    ),
    estimator_name="ML$_{\:All}$",
)
display_ML_model.plot(ax=ax)
ax.set_aspect('auto')

precision, recall, _ = precision_recall_curve(
    y_test_specific.values[indexes], weird_probabilities
)

display = PrecisionRecallDisplay(
    precision_dlbcl,
    recall_dlbcl,
    average_precision=average_precision_score_dlbcl,
    estimator_name="ML$_{\:DLBCL}$",
)
display.plot(ax=ax)
ax.set_aspect('auto')

display_IPI_model = PrecisionRecallDisplay(
    precision_ipi,
    recall_ipi,
    average_precision=average_precision_score_ipi,
    estimator_name="ML$_{\:IPI}$",
)
display_IPI_model.plot(ax=ax)
ax.set_aspect('auto')


precision, recall, _ = precision_recall_curve(
    y_test_specific.values, test_specific["lr_probs"]
)

display_LR_model = PrecisionRecallDisplay(
    precision,
    recall,
    average_precision=average_precision_score(
        y_test_specific.values, test_specific["lr_probs"]
    ),
    estimator_name="Logistic Regression",
)
display_LR_model.plot(ax=ax)
ax.set_aspect('auto')

precision, recall, _ = precision_recall_curve(
    y_test_specific.values, test_specific["tabpfn_probs"]
)

display_LR_model = PrecisionRecallDisplay(
    precision,
    recall,
    average_precision=average_precision_score(
        y_test_specific.values, test_specific["tabpfn_probs"]
    ),
    estimator_name="TabPFN",
)
display_LR_model.plot(ax=ax)
ax.set_aspect('auto')

pos_rate = y_test_specific.mean()  # proportion of positives

ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)

ax.hlines(pos_rate, xmin=0, xmax=1, colors="gray", linestyles="--", label=f"Chance (AP = {pos_rate:.2f})")

plt.xlabel("Recall", fontsize=15)
plt.ylabel("Precision", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
ax.grid(True, linestyle="--", alpha =0.3)


plt.savefig("plots/pr_auc.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/pr_auc.pdf", bbox_inches="tight")
plt.savefig("plots/pr_auc.svg", bbox_inches="tight")

# make NCCN and IPI comparison

precision_ml, recall_ml, thresholds_ml = precision_recall_curve(
    y_test_specific.values, [x[1] for x in y_pred]
)

precision_nccn, recall_nccn, thresholds_nccn = precision_recall_curve(
    y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
)

### find matches

recall_idx = (np.abs(recall_ml - recall_nccn[6])).argmin()

precision_idx = (np.abs(precision_ml - precision_nccn[6])).argmin()


thresholds_ml[precision_idx]

recall_ml[precision_idx]
precision_ml[precision_idx]

recall_nccn[6]

fig, ax = plt.subplots()

display_NCCN = PrecisionRecallDisplay(
    precision_nccn,
    recall_nccn,
    average_precision=average_precision_score(
        y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
    ),
    estimator_name="NCCN IPI",
)
fig = display_NCCN.plot(ax=ax)
ax.set_aspect('auto')


display_ML_model = PrecisionRecallDisplay(
    precision_ml,
    recall_ml,
    average_precision=average_precision_score(
        y_test_specific.values, [x[1] for x in y_pred]
    ),
    estimator_name="ML$_{\:All}$",
)
display_ML_model.plot(ax=ax)
ax.set_aspect('auto')

import seaborn as sns

helper_df = (
    pd.DataFrame(
        [recall_nccn[6], recall_ml[recall_idx]], [precision_nccn[6], precision_ml[recall_idx]]
    )
    .reset_index()
    .rename(columns={"index": "Precision", 0: "Recall"})
)

sns.lineplot(
    data=helper_df, x="Recall", y="Precision", dashes=(2, 2), ax=ax, color="black"
)

helper_df_2 = (
    pd.DataFrame(
        [recall_nccn[6], recall_ml[precision_idx]], [precision_nccn[6], precision_ml[precision_idx]]
    )
    .reset_index()
    .rename(columns={"index": "Precision", 0: "Recall"})
)

sns.lineplot(
    data=helper_df_2,
    x="Recall",
    y="Precision",
    dashes=(2, 2),
    ax=ax,
    color="black",
)
ax.set_aspect('auto')


pos_rate = y_test_specific.mean()  # proportion of positives

ax.hlines(pos_rate, xmin=-0.02, xmax=1.02, colors="gray", linestyles="--", label=f"Chance (AP = {pos_rate:.2f})")


# plt.plot([recall_nccn[6], recall_ml[precision_idx]],
#          [precision_nccn[6], precision_nccn[6]],
#          "k--", solid_capstyle="butt")

# plt.plot([recall_ml[recall_idx], recall_ml[recall_idx]],
#          [precision_nccn[6], precision_ml[recall_idx]],
#          "k--", solid_capstyle="butt")

# plt.scatter([recall_ml[precision_idx]], [precision_nccn[6]], color="black", zorder = 5, s = 5)
# plt.scatter([recall_nccn[6]], [precision_ml[recall_idx]], color="black", zorder = 5, s = 5)


plt.xlabel("Recall", fontsize=15)
plt.ylabel("Precision", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.ylim(bottom = 0.25)
plt.grid(alpha=0.3, linestyle='--', axis='y')

plt.savefig("plots/ml_compared_to_nccn_pr.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/ml_compared_to_nccn_pr.pdf", bbox_inches="tight")
plt.savefig("plots/ml_compared_to_nccn_pr.svg", bbox_inches="tight")

## SAME FOR AUC


fig, ax = plt.subplots()

fpr, tpr, _ = roc_curve(y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN)

display_NCCN = RocCurveDisplay(
    fpr=fpr,
    tpr=tpr,
    roc_auc=roc_auc_score(
        y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
    ),
    estimator_name="NCCN IPI",
)
display_NCCN.plot(ax = ax)

ax.set_aspect('auto')

fpr, tpr, _ = roc_curve(y_test_specific.values, [x[1] for x in y_pred])

display_ML_model = RocCurveDisplay(
    fpr=fpr,
    tpr=tpr,
    roc_auc=roc_auc_score(y_test_specific.values, [x[1] for x in y_pred]),
    estimator_name="ML$_{\:All}$",
)
display_ML_model.plot(ax = ax)
ax.set_aspect('auto')

fpr, tpr, _ = roc_curve(y_test_specific.values[indexes], weird_probabilities)

display = RocCurveDisplay(
    fpr=fpr_dlbcl,
    tpr=tpr_dlbcl,
    roc_auc=roc_auc_score_dlbcl,
    estimator_name="ML$_{\:DLBCL}$",
)
display.plot(ax = ax)
ax.set_aspect('auto')

display_IPI_model = RocCurveDisplay(
    fpr=fpr_ipi,
    tpr=tpr_ipi,
    roc_auc=roc_auc_score_ipi,
    estimator_name="ML$_{\:IPI}$",
)
display_IPI_model.plot(ax = ax)
ax.set_aspect('auto')

fpr, tpr, _ = roc_curve(y_test_specific, test_specific["lr_probs"])

display = RocCurveDisplay(
    fpr=fpr,
    tpr=tpr,
    roc_auc=roc_auc_score(y_test_specific, test_specific["lr_probs"]),
    estimator_name="Logistic Regression",
)
display.plot(ax = ax)
ax.set_aspect('auto')

fpr, tpr, _ = roc_curve(y_test_specific, test_specific["tabpfn_probs"])

display = RocCurveDisplay(
    fpr=fpr,
    tpr=tpr,
    roc_auc=roc_auc_score(y_test_specific, test_specific["lr_probs"]),
    estimator_name="TabPFN",
)
display.plot(ax = ax)
ax.set_aspect('auto')

ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance (AUC = 0.50)")

plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
ax.grid(True, linestyle="--", alpha =0.3)


ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)


plt.savefig("plots/roc_auc.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/roc_auc.pdf", bbox_inches="tight")
plt.savefig("plots/roc_auc.svg", bbox_inches="tight")


y_pred = bst.predict_proba(X_test_specific).astype(float)
y_pred = [1 if x[1] > 0.3 else 0 for x in y_pred]
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


y_pred = bst.predict_proba(X_test).astype(float)
y_pred = [1 if x[1] > 0.5 else 0 for x in y_pred]
y_test_specific = test[outcomes[-1]]

f1 = f1_score(y_test.values, y_pred)
roc_auc = roc_auc_score(
    y_test.values,
    bst.predict_proba(X_test).astype(float)[:, 1],
)
recall = recall_score(y_test.values, y_pred)
specificity = recall_score(y_test.values, y_pred, pos_label=0)
precision = precision_score(y_test.values, y_pred)
pr_auc = average_precision_score(
    y_test.values,
    bst.predict_proba(X_test).astype(float)[:, 1],
)
mcc = matthews_corrcoef(y_test.values, y_pred)

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test.values, y_pred))


# for the IPI models

y_pred = [
    1 if x >= 6 else 0
    for x in test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"]
]

import math

y_pred = [
    0 if x < 4 else 1
    for x in test_specific[test_specific["CNS_IPI_diagnosis"].notnull()][
        "CNS_IPI_diagnosis"
    ]
]

y_pred = np.array(y_pred)

y_test_specific = test_specific[outcomes[-1]]

f1 = f1_score(y_test_specific.values[indexes], y_pred)
roc_auc = roc_auc_score(y_test_specific.values[indexes], weird_probabilities)
recall = recall_score(y_test_specific.values[indexes], y_pred)
specificity = recall_score(y_test_specific.values[indexes], y_pred, pos_label=0)
precision = precision_score(y_test_specific.values[indexes], y_pred)
pr_auc = average_precision_score(y_test_specific.values[indexes], weird_probabilities)
mcc = matthews_corrcoef(y_test_specific.values[indexes], y_pred)

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test_specific.values[indexes], y_pred))


y_test_specific = test_specific[outcomes[-1]]


y_pred = [
    1 if x >= 6 else 0
    for x in test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"][indexes_NCCN]
]


f1 = f1_score(y_test_specific.values[indexes_NCCN], y_pred)
roc_auc = roc_auc_score(y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN)
recall = recall_score(y_test_specific.values[indexes_NCCN], y_pred)
specificity = recall_score(y_test_specific.values[indexes_NCCN], y_pred, pos_label=0)
precision = precision_score(y_test_specific.values[indexes_NCCN], y_pred)
pr_auc = average_precision_score(
    y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
)
mcc = matthews_corrcoef(y_test_specific.values[indexes_NCCN], y_pred)

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test_specific.values[indexes_NCCN], y_pred))


import shap

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
    "Number of involved nodal sites (diagnosis)",
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
    "Ann Arbor Stage (diagnosis)",
    "Extranodal disease (diagnosis)",
]

[x for x in X_test_specific.columns if "extranodal" in x]

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
    "pred_RKKP_AA_stage_diagnosis_fallback_-1",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
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


plt.savefig("plots/shap_values_ipi_only.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/shap_values_ipi_only.pdf", bbox_inches="tight")
