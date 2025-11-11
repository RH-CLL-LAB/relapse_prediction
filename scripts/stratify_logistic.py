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

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    DetCurveDisplay,
)
import matplotlib.patches as mpatches

import shap

import joblib

from tqdm import tqdm
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from helpers.constants import *
from helpers.processing_helper import *
from helpers.sql_helper import *

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

sns.set_context("paper")

seed = 46

DLBCL_ONLY = False


WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]

feature_matrix = feature_matrix[~feature_matrix["patientid"].isin(wrong_patientids)].reset_index(drop = True)

feature_matrix.replace(-1, np.nan, inplace=True)

features = pd.read_csv("results/feature_names_all.csv")["features"].values
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)

if DLBCL_ONLY:
    train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[-1]
# outcome = outcome_column[0]
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

test_with_treatment = test.merge(
    WIDE_DATA[["patientid", "regime_1_chemo_type_1st_line"]]
)

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

bst = XGBClassifier(
    # missing=-1,
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


NCCN_IPIS = [
    "pred_RKKP_age_diagnosis_fallback_-1",
    "pred_RKKP_LDH_diagnosis_fallback_-1",
    "pred_RKKP_AA_stage_diagnosis_fallback_-1",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
    "pred_RKKP_PS_diagnosis_fallback_-1",
]

bst = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # handle NaN
    ("scaler", StandardScaler()),                  # scale features
    ("logreg", LogisticRegression(max_iter=1000, solver="saga"))
])

# Fit on training data
bst.fit(X_train_smtom, y_train_smtom)


results, best_threshold = check_performance_across_thresholds(X_test, y_test, bst, y_pred_proba=[])

f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test, y_test, bst, 0.5, y_pred_proba=[]
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

## SAVE THE PREDICITONS
outfile = "results/lr_predictions.pkl"
out = {
        "y_true": y_test,
        "y_pred_proba": bst.predict_proba(X_test)[:, 1] ,
    }
joblib.dump(out, outfile)



results, best_threshold = check_performance_across_thresholds(
    X_test_specific, y_test_specific, bst, y_pred_proba=[]
)

y_pred = bst.predict_proba(X_test_specific).astype(float)
y_pred = [1 if x[1] > 0.5 else 0 for x in y_pred]

test_specific["model_highrisk"] = y_pred

from sklearn.utils import resample

def stratified_bootstrap_metrics(
    y_true, y_pred_proba, y_pred_label, n_bootstraps=1000, seed=42
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

y_pred_proba_LR = bst.predict_proba(X_test_specific)[:, 1]

f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test_specific, y_test_specific, bst, 0.5, y_pred_proba=[]
)

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test_specific.values, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test_specific.values, y_pred)).plot()


bst = XGBClassifier(
    # missing=-1,
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


NCCN_IPIS = [
    "pred_RKKP_age_diagnosis_fallback_-1",
    "pred_RKKP_LDH_diagnosis_fallback_-1",
    "pred_RKKP_AA_stage_diagnosis_fallback_-1",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
    "pred_RKKP_PS_diagnosis_fallback_-1",
]

bst.fit(X_train_smtom, y_train_smtom)

y_pred_proba_xg = bst.predict_proba(X_test_specific)[:, 1]


test_specific["estimated_probability_LR"] = y_pred_proba_LR.astype(float)
test_specific["estimated_probability_XG"] = y_pred_proba_xg.astype(float)

test_specific["ML_risk_group"] = pd.Categorical(
    test_specific["estimated_probability_XG"].apply(make_prediction_categorical),
    ordered=True,
    categories=["Low", "Low-Intermediate", "Intermediate-High", "High"],
)

test_specific["estimated_probability_XG"] = pd.to_numeric(test_specific["estimated_probability_XG"]) 
test_specific["estimated_probability_LR"] = pd.to_numeric(test_specific["estimated_probability_LR"]) 

print(test_specific["estimated_probability_XG"].dtype)

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

figure_axis = sns.scatterplot(
    data=test_specific,
    y="estimated_probability_LR",
    x="estimated_probability_XG",
    hue=outcome,
    hue_order=[1, 0],
    #style=outcome,
    #jitter=0.35,
    palette=sns.color_palette("Set1"),
    #dodge=True,
    size=7,
)
plt.fill_between([0, 0.1], -0., 1, alpha=0.4, color=colors[0], label="Low")
plt.fill_between(
    [0.1, 0.3], -0., 1, alpha=0.4, color=colors[1], label="Low-Intermediate"
)
plt.fill_between(
    [0.3, 0.65], -0., 1, alpha=0.4, color=colors[2], label="Intermediate-High"
)
plt.fill_between([0.65, 1], -0., 1, alpha=0.4, color=colors[3], label="High")
# plt.hlines([0.1, 0.3, 0.65], 0, 1, colors="black", linestyles="dotted")
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
plt.xlabel("Estimated probability: ML$_{\:All}$ model")
plt.ylabel("Estimated probability: Logistic Regression model")
plt.plot([0,1],[0,1], linestyle="--", color="black", alpha=0.5)

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


# Thresholds for defining quadrants
thresh_x = 0.5
thresh_y = 0.5

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=test_specific,
    x="estimated_probability_XG",
    y="estimated_probability_LR",
    hue=outcome,
    palette={0: "steelblue", 1: "firebrick"},
    alpha=0.7,
    s=30
)

# Quadrant lines
plt.axvline(thresh_x, color="black", linestyle="--")
plt.axhline(thresh_y, color="black", linestyle="--")

# Labels
plt.xlabel("Estimated probability (ML model)")
plt.ylabel("Estimated probability (Logistic Regression)")
plt.title("Quadrant Analysis: Concordance vs Discordance")

plt.legend(title="Outcome", labels=["Treatment Success", "Treatment Failure"])
plt.tight_layout()
plt.show()

# Assign quadrants
def quadrant(row):
    if row["estimated_probability_XG"] >= thresh_x and row["estimated_probability_LR"] >= thresh_y:
        return "Both High"
    elif row["estimated_probability_XG"] < thresh_x and row["estimated_probability_LR"] < thresh_y:
        return "Both Low"
    elif row["estimated_probability_XG"] >= thresh_x and row["estimated_probability_LR"] < thresh_y:
        return "ML only High"
    else:
        return "LR only High"

test_specific["quadrant"] = test_specific.apply(quadrant, axis=1)

# Barplot
plt.figure(figsize=(7,5))
sns.countplot(
    data=test_specific,
    x="quadrant",
    hue=outcome,
    palette={0: "steelblue", 1: "firebrick"}
)

plt.xlabel("Quadrant")
plt.ylabel("Number of patients")
plt.title("Outcomes in concordant and discordant quadrants")
plt.legend(title="Outcome", labels=["Treatment Success", "Treatment Failure"])
plt.tight_layout()
plt.show()

# Define bins and labels
bins = [0.0, 0.1, 0.3, 0.65, 1.0]
labels = ["Low", "Low-Intermediate", "Intermediate-High", "High"]

# Assign groups
test_specific["ML_group"] = pd.cut(
    test_specific["estimated_probability_XG"], bins=bins, labels=labels, include_lowest=True
)
test_specific["LR_group"] = pd.cut(
    test_specific["estimated_probability_LR"], bins=bins, labels=labels, include_lowest=True
)

# Combine groups
test_specific["concordance"] = test_specific["ML_group"].astype(str) + " vs " + test_specific["LR_group"].astype(str)

plt.figure(figsize=(12,6))
sns.countplot(
    data=test_specific,
    x="concordance",
    hue=outcome,
    palette={0:"steelblue",1:"firebrick"}
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("ML risk group vs LR risk group")
plt.ylabel("Number of patients")
plt.title("Concordance and discordance between ML and Logistic Regression models")
plt.legend(title="Outcome", labels=["Treatment Success","Treatment Failure"])
plt.tight_layout()
plt.show()