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

pd.Categorical(WIDE_DATA["subtype"]).categories


# PLOT

test["pred_RKKP_subtype_fallback_-1"].value_counts()



plotting_data = pd.DataFrame(
    [
        {
            "subtype": "DLBCL",
            "count": 1060,
            "disease_specific_ap": 0.66,
            "all_ap": 0.66,
        },
        {"subtype": "FL", "count": 332, "disease_specific_ap": 0.38, "all_ap": 0.42},
        {"subtype": "cHL", "count": 303, "disease_specific_ap": 0.56, "all_ap": 0.58},
        {"subtype": "MZL", "count": 148, "disease_specific_ap": 0.35, "all_ap": 0.46},
        {"subtype": "MCL", "count": 139, "disease_specific_ap": 0.63, "all_ap": 0.62},
        {"subtype": "WM", "count": 107, "disease_specific_ap": 0.50, "all_ap": 0.48},
        {"subtype": "NHL", "count": 63, "disease_specific_ap": 0.46, "all_ap": 0.73, "all_ap_upper": 0.87, "all_ap_lower": 0.59},
        {"subtype": "OL", "count": 59, "disease_specific_ap": 0.50, "all_ap": 0.51, "all_ap_upper": 0.87, "all_ap_lower": 0.59},
        # {"subtype": "SLL", "count": 40, "disease_specific_ap": 0.58, "all_ap": 0.59},
        # {"subtype": "HD-LP", "count": 21, "disease_specific_ap": 0.58, "all_ap": 0.59},
    ]
)

# 0.53

plotting_data = pd.DataFrame(
    [
        {
            "subtype": "DLBCL",
            "count": 1060,
            "disease_specific_ap": 0.66,
            "all_ap": 0.67,
        },
        {"subtype": "FL", "count": 332, "disease_specific_ap": 0.38, "all_ap": 0.42},
        {"subtype": "cHL", "count": 303, "disease_specific_ap": 0.52, "all_ap": 0.58},
        {"subtype": "MZL", "count": 148, "disease_specific_ap": 0.41, "all_ap": 0.46},
        {"subtype": "MCL", "count": 139, "disease_specific_ap": 0.62, "all_ap": 0.62},
        {"subtype": "WM", "count": 107, "disease_specific_ap": 0.54, "all_ap": 0.48},
        {"subtype": "NHL", "count": 63, "disease_specific_ap": 0.42, "all_ap": 0.73},
        # {"subtype": "OL", "count": 59, "disease_specific_ap": 0.50, "all_ap": 0.51},
        # {"subtype": "SLL", "count": 40, "disease_specific_ap": 0.58, "all_ap": 0.51},
        # {"subtype": "HD-LP", "count": 21, "disease_specific_ap": 0.58, "all_ap": 0.59},
    ]
)

plotting_data = plotting_data.rename(
    columns={"subtype": "Subtype", "count": "Sample size"}
)

# Set white background and ticks
sns.set_style("white")

# Optional: use "notebook" context for standard sizing
sns.set_context("notebook", font_scale=0.85)

# Create your plot
fig, ax = plt.subplots(figsize=(11 * 0.6, 8 * 0.6))

# plt.figure(figsize=(11*0.7, 8*0.7))
# or "paper" / "talk" / "poster"
import matplotlib.patches as mpatches

plotting_data["size_scaled"] = np.sqrt(plotting_data["Sample size"])

# Scatter plot with size scaling
scatter = sns.scatterplot(
    data=plotting_data,
    x="disease_specific_ap",
    y="all_ap",
    size="Sample size",
    hue="Subtype",
    sizes=(100, 5000),
    # legend='full',
    legend=False,  # Turn off default size legend
    alpha=0.8,
    palette="tab10",
    ax=ax,
)
# Scatter plot with size scaling
scatter = sns.scatterplot(
    data=plotting_data,
    x="disease_specific_ap",
    y="all_ap",
    size=1,
    # hue='Subtype',
    # sizes=(100, 5000),
    color="black",
    marker="+",
    # legend='full',
    legend=False,  # Turn off default size legend
    alpha=1,
    palette="tab10",
    ax=ax,
)
# Manually add only the hue legend (colors)
# Get unique subtypes and plot empty handles for legend
subtypes = plotting_data["Subtype"].unique()
palette = sns.color_palette("tab10", len(subtypes))

handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=palette[i],
        markersize=10,
        label=subtypes[i],
    )
    for i in range(len(subtypes))
]

fig.legend(
    handles=handles,
    title="Subtype",
    #loc="upper left",
    frameon=True,
    labelspacing=0.7,
    bbox_to_anchor=(1.15, 0.7),
)

# Make all 4 spines visible
for spine in ax.spines.values():
    spine.set_visible(True)

# ✅ Force tick marks to be visible on all axes
ax.tick_params(
    axis="both",
    which="both",  # major and minor ticks
    direction="out",
    length=6,
    width=1,
    bottom=True,
    top=False,
    left=True,
    right=False,
)

# Diagonal reference line
max_val = max(plotting_data["disease_specific_ap"].max(), plotting_data["all_ap"].max())
ax.plot(
    [0.35 - 0.02, max_val + 0.02], [0.35 - 0.02, max_val + 0.02], "--", color="gray"
)

# Annotations (optional)
ax.text(
    0.44,
    0.64,
    "ML$_{\: All}$ model\nperforms better",
    ha="center",
    fontsize=10,
    alpha=0.7,
)
ax.text(
    0.65,
    0.40,
    "Subtype specific model\nperforms better",
    ha="center",
    fontsize=10,
    alpha=0.7,
)

arrow1 = mpatches.FancyArrow(
    0.53,
    0.54,
    -0.15 * 0.5,
    0.15 * 0.5,
    width=0.015,
    head_width=0.05,
    head_length=0.05,
    length_includes_head=True,
    color="gray",
    alpha=0.7,
)
arrow2 = mpatches.FancyArrow(
    0.54,
    0.53,
    0.15 * 0.5,
    -0.15 * 0.5,
    width=0.015,
    head_width=0.05,
    head_length=0.05,
    length_includes_head=True,
    color="gray",
    alpha=0.7,
)

fig.gca().add_patch(arrow1)
fig.gca().add_patch(arrow2)

ax.set_xlabel("PR-AUC for the subtype-specific models")
ax.set_ylabel("PR-AUC for the ML$_{\: All}$ model")
# plt.title('AP for Single-Cancer vs Pan-Cancer Models')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', labelspacing = 0.72, handletextpad=1.0, borderaxespad=0)

ax.set_xlim(0.35 - 0.01, max_val + 0.01)
ax.set_ylim(0.35 - 0.01, max_val + 0.01)
ax.tick_params(direction="out", length=10, width=3)

# Force all spines and ticks to show
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)

ax.tick_params(direction="out", length=6, width=1)

fig.tight_layout()
fig.savefig("plots/subtype_specific_comparison.svg", bbox_inches="tight")
fig.savefig("plots/subtype_specific_comparison.pdf", bbox_inches="tight")
fig.savefig("plots/subtype_specific_comparison.png", bbox_inches="tight")


sns.set_context("notebook", font_scale=1)

# Create your plot
fig, ax = plt.subplots(figsize=(11 * 0.5, 8 * 0.5))

# plt.figure(figsize=(11*0.7, 8*0.7))
# or "paper" / "talk" / "poster"
import matplotlib.patches as mpatches

plotting_data["size_scaled"] = np.sqrt(plotting_data["Sample size"])

# Scatter plot with size scaling
scatter = sns.scatterplot(
    data=plotting_data,
    x="disease_specific_ap",
    y="all_ap",
    size="Sample size",
    hue="Subtype",
    sizes=(100, 5000),
    # legend='full',
    legend=False,  # Turn off default size legend
    alpha=0.8,
    palette="tab10",
    ax=ax,
)
# Scatter plot with size scaling
scatter = sns.scatterplot(
    data=plotting_data,
    x="disease_specific_ap",
    y="all_ap",
    size=1,
    # hue='Subtype',
    # sizes=(100, 5000),
    color="black",
    marker="+",
    # legend='full',
    legend=False,  # Turn off default size legend
    alpha=1,
    palette="tab10",
    ax=ax,
)
# Manually add only the hue legend (colors)
# Get unique subtypes and plot empty handles for legend
subtypes = plotting_data["Subtype"].unique()
palette = sns.color_palette("tab10", len(subtypes))

handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=palette[i],
        markersize=10,
        label=subtypes[i],
    )
    for i in range(len(subtypes))
]

fig.legend(
    handles=handles,
    title="Subtype",
    #loc="upper left",
    frameon=True,
    labelspacing=0.7,
    bbox_to_anchor=(1.17, 0.78),
    fontsize = 10
)

# Make all 4 spines visible
for spine in ax.spines.values():
    spine.set_visible(True)

# ✅ Force tick marks to be visible on all axes
ax.tick_params(
    axis="both",
    which="both",  # major and minor ticks
    direction="out",
    length=6,
    width=1,
    bottom=True,
    top=False,
    left=True,
    right=False,
)

# Diagonal reference line
max_val = max(plotting_data["disease_specific_ap"].max(), plotting_data["all_ap"].max())
ax.plot(
    [0.35 - 0.02, max_val + 0.02], [0.35 - 0.02, max_val + 0.02], "--", color="gray"
)

# Annotations (optional)
ax.text(
    0.44,
    0.64,
    "ML$_{\: All}$ model\nperforms better",
    ha="center",
    fontsize=11,
    alpha=0.7,
)
ax.text(
    0.65,
    0.40,
    "Subtype specific model\nperforms better",
    ha="center",
    fontsize=11,
    alpha=0.7,
)

arrow1 = mpatches.FancyArrow(
    0.53,
    0.54,
    -0.15 * 0.5,
    0.15 * 0.5,
    width=0.015,
    head_width=0.05,
    head_length=0.05,
    length_includes_head=True,
    color="gray",
    alpha=0.7,
)
arrow2 = mpatches.FancyArrow(
    0.54,
    0.53,
    0.15 * 0.5,
    -0.15 * 0.5,
    width=0.015,
    head_width=0.05,
    head_length=0.05,
    length_includes_head=True,
    color="gray",
    alpha=0.7,
)

fig.gca().add_patch(arrow1)
fig.gca().add_patch(arrow2)

ax.set_xlabel("PR-AUC for the subtype-specific models")
ax.set_ylabel("PR-AUC for the ML$_{\: All}$ model")
# plt.title('AP for Single-Cancer vs Pan-Cancer Models')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', labelspacing = 0.72, handletextpad=1.0, borderaxespad=0)

ax.set_xlim(0.35 - 0.01, max_val + 0.01)
ax.set_ylim(0.35 - 0.01, max_val + 0.01)
ax.tick_params(direction="out", length=10, width=3)

# Force all spines and ticks to show
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)

ax.tick_params(direction="out", length=6, width=1)

fig.tight_layout()
fig.savefig("plots/subtype_specific_comparison_talk.svg", bbox_inches="tight")
fig.savefig("plots/subtype_specific_comparison_talk.pdf", bbox_inches="tight")
fig.savefig("plots/subtype_specific_comparison_talk.png", bbox_inches="tight")



plt.show()

pd.Categorical(WIDE_DATA["subtype"]).categories


# PLOT

test["pred_RKKP_subtype_fallback_-1"].value_counts()

subtype_number = 0

train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)


# train = train[train["pred_RKKP_subtype_fallback_-1"].isin([2,6])].reset_index(
#     drop=True
# )


# train = train[train["pred_RKKP_subtype_fallback_-1"] == subtype_number].reset_index(
#     drop=True
# )
features = pd.read_csv("results/feature_names_all.csv")["features"].values


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


test_specific = test[
    test["pred_RKKP_subtype_fallback_-1"] != 0
].reset_index(drop=True)

# test_specific = test[test["pred_RKKP_subtype_fallback_-1"].isin([2,6])].reset_index(drop = True)

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

plot_confusion_matrix(confusion_matrix(y_test_specific.values, y_pred))
plt.savefig("plots/cm_treatment_failure_2_years_ml_all_0.3_OL.pdf", bbox_inches="tight")


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
y_pred_proba = bst.predict_proba(X_test_specific)[:, 1]  # Get probabilities for class 1
y_pred_label = (y_pred_proba >= 0.3).astype(int)  # Apply 0.5 threshold (or whatever you used)

results = stratified_bootstrap_metrics(y_test_specific, y_pred_proba, y_pred_label)

for metric, stats in results.items():
    print(f"{metric}: {stats['mean']:.3f} (95% CI: {stats['ci_lower']:.3f}–{stats['ci_upper']:.3f})")


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
