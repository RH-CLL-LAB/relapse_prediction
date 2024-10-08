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


seed = 46

feature_matrix = pd.read_pickle("results/feature_matrix.pkl")

# split to have a completely untouched test set

feature_matrix, test = train_test_split(
    feature_matrix,
    test_size=0.15,
    stratify=feature_matrix["group"],
    random_state=seed,
)

outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[-1]
col_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"]
col_to_leave.extend(outcome_column)

ipi_cols = [x for x in feature_matrix.columns if "NCCN_" in x]
col_to_leave.extend(ipi_cols)

train, test = train_test_split(
    feature_matrix,
    test_size=0.20,
    stratify=feature_matrix["group"],
    random_state=seed,
)

predictor_columns = [x for x in train.columns if x not in col_to_leave]

predictor_columns = [
    x
    for x in predictor_columns
    if x not in ["pred_RKKP_subtype_fallback_-1", "pred_RKKP_hospital_fallback_-1"]
]


def clip_values(data: pd.DataFrame, column: str):
    relevant_data = data[data[column] != -1][column]

    lower_limit, upper_limit = relevant_data.quantile(0.05), relevant_data.quantile(
        0.95
    )
    data.loc[data[column] != -1, column] = data[data[column] != -1][column].clip(
        lower=lower_limit, upper=upper_limit
    )


for column in tqdm(predictor_columns):
    clip_values(train, column)
    clip_values(test, column)

smtom = SMOTETomek(random_state=seed)
X_train_smtom, y_train_smtom = smtom.fit_resample(
    X=train[[x for x in train.columns if x not in col_to_leave]], y=train[outcome]
)

X_train_smtom = train[[x for x in train.columns if x not in col_to_leave]]
y_train_smtom = train[outcome]


test_specific = test[test["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

included_treatments = ["chop", "choep", "maxichop"]  # "cop", "minichop"

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_specific = test_specific.merge(
    WIDE_DATA[["patientid", "regime_1_chemo_type_1st_line"]]
)

test_specific = test_specific[
    test_specific["regime_1_chemo_type_1st_line"].isin(included_treatments)
].reset_index(drop=True)

test_specific = test_specific.drop(columns="regime_1_chemo_type_1st_line")

X_test_specific = test_specific[
    [x for x in test_specific.columns if x not in col_to_leave]
]
y_test_specific = test_specific[outcome]

X_test = test[[x for x in test.columns if x not in col_to_leave]]
y_test = test[outcome]

bst = XGBClassifier(
    missing=-1,
    n_estimators=3000,  # was 2000 before
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=3,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    reg_alpha=10,
    nthread=10,
    random_state=seed,
)
# fit model

bst.fit(X_train_smtom, y_train_smtom)


def check_performance(X, y, threshold=0.5):
    y_pred = bst.predict_proba(X).astype(float)
    y_pred = [1 if x[1] > threshold else 0 for x in y_pred]

    f1 = f1_score(y.values, y_pred)
    roc_auc = roc_auc_score(y.values, y_pred)
    recall = recall_score(y.values, y_pred)
    specificity = recall_score(y.values, y_pred, pos_label=0)
    precision = precision_score(y.values, y_pred, zero_division=1)
    pr_auc = average_precision_score(y.values, y_pred)
    mcc = matthews_corrcoef(y.values, y_pred)
    return f1, roc_auc, recall, specificity, precision, pr_auc, mcc


import matplotlib.pyplot as plt
import seaborn as sns


def check_performance_across_thresholds(X, y):
    list_of_results = []
    for threshold in np.linspace(0, 1, num=100):
        f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
            X_test, y_test, threshold=threshold
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
    X_test, y_test, best_threshold
)
y_pred = bst.predict_proba(X_test).astype(float)
y_pred = [1 if x[1] > best_threshold else 0 for x in y_pred]

print(f"F1: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"PR-AUC: {pr_auc}")
print(f"MCC: {mcc}")
print(confusion_matrix(y_test.values, y_pred))

results, best_threshold = check_performance_across_thresholds(
    X_test_specific, y_test_specific
)

y_pred = bst.predict_proba(X_test_specific).astype(float)
y_pred = [1 if x[1] > best_threshold else 0 for x in y_pred]

f1 = f1_score(y_test_specific.values, y_pred)
roc_auc = roc_auc_score(y_test_specific.values, y_pred)
recall = recall_score(y_test_specific.values, y_pred)
specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
precision = precision_score(y_test_specific.values, y_pred)
pr_auc = average_precision_score(y_test_specific.values, y_pred)
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

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X_test.columns,
)

# compute SHAP values
explainer = shap.TreeExplainer(bst)
shap_values = explainer(X_test_specific)

shap.plots.beeswarm(shap_values[:, :], max_display=30, show=False)
plt.yticks()
# plt.xlim((-0.75, 0.75))

shap.plots.scatter(
    shap_values[:, "pred_RKKP_AA_stage_diagnosis_fallback_-1"], show=False
)

shap.plots.scatter(
    shap_values[
        :, "pred_poly_pharmacy_atc_level_5_within_0_to_1825_days_count_fallback_-1"
    ],
    show=False,
)

shap.plots.scatter(
    shap_values[
        :, "pred_poly_pharmacy_atc_level_1_within_0_to_1825_days_count_fallback_-1"
    ],
    show=False,
)


shap.plots.scatter(
    shap_values[
        :,
        "pred_PERSIMUNE_microbiology_analysis_Epstein-Barr Virus (EBV) IgG (EBNA)_within_0_to_1825_days_count_fallback_-1",
    ],
    show=False,
)


shap.plots.scatter(
    shap_values[
        :,
        "pred_RKKP_LDH_diagnosis_fallback_-1",
    ],
    show=False,
)
plt.xlim(-1, 4000)


shap.plots.scatter(
    shap_values[:, "pred_RKKP_sex_fallback_-1"],
    show=False,
)

results_dataframes = []
results_stratified = []
for i in range(5, 10):
    train, test = train_test_split(
        feature_matrix, test_size=0.2, random_state=i, stratify=feature_matrix["group"]
    )

    from tqdm import tqdm

    for column in tqdm(predictor_columns):
        clip_values(train, column)
        clip_values(test, column)

    smtom = SMOTETomek(random_state=seed)
    X_train_smtom, y_train_smtom = smtom.fit_resample(
        X=train[[x for x in train.columns if x not in col_to_leave]], y=train[outcome]
    )

    X_train_smtom = train[[x for x in train.columns if x not in col_to_leave]]
    y_train_smtom = train[outcome]
    bst = XGBClassifier(
        missing=-1,
        n_estimators=3000,  # was 2000 before
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        reg_alpha=10,
        nthread=10,  # was 6 before-could bumpt this for faster fitting probably?
        # scale_pos_weight=scale_pos_weight,
        random_state=i,
    )

    bst.fit(X_train_smtom, y_train_smtom)

    test_specific = test[test["pred_RKKP_subtype_fallback_-1"] == 0]

    included_treatments = ["chop", "choep", "maxichop"]

    WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

    test_specific = test_specific.merge(
        WIDE_DATA[["patientid", "regime_1_chemo_type_1st_line"]]
    )

    test_specific = test_specific[
        test_specific["regime_1_chemo_type_1st_line"].isin(included_treatments)
    ].reset_index(drop=True)

    test_specific = test_specific.drop(columns="regime_1_chemo_type_1st_line")

    X_test_specific = test_specific[
        [x for x in test_specific.columns if x not in col_to_leave]
    ]
    y_test_specific = test_specific[outcome]

    X_test = test[[x for x in test.columns if x not in col_to_leave]]
    y_test = test[outcome]

    results, best_threshold = check_performance_across_thresholds(
        X_test_specific, y_test_specific
    )

    results_dataframes.append(results)

    y_pred = bst.predict_proba(X_test_specific).astype(float)
    y_pred = [1 if x[1] > best_threshold else 0 for x in y_pred]

    f1 = f1_score(y_test_specific.values, y_pred)
    roc_auc = roc_auc_score(y_test_specific.values, y_pred)
    recall = recall_score(y_test_specific.values, y_pred)
    precision = precision_score(y_test_specific.values, y_pred, zero_division=1)
    specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
    pr_auc = average_precision_score(y_test_specific.values, y_pred)
    mcc = matthews_corrcoef(y_test_specific.values, y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(confusion_matrix(y_test_specific.values, y_pred))

    results_stratified.append(
        {
            "threshold": best_threshold,
            "f1": f1,
            "roc_auc": roc_auc,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "pr_auc": pr_auc,
            "mcc": mcc,
            "confusion_matrix": confusion_matrix(y_test_specific.values, y_pred),
            "seed": i,
        }
    )

import pickle as pkl

with open("results/results_stratified.pkl", "rb") as f:
    results_stratified = pkl.load(f)

with open("results/results_stratified.pkl", "wb") as f:
    pkl.dump(results_stratified, f)

for i, x in enumerate(results_stratified):
    if i == 0:
        confusion_summed = x["confusion_matrix"]
    else:
        confusion_summed += x["confusion_matrix"]

average_confusion_matrix = confusion_summed / 5

average_confusion_matrix

results_stratified_df = pd.DataFrame(results_stratified)

sns.boxplot(
    results_stratified_df[
        [
            x
            for x in results_stratified_df.columns
            if x not in ["confusion_matrix", "seed", "threshold"]
        ]
    ].melt(),
    x="variable",
    y="value",
)

results_stratified_df[
    [x for x in results_stratified_df.columns if x not in ["confusion_matrix", "seed"]]
].melt().groupby("variable").agg(mean=("value", "mean"), std=("value", "std"))

all_results = pd.concat(results_dataframes)

all_results = all_results.melt(id_vars="threshold")

sns.set_theme(context="paper", style="whitegrid")
sns.lineplot(
    data=all_results[all_results["variable"].isin(["mcc", "recall", "specificity"])],
    x="threshold",
    y="value",
    hue="variable",
)
# plt.xlim((0.1, 0.9))
