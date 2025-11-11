from helpers.constants import *
import pandas as pd
import datetime
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
import seaborn as sns
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def clip_values(train: pd.DataFrame, test: pd.DataFrame, column: str):
    relevant_data = train[train[column] != -1][column]

    lower_limit, upper_limit = relevant_data.quantile(0.01), relevant_data.quantile(
        0.99
    )
    train.loc[train[column] != -1, column] = train[train[column] != -1][column].clip(
        lower=lower_limit, upper=upper_limit
    )
    test.loc[test[column] != -1, column] = test[test[column] != -1][column].clip(
        lower=lower_limit, upper=upper_limit
    )


def check_performance(X, y, model, threshold=0.5, y_pred_proba=[]):
    if len(y_pred_proba):
        y_pred = [1 if x > threshold else 0 for x in y_pred_proba]
    else:
        y_pred_proba = model.predict_proba(X).astype(float)[:, 1]
        y_pred = [1 if x > threshold else 0 for x in y_pred_proba]

    f1 = f1_score(y.values, y_pred)
    roc_auc = roc_auc_score(y.values, y_pred_proba)
    recall = recall_score(y.values, y_pred)
    specificity = recall_score(y.values, y_pred, pos_label=0)
    precision = precision_score(y.values, y_pred, zero_division=1)
    pr_auc = average_precision_score(
        y.values, y_pred_proba
    )
    mcc = matthews_corrcoef(y.values, y_pred)
    return f1, roc_auc, recall, specificity, precision, pr_auc, mcc


def check_performance_across_thresholds(X, y, model, y_pred_proba = []):
    list_of_results = []
    for threshold in tqdm(np.linspace(0, 1, num=100)):
        f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
            X, y, model, threshold=threshold, y_pred_proba=y_pred_proba
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


# Define NCCN IPI calculation function
def calculate_NCCN_IPI(age, ldh, aa_stage, extranodal, ps):
    if any(map(math.isnan, [age, ldh, aa_stage, ps])):
        return pd.NA

    total_score = sum(
        [
            3 if age > 75 else 2 if age > 60 else 1 if age > 40 else 0,
            2
            if ldh / (255 if age >= 70 else 205) > 3
            else 1
            if ldh / (255 if age >= 70 else 205) > 1
            else 0,
            1 if aa_stage > 2 else 0,
            1 if extranodal == 1 else 0,
            1 if ps >= 2 else 0,
        ]
    )
    return total_score


def calculate_CNS_IPI(age, ldh, aa_stage, extranodal, ps, kidneys_diagnosis):
    # NOTE NOW RETURNING NANS FOR PATIENTS WITH MISSING VALUES

    if any(
        [
            math.isnan(age),
            math.isnan(ldh),
            math.isnan(aa_stage),
            # math.isnan(extranodal),
            math.isnan(ps),
            math.isnan(kidneys_diagnosis),
        ]
    ):
        return pd.NA
    total_score = 0
    if age > 60:
        total_score += 1
    if age < 70:
        upper_limit = 205
    else:
        upper_limit = 255

    if ldh > upper_limit:
        total_score += 1

    if ps > 1:
        total_score += 1

    if aa_stage > 2:
        total_score += 1
    if extranodal == 1:
        total_score += 1
    if kidneys_diagnosis:
        total_score += 1

    return total_score


def get_features_and_outcomes(
    train, test, WIDE_DATA, outcome, feature_list, specific_immunotherapy=False, none_chop_like = False, only_DLBCL_filter = False,
):
    X_train_smtom = train[[x for x in train.columns if x in feature_list]]
    y_train_smtom = train[outcome]

    test_specific = test[test["pred_RKKP_subtype_fallback_-1"] == 0]

    included_treatments = ["chop", "choep", "maxichop"]

    test_specific = test_specific.merge(
        WIDE_DATA[
            ["patientid", "regime_1_chemo_type_1st_line", "immunotherapy_type_1st_line"]
        ]
    )
    if only_DLBCL_filter == False:
        if none_chop_like:
            test_specific = test_specific[
            ~test_specific["regime_1_chemo_type_1st_line"].isin(included_treatments)
        ].reset_index(drop=True)
        else:
            test_specific = test_specific[
                test_specific["regime_1_chemo_type_1st_line"].isin(included_treatments)
            ].reset_index(drop=True)

        if specific_immunotherapy:
            test_specific = test_specific[
                test_specific["immunotherapy_type_1st_line"] == "rituximab"
            ].reset_index(drop=True)

    test_specific = test_specific.drop(
        columns=["regime_1_chemo_type_1st_line", "immunotherapy_type_1st_line"]
    )

    X_test_specific = test_specific[
        [x for x in test_specific.columns if x in feature_list]
    ]
    y_test_specific = test_specific[outcome]

    X_test = test[[x for x in test.columns if x in feature_list]]
    y_test = test[outcome]

    return (
        X_train_smtom,
        y_train_smtom,
        X_test,
        y_test,
        X_test_specific,
        y_test_specific,
        test_specific,
    )

def plot_confusion_matrix(cm, tick_labels = ["Treatment Success", "Treatment Failure"]):
    fig, ax = plt.subplots()

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=False, fmt="magma",
                xticklabels=tick_labels,
                yticklabels=tick_labels,
                cbar=True, ax=ax,
                vmin=0,
                vmax=1)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Row-normalized proportion", rotation=270, labelpad=15)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j]
            text_color = "white" if value < 0.5 else "black"
            ax.text(j+0.5, i+0.5,
                    f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                    ha="center", va="center", color=text_color)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    return fig