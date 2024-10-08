import optuna
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

# gotta concatenate everything by strings

feature_matrix, test = train_test_split(
    feature_matrix,
    test_size=0.15,
    stratify=feature_matrix["group"],
    random_state=46,
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


def objective(trial):
    results_stratified = []
    results_dataframes = []
    n_estimators = trial.suggest_int("n_estimators", 1, 5)
    max_depth = trial.suggest_int("max_depth", 2, 10)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 5)
    subsample = trial.suggest_int("subsample", 8, 10)
    colsample_bytree = trial.suggest_int("colsample_bytree", 8, 10)

    for i in range(5, 10):
        train, test = train_test_split(
            train,
            test_size=0.2,
            random_state=i,
            stratify=train["group"],
        )

        from tqdm import tqdm

        for column in predictor_columns:
            clip_values(train, column)
            clip_values(test, column)

        smtom = SMOTETomek(random_state=seed)
        X_train_smtom, y_train_smtom = smtom.fit_resample(
            X=train[[x for x in train.columns if x not in col_to_leave]],
            y=train[outcome],
        )

        X_train_smtom = train[[x for x in train.columns if x not in col_to_leave]]
        y_train_smtom = train[outcome]
        from xgboost import XGBClassifier

        bst = XGBClassifier(
            missing=-1,
            n_estimators=n_estimators * 1000,  # was 2000 before
            learning_rate=0.001,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=0,
            subsample=subsample / 10,
            colsample_bytree=colsample_bytree / 10,
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

        # test_specific = test_specific[
        #     test_specific["pred_RKKP_hospital_fallback_-1"].isin([4, 5, 7, 9, 10])
        # ]
        X_test_specific = test_specific[
            [x for x in test_specific.columns if x not in col_to_leave]
        ]
        y_test_specific = test_specific[outcome]

        X_test = test[[x for x in test.columns if x not in col_to_leave]]
        y_test = test[outcome]

        # feature_matrix_test_specific = feature_matrix_test_specific[
        #     feature_matrix_test_specific[
        #         "outc_proxy_death_within_0_to_730_days_max_fallback_0"
        #     ]
        #     == 0
        # ].reset_index(drop=True)

        list_of_results = []

        for threshold in np.linspace(0, 1, num=100):
            y_pred = bst.predict_proba(X_test_specific).astype(float)
            y_pred = [1 if x[1] > threshold else 0 for x in y_pred]

            f1 = f1_score(y_test_specific.values, y_pred)
            roc_auc = roc_auc_score(y_test_specific.values, y_pred)
            recall = recall_score(y_test_specific.values, y_pred)
            precision = precision_score(y_test_specific.values, y_pred, zero_division=1)
            specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
            pr_auc = average_precision_score(y_test_specific.values, y_pred)
            mcc = matthews_corrcoef(y_test_specific.values, y_pred)

            list_of_results.append(
                {
                    "threshold": threshold,
                    "f1": f1,
                    "roc_auc": roc_auc,
                    "recall": recall,
                    "precision": precision,
                    "specificity": specificity,
                    "pr_auc": pr_auc,
                    "mcc": mcc,
                }
            )

        results = pd.DataFrame(list_of_results)

        results_dataframes.append(results)

        best_threshold = results[results["mcc"] == results["mcc"].max()][
            "threshold"
        ].values[-1]

        # results = results.melt(id_vars="threshold")

        # import seaborn as sns

        # sns.lineplot(data=results, x="threshold", y="value", hue="variable")
        # import matplotlib.pyplot as plt

        # plt.show()

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

    mccs = [x["mcc"] for x in results_stratified]
    return sum(mccs) / len(mccs)


import joblib


def run_optimization(objective):
    study = optuna.create_study(study_name="minimization", direction="maximize")
    study.optimize(lambda trial: objective(trial=trial), n_trials=100, n_jobs=10)
    joblib.dump(study, "best_params.pkl")


run_optimization(objective=objective)
