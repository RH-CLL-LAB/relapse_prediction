from sklearn.model_selection import train_test_split, StratifiedKFold
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
from xgboost import XGBClassifier


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")

seed = 46
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")
# feature_matrix = feature_matrix[feature_matrix["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop = True)
features = pd.read_csv("results/feature_names_all.csv")["features"].values

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


def check_performance(X, y, threshold=0.5):
    y_pred = bst.predict_proba(X).astype(float)
    y_pred = [1 if x[1] > threshold else 0 for x in y_pred]

    f1 = f1_score(y.values, y_pred)
    roc_auc = roc_auc_score(
        y.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
    recall = recall_score(y.values, y_pred)
    specificity = recall_score(y.values, y_pred, pos_label=0)
    precision = precision_score(y.values, y_pred, zero_division=1)
    pr_auc = average_precision_score(
        y.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
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


results_dataframes = []
results_stratified = []
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(
    drop=True
)
train_splitter = train.copy()


skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
skf.get_n_splits(X=train[features], y=train["group"])

for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    train = train_splitter.iloc[train_index]
    test = train_splitter.iloc[test_index]

    X_train_smtom = train[[x for x in train.columns if x not in col_to_leave]]
    y_train_smtom = train[outcome]

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

    def calculate_CNS_IPI(age, ldh, aa_stage, extranodal, ps, kidneys_diagnosis):
        # NOTE NOW RETURNING NANS FOR PATIENTS WITH MISSING VALUES
        import math

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

    test_specific.loc[
        test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] == -1,
        "pred_RKKP_NCCN_IPI_diagnosis_fallback_-1",
    ] = None
    y_pred = [
        1 if x >= 6 else 0
        for x in test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"]
    ]

    # y_pred = [
    #     0 if (pd.isnull(x)) or x < 4 else 1 for x in test_specific["CNS_IPI_diagnosis"]
    # ]

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

    average_precision_score(y_test_specific.values[indexes], weird_probabilities)
    average_precision_score(
        y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
    )
    roc_auc_score(y_test_specific.values[indexes], weird_probabilities)
    roc_auc_score(y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN)

    f1 = f1_score(y_test_specific.values, y_pred)
    roc_auc = roc_auc_score(
        y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
    )
    recall = recall_score(y_test_specific.values, y_pred)
    precision = precision_score(y_test_specific.values, y_pred, zero_division=1)
    specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
    pr_auc = average_precision_score(
        y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
    )
    mcc = matthews_corrcoef(y_test_specific.values, y_pred)
    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    # print(f"Best Threshold: {best_threshold}")
    print(cm)

    results_stratified.append(
        {
            "threshold": 0.5,
            "f1": f1,
            "roc_auc": roc_auc,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "pr_auc": pr_auc,
            "mcc": mcc,
            "confusion_matrix": cm,
            "seed": i,
        }
    )

# CNS

for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    train = train_splitter.iloc[train_index]
    test = train_splitter.iloc[test_index]

    X_train_smtom = train[[x for x in train.columns if x not in col_to_leave]]
    y_train_smtom = train[outcome]

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

    def calculate_CNS_IPI(age, ldh, aa_stage, extranodal, ps, kidneys_diagnosis):
        # NOTE NOW RETURNING NANS FOR PATIENTS WITH MISSING VALUES
        import math

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

    weird_probabilities = (test_specific["CNS_IPI_diagnosis"] / 7).values
    # fix nans - should we exclude them? probably yes
    weird_probabilities = [
        (i, x) for i, x in enumerate(weird_probabilities) if pd.notnull(x)
    ]
    indexes = [x[0] for x in weird_probabilities]
    weird_probabilities = [x[1] for x in weird_probabilities]

    y_pred = [
        0 if x < 4 else 1
        for x in test_specific[test_specific["CNS_IPI_diagnosis"].notnull()][
            "CNS_IPI_diagnosis"
        ]
    ]

    weird_probabilities_NCCN = (
        test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] / 9
    ).values
    weird_probabilities_NCCN = [
        (i, x) for i, x in enumerate(weird_probabilities_NCCN) if pd.notnull(x)
    ]
    indexes_NCCN = [x[0] for x in weird_probabilities_NCCN]
    weird_probabilities_NCCN = [x[1] for x in weird_probabilities_NCCN]

    from sklearn.metrics import precision_recall_curve, roc_curve

    average_precision_score(y_test_specific.values[indexes], weird_probabilities)
    average_precision_score(
        y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN
    )
    roc_auc_score(y_test_specific.values[indexes], weird_probabilities)
    roc_auc_score(y_test_specific.values[indexes_NCCN], weird_probabilities_NCCN)

    f1 = f1_score(y_test_specific.values[indexes], y_pred)
    roc_auc = roc_auc_score(y_test_specific.values[indexes], weird_probabilities)
    recall = recall_score(y_test_specific.values[indexes], y_pred)
    precision = precision_score(
        y_test_specific.values[indexes], y_pred, zero_division=1
    )
    specificity = recall_score(y_test_specific.values[indexes], y_pred, pos_label=0)
    pr_auc = average_precision_score(
        y_test_specific.values[indexes], weird_probabilities
    )
    mcc = matthews_corrcoef(y_test_specific.values[indexes], y_pred)
    cm = confusion_matrix(y_test_specific.values[indexes], y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    # print(f"Best Threshold: {best_threshold}")
    print(cm)

    results_stratified.append(
        {
            "threshold": 0.5,
            "f1": f1,
            "roc_auc": roc_auc,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "pr_auc": pr_auc,
            "mcc": mcc,
            "confusion_matrix": cm,
            "seed": i,
        }
    )


results_stratified

results_dataframes = []
results_stratified = []
train_splitter = train.copy()
features_original = features.copy()

for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    train = train_splitter.iloc[train_index]
    test = train_splitter.iloc[test_index]
    features = features_original.copy()

    supplemental_columns = [
        "pred_RKKP_tumor_diameter_diagnosis_fallback_-1",
        # "pred_RKKP_b_symptoms_diagnosis_fallback_-1",
        "pred_RKKP_LDH_diagnosis_fallback_-1",
        "pred_RKKP_ALB_diagnosis_fallback_-1",
        "pred_RKKP_TRC_diagnosis_fallback_-1",
        "pred_RKKP_AA_stage_diagnosis_fallback_-1",
        "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
        # names[0],
        # names[2],
    ]
    features = list(features)

    for j in supplemental_columns:
        if j not in features:
            features.append(j)

    from tqdm import tqdm

    features = [
        "pred_RKKP_age_diagnosis_fallback_-1",
        "pred_RKKP_LDH_diagnosis_fallback_-1",  # needs to be normalized
        "pred_RKKP_AA_stage_diagnosis_fallback_-1",
        "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
        "pred_RKKP_PS_diagnosis_fallback_-1",
    ]
    # lab_measurement_features = [x for x in feature_matrix.columns if "labmeasurements" in x]

    for column in tqdm(features):
        clip_values(train, test, column)

    # supplemental_columns = [
    #     "pred_RKKP_hospital_fallback_-1",
    #     "pred_RKKP_subtype_fallback_-1",
    #     "pred_RKKP_sex_fallback_-1",
    # ]

    # features = list(features)
    # features.extend(supplemental_columns)

    X_train_smtom = train[[x for x in train.columns if x in features]]
    y_train_smtom = train[outcome]
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
        nthread=10,  # was 6 before-could bumpt this for faster fitting probably?
        # scale_pos_weight=scale_pos_weight,
        random_state=i,
    )

    # bst = HistGradientBoostingClassifier(learning_rate=0.01, max_iter=3000)
    # from sklearn.linear_model import LogisticRegression

    # bst = LogisticRegression(max_iter=3000)

    # bst = LGBMClassifier(learning_rate=0.01, n_estimators=3000)

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

    X_test_specific = test_specific[[x for x in test_specific.columns if x in features]]
    y_test_specific = test_specific[outcome]

    X_test = test[[x for x in test.columns if x in features]]
    y_test = test[outcome]

    results, best_threshold = check_performance_across_thresholds(
        X_test_specific, y_test_specific
    )
    tested_threshold = 0.5

    results_dataframes.append(results)

    y_pred = bst.predict_proba(X_test_specific).astype(float)
    y_pred = [1 if x[1] > tested_threshold else 0 for x in y_pred]

    f1 = f1_score(y_test_specific.values, y_pred)
    roc_auc = roc_auc_score(
        y_test_specific.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
    recall = recall_score(y_test_specific.values, y_pred)
    precision = precision_score(y_test_specific.values, y_pred, zero_division=1)
    specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
    pr_auc = average_precision_score(
        y_test_specific.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
    mcc = matthews_corrcoef(y_test_specific.values, y_pred)
    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(f"Best Threshold: {best_threshold}")
    print(cm)

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
            "confusion_matrix": cm,
            "seed": i,
        }
    )

results_dataframes = []
results_stratified = []
train_splitter = train.copy()
features_original = features.copy()
for i, (train_index, test_index) in enumerate(
    skf.split(train_splitter[features], train_splitter["group"])
):
    train = train_splitter.iloc[train_index]
    test = train_splitter.iloc[test_index]
    features = features_original.copy()
    # train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)
    # test = test[test["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

    supplemental_columns = [
        "pred_RKKP_tumor_diameter_diagnosis_fallback_-1",
        # "pred_RKKP_b_symptoms_diagnosis_fallback_-1",
        "pred_RKKP_LDH_diagnosis_fallback_-1",
        "pred_RKKP_ALB_diagnosis_fallback_-1",
        "pred_RKKP_TRC_diagnosis_fallback_-1",
        "pred_RKKP_AA_stage_diagnosis_fallback_-1",
        "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
        # names[0],
        # names[2],
    ]
    features = list(features)

    for j in supplemental_columns:
        if j not in features:
            features.append(j)

    from tqdm import tqdm

    # lab_measurement_features = [x for x in feature_matrix.columns if "labmeasurements" in x]

    for column in tqdm(features):
        clip_values(train, test, column)

    # supplemental_columns = [
    #     "pred_RKKP_hospital_fallback_-1",
    #     "pred_RKKP_subtype_fallback_-1",
    #     "pred_RKKP_sex_fallback_-1",
    # ]

    # features = list(features)
    # features.extend(supplemental_columns)

    X_train_smtom = train[[x for x in train.columns if x in features]]
    y_train_smtom = train[outcome]
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

    X_test_specific = test_specific[[x for x in test_specific.columns if x in features]]
    y_test_specific = test_specific[outcome]

    X_test = test[[x for x in test.columns if x in features]]
    y_test = test[outcome]

    results, best_threshold = check_performance_across_thresholds(
        X_test_specific, y_test_specific
    )
    tested_threshold = 0.3

    results_dataframes.append(results)

    y_pred = bst.predict_proba(X_test_specific).astype(float)
    y_pred = [1 if x[1] > tested_threshold else 0 for x in y_pred]

    f1 = f1_score(y_test_specific.values, y_pred)
    roc_auc = roc_auc_score(
        y_test_specific.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
    recall = recall_score(y_test_specific.values, y_pred)
    precision = precision_score(y_test_specific.values, y_pred, zero_division=1)
    specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
    pr_auc = average_precision_score(
        y_test_specific.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
    mcc = matthews_corrcoef(y_test_specific.values, y_pred)
    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(f"Best Threshold: {best_threshold}")
    print(cm)

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
            "confusion_matrix": cm,
            "seed": i,
        }
    )


results_dataframes = []
results_stratified = []
train_splitter = train.copy()
features_original = features.copy()
features_original = list(feature_matrix.columns)
features_original = [x for x in features_original if x not in col_to_leave]
supplemental_columns = [
    "pred_RKKP_hospital_fallback_-1",
    "pred_RKKP_subtype_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
]
features_original = [x for x in features_original if x not in supplemental_columns]
for i in range(5):
    features = features_original.copy()
    train, test = train_test_split(
        train_splitter, test_size=0.2, random_state=i, stratify=train_splitter["group"]
    )
    # train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)
    # test = test[test["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

    from tqdm import tqdm

    # lab_measurement_features = [x for x in feature_matrix.columns if "labmeasurements" in x]

    for column in tqdm(features):
        clip_values(train, test, column)

    # supplemental_columns = [
    #     "pred_RKKP_hospital_fallback_-1",
    #     "pred_RKKP_subtype_fallback_-1",
    #     "pred_RKKP_sex_fallback_-1",
    # ]

    features = list(features)
    features.extend(supplemental_columns)

    X_train_smtom = train[[x for x in train.columns if x in features]]
    y_train_smtom = train[outcome]
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

    X_test_specific = test_specific[[x for x in test_specific.columns if x in features]]
    y_test_specific = test_specific[outcome]

    X_test = test[[x for x in test.columns if x in features]]
    y_test = test[outcome]

    results, best_threshold = check_performance_across_thresholds(
        X_test_specific, y_test_specific
    )
    tested_threshold = 0.5

    results_dataframes.append(results)

    y_pred = bst.predict_proba(X_test_specific).astype(float)
    y_pred = [1 if x[1] > tested_threshold else 0 for x in y_pred]

    f1 = f1_score(y_test_specific.values, y_pred)
    roc_auc = roc_auc_score(
        y_test_specific.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
    recall = recall_score(y_test_specific.values, y_pred)
    precision = precision_score(y_test_specific.values, y_pred, zero_division=1)
    specificity = recall_score(y_test_specific.values, y_pred, pos_label=0)
    pr_auc = average_precision_score(
        y_test_specific.values, bst.predict_proba(X_test_specific).astype(float)[:, 1]
    )
    mcc = matthews_corrcoef(y_test_specific.values, y_pred)
    cm = confusion_matrix(y_test_specific.values, y_pred)

    print(f"F1: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"PR-AUC: {pr_auc}")
    print(f"MCC: {mcc}")
    print(f"Best Threshold: {best_threshold}")
    print(cm)

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
            "confusion_matrix": cm,
            "seed": i,
        }
    )


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
