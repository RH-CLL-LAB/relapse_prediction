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
from tqdm import tqdm

feature_matrix = pd.read_pickle("results/feature_matrix.pkl")


outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[1]
col_to_leave = [
    "patientid",
    "timestamp",
    "pred_time_uuid",
    outcome_column[0],
    outcome_column[1],
]

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


diseases = ["train_only_on_dlbcl", "train_on_all_diseases"]
outcomes = ["train_only_on_real_outcome", "train_on_proxy_outcome"]
feature_lengths = ["short_features", "long_features"]

results = []

for disease in diseases:
    for outcome_column in outcomes:
        for feature_length in feature_lengths:
            feature_matrix_copy = feature_matrix.copy()
            if disease == "train_only_on_dlbcl":
                feature_matrix_copy = feature_matrix_copy[
                    feature_matrix_copy["pred_RKKP_subtype_fallback_-1"] == 0
                ].reset_index(drop=True)
            if outcome_column == "train_only_on_real_outcome":
                feature_matrix_copy.loc[
                    feature_matrix_copy[
                        "outc_proxy_death_within_0_to_730_days_max_fallback_0"
                    ]
                    == 1,
                    "outc_relapse_within_0_to_730_days_max_fallback_0",
                ] = 0
            if feature_length == "short_features":
                non_long_features = [
                    x
                    for x in feature_matrix_copy.columns
                    if "_1825_" not in x and "_365_" not in x
                ]
                feature_matrix_copy = feature_matrix_copy[non_long_features]

            X = feature_matrix_copy[
                [x for x in feature_matrix_copy.columns if x not in col_to_leave]
            ]

            y = feature_matrix_copy[outcome]
            seed = 42

            print(
                f"Now running: \n disease: {disease} \n outcome: {outcome_column} \n feature_length: {feature_length}"
            )
            for i in tqdm(range(5)):
                seed += 1
                np.random.seed(seed)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=seed
                )

                (
                    dummy_X_train,
                    dummy_X_test,
                    dummy_y_train,
                    dummy_y_test,
                ) = train_test_split(
                    feature_matrix_copy,
                    feature_matrix_copy[
                        "outc_proxy_death_within_0_to_730_days_max_fallback_0"
                    ],
                    test_size=0.20,
                    random_state=seed,
                )

                indexer = dummy_y_test == 0

                bst = XGBClassifier(
                    missing=-1,
                    n_estimators=1000,  # 2000 was great, but it's just too slow
                    learning_rate=0.30,
                    max_depth=3,
                    min_child_weight=5,
                    gamma=0,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    nthread=6,
                    scale_pos_weight=1,
                )

                # ct = ColumnTransformer([])

                # pipe = Pipeline([("scaler", Standar)])

                # bst = RandomForestClassifier(class_weight="balanced")
                # bst = SVC(class_weight="balanced")
                # bst = RidgeClassifier(class_weight="balanced")
                # bst = DecisionTreeClassifier(class_weight="balanced")
                # fit model

                bst.fit(X_train, y_train)

                # NOTE: only for LYFO patients, but we need to see if it just predicts death

                X_test_specific = X_test[
                    (X_test["pred_RKKP_subtype_fallback_-1"] == 0) & (indexer)
                ].reset_index()  # need to exclude proxy stuff here

                X_test_specific = X_test[
                    X_test["pred_RKKP_subtype_fallback_-1"] == 0
                ].reset_index()  # need to exclude proxy stuff here

                index = X_test_specific["index"].values
                X_test_specific = X_test_specific[
                    [x for x in X_test_specific.columns if "index" not in x]
                ]

                y_test_specific = y_test[index]
                counter = 0
                for x_, y_ in [(X_test, y_test), (X_test_specific, y_test_specific)]:
                    y_pred = bst.predict(x_).astype(float)

                    f1 = f1_score(y_.values, y_pred)
                    roc_auc = roc_auc_score(y_.values, y_pred)
                    recall = recall_score(y_.values, y_pred)
                    precision = precision_score(y_.values, y_pred)
                    pr_auc = average_precision_score(y_.values, y_pred)
                    mcc = matthews_corrcoef(y_.values, y_pred)

                    cohort = "full_cohort"

                    if counter == 1:
                        cohort = "non_proxy_only_dlbcl"

                    result = {
                        "disease": disease,
                        "outcome": outcome_column,
                        "feature_length": feature_length,
                        "cohort": cohort,
                        "seed": seed,
                        "f1": f1,
                        "roc_auc": roc_auc,
                        "recall": recall,
                        "precision": precision,
                        "pr_auc": pr_auc,
                        "mcc": mcc,
                        "confusion_matrix": confusion_matrix(y_.values, y_pred),
                    }
                    results.append(result)

                    print(f"F1: {f1}")
                    print(f"ROC-AUC: {roc_auc}")
                    print(f"Recall: {recall}")
                    print(f"Precision: {precision}")
                    print(f"PR-AUC: {pr_auc}")
                    print(f"MCC: {mcc}")
                    print(confusion_matrix(y_.values, y_pred))
                    counter += 1
pd.DataFrame(results).to_pickle("results/initial_tests_test_death.pkl")


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

# import seaborn as sns

# sns.barplot(data=score_df, y="score", x="days")
# score_df

# sns.barplot(data=score_df, y="score", x="data_modality")


# sns.barplot(data=score_df, y="score", x="data_modality", estimator="max")

# score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)

# score_df

# feature_matrix

# import seaborn as sns

# order = score_df["index"].head(30)

# score_df.head(20)

# sns.barplot(data=score_df.head(30), x="score", y="index", order=order)
