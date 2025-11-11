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

from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    DetCurveDisplay,
)
import shap


from tqdm import tqdm
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from helpers.constants import *
from helpers.processing_helper import *
from helpers.sql_helper import *

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
#outcome = outcome_column[3]
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
    only_DLBCL_filter=False
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

bst.fit(X_train_smtom, y_train_smtom)
y_pred = bst.predict_proba(X_test_specific).astype(float)
y_pred_label = [1 if x[1] > 0.3 else 0 for x in y_pred]
plot_confusion_matrix(confusion_matrix(y_test_specific.values, y_pred_label))
#plt.savefig("plots/cm_treatment_failure_2_years_ml_all_0.5_no_chop.pdf", bbox_inches="tight")
plt.savefig("plots/cm_treatment_failure_2_years_ml_all_0.3.pdf", bbox_inches="tight")


### for other lymphomas

for subtype_number, subtype_name in enumerate(pd.Categorical(WIDE_DATA["subtype"]).categories):
    for threshold in [0.3, 0.5]:

        test_specific = test[
            test["pred_RKKP_subtype_fallback_-1"] == subtype_number
        ].reset_index(drop=True)

        y_pred = bst.predict_proba(X_test_specific).astype(float)
        y_pred_label = [1 if x[1] > threshold else 0 for x in y_pred]
        plot_confusion_matrix(confusion_matrix(y_test_specific.values, y_pred_label))
        #plt.savefig("plots/cm_treatment_failure_2_years_ml_all_0.5_no_chop.pdf", bbox_inches="tight")
        plt.savefig(f"plots/cm_treatment_failure_2_years_ml_all_{threshold}_{subtype_name}.pdf", bbox_inches="tight")

for threshold in [0.3, 0.5]:
    test_specific = test[test["pred_RKKP_subtype_fallback_-1"] != 0].reset_index(drop=True)
    y_pred = bst.predict_proba(X_test_specific).astype(float)
    y_pred_label = [1 if x[1] > threshold else 0 for x in y_pred]
    plot_confusion_matrix(confusion_matrix(y_test_specific.values, y_pred_label))
    #plt.savefig("plots/cm_treatment_failure_2_years_ml_all_0.5_no_chop.pdf", bbox_inches="tight")
    plt.savefig(f"plots/cm_treatment_failure_2_years_ml_all_{threshold}_OL.pdf", bbox_inches="tight")


