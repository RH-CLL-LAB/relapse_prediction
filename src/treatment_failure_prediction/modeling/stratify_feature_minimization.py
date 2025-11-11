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

from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import StratifiedKFold


min_features_to_select = 1  # Minimum number of features to consider
cv = StratifiedKFold(5, random_state=seed)

rfecv = RFECV(
    estimator=bst,
    step=1,
    cv=cv,
    scoring="average_precision",
    min_features_to_select=min_features_to_select,
    n_jobs=10,
    verbose=1,
)
rfecv.fit(X_train_smtom, y_train_smtom)

import pickle

#
# Create your model here (same as above)
#

# Save to file in the current working directory
pkl_filename = "RFECV_model.pkl"
with open(pkl_filename, "wb") as file:
    pickle.dump(rfecv, file)

with open(pkl_filename, "rb") as file:
    rfecv = pickle.load(file)

print(f"Optimal number of features: {rfecv.n_features_}")

data = {
    key: value
    for key, value in rfecv.cv_results_.items()
    if key in ["mean_test_score", "std_test_score"]
}
data["n_features"] = np.array([x for x in range(len(X_train_smtom.columns))])
cv_results = pd.DataFrame(data)
plt.figure(figsize=(9, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Average precision (PR-AUC)")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
    capsize=2,
)
plt.grid(visible=True)
# plt.title("Recursive Feature Elimination \nwith correlated features")
plt.savefig("plots/recursive_feature_elimination.pdf")

X_train_smtom.columns[~rfecv.get_support()]

selector = RFE(bst, n_features_to_select=50, step=1)
selector.fit(X_train_smtom, y_train_smtom)

X_train_smtom.columns[~selector.get_support()]

selector.get_feature_names_out()

# Save to file in the current working directory
pkl_filename = "RFE50_model.pkl"
with open(pkl_filename, "wb") as file:
    pickle.dump(selector, file)


selector = RFE(bst, n_features_to_select=30, step=1, verbose=True)
selector.fit(X_train_smtom, y_train_smtom)

X_train_smtom.columns[~selector.get_support()]

selector.get_feature_names_out()

# Save to file in the current working directory
pkl_filename = "RFE30_model.pkl"

with open(pkl_filename, "rb") as file:
    selector = pickle.load(file)

with open(pkl_filename, "wb") as file:
    pickle.dump(selector, file)

features_30 = selector.get_feature_names_out()

features_30


pkl_filename = "RFE50_model.pkl"
with open(pkl_filename, "rb") as file:
    selector = pickle.load(file)

features_50 = selector.get_feature_names_out()


bst.fit(X_train_smtom[features_30], y_train_smtom)

results, best_threshold = check_performance_across_thresholds(
    X_test[features_30], y_test, bst
)

f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test[features_30], y_test, bst, 0.5
)
y_pred = bst.predict_proba(X_test[features_30]).astype(float)
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
    X_test_specific[features_30], y_test_specific, bst
)

y_pred = bst.predict_proba(X_test_specific[features_30]).astype(float)
y_pred = [1 if x[1] > 0.5 else 0 for x in y_pred]

test_specific["model_highrisk"] = y_pred

f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test_specific[features_30], y_test_specific, bst, 0.5
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


test_specific["y_pred"] = y_pred

outcomes = [x for x in test_specific.columns if "outc" in x]

test_specific["outc_succesful_treatment_label_within_0_to_1825_days_max_fallback_0"] = (
    test_specific[outcomes[1]] + test_specific[outcomes[3]]
).apply(lambda x: min(x, 1))

# test_specific = test_specific.merge(
#     WIDE_DATA[["patientid", "CNS_IPI_diagnosis"]]
# ).reset_index(drop=True)


y_pred = [
    1 if x >= 6 else 0
    for x in test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"]
]

test_specific["NCCN_highrisk"] = y_pred

# y_pred = [
#     0 if (pd.isnull(x)) or x < 4 else 1 for x in test_specific["CNS_IPI_diagnosis"]
# ]


y_pred = bst.predict_proba(X_test_specific).astype(float)
y_pred = [1 if x[1] > 0.2 else 0 for x in y_pred]

y_test_specific = test_specific[outcomes[1]]

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

# Filter to high-risk groups
ml_all_highrisk = test_specific[test_specific["model_highrisk"] == 1]
nccn_highrisk = test_specific[test_specific["NCCN_highrisk"] == 1]
# Compute event rates
ml_all_event_rate = ml_all_highrisk[outcome].mean()
nccn_event_rate = nccn_highrisk[outcome].mean()
# Compute absolute risk difference and NNT
absolute_risk_difference = ml_all_event_rate - nccn_event_rate
nnt = 1 / absolute_risk_difference if absolute_risk_difference != 0 else float("inf")
# Output results
print(f"ML_All event rate in high-risk group: {ml_all_event_rate:.3f}")
print(f"NCCN IPI event rate in high-risk group: {nccn_event_rate:.3f}")
print(f"Absolute risk difference: {absolute_risk_difference:.3f}")
print(f"Estimated NNT (ML_All vs NCCN IPI): {nnt:.1f}")

# COMORBIDITY AND AE

feature_names = [
    "Performance status (diagnosis)",
    "Age (diagnosis)",
    "LDH (diagnosis)",
    "Count of CT-scans of cerebrum (90 days)",
    "Count of MR-scans of cerebrum (365 days)",
    "Count of treatments with blood or blood products (1095 days)",
    "Count of treatments with relation to blood, hematopoietic organs lymphatic tissue (90 days)",
    "Sex",
    "Platelets (diagnosis)",
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

new_renaming_dict = {
    "pred_RKKP_n_extranodal_regions_diagnosis_fallback_-1": "Count of extranodal regions (diagnosis)",
    "pred_RKKP_tumor_diameter_diagnosis_fallback_-1": "Tumor diameter (diagnosis)",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1": "Extranodal disease (diagnosis)",
    "pred_RKKP_subtype_fallback_-1": "Lymphoma subtype (diagnosis)",
    "pred_RKKP_AA_stage_diagnosis_fallback_-1": "Ann Arbor Stage (diagnosis)",
    "pred_RKKP_HB_diagnosis_fallback_-1": "Hemoglobin (diagnosis)",
    "pred_RKKP_ALB_diagnosis_fallback_-1": "Albumin (diagnosis)",
    "pred_RKKP_IgM_uM_diagnosis_fallback_-1": "Immunoglobulin M (diagnosis)",
    "pred_labmeasurements_ALT_within_0_to_1825_days_count_fallback_-1": "Count of alanine transaminase tests (1095 days)",
    "pred_labmeasurements_LEUK_within_0_to_90_days_min_fallback_-1": "Minimum of leukocytes (90 days)",
    "pred_labmeasurements_LEUK_within_0_to_90_days_latest_fallback_-1": "Latest value of leukocytes (90 days)",
    "pred_labmeasurements_NEU_within_0_to_90_days_latest_fallback_-1": "Latest value of neutrophils (90 days)",
    "pred_labmeasurements_IGM_within_0_to_1825_days_min_fallback_-1": "Minimum of immunoglobulin M (1095 days)",
    "pred_labmeasurements_GLUC_within_0_to_90_days_min_fallback_-1": "Minimum of blood glucose (90 days)",
    "pred_labmeasurements_KOL_within_0_to_1825_days_max_fallback_-1": "Maximum of cholesterol (1095 days)",
    "pred_labmeasurements_LDL_within_0_to_1825_days_count_fallback_-1": "Count of low-density lipoprotein tests (1095 days)",
    "pred_labmeasurements_HAPTO_within_0_to_90_days_count_fallback_-1": "Count of haptoglobin tests (90 days)",
    "pred_pathology_concat_bone_marrow_within_0_to_90_days_count_fallback_-1": "Count of pathology codes referring to bone marrow (90 days)",
    "pred_pathology_concat_no_problem_cells_within_0_to_365_days_count_fallback_-1": "Count of pathology codes referring to no problem cells (365 days)",
    "pred_pathology_concat_no_problem_cells_within_0_to_1825_days_count_fallback_-1": "Count of pathology codes referring to no problem cells (1095 days)",
    "pred_pathology_concat_text_specification_within_0_to_90_days_count_fallback_-1": "Count of pathology codes with text specification (90 days)",
    "pred_ord_medicine_poly_pharmacy_atc_level_4_within_0_to_90_days_count_fallback_-1": "Unique count of level 4 ATC codes from prescription medicine (90 days)",
    "pred_ord_medicine_poly_pharmacy_atc_level_4_within_0_to_365_days_count_fallback_-1": "Unique count of level 4 ATC codes from prescription medicine (365 days)",
    "pred_diagnoses_all_comorbidity_all_within_0_to_1825_days_count_fallback_-1": "Unique count of diagnosis codes (1095 days)",
    "pred_sks_referals_Radiologiske procedurer_within_0_to_1825_days_count_fallback_-1": "Count of referals for radiological procedures (1095 days)",
    "pred_sks_referals_CT-skanning af abdomen_within_0_to_365_days_count_fallback_-1": "Count of referals for CT scans of abdomen (365 days)",
    "pred_sks_referals_Endoskopier genn. naturlige og kunstige legemsåbninger_within_0_to_365_days_count_fallback_-1": "Count of referals for endoscopy (365 days)",
    "pred_sks_referals_Klinisk kontrol_within_0_to_90_days_count_fallback_-1": "Count of referals for clinical control (90 days)",
    "pred_sks_referals_Kroppen_within_0_to_90_days_sum_fallback_-1": "Sum of days of referals regarding the body (90 days)",
    "pred_sks_referals_Op. på perifere kar og lymfesystem_within_0_to_90_days_sum_fallback_-1": "Sum of days of referals for surgical procedures on peripheral and lymphatic vessels (90 days)",
    "pred_sks_referals_Skriftlig kommunikation_within_0_to_365_days_count_fallback_-1": "Count of referals for written communication (365 days)",
    "pred_sks_referals_Klinisk undersøgelse_within_0_to_90_days_count_fallback_-1": "Count of referals for clinical examination (90 days)",
    "pred_sks_referals_Brysthule_comma_ lunger og respiration_within_0_to_365_days_count_fallback_-1": "Count of referals regarding chest cavity, lungs, or respiration (365 days)",
    "pred_sks_referals_Op. på fordøjelsesorganer og milt_within_0_to_1825_days_sum_fallback_-1": "Sum of days of referals for surgical procedures on digestive organs and spleen (1095 days)",
    "pred_sks_referals_Mammografi_comma_ screening_within_0_to_1825_days_count_fallback_-1": "Count of referals for mammography screening (1095 days)",
    "pred_sks_referals_Mammografi_comma_ screening_within_0_to_1825_days_sum_fallback_-1": "Sum of days of referals for mammography screening (1095 days)",
    "pred_sks_referals_Otoskopi_within_0_to_365_days_sum_fallback_-1": "Sum of days of referals for otoscopy (365 days)",
    "pred_sks_referals_Observation af patient efter undersøgelse/behandling_within_0_to_90_days_count_fallback_-1": "Count of referals for observation of patient after examination or treatment (90 days)",
    "pred_sks_referals_unique_niveau9_tekst_within_0_to_90_days_count_fallback_-1": "Unique count of referals (90 days)",
    "pred_sks_at_the_hospital_Basis cytostatisk behandling_within_0_to_365_days_count_fallback_-1": "Count of basic cytostatic treatment (365 days)",
    "pred_sks_at_the_hospital_Ultralyd-undersøgelser af abdomen_within_0_to_1825_days_count_fallback_-1": "Count of ultra-sound examinations of the abdomen (1095 days)",
    "pred_sks_at_the_hospital_Perifere kredsløb og lymfesystem_within_0_to_1825_days_count_fallback_-1": "Count of examinations regarding peripheral and lymphatic vessels (1095 days)",
    "pred_sks_at_the_hospital_Generelle pædagogiske interventioner_within_0_to_1825_days_count_fallback_-1": "Count of general pedagogical interventions (1095 days)",
    "pred_sks_at_the_hospital_Entero- og koloskopier_within_0_to_90_days_sum_fallback_-1": "Sum of days in hospital for enteroscopies and colonoscopies (90 days)",
    "pred_sks_at_the_hospital_Iltbehandling_within_0_to_365_days_count_fallback_-1": "Count of oxygen treatments (365 days)",
    "pred_SDS_pato_Knoglemarv_within_0_to_90_days_count_fallback_-1": "Count of bone marrow pathologies (90 days)",
    "pred_ordered_medicine_musculo-skeletal system_within_0_to_90_days_count_fallback_-1": "Count of ordered medicine regarding the musculo-skeletal system (90 days)",
    "pred_ordered_medicine_other analgesics and antipyretics_within_0_to_1825_days_count_fallback_-1": "Count of orders for other analgesics and antipyretics (1095 days)",
    "pred_ordered_medicine_paracetamol_within_0_to_1825_days_count_fallback_-1": "Count of paracetamol orders (1095 days)",
    "pred_ordered_medicine_diuretics_within_0_to_365_days_count_fallback_-1": "Count of diuretics orders (365 days)",
    "pred_ordered_medicine_lipid modifying agents_within_0_to_1825_days_count_fallback_-1": "Count of orders for lipid modifying agents (1095 days)",
    "pred_ordered_medicine_genito urinary system and sex hormones_within_0_to_1825_days_count_fallback_-1": "Count of orders for sex hormones and genito urinary system (1095 days)",
    "pred_ordered_medicine_mineral supplements_within_0_to_365_days_count_fallback_-1": "Count of orders for mineral supplements (365 days)",
    "pred_ordered_medicine_dicloxacillin_within_0_to_1825_days_count_fallback_-1": "Count of orders for dicloxacillin (1095 days)",
    "pred_ordered_medicine_angiotensin ii antagonists_comma_ plain_within_0_to_365_days_count_fallback_-1": "Count of orders for plain angiotensin II antagonists (365 days)",
    "pred_ordered_medicine_lansoprazole_within_0_to_1825_days_count_fallback_-1": "Count of orders for lansoprazole (1095 days)",
    "pred_PERSIMUNE_leukocytes_all_within_0_to_365_days_max_fallback_-1": "Max leukocyte count across all blood tests (365 days)",
    "pred_PERSIMUNE_microbiology_culture_rare_within_0_to_90_days_count_fallback_-1": "Count of rare microbiology cultures (90 days)",
    "pred_diagnoses_all_rare_within_0_to_365_days_count_fallback_-1": "Count of rare diagnoses (365 days)",
    "pred_diagnoses_all_Kontakt mhp radiologisk undersøgelse_within_0_to_1825_days_count_fallback_-1": "Count of contacts in preparation for radiological examinations (1095 days)",
    "pred_diagnoses_all_Diffust storcellet B-celle lymfom_within_0_to_90_days_count_fallback_-1": "Count of Diffuse Large B-Cell Lymphoma diagnoses (90 days)",
}


rename_dict = {
    feature_names_original[i]: feature_names[i] for i in range(len(feature_names))
}

rename_dict.update(new_renaming_dict)

X_test_specific_renamed = X_test_specific[features_30].rename(columns=rename_dict)

X_train_smtom_renamed = X_train_smtom[features_30].rename(columns=rename_dict)

# Initialize an empty DataFrame to store correlations
features = X_train_smtom_renamed.columns
corr_matrix = pd.DataFrame(index=features, columns=features, dtype=float)
# Compute pairwise correlations excluding -1 values
for i in tqdm(features):
    for j in features:
        valid_mask = (X_train_smtom_renamed[i] != -1) & (X_train_smtom_renamed[j] != -1)
        if valid_mask.sum() > 1:  # Need at least 2 points
            corr_matrix.loc[i, j] = X_train_smtom_renamed.loc[valid_mask, i].corr(
                X_train_smtom_renamed.loc[valid_mask, j]
            )
        else:
            corr_matrix.loc[i, j] = np.nan
# Plot heatmap

corr_clean = corr_matrix.astype(float).copy()
corr_clean = corr_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
plt.figure(figsize=(18, 14))
sns.clustermap(
    corr_clean.astype(float),
    cmap="vlag",
    center=0,
    linewidths=0.5,
    figsize=(20, 18),
    xticklabels=True,
    yticklabels=True,
    cbar_pos=(1.02, 0.4, 0.02, 0.4),
)
plt.title("Correlation Coefficient")
plt.savefig("plots/feature_correlation_heatmap.pdf", bbox_inches="tight")
plt.show()

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

feature_names = list(X_test_specific_renamed.columns)
vals = np.abs(shap_values.values).mean(0)

feature_importance = pd.DataFrame(
    list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
)
feature_importance.sort_values(
    by=["feature_importance_vals"], ascending=False, inplace=True
)

feature_importance_df = feature_importance.reset_index(drop=True)

feature_importance_df["feature_importance_vals_rounded"] = feature_importance_df[
    "feature_importance_vals"
].round(2)

feature_importance_df["latex_string"] = feature_importance_df.apply(
    lambda x: f"{x['col_name']} & {x['feature_importance_vals_rounded']:.2f} \\", axis=1
)

feature_importance_df["latex_string"]

feature_importance_df.to_csv("tables/feature_importance_df.csv", index=False, sep=";")


figure = shap.plots.bar(
    shap_values,
    # X_test_specific_renamed,
    # feature_names=feature_names,
    max_display=len(X_test_specific_renamed.columns),
    show=False,
)

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
