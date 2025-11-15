import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score, matthews_corrcoef, confusion_matrix
)
from xgboost import XGBClassifier

from helpers.constants import *
from helpers.processing_helper import *

sns.set_context("paper")

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
seed = 46
DLBCL_ONLY = False

# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------
WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]

feature_matrix = (
    pd.read_pickle("results/feature_matrix_all.pkl")
    .query("patientid not in @wrong_patientids")
    .reset_index(drop=True)
)
feature_matrix.replace(-1, np.nan, inplace=True)

test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)

if DLBCL_ONLY:
    train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

# ---------------------------------------------------------------------
# Outcome and feature setup
# ---------------------------------------------------------------------
outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[-1]
col_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"] + outcome_column
col_to_leave.extend([x for x in feature_matrix.columns if "NCCN_" in x])

features = list(pd.read_csv("results/feature_names_all.csv")["features"].values)
supplemental_columns = [
    "pred_RKKP_hospital_fallback_-1",
    "pred_RKKP_subtype_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
]
for col in supplemental_columns:
    if col not in features:
        features.append(col)

for col in tqdm(features, desc="Clipping feature values"):
    clip_values(train, test, col)

# ---------------------------------------------------------------------
# Prepare training/test matrices
# ---------------------------------------------------------------------
(
    X_train_smtom,
    y_train_smtom,
    X_test,
    y_test,
    X_test_specific,
    y_test_specific,
    test_specific,
) = get_features_and_outcomes(train, test, WIDE_DATA, outcome, features)

# ---------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------
bst = XGBClassifier(
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

# ---------------------------------------------------------------------
# Recursive feature elimination (RFECV)
# ---------------------------------------------------------------------
cv = StratifiedKFold(5, random_state=seed, shuffle=True)
rfecv = RFECV(
    estimator=bst,
    step=1,
    cv=cv,
    scoring="average_precision",
    min_features_to_select=1,
    n_jobs=10,
    verbose=1,
)
rfecv.fit(X_train_smtom, y_train_smtom)

with open("results/models/RFECV_model.pkl", "wb") as f:
    pickle.dump(rfecv, f)

print(f"Optimal number of features: {rfecv.n_features_}")

# Plot performance by number of features
n_feat = range(len(rfecv.cv_results_["mean_test_score"]))
plt.figure(figsize=(9, 6))
plt.errorbar(
    n_feat,
    rfecv.cv_results_["mean_test_score"],
    yerr=rfecv.cv_results_["std_test_score"],
    capsize=2,
)
plt.xlabel("Number of features selected")
plt.ylabel("Average Precision (PR-AUC)")
plt.grid(True)
plt.savefig("plots/recursive_feature_elimination.pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# Fixed-size RFE runs (30 and 50 features)
# ---------------------------------------------------------------------
def fit_rfe_model(n_features, X, y, out_path):
    selector = RFE(bst, n_features_to_select=n_features, step=1, verbose=True)
    selector.fit(X, y)
    with open(out_path, "wb") as f:
        pickle.dump(selector, f)
    return selector

selector_30 = fit_rfe_model(30, X_train_smtom, y_train_smtom, "results/models/RFE30_model.pkl")
selector_50 = fit_rfe_model(50, X_train_smtom, y_train_smtom, "results/models/RFE50_model.pkl")

features_30 = selector_30.get_feature_names_out()
features_50 = selector_50.get_feature_names_out()

# ---------------------------------------------------------------------
# Evaluate model using 30-feature subset
# ---------------------------------------------------------------------
bst.fit(X_train_smtom[features_30], y_train_smtom)
results, best_threshold = check_performance_across_thresholds(
    X_test_specific[features_30], y_test_specific, bst
)

# Fixed threshold evaluation
f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test_specific[features_30], y_test_specific, bst, 0.5
)
y_pred_probs = bst.predict_proba(X_test_specific[features_30])[:, 1]
y_pred = (y_pred_probs > 0.5).astype(int)

print(f"\n--- Model performance (30 features) ---")
print(f"F1: {f1:.3f} | ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")
print(f"Recall: {recall:.3f} | Precision: {precision:.3f} | Specificity: {specificity:.3f}")
print(f"MCC: {mcc:.3f}")
print(confusion_matrix(y_test_specific.values, y_pred))

test_specific["model_highrisk"] = y_pred

# ---------------------------------------------------------------------
# Compare ML vs NCCN high-risk
# ---------------------------------------------------------------------
test_specific["NCCN_highrisk"] = (
    test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] >= 6
).astype(int)

ml_rate = test_specific.loc[test_specific["model_highrisk"] == 1, outcome].mean()
nccn_rate = test_specific.loc[test_specific["NCCN_highrisk"] == 1, outcome].mean()
diff = ml_rate - nccn_rate
nnt = 1 / diff if diff != 0 else float("inf")

print(f"\nML high-risk event rate: {ml_rate:.3f}")
print(f"NCCN high-risk event rate: {nccn_rate:.3f}")
print(f"Absolute risk difference: {diff:.3f}")
print(f"Estimated NNT: {nnt:.1f}")

# ---------------------------------------------------------------------
# Correlation heatmap (30 selected features)
# ---------------------------------------------------------------------
renamed = X_train_smtom[features_30].copy()
corr = renamed.corr(numeric_only=True).fillna(0)
sns.clustermap(
    corr, cmap="vlag", center=0, linewidths=0.5, figsize=(20, 18),
    cbar_pos=(1.02, 0.4, 0.02, 0.4),
)
plt.title("Correlation between selected features")
plt.savefig("plots/feature_correlation_heatmap.pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------
explainer = shap.TreeExplainer(bst)
shap_values = explainer(X_test_specific[features_30])

# Summary plot
shap.summary_plot(
    shap_values, X_test_specific[features_30],
    max_display=20, show=False
)
plt.savefig("plots/shap_values.pdf", bbox_inches="tight")
plt.close()

# Bar plot
shap.plots.bar(shap_values, max_display=len(features_30), show=False)
plt.savefig("plots/shap_bar_values.pdf", bbox_inches="tight")
plt.close()

# Feature importance table
mean_abs = np.abs(shap_values.values).mean(0)
importance_df = (
    pd.DataFrame({"feature": features_30, "importance": mean_abs})
    .sort_values("importance", ascending=False)
)
importance_df["importance_rounded"] = importance_df["importance"].round(2)
importance_df["latex_string"] = importance_df.apply(
    lambda x: f"{x['feature']} & {x['importance_rounded']:.2f} \\\\", axis=1
)
importance_df.to_csv("tables/feature_importance_df.csv", sep=";", index=False)

# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------
bst.save_model("results/models/model_all.json")
test_specific.to_csv("results/test_specific.csv", index=False)
X_test_specific.to_csv("results/X_test_specific.csv", index=False)
X_test.to_csv("results/X_test.csv", index=False)
print("\n✅ Finished feature minimization and analysis.")
