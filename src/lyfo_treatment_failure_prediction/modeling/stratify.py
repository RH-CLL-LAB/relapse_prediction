"""
Train an XGBoost model (DLBCL-only training by default), evaluate on:
  1) all test patients (X_test, y_test)
  2) subtype-specific test set (X_test_specific, y_test_specific)

Also computes:
  - NCCN IPI comparators (scaled as probabilities and thresholded at 6)
  - Bootstrap 95% CIs for key metrics
  - Feature correlation visuals
  - SHAP summary and bar plots
  - Saves useful CSVs and model artifacts
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

# ---- helpers (fixed imports) ----
from helpers.constants import supplemental_columns  # + colors if you need them elsewhere
from helpers.processing_helper import (
    get_features_and_outcomes,
    clip_values,
    plot_confusion_matrix,
    check_performance,
    check_performance_across_thresholds,
)

# ---- viz defaults ----
sns.set_context("paper")
plt.rcParams.update({"mathtext.default": "regular"})

# ---- params ----
seed = 46
DLBCL_ONLY = True  # train on DLBCL only

# ---- I/O and dirs ----
os.makedirs("plots", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("tables", exist_ok=True)
os.makedirs("data", exist_ok=True)

# =============================================================================
# 1) Load & prepare data
# =============================================================================
WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]

# remove missing-age patients
wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]
feature_matrix = (
    feature_matrix[~feature_matrix["patientid"].isin(wrong_patientids)]
    .reset_index(drop=True)
)
# replace sentinel -1 with NaN (robust for metrics/calibration, and corr will skip)
feature_matrix.replace(-1, np.nan, inplace=True)

# base features list
features = pd.read_csv("results/feature_names_all.csv")["features"].tolist()

# split
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)

# if requested, train only on DLBCL (subtype code assumed == 0)
if DLBCL_ONLY and "pred_RKKP_subtype_fallback_-1" in train.columns:
    train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

# pick outcome
outcome_columns = [c for c in feature_matrix.columns if "outc" in c]
# default: the first one (consistent with your original script)
outcome = outcome_columns[0]

# ensure supplemental columns are included
for col in supplemental_columns:
    if col not in features:
        features.append(col)

# robustify via clipping
for col in tqdm(features, desc="Clipping outliers"):
    clip_values(train, test, col)

# =============================================================================
# 2) Build matrices via helper
# =============================================================================
(
    X_train_smtom,
    y_train_smtom,
    X_test,
    y_test,
    X_test_specific,
    y_test_specific,
    test_specific,
) = get_features_and_outcomes(
    train=train,
    test=test,
    WIDE_DATA=WIDE_DATA,
    outcome=outcome,
    features=features,
    specific_immunotherapy=False,
    none_chop_like=False,
    only_DLBCL_filter=False,  # already filtered above if DLBCL_ONLY
)

# =============================================================================
# 3) Train model
# =============================================================================
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
bst.fit(X_train_smtom, y_train_smtom)

# save useful per-patient predictions for DLBCL model
y_pred_proba_specific = bst.predict_proba(X_test_specific)[:, 1].astype(float)
test_specific["ml_dlbcl_pred_proba"] = y_pred_proba_specific
test_specific.to_csv("data/test_specific_ml_dlbcl.csv", index=False)

# quick CM (DLBCL specific set) at 0.30, plus pretty plot
y_pred_label_specific_03 = (y_pred_proba_specific > 0.30).astype(int)
plot_confusion_matrix(confusion_matrix(y_test_specific.values, y_pred_label_specific_03))
plt.savefig("plots/cm_treatment_failure_2_years_ml_all_0.3_no_chop.pdf", bbox_inches="tight")

# =============================================================================
# 4) Metric sweeps + point estimates (helpers)
# =============================================================================
_ = check_performance_across_thresholds(X_test, y_test, bst, y_pred_proba=[])
f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test, y_test, bst, 0.5, y_pred_proba=[]
)
y_pred_test = (bst.predict_proba(X_test)[:, 1] > 0.5).astype(int)
print(f"\n=== All test (thr=0.50) ===")
print(f"F1: {f1:.3f} | ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")
print(f"Recall: {recall:.3f} | Precision: {precision:.3f} | Specificity: {specificity:.3f} | MCC: {mcc:.3f}")
print(confusion_matrix(y_test.values, y_pred_test))
ConfusionMatrixDisplay(confusion_matrix(y_test.values, y_pred_test)).plot()

_ = check_performance_across_thresholds(X_test_specific, y_test_specific, bst, y_pred_proba=[])
# your script used 0.59 here — keep that choice
y_pred_specific_059 = (y_pred_proba_specific > 0.59).astype(int)
test_specific["model_highrisk"] = y_pred_specific_059

f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(
    X_test_specific, y_test_specific, bst, 0.59, y_pred_proba=[]
)
print(f"\n=== DLBCL-specific test (thr=0.59) ===")
print(f"F1: {f1:.3f} | ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")
print(f"Recall: {recall:.3f} | Precision: {precision:.3f} | Specificity: {specificity:.3f} | MCC: {mcc:.3f}")
print(confusion_matrix(y_test_specific.values, y_pred_specific_059))
ConfusionMatrixDisplay(confusion_matrix(y_test_specific.values, y_pred_specific_059)).plot()

# =============================================================================
# 5) Bootstrap CIs (local helper)
# =============================================================================
def stratified_bootstrap_metrics(y_true, y_pred_proba, threshold, n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    def summarize(x):
        x = np.array(x, dtype=float)
        return {
            "mean": float(np.nanmean(x)),
            "ci_lower": float(np.nanpercentile(x, 2.5)),
            "ci_upper": float(np.nanpercentile(x, 97.5)),
        }

    out = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": [], "specificity": [], "mcc": []}

    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        bs = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        yb = y_true[bs]
        pb = y_pred_proba[bs]
        yp = (pb >= threshold).astype(int)

        cm = confusion_matrix(yb, yp, labels=[0, 1]).ravel()
        tn, fp, fn, tp = cm if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        try:
            out["roc_auc"].append(roc_auc_score(yb, pb))
            out["pr_auc"].append(average_precision_score(yb, pb))
            out["precision"].append(precision_score(yb, yp, zero_division=0))
            out["recall"].append(recall_score(yb, yp, zero_division=0))
            out["specificity"].append(specificity)
            out["mcc"].append(matthews_corrcoef(yb, yp))
        except Exception:
            continue

    return {k: summarize(v) for k, v in out.items()}

# All-test bootstrap (thr=0.30 here to mirror the earlier section; adjust if needed)
proba_all = bst.predict_proba(X_test)[:, 1]
thr_all = 0.30
ci_all = stratified_bootstrap_metrics(y_test, proba_all, thr_all)
print("\n=== All test bootstrap (thr=0.30) ===")
for k, s in ci_all.items():
    print(f"{k:12s}: {s['mean']:.3f} (95% CI: {s['ci_lower']:.3f}–{s['ci_upper']:.3f})")

# DLBCL-specific bootstrap (thr=0.50 as in your script’s later section)
proba_spec = y_pred_proba_specific
thr_spec = 0.50
ci_spec = stratified_bootstrap_metrics(y_test_specific, proba_spec, thr_spec)
print("\n=== DLBCL-specific bootstrap (thr=0.50) ===")
for k, s in ci_spec.items():
    print(f"{k:12s}: {s['mean']:.3f} (95% CI: {s['ci_lower']:.3f}–{s['ci_upper']:.3f})")

# =============================================================================
# 6) NCCN IPI comparators (scaled probabilities and label at >=6)
# =============================================================================
nccn_raw = test_specific["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"].copy()
nccn_scaled = (nccn_raw / 9).astype(float)  # keep NaN where missing
test_specific["nccn_ipi_pred_proba"] = nccn_scaled
test_specific.to_csv("data/test_specific_nccn_ipi.csv", index=False)

# evaluate where NCCN is present
idx_nccn = nccn_scaled.notna().to_numpy().nonzero()[0]
y_true_nccn = y_test_specific.iloc[idx_nccn].values
proba_nccn = nccn_scaled.iloc[idx_nccn].values
label_nccn = (nccn_raw.iloc[idx_nccn].values >= 6).astype(int)

ci_nccn = stratified_bootstrap_metrics(y_true_nccn, proba_nccn, threshold=0.5)  # threshold only used for discrete metrics
print("\n=== NCCN IPI (where available) ===")
for k, s in ci_nccn.items():
    print(f"{k:12s}: {s['mean']:.3f} (95% CI: {s['ci_lower']:.3f}–{s['ci_upper']:.3f})")

print("\nNCCN point metrics:")
print(f"PR-AUC: {average_precision_score(y_true_nccn, proba_nccn):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_true_nccn, proba_nccn):.3f}")
print(f"F1:     {f1_score(y_true_nccn, label_nccn):d}.{int(round((f1_score(y_true_nccn, label_nccn)%1)*1000)):03d}")
print(confusion_matrix(y_true_nccn, label_nccn))
ConfusionMatrixDisplay(confusion_matrix(y_true_nccn, label_nccn)).plot()
plot_confusion_matrix(confusion_matrix(y_true_nccn, label_nccn))
plt.savefig("plots/cm_treatment_failure_2_years_nccn_ipi.pdf", bbox_inches="tight")

# =============================================================================
# 7) Additional outcome variants (optional — keep minimal example)
# =============================================================================
# Example: use a different outcome if you need (kept from previous script)
# y_test_alt = test_specific[outcome_columns[3]]

# =============================================================================
# 8) Feature correlations (train set, pairwise-complete)
# =============================================================================
X_train_renamed = X_train_smtom.copy()  # already numeric; NaNs allowed
corr = X_train_renamed.corr(method="pearson", min_periods=2)
# histogram of absolute pairwise corr
upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).abs().stack()
bins = np.arange(0, 1.1, 0.1)
plt.figure(figsize=(6, 4))
plt.hist(upper, bins=bins, edgecolor="black", alpha=0.8)
plt.xlabel("Absolute correlation between feature pairs")
plt.ylabel("Number of feature pairs")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/feature_correlation_histogram.pdf", bbox_inches="tight")
plt.close()

# clustered heatmap
corr_clean = corr.replace([np.inf, -np.inf], np.nan).fillna(0)
sns.clustermap(
    corr_clean,
    cmap="vlag",
    center=0,
    linewidths=0.5,
    figsize=(20, 18),
    xticklabels=True,
    yticklabels=True,
    cbar_pos=(1.02, 0.4, 0.02, 0.4),
)
plt.savefig("plots/feature_correlation_heatmap.pdf", bbox_inches="tight")
plt.close()

# =============================================================================
# 9) SHAP (summary + bar + stability vs train)
# =============================================================================
import shap  # import only where needed to speed import
explainer = shap.TreeExplainer(bst)

# For readability in figs, optionally rename columns (if you have a mapping)
X_test_shap = X_test_specific.copy()

# summary plot
shap_values = explainer(X_test_shap)
fig = shap.summary_plot(shap_values, X_test_shap, max_display=20, show=False)
plt.savefig("plots/shap_values.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/shap_values.pdf", bbox_inches="tight")
plt.savefig("plots/shap_values.svg", bbox_inches="tight")
plt.close()

# bar plot (global importance)
fig = shap.plots.bar(shap_values, max_display=min(50, X_test_shap.shape[1]), show=False)
plt.savefig("plots/shap_values_bar.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/shap_values_bar.pdf", bbox_inches="tight")
plt.close()

# train vs test mean|SHAP|
shap_train = explainer.shap_values(X_train_smtom)
shap_test = explainer.shap_values(X_test_shap)
mean_train = np.abs(shap_train).mean(axis=0)
mean_test = np.abs(shap_test).mean(axis=0)

plt.figure(figsize=(8, 8))
plt.scatter(mean_train, mean_test, alpha=0.7)
lims = [min(mean_train.min(), mean_test.min()), max(mean_train.max(), mean_test.max())]
plt.plot(lims, lims, "k--", alpha=0.8)
plt.xlabel("Mean |SHAP| (Training set)")
plt.ylabel("Mean |SHAP| (Test set)")
plt.tight_layout()
plt.savefig("plots/shap_train_test_scatter.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/shap_train_test_scatter.pdf", bbox_inches="tight")
plt.close()

# top 20 features by test SHAP
top_idx = np.argsort(mean_test)[-20:]
feat_names = (
    list(X_train_smtom.columns)
    if hasattr(X_train_smtom, "columns")
    else [f"f{i}" for i in range(X_train_smtom.shape[1])]
)
df_bar = pd.DataFrame(
    {"Feature": np.array(feat_names)[top_idx], "Train": mean_train[top_idx], "Test": mean_test[top_idx]}
).sort_values("Test", ascending=False)
df_melt = df_bar.melt(id_vars="Feature", var_name="Dataset", value_name="Mean |SHAP|")
sns.set_style("whitegrid")
plt.figure(figsize=(11, 6))
sns.barplot(data=df_melt, x="Mean |SHAP|", y="Feature", hue="Dataset", dodge=True)
plt.xlabel("Mean |SHAP| Value")
plt.ylabel("")
plt.legend(title="Dataset")
plt.tight_layout()
plt.savefig("plots/shap_train_test_bar.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/shap_train_test_bar.pdf", bbox_inches="tight")
plt.close()

# =============================================================================
# 10) Save artifacts
# =============================================================================
bst.save_model("results/models/model_all.json")
test_specific.to_csv("results/test_specific.csv", index=False)
test.to_csv("results/test.csv", index=False)
X_test_specific.to_csv("results/X_test_specific.csv", index=False)
X_test.to_csv("results/X_test.csv", index=False)

print("\n✅ stratify.py complete. Plots saved to /plots and data to /results & /data.\n")
