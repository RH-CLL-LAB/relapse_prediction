"""
Evaluate TabPFN model performance (global and subtype-specific),
produce performance metrics with bootstrap CIs, and save confusion matrices.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
)

# --- Custom helpers ---
from helpers.constants import supplemental_columns
from helpers.processing_helper import (
    get_features_and_outcomes,
    clip_values,
)
# Note: Move your stratified_bootstrap_metrics() helper into processing_helper.py
# if not already done.

sns.set_context("paper")
seed = 46
os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]

# Remove patients without diagnosis age
wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]
feature_matrix = feature_matrix[~feature_matrix["patientid"].isin(wrong_patientids)].reset_index(drop=True)
feature_matrix.replace(-1, np.nan, inplace=True)

# Train/test split
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)

# ---------------------------------------------------------------------
# 2. Setup features & outcome
# ---------------------------------------------------------------------
outcome_column = [c for c in feature_matrix.columns if "outc" in c]
outcome = outcome_column[-1]

features = pd.read_csv("results/feature_names_all.csv")["features"].tolist()

# Ensure supplemental columns exist
for col in supplemental_columns:
    if col not in features:
        features.append(col)

# Clip values for robustness
for col in tqdm(features, desc="Clipping outliers"):
    clip_values(train, test, col)

# ---------------------------------------------------------------------
# 3. Attach TabPFN predictions (align by patientid and pred_time_uuid)
# ---------------------------------------------------------------------
predictions = joblib.load("results/tabpfn_predictions.pkl")

if isinstance(predictions, dict) and "y_pred_proba" in predictions:
    y_pred_proba = np.asarray(predictions["y_pred_proba"]).astype(float)
else:
    raise ValueError("Invalid format: tabpfn_predictions.pkl must contain 'y_pred_proba' array.")

key_cols = ["patientid"]
if "pred_time_uuid" in test.columns:
    key_cols.append("pred_time_uuid")

pred_df = test[key_cols].copy()
pred_df["y_pred_proba"] = y_pred_proba[: len(pred_df)]
test = test.merge(pred_df, on=key_cols, how="left")

# ---------------------------------------------------------------------
# 4. Prepare outcome matrices
# ---------------------------------------------------------------------
(
    X_train,
    y_train,
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
)

# Re-attach predictions to test_specific via keys
test_specific = test_specific.merge(test[key_cols + ["y_pred_proba"]], on=key_cols, how="left")

# ---------------------------------------------------------------------
# 5. Metric helpers
# ---------------------------------------------------------------------
def metrics_from_proba(y_true, y_proba, thr=0.5):
    """Compute main classification metrics from probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cm": cm,
    }


def sweep_best_threshold(y_true, y_proba, scorer="mcc", n=501):
    """Find threshold maximizing a metric (MCC or F1)."""
    grid = np.linspace(0, 1, n)
    best_val, best_thr = -np.inf, 0.5
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        try:
            score = matthews_corrcoef(y_true, y_pred) if scorer == "mcc" else f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            score = np.nan
        if score > best_val:
            best_val, best_thr = score, t
    return best_thr


def stratified_bootstrap_metrics(y_true, y_proba, thr, n_bootstraps=1000, seed=46):
    """Bootstrap 95% CIs for classification metrics."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    idx_pos = np.flatnonzero(y_true == 1)
    idx_neg = np.flatnonzero(y_true == 0)

    metrics = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": [], "specificity": [], "mcc": []}

    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        bs = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        yb = y_true[bs]
        pb = y_proba[bs]
        yp = (pb >= thr).astype(int)
        cm = confusion_matrix(yb, yp, labels=[0, 1]).ravel()
        tn, fp, fn, tp = cm if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        try:
            metrics["roc_auc"].append(roc_auc_score(yb, pb))
            metrics["pr_auc"].append(average_precision_score(yb, pb))
            metrics["precision"].append(precision_score(yb, yp, zero_division=0))
            metrics["recall"].append(recall_score(yb, yp, zero_division=0))
            metrics["specificity"].append(specificity)
            metrics["mcc"].append(matthews_corrcoef(yb, yp))
        except Exception:
            continue

    def summary(x):
        x = np.array(x, dtype=float)
        return {
            "mean": float(np.nanmean(x)),
            "ci_lower": float(np.nanpercentile(x, 2.5)),
            "ci_upper": float(np.nanpercentile(x, 97.5)),
        }

    return {k: summary(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------
# 6. Evaluate TabPFN results
# ---------------------------------------------------------------------
print("\nEvaluating TabPFN probabilities...\n")

# --- All test patients ---
proba_all = test["y_pred_proba"].values
thr_all = sweep_best_threshold(y_test.values, proba_all)
m_all = metrics_from_proba(y_test.values, proba_all, thr_all)
ci_all = stratified_bootstrap_metrics(y_test.values, proba_all, thr_all)

# --- Subtype-specific subset ---
proba_spec = test_specific["y_pred_proba"].values
thr_spec = sweep_best_threshold(y_test_specific.values, proba_spec)
m_spec = metrics_from_proba(y_test_specific.values, proba_spec, thr_spec)
ci_spec = stratified_bootstrap_metrics(y_test_specific.values, proba_spec, thr_spec)

# ---------------------------------------------------------------------
# 7. Print & save results
# ---------------------------------------------------------------------
def print_metrics(title, m, ci):
    print(f"\n=== {title} ===")
    for k in ["f1", "roc_auc", "pr_auc", "precision", "recall", "specificity", "mcc"]:
        mean = ci[k]["mean"]
        lo = ci[k]["ci_lower"]
        hi = ci[k]["ci_upper"]
        print(f"{k:12s}: {mean:.3f} (95% CI: {lo:.3f}–{hi:.3f})")

print_metrics("All test patients", m_all, ci_all)
print_metrics("Subtype-specific subset", m_spec, ci_spec)

# ---------------------------------------------------------------------
# 8. Confusion matrices
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay(m_all["cm"]).plot(ax=ax, colorbar=False)
plt.title(f"All test (thr={thr_all:.2f})")
plt.tight_layout()
plt.savefig("plots/cm_tabpfn_all.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay(m_spec["cm"]).plot(ax=ax, colorbar=False)
plt.title(f"Subtype-specific (thr={thr_spec:.2f})")
plt.tight_layout()
plt.savefig("plots/cm_tabpfn_specific.pdf", bbox_inches="tight")

print("\n✅ Evaluation complete. Results and plots saved in /plots.\n")
