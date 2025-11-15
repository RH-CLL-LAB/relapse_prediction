import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

sns.set_context("paper")
os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------------------
# Load data and models
# ---------------------------------------------------------------------
data = pd.read_csv("data/test_specific_ml_ipi_and_comparators.csv")
y_true = data["outc_treatment_failure_label_within_0_to_730_days_max_fallback_0"]

comparators = {}

# IPI-only model (already in CSV)
comparators["ML$_{IPI}$"] = data["y_pred_proba_ml_ipi"]

# Logistic Regression and TabPFN
if "lr_probs" in data.columns:
    comparators["Logistic Regression"] = data["lr_probs"]
if "tabpfn_probs" in data.columns:
    comparators["TabPFN"] = data["tabpfn_probs"]

# Full model (ML_All)
if os.path.exists("results/model_all.json"):
    bst_all = XGBClassifier()
    bst_all.load_model("results/model_all.json")
    X = data.filter(like="pred_")
    comparators["ML$_{All}$"] = bst_all.predict_proba(X)[:, 1]

# DLBCL-only model (ML_DLBCL)
if os.path.exists("results/model_dlbcl.json"):
    bst_dlbcl = XGBClassifier()
    bst_dlbcl.load_model("results/model_dlbcl.json")
    X = data.filter(like="pred_")
    comparators["ML$_{DLBCL}$"] = bst_dlbcl.predict_proba(X)[:, 1]

# NCCN and CNS IPI as clinical baselines, if available
if "pred_RKKP_NCCN_IPI_diagnosis_fallback_-1" in data.columns:
    comparators["NCCN IPI"] = data["pred_RKKP_NCCN_IPI_diagnosis_fallback_-1"] / 9.0
if "CNS_IPI_diagnosis" in data.columns:
    comparators["CNS IPI"] = data["CNS_IPI_diagnosis"] / 7.0

# ---------------------------------------------------------------------
# Precision–Recall curve
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for name, y_pred in comparators.items():
    mask = ~pd.isna(y_pred)
    display = PrecisionRecallDisplay.from_predictions(
        y_true[mask], y_pred[mask], name=name, ax=ax
    )
    display.plot(ax=ax)

pos_rate = y_true.mean()
ax.hlines(pos_rate, 0, 1, colors="gray", linestyles="--", label=f"Chance (AP = {pos_rate:.2f})")
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_xlabel("Recall", fontsize=14)
ax.set_ylabel("Precision", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("plots/pr_auc_full.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/pr_auc_full.pdf", bbox_inches="tight")
plt.savefig("plots/pr_auc_full.svg", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for name, y_pred in comparators.items():
    mask = ~pd.isna(y_pred)
    display = RocCurveDisplay.from_predictions(
        y_true[mask], y_pred[mask], name=name, ax=ax
    )
    display.plot(ax=ax)

ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance (AUC = 0.50)")
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_xlabel("False Positive Rate", fontsize=14)
ax.set_ylabel("True Positive Rate", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("plots/roc_auc_full.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/roc_auc_full.pdf", bbox_inches="tight")
plt.savefig("plots/roc_auc_full.svg", bbox_inches="tight")
plt.close()
