import joblib, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix, f1_score
)
from xgboost import XGBClassifier

# Load model and data
bst = XGBClassifier()
bst.load_model("results/model_ipi_only.json")
data = pd.read_csv("data/test_specific_ml_ipi_and_comparators.csv")

y_true = data["outc_treatment_failure_label_within_0_to_730_days_max_fallback_0"]
y_pred_proba = data["y_pred_proba_ml_ipi"]
y_pred_label = (y_pred_proba >= 0.5).astype(int)

# Standard metrics
f1 = f1_score(y_true, y_pred_label)
roc_auc = roc_auc_score(y_true, y_pred_proba)
pr_auc = average_precision_score(y_true, y_pred_proba)
precision = precision_score(y_true, y_pred_label)
recall = recall_score(y_true, y_pred_label)
mcc = matthews_corrcoef(y_true, y_pred_label)
cm = confusion_matrix(y_true, y_pred_label)
print(f"F1={f1:.3f} ROC-AUC={roc_auc:.3f} PR-AUC={pr_auc:.3f}")
print(cm)

# Bootstrap confidence intervals
def stratified_bootstrap_metrics(y_true, y_proba, y_label, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_proba, y_label = map(np.array, (y_true, y_proba, y_label))
    pos_idx, neg_idx = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    metrics = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": [], "specificity": [], "mcc": []}
    for _ in tqdm(range(n_bootstraps)):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True)
        ])
        rng.shuffle(idx)
        yt, yp, yl = y_true[idx], y_proba[idx], y_label[idx]
        cm = confusion_matrix(yt, yl)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics["specificity"].append(tn / (tn + fp + 1e-9))
        metrics["roc_auc"].append(roc_auc_score(yt, yp))
        metrics["pr_auc"].append(average_precision_score(yt, yp))
        metrics["precision"].append(precision_score(yt, yl, zero_division=0))
        metrics["recall"].append(recall_score(yt, yl, zero_division=0))
        metrics["mcc"].append(matthews_corrcoef(yt, yl))
    return {m: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) for m, v in metrics.items()}

summary = stratified_bootstrap_metrics(y_true, y_pred_proba, y_pred_label)
print(summary)
