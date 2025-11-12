import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

from helpers.constants import *
from helpers.processing_helper import *
from helpers.sql_helper import *

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
sns.set_context("paper")
seed = 46
DLBCL_ONLY = False
os.makedirs("results", exist_ok=True)

def make_prediction_categorical(y_prob):
    if pd.isnull(y_prob):
        return None
    if y_prob < 0.1:
        return "Low"
    elif y_prob < 0.3:
        return "Low-Intermediate"
    elif y_prob < 0.65:
        return "Intermediate-High"
    else:
        return "High"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")
test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]
feature_matrix = pd.read_pickle("results/feature_matrix_all.pkl")

# Filter invalid patient IDs
wrong_patientids = WIDE_DATA[WIDE_DATA["age_diagnosis"].isna()]["patientid"]
feature_matrix = feature_matrix[~feature_matrix["patientid"].isin(wrong_patientids)].reset_index(drop=True)
feature_matrix.replace(-1, np.nan, inplace=True)

# Train/test split
test = feature_matrix[feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)
train = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)

if DLBCL_ONLY:
    train = train[train["pred_RKKP_subtype_fallback_-1"] == 0].reset_index(drop=True)

# ---------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------
outcome_column = [x for x in feature_matrix if "outc" in x]
outcome = outcome_column[-1]

col_to_leave = ["patientid", "timestamp", "pred_time_uuid", "group"]
col_to_leave.extend(outcome_column)
col_to_leave.extend([x for x in feature_matrix.columns if "NCCN_" in x])

features = list(pd.read_csv("results/feature_names_all.csv")["features"].values)

# Define supplemental columns BEFORE using them
supplemental_columns = [
    "pred_RKKP_hospital_fallback_-1",
    "pred_RKKP_subtype_fallback_-1",
    "pred_RKKP_sex_fallback_-1",
]

for col in supplemental_columns:
    if col not in features:
        features.append(col)

# Clip values
for col in tqdm(features, desc="Clipping feature values"):
    clip_values(train, test, col)

test_with_treatment = test.merge(WIDE_DATA[["patientid", "regime_1_chemo_type_1st_line"]])

# ---------------------------------------------------------------------
# Feature and outcome extraction
# ---------------------------------------------------------------------
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
)

# ---------------------------------------------------------------------
# Logistic Regression Pipeline
# ---------------------------------------------------------------------
logreg_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, solver="saga", random_state=seed))
])

logreg_pipeline.fit(X_train_smtom, y_train_smtom)

# ---------------------------------------------------------------------
# Evaluate performance
# ---------------------------------------------------------------------
results, best_threshold = check_performance_across_thresholds(X_test, y_test, logreg_pipeline, y_pred_proba=[])
f1, roc_auc, recall, specificity, precision, pr_auc, mcc = check_performance(X_test, y_test, logreg_pipeline, 0.5, y_pred_proba=[])

print(f"\n=== Logistic Regression (global test set) ===")
for name, val in zip(
    ["F1", "ROC-AUC", "Recall", "Specificity", "Precision", "PR-AUC", "MCC"],
    [f1, roc_auc, recall, specificity, precision, pr_auc, mcc]
):
    print(f"{name}: {val:.3f}")

ConfusionMatrixDisplay(confusion_matrix(y_test.values, logreg_pipeline.predict(X_test))).plot()

# Save predictions
joblib.dump(
    {"y_true": y_test, "y_pred_proba": logreg_pipeline.predict_proba(X_test)[:, 1]},
    "results/lr_predictions.pkl"
)

# ---------------------------------------------------------------------
# Stratified bootstrap metrics
# ---------------------------------------------------------------------
def stratified_bootstrap_metrics(y_true, y_pred_proba, y_pred_label, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_pred_proba, y_pred_label = map(np.array, (y_true, y_pred_proba, y_pred_label))
    pos_idx, neg_idx = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    metrics = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": [], "specificity": [], "mcc": []}

    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True)
        ])
        rng.shuffle(idx)
        yt, yp, yl = y_true[idx], y_pred_proba[idx], y_pred_label[idx]

        cm = confusion_matrix(yt, yl)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) else 0

        metrics["roc_auc"].append(roc_auc_score(yt, yp))
        metrics["pr_auc"].append(average_precision_score(yt, yp))
        metrics["precision"].append(precision_score(yt, yl, zero_division=0))
        metrics["recall"].append(recall_score(yt, yl, zero_division=0))
        metrics["specificity"].append(specificity)
        metrics["mcc"].append(matthews_corrcoef(yt, yl))

    def summarize(v):
        return {"mean": np.mean(v), "ci_lower": np.percentile(v, 2.5), "ci_upper": np.percentile(v, 97.5)}

    return {m: summarize(v) for m, v in metrics.items()}

# Evaluate bootstrap CIs
for dataset, (X, y) in {"Test": (X_test, y_test), "Specific": (X_test_specific, y_test_specific)}.items():
    y_pred_proba = logreg_pipeline.predict_proba(X)[:, 1]
    y_pred_label = (y_pred_proba >= 0.5).astype(int)
    res = stratified_bootstrap_metrics(y, y_pred_proba, y_pred_label)
    print(f"\n=== Bootstrap metrics ({dataset}) ===")
    for metric, stats in res.items():
        print(f"{metric}: {stats['mean']:.3f} (95% CI: {stats['ci_lower']:.3f}–{stats['ci_upper']:.3f})")

# ---------------------------------------------------------------------
# Compare LR vs XGBoost
# ---------------------------------------------------------------------
xgb = XGBClassifier(
    n_estimators=3000, learning_rate=0.01, max_depth=8, min_child_weight=3,
    gamma=0, subsample=1, colsample_bytree=0.9, objective="binary:logistic",
    reg_alpha=10, nthread=10, random_state=seed
)
xgb.fit(X_train_smtom, y_train_smtom)

test_specific["estimated_probability_LR"] = logreg_pipeline.predict_proba(X_test_specific)[:, 1]
test_specific["estimated_probability_XG"] = xgb.predict_proba(X_test_specific)[:, 1]
test_specific["ML_risk_group"] = pd.Categorical(
    test_specific["estimated_probability_XG"].apply(make_prediction_categorical),
    ordered=True,
    categories=["Low", "Low-Intermediate", "Intermediate-High", "High"],
)

# ---------------------------------------------------------------------
# Visualization: Scatter + Quadrant + Concordance
# ---------------------------------------------------------------------
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)}, style="white")
plt.figure()
sns.scatterplot(
    data=test_specific,
    y="estimated_probability_LR",
    x="estimated_probability_XG",
    hue=outcome,
    hue_order=[1, 0],
    palette=sns.color_palette("Set1"),
)
plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
plt.xlabel("Estimated probability (XGBoost)")
plt.ylabel("Estimated probability (Logistic Regression)")
plt.title("Comparison of ML models")
plt.show()
