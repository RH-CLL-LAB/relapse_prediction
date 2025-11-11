"""
Paired bootstrap significance testing for ML metrics using precomputed
prediction probabilities.

Core idea: For each bootstrap replicate, resample the dataset (optionally
stratified), evaluate all models on the same resample, compute Δ = metric(model)
− metric(baseline) for chosen pairs, and form percentile CIs for Δ.

This avoids CI-overlap heuristics and does not assume normality.

Dependencies: numpy, pandas, scikit-learn

Example usage:

    import pandas as pd
    from paired_bootstrap_delta_significance import bootstrap_compare

    # df must include the true label column `y_true` and one column per model
    # containing predicted probabilities for the positive class
    df = pd.read_csv("my_probs.csv")

    results = bootstrap_compare(
        df=df,
        y_col="y_true",
        proba_cols={
            "NCCN_IPI": "nccn_proba",
            "ML_05_IPI": "ml05_ipi_proba",
            "ML_05_DLBCL": "ml05_dlbcl_proba",
            "ML_05_All": "ml05_all_proba",
        },
        baseline="NCCN_IPI",
        pairs=None,                     # None ⇒ compare every model vs baseline
        metrics=("roc_auc","pr_auc","precision","recall","specificity","mcc"),
        B=2000,
        alpha=0.05,
        thresholds=0.5,                 # fixed threshold used for thresholded metrics
        stratify=True,
        random_state=17,
        fdr_method="bh"                 # Benjamini–Hochberg on p-values (optional)
    )

    # results["delta"]: table of Δ estimates and CIs per metric × pair
    # results["levels"]: per-model performance with CIs (optional reporting)
    print(results["delta"].sort_values(["metric","pair"]))

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
)


# ---------------------------- Metric helpers ---------------------------- #

def _binarize(proba: np.ndarray, thr: float) -> np.ndarray:
    return (proba >= thr).astype(int)


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Confusion matrix order: tn, fp, fn, tp with labels=[0,1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = (tn + fp)
    return float(tn / denom) if denom > 0 else np.nan


MetricFunc = Callable[[np.ndarray, np.ndarray], float]


def make_metric_functions(
    thresholds: Union[float, Dict[str, float]] = 0.5,
) -> Dict[str, Callable[[np.ndarray, np.ndarray, Optional[str]], float]]:
    """Return metric evaluators given a threshold spec.

    thresholds: single float applied to all models for thresholded metrics, or
                dict mapping model_name -> threshold.
    Returns functions f(y_true, proba, model_name) -> metric.
    """

    def _thr_for(model: Optional[str]) -> float:
        if isinstance(thresholds, dict):
            if model is None:
                raise ValueError("Model name required when thresholds is a dict.")
            return float(thresholds[model])
        return float(thresholds)

    def roc_auc(y, p, m):
        # handle constant vectors robustly
        try:
            return float(roc_auc_score(y, p))
        except ValueError:
            return np.nan

    def pr_auc(y, p, m):
        # average_precision_score is a standard estimator of PR-AUC
        try:
            return float(average_precision_score(y, p))
        except ValueError:
            return np.nan

    def precision(y, p, m):
        thr = _thr_for(m)
        yp = _binarize(p, thr)
        return float(precision_score(y, yp, zero_division=0))

    def recall(y, p, m):
        thr = _thr_for(m)
        yp = _binarize(p, thr)
        return float(recall_score(y, yp, zero_division=0))

    def mcc(y, p, m):
        thr = _thr_for(m)
        yp = _binarize(p, thr)
        return float(matthews_corrcoef(y, yp))

    def specificity(y, p, m):
        thr = _thr_for(m)
        yp = _binarize(p, thr)
        return float(_specificity(y, yp))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "mcc": mcc,
    }


# ---------------------------- Bootstrap engine ---------------------------- #

@dataclass
class BootstrapResult:
    delta: pd.DataFrame  # CI and p-values for differences
    levels: pd.DataFrame  # optional: per-model metric levels with CIs


def _percentile_ci(arr: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo, hi = np.percentile(arr, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _bootstrap_pvalue_two_sided(deltas: np.ndarray) -> float:
    # Two-sided bootstrap p-value: 2 * min(Pr(Δ<=0), Pr(Δ>=0)) with small-sample correction
    n = len(deltas)
    le0 = np.sum(deltas <= 0)
    ge0 = np.sum(deltas >= 0)
    p = 2 * min((le0 + 1) / (n + 1), (ge0 + 1) / (n + 1))
    return min(1.0, float(p))


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    # Returns BH-adjusted p-values (FDR)
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    cummin = 1.0
    for i, idx in enumerate(order[::-1], start=1):
        rank = n - i + 1
        adj = p[idx] * n / rank
        cummin = min(cummin, adj)
        ranked[idx] = cummin
    return np.clip(ranked, 0, 1)


def bootstrap_compare(
    df: pd.DataFrame,
    y_col: str,
    proba_cols: Dict[str, str],
    baseline: str,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    metrics: Sequence[str] = ("roc_auc", "pr_auc", "precision", "recall", "specificity", "mcc"),
    B: int = 2000,
    alpha: float = 0.05,
    thresholds: Union[float, Dict[str, float]] = 0.5,
    stratify: bool = True,
    random_state: Optional[int] = None,
    fdr_method: Optional[str] = None,  # "bh" for Benjamini–Hochberg, or None
) -> BootstrapResult:
    """Paired bootstrap comparisons of models using precomputed probabilities.

    Parameters
    ----------
    df : DataFrame with y_col and one column per model (proba_cols values)
    y_col : name of binary ground-truth column (0/1 or False/True)
    proba_cols : mapping {model_name: column_name_with_positive_class_prob}
    baseline : model name used as comparator in Δ = model − baseline
    pairs : optional explicit list of (model, baseline_or_other). If None,
            compares every model != baseline against `baseline`.
    metrics : iterable of metric names: among
              {"roc_auc","pr_auc","precision","recall","specificity","mcc"}
    B : number of bootstrap replicates
    alpha : significance level for CIs (e.g., 0.05 for 95% CI)
    thresholds : float or {model: float} threshold(s) for thresholded metrics
    stratify : whether to bootstrap with class-stratified resampling
    random_state : optional seed for reproducibility
    fdr_method : optional multiple-testing correction ("bh" for FDR)

    Returns
    -------
    BootstrapResult with two tables:
      - delta: rows for each metric × pair, with Δ mean, CI, p-value, stars
      - levels: per-model per-metric levels with CIs (useful for the figure)
    """

    rng = np.random.default_rng(random_state)

    # Validate inputs
    assert baseline in proba_cols, f"baseline '{baseline}' not in proba_cols"
    model_names = list(proba_cols.keys())
    if pairs is None:
        pairs = [(m, baseline) for m in model_names if m != baseline]

    # Prepare data arrays
    y = df[y_col].astype(int).to_numpy()
    P = {m: df[c].to_numpy(dtype=float) for m, c in proba_cols.items()}

    # Metric functions
    mf = make_metric_functions(thresholds)
    bad = set(metrics) - set(mf.keys())
    if bad:
        raise ValueError(f"Unknown metrics: {sorted(bad)}")

    n = len(y)
    if stratify:
        idx_pos = np.flatnonzero(y == 1)
        idx_neg = np.flatnonzero(y == 0)

    # Storage
    # Levels per model/metric across bootstraps
    level_samples: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in metrics} for m in model_names}
    # Deltas per pair/metric across bootstraps
    delta_samples: Dict[Tuple[str, str], Dict[str, List[float]]] = {pair: {k: [] for k in metrics} for pair in pairs}

    for b in range(B):
        # Sample indices
        if stratify:
            pos_bs = rng.choice(idx_pos, size=idx_pos.size, replace=True)
            neg_bs = rng.choice(idx_neg, size=idx_neg.size, replace=True)
            bs_idx = np.concatenate([pos_bs, neg_bs])
        else:
            bs_idx = rng.choice(np.arange(n), size=n, replace=True)

        yb = y[bs_idx]
        # Evaluate all models on the same resample (paired bootstrap)
        perf: Dict[str, Dict[str, float]] = {}
        for m in model_names:
            pb = P[m][bs_idx]
            perf[m] = {k: mf[k](yb, pb, m) for k in metrics}
            for k in metrics:
                level_samples[m][k].append(perf[m][k])

        # Deltas for requested pairs
        for (m1, m2) in pairs:
            for k in metrics:
                v = perf[m1][k] - perf[m2][k]
                delta_samples[(m1, m2)][k].append(v)

    # Assemble levels summary
    level_rows = []
    for m in model_names:
        for k in metrics:
            arr = np.asarray(level_samples[m][k], dtype=float)
            mean = float(np.nanmean(arr))
            lo, hi = _percentile_ci(arr, alpha)
            level_rows.append({
                "model": m,
                "metric": k,
                "mean": mean,
                f"ci_low_{int((1-alpha)*100)}": lo,
                f"ci_high_{int((1-alpha)*100)}": hi,
            })
    levels_df = pd.DataFrame(level_rows)

    # Assemble delta summary
    delta_rows = []
    for pair in pairs:
        for k in metrics:
            arr = np.asarray(delta_samples[pair][k], dtype=float)
            mean = float(np.nanmean(arr))
            lo, hi = _percentile_ci(arr, alpha)
            pval = _bootstrap_pvalue_two_sided(arr)
            delta_rows.append({
                "pair": f"{pair[0]} - {pair[1]}",
                "model": pair[0],
                "baseline": pair[1],
                "metric": k,
                "delta_mean": mean,
                f"delta_ci_low_{int((1-alpha)*100)}": lo,
                f"delta_ci_high_{int((1-alpha)*100)}": hi,
                "p_value": pval,
            })
    delta_df = pd.DataFrame(delta_rows)

    # Optional multiple-testing correction
    if fdr_method is not None:
        if fdr_method.lower() == "bh":
            delta_df["p_adj"] = _benjamini_hochberg(delta_df["p_value"].to_numpy())
        else:
            raise ValueError("Unsupported fdr_method. Use 'bh' or None.")
        use_col = "p_adj"
    else:
        use_col = "p_value"

    # Stars based on the CI for Δ (excludes 0) — primary criterion
    def star_by_ci(lo: float, hi: float) -> str:
        return "★" if (lo > 0) or (hi < 0) else ""

    delta_df["star"] = [
        star_by_ci(row[f"delta_ci_low_{int((1-alpha)*100)}"], row[f"delta_ci_high_{int((1-alpha)*100)}"])  # type: ignore
        for _, row in delta_df.iterrows()
    ]

    # Optional secondary stars by adjusted p-value thresholds (if desired)
    def star_by_p(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    delta_df["stars_p"] = delta_df[use_col].apply(star_by_p)

    # Order columns nicely
    ci_low = f"delta_ci_low_{int((1-alpha)*100)}"
    ci_high = f"delta_ci_high_{int((1-alpha)*100)}"
    delta_df = delta_df[
        [
            "metric", "pair", "model", "baseline",
            "delta_mean", ci_low, ci_high, "p_value"
        ] + (["p_adj"] if "p_adj" in delta_df.columns else []) + ["star", "stars_p"]
    ].sort_values(["metric", "pair"]).reset_index(drop=True)

    return BootstrapResult(delta=delta_df, levels=levels_df)

# load data
import os

dir_list = os.listdir("data/")
test_specific_file_names = [x for x in dir_list if "test_specific" in x]

read_files = [pd.read_csv("data/"+x) for x in test_specific_file_names]

read_files[0].columns

only_relevant_columns = []

for i in read_files:
    colnames = [x for x in i.columns if "patientid" in x or "pred_proba" in x or "_probs" in x or "outc_" in x]
    only_relevant_columns.append(i[colnames])

data = pd.merge(only_relevant_columns[0], only_relevant_columns[1])
data = pd.merge(data, only_relevant_columns[2])
data = pd.merge(data, only_relevant_columns[3])

results = bootstrap_compare(
        df=data[data["nccn_ipi_pred_proba"].notna()],
        y_col="outc_succesful_treatment_label_within_0_to_730_days_max_fallback_0",
        proba_cols={
            "LR_05": "lr_probs",
            "NCCN_IPI": "nccn_ipi_pred_proba",
            "ML_05_IPI": "y_pred_proba_ml_ipi",
            "ML_05_DLBCL": "ml_dlbcl_pred_proba",
            "ML_05_All": "ml_all_pred_proba",
            "TabPFN_0.5": "tabpfn_probs"
        },
        baseline="NCCN_IPI",
        pairs=None,                     # None ⇒ compare every model vs baseline
        metrics=("roc_auc","pr_auc","precision","recall","specificity","mcc"),
        B=2000,
        alpha=0.05,
        thresholds={
            "LR_05": 0.5,
            "NCCN_IPI": 0.6,
            "ML_05_IPI": 0.5,
            "ML_05_DLBCL": 0.5,
            "ML_05_All": 0.5,
            "TabPFN_0.5": 0.5
        },                # fixed threshold used for thresholded metrics
        stratify=True,
        random_state=17,
        fdr_method="bh"                 # Benjamini–Hochberg on p-values (optional)
    )
pd.DataFrame(results.delta)

    # results["delta"]: table of Δ estimates and CIs per metric × pair
    # results["levels"]: per-model performance with CIs (optional reporting)
#print(results["delta"].sort_values(["metric","pair"]))

#### PREP FOR OTHER OUTCOMES

dir_list = os.listdir("data/individual_outcomes")
test_specific_file_names = [x for x in dir_list if "test_specific" in x]

read_files = [pd.read_csv("data/individual_outcomes/"+x) for x in test_specific_file_names]

only_relevant_columns = []

for i in read_files:
    colnames = [x for x in i.columns if "patientid" in x or "pred_proba" in x or "_probs" in x or "outc_" in x]
    only_relevant_columns.append(i[colnames])

data = pd.merge(data, only_relevant_columns[0])
data = pd.merge(data, only_relevant_columns[1])
data = pd.merge(data, only_relevant_columns[2])
data = pd.merge(data, only_relevant_columns[3])
data = pd.merge(data, only_relevant_columns[4])

#### FOR DEATH (2 years)


results = bootstrap_compare(
        df=data[data["nccn_ipi_pred_proba"].notna()],
        y_col="outc_dead_label_within_0_to_730_days_max_fallback_0",
        proba_cols={
            "NCCN_IPI": "nccn_ipi_pred_proba",
            "ML_03_All": "ml_all_pred_proba",
            "ML_05_All": "ml_all_pred_proba",
            "ML_05_Death": "ml_outc_dead_label_within_0_to_730_days_max_fallback_0"
        },
        baseline="NCCN_IPI",
        pairs=None,                     # None ⇒ compare every model vs baseline
        metrics=("roc_auc","pr_auc","precision","recall","specificity","mcc"),
        B=2000,
        alpha=0.05,
        thresholds={
            "NCCN_IPI": 0.6,
            "ML_05_All": 0.5,
            "ML_03_All": 0.3,
            "ML_05_Death": 0.5,
        },                # fixed threshold used for thresholded metrics
        stratify=True,
        random_state=17,
        fdr_method="bh"                 # Benjamini–Hochberg on p-values (optional)
    )

pd.DataFrame(results.delta)


#### FOR DEATH (5 years)


results = bootstrap_compare(
        df=data[data["nccn_ipi_pred_proba"].notna()],
        y_col="outc_dead_label_within_0_to_1825_days_max_fallback_0",
        proba_cols={
            "NCCN_IPI": "nccn_ipi_pred_proba",
            "ML_03_All": "ml_all_pred_proba",
            "ML_05_All": "ml_all_pred_proba",
            "ML_05_Death": "ml_outc_dead_label_within_0_to_1825_days_max_fallback_0"
        },
        baseline="NCCN_IPI",
        pairs=None,                     # None ⇒ compare every model vs baseline
        metrics=("roc_auc","pr_auc","precision","recall","specificity","mcc"),
        B=2000,
        alpha=0.05,
        thresholds={
            "NCCN_IPI": 0.6,
            "ML_05_All": 0.5,
            "ML_03_All": 0.3,
            "ML_05_Death": 0.5,
        },                # fixed threshold used for thresholded metrics
        stratify=True,
        random_state=17,
        fdr_method="bh"                 # Benjamini–Hochberg on p-values (optional)
    )

pd.DataFrame(results.delta)


#### FOR RELAPSE (2 years)


results = bootstrap_compare(
        df=data[data["nccn_ipi_pred_proba"].notna()],
        y_col="outc_relapse_within_0_to_730_days_max_fallback_0",
        proba_cols={
            "NCCN_IPI": "nccn_ipi_pred_proba",
            "ML_03_All": "ml_all_pred_proba",
            "ML_05_All": "ml_all_pred_proba",
            "ML_05_Relapse": "ml_outc_relapse_within_0_to_730_days_max_fallback_0"
        },
        baseline="NCCN_IPI",
        pairs=None,                     # None ⇒ compare every model vs baseline
        metrics=("roc_auc","pr_auc","precision","recall","specificity","mcc"),
        B=2000,
        alpha=0.05,
        thresholds={
            "NCCN_IPI": 0.6,
            "ML_05_All": 0.5,
            "ML_03_All": 0.3,
            "ML_05_Relapse": 0.5,
        },                # fixed threshold used for thresholded metrics
        stratify=True,
        random_state=17,
        fdr_method="bh"                 # Benjamini–Hochberg on p-values (optional)
    )

pd.DataFrame(results.delta)


#### FOR RELAPSE (5 years)


results = bootstrap_compare(
        df=data[data["nccn_ipi_pred_proba"].notna()],
        y_col="outc_relapse_within_0_to_1825_days_max_fallback_0",
        proba_cols={
            "NCCN_IPI": "nccn_ipi_pred_proba",
            "ML_03_All": "ml_all_pred_proba",
            "ML_05_All": "ml_all_pred_proba",
            "ML_05_Relapse": "ml_outc_relapse_within_0_to_1825_days_max_fallback_0"
        },
        baseline="NCCN_IPI",
        pairs=None,                     # None ⇒ compare every model vs baseline
        metrics=("roc_auc","pr_auc","precision","recall","specificity","mcc"),
        B=2000,
        alpha=0.05,
        thresholds={
            "NCCN_IPI": 0.6,
            "ML_05_All": 0.5,
            "ML_03_All": 0.3,
            "ML_05_Relapse": 0.5,
        },                # fixed threshold used for thresholded metrics
        stratify=True,
        random_state=17,
        fdr_method="bh"                 # Benjamini–Hochberg on p-values (optional)
    )

pd.DataFrame(results.delta)