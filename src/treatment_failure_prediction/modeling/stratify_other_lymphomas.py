# --- Config ---------------------------------------------------------------
SUBTYPE_COL = "pred_RKKP_subtype_fallback_-1"   # coded as in WIDE_DATA categories
OUTCOME = [c for c in feature_matrix.columns if "outc_" in c][-1]  # last outcome
N_BOOT = 1000
RANDOM_STATE = 46
PLOT_MIN = 0.35  # lower bound of axes

# --- Utilities ------------------------------------------------------------
from sklearn.metrics import average_precision_score
rng = np.random.default_rng(RANDOM_STATE)

def ap_with_ci(y, p, n_boot=N_BOOT):
    mask = (~pd.isna(y)) & (~pd.isna(p))
    y = y[mask].astype(int).values
    p = p[mask].astype(float).values
    if y.sum() == 0 or y.sum() == len(y):  # degenerate
        return np.nan, (np.nan, np.nan)
    ap = average_precision_score(y, p)
    # stratified bootstrap
    idx_pos = np.where(y==1)[0]
    idx_neg = np.where(y==0)[0]
    boots = []
    for _ in range(n_boot):
        bs = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True)
        ])
        boots.append(average_precision_score(y[bs], p[bs]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(ap), (float(lo), float(hi))

def fit_xgb(X, y, seed=RANDOM_STATE):
    clf = XGBClassifier(
        missing=-1,
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        subsample=1,
        colsample_bytree=0.9,
        objective="binary:logistic",
        reg_alpha=10,
        nthread=10,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf

# --- Prepare splits and features -----------------------------------------
features_list = list(pd.read_csv("results/feature_names_all.csv")["features"].values)

# Add your supplemental columns except the subtype itself (no variance within a subtype)
supplemental_columns = ["pred_RKKP_hospital_fallback_-1", "pred_RKKP_sex_fallback_-1"]
for col in supplemental_columns:
    if col not in features_list:
        features_list.append(col)

train_all = feature_matrix[~feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)
test_all  = feature_matrix[ feature_matrix["patientid"].isin(test_patientids)].reset_index(drop=True)

# Clip values once for all chosen features
for col in tqdm(features_list):
    clip_values(train_all, test_all, col)

# Pan-cancer (ML_All) model trained on ALL train
X_train_all = train_all[features_list]
y_train_all = train_all[OUTCOME]
ml_all = fit_xgb(X_train_all, y_train_all)

# --- Compute per-subtype APs ---------------------------------------------
subtype_codes = pd.Categorical(WIDE_DATA["subtype"]).categories
rows = []
for code_idx, subtype_name in enumerate(subtype_codes):
    # Subset test rows for this subtype
    test_sub = test_all[test_all[SUBTYPE_COL] == code_idx].reset_index(drop=True)
    if test_sub.empty:
        continue

    X_sub = test_sub[features_list]
    y_sub = test_sub[OUTCOME].astype(int)

    # ML_All probs on this subtype
    p_all = pd.Series(ml_all.predict_proba(X_sub)[:,1], index=y_sub.index)
    ap_all, (lo_all, hi_all) = ap_with_ci(y_sub, p_all)

    # Train a subtype-specific model on matching train rows (optional: if you want true disease-specific)
    train_sub = train_all[train_all[SUBTYPE_COL] == code_idx].reset_index(drop=True)
    if len(train_sub) >= 50:  # small guard
        X_train_sub = train_sub[features_list]
        y_train_sub = train_sub[OUTCOME].astype(int)
        ml_sub = fit_xgb(X_train_sub, y_train_sub)
        p_sub = pd.Series(ml_sub.predict_proba(X_sub)[:,1], index=y_sub.index)
        ap_sub, (lo_sub, hi_sub) = ap_with_ci(y_sub, p_sub)
    else:
        ap_sub, lo_sub, hi_sub = (np.nan, np.nan, np.nan)

    rows.append({
        "Subtype": str(subtype_name),
        "Sample size": int(len(test_sub)),
        "disease_specific_ap": ap_sub,
        "disease_specific_ap_low": lo_sub,
        "disease_specific_ap_high": hi_sub,
        "all_ap": ap_all,
        "all_ap_low": lo_all,
        "all_ap_high": hi_all,
    })

plotting_data = pd.DataFrame(rows)
plotting_data = plotting_data.sort_values("Sample size", ascending=False).reset_index(drop=True)

# --- Scatter plot with (optional) error bars -----------------------------
sns.set_style("white")
sns.set_context("notebook", font_scale=0.9)
fig, ax = plt.subplots(figsize=(6.6, 4.8))

palette = sns.color_palette("tab10", len(plotting_data))
for i, row in plotting_data.iterrows():
    ax.scatter(row["disease_specific_ap"], row["all_ap"], s=10 + 0.9*np.sqrt(row["Sample size"])*20,
               color=palette[i], alpha=0.85, label=row["Subtype"])
    # error bars if available
    if not np.isnan(row["disease_specific_ap_low"]):
        ax.vlines([row["disease_specific_ap"]], row["all_ap_low"], row["all_ap_high"], color=palette[i], alpha=0.25, linewidth=1)
        ax.hlines([row["all_ap"]], row["disease_specific_ap_low"], row["disease_specific_ap_high"], color=palette[i], alpha=0.25, linewidth=1)
    # plus marker overlay
    ax.scatter(row["disease_specific_ap"], row["all_ap"], marker="+", color="black", s=40, linewidths=1)

# Diagonal
lo = max(PLOT_MIN, min(plotting_data["disease_specific_ap"].min(), plotting_data["all_ap"].min()) - 0.01)
hi = max(plotting_data["disease_specific_ap"].max(), plotting_data["all_ap"].max()) + 0.01
ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.7)

# Labels/legends
ax.set_xlabel("PR-AUC for the subtype-specific models")
ax.set_ylabel("PR-AUC for the ML$_{All}$ model")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.grid(True, linestyle="--", alpha=0.3)
leg = ax.legend(title="Subtype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
fig.tight_layout()
fig.savefig("plots/subtype_specific_comparison.svg", bbox_inches="tight")
fig.savefig("plots/subtype_specific_comparison.pdf", bbox_inches="tight")
fig.savefig("plots/subtype_specific_comparison.png", bbox_inches="tight")
