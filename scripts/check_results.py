import pandas as pd
import seaborn as sns

df = pd.read_pickle("results/initial_tests_test_death.pkl")

df.groupby(["outcome", "disease", "cohort", "feature_length"]).agg(
    f1_mean=("f1", "mean"),
    mcc_mean=("mcc", "mean"),
    precision_mean=("precision", "mean"),
    recall_mean=("recall", "mean"),
).reset_index()


df.groupby(["outcome", "disease", "cohort"]).agg(
    f1_mean=("f1", "mean"),
    mcc_mean=("mcc", "mean"),
    precision_mean=("precision", "mean"),
    recall_mean=("recall", "mean"),
).reset_index()

df[
    (df["cohort"] == "non_proxy_only_dlbcl")
    & (df["outcome"] == "train_on_proxy_outcome")
].groupby(["feature_length"]).agg(
    f1_mean=("f1", "mean"),
    mcc_mean=("mcc", "mean"),
    precision_mean=("precision", "mean"),
    recall_mean=("recall", "mean"),
).reset_index()

df[df["mcc"] == df["mcc"].max()]

sns.catplot(
    data=df,
    x="outcome",
    y="f1",
    hue="disease",
    kind="bar",
    col="cohort",
    row="feature_length",
)
