# TP FP TN FN
y_pred = bst.predict_proba(X_test_specific)
adjusted_y_pred_binary = np.argmax(y_pred, axis=1)

print(classification_report(y_test_specific, adjusted_y_pred_binary))

test_specific["y_pred"] = adjusted_y_pred_binary

test_specific = test_specific.reset_index(drop=True)

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_specific_plotting = test_specific[["patientid", "y_pred"]].merge(WIDE_DATA)
test_specific_plotting["date_event"] = test_specific_plotting["relapse_date"]
test_specific_plotting.loc[test_specific_plotting["date_event"].notna(), "event"] = 1

test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = test_specific_plotting[test_specific_plotting["relapse_date"].isna()]["date_death"]
test_specific_plotting.loc[
    (test_specific_plotting["date_event"].notna())
    & (test_specific_plotting["event"].isna()),
    "event",
] = 2
test_specific_plotting.loc[test_specific_plotting["date_event"].isna(), "event"] = 0
test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = pd.to_datetime("2022-01-01")
test_specific_plotting["days_to_event"] = (
    test_specific_plotting["date_event"]
    - test_specific_plotting["date_treatment_1st_line"]
).dt.days
test_specific_plotting["group"] = test_specific_plotting["y_pred"].apply(
    lambda x: 1 if x > 0 else 0
)
test_specific_plotting[
    ["patientid", "days_to_event", "event", "group", "NCCN_IPI_diagnosis"]
].to_csv("km_data_lyfo_FCR.csv", index=False)


test["y_pred"] = adjusted_y_pred_binary

test = test.reset_index(drop=True)

WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_specific_plotting = test[["patientid", "y_pred"]].merge(WIDE_DATA)
test_specific_plotting["date_event"] = test_specific_plotting["relapse_date"]
test_specific_plotting.loc[test_specific_plotting["date_event"].notna(), "event"] = 1

test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = test_specific_plotting[test_specific_plotting["relapse_date"].isna()]["date_death"]
test_specific_plotting.loc[
    (test_specific_plotting["date_event"].notna())
    & (test_specific_plotting["event"].isna()),
    "event",
] = 2
test_specific_plotting.loc[test_specific_plotting["date_event"].isna(), "event"] = 0
test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = pd.to_datetime("2022-01-01")
test_specific_plotting["days_to_event"] = (
    test_specific_plotting["date_event"]
    - test_specific_plotting["date_treatment_1st_line"]
).dt.days
test_specific_plotting["group"] = test_specific_plotting["y_pred"].apply(
    lambda x: 1 if x > 0 else 0
)
test_specific_plotting[
    ["patientid", "days_to_event", "event", "group", "NCCN_IPI_diagnosis"]
].to_csv("km_data_lyfo_FCR.csv", index=False)


WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_specific_plotting = test[["patientid", "y_pred"]].merge(WIDE_DATA)
test_specific_plotting["date_event"] = test_specific_plotting["date_death"]
test_specific_plotting.loc[test_specific_plotting["date_event"].notna(), "event"] = 1
test_specific_plotting.loc[test_specific_plotting["date_event"].isna(), "event"] = 0
test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = pd.to_datetime("2022-01-01")

test_specific_plotting["days_to_event"] = (
    test_specific_plotting["date_event"]
    - test_specific_plotting["date_treatment_1st_line"]
).dt.days
test_specific_plotting["group"] = test_specific_plotting["y_pred"].apply(
    lambda x: 1 if x == 0 else 0
)


test_specific_plotting[
    ["patientid", "days_to_event", "event", "group", "NCCN_IPI_diagnosis"]
].to_csv("km_data_lyfo_OS.csv", index=False)


WIDE_DATA = pd.read_pickle("data/WIDE_DATA.pkl")

test_specific["y_pred"] = adjusted_y_pred_binary
test_specific_plotting = test_specific[["patientid", "y_pred"]].merge(WIDE_DATA)
test_specific_plotting["date_event"] = test_specific_plotting["date_death"]
test_specific_plotting.loc[test_specific_plotting["date_event"].notna(), "event"] = 1
test_specific_plotting.loc[test_specific_plotting["date_event"].isna(), "event"] = 0
test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = pd.to_datetime("2022-01-01")

test_specific_plotting["days_to_event"] = (
    test_specific_plotting["date_event"]
    - test_specific_plotting["date_treatment_1st_line"]
).dt.days
test_specific_plotting["group"] = test_specific_plotting["y_pred"].apply(
    lambda x: 1 if x == 0 else 0
)


test_specific_plotting[
    ["patientid", "days_to_event", "event", "group", "NCCN_IPI_diagnosis"]
].to_csv("km_data_lyfo_OS.csv", index=False)


test_specific_plotting = test[["patientid", "y_pred"]].merge(WIDE_DATA)
test_specific_plotting["date_event"] = test_specific_plotting.apply(
    lambda x: min(x["relapse_date"], x["date_death"]), axis=1
)
test_specific_plotting.loc[test_specific_plotting["date_event"].notna(), "event"] = 1
test_specific_plotting.loc[test_specific_plotting["date_event"].isna(), "event"] = 0
test_specific_plotting.loc[
    test_specific_plotting["date_event"].isna(), "date_event"
] = pd.to_datetime("2022-01-01")

test_specific_plotting["days_to_event"] = (
    test_specific_plotting["date_event"]
    - test_specific_plotting["date_treatment_1st_line"]
).dt.days
test_specific_plotting["group"] = test_specific["y_pred"].apply(
    lambda x: 1 if x == 0 else 0
)
test_specific_plotting[
    ["patientid", "days_to_event", "event", "group", "RIPI_diagnosis"]
].to_csv("km_data_lyfo_FCR.csv", index=False)
