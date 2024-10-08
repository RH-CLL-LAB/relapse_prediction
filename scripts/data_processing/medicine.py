from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort, WIDE_DATA
from data_processing.lookup_tables import ATC_LOOKUP_TABLE
from tqdm import tqdm

tqdm.pandas()


def normalize_units(data, variable_code, unit, factor):
    data.loc[
        (data["variable_code"] == variable_code) & (data["unit"] == unit),
        "value_numeric",
    ] = (
        data.loc[(data["variable_code"] == variable_code) & (data["unit"] == unit)][
            "value_numeric"
        ]
        * factor
    )


# cumulative dosage is
# value * days

# try to get at the diversity of medication taken
# number of unique atc codes

# also calculate days from treatment till first
# occurence of atc and last occurence

# NOTE: dates need to be clamped - if end_date is within
# the lookback, the start_date needs to be moved within the window

medicine_dict = {
    "RECEPTDATA_CLEAN": {
        "atc_kode": "variable_code",
        "expdato": "timestamp",
        "patientid": "patientid",
        "styrke": "value",
        "Unit": "unit",
    },
    "adm_medicine": {
        "patientid": "patientid",
        "d_adm_date": "timestamp",
        "d_ord_slut_date": "end_date",
        "c_atc": "variable_code",
        "v_adm_dosis": "value",
        "v_adm_dosis_enhed": "unit",
    },
    # consider adding number of days
    "SP_OrdineretMedicin": {
        "patientid": "patientid",
        "order_start_time": "timestamp",
        "order_end_time": "end_date",
        "atc": "variable_code",
        "hv_discrete_dose": "value",  # most likely wrong - doesn't match with strings
    },
    "SDS_epikur": {
        "atc": "variable_code",
        "eksd": "timestamp",
        "doso": "value",  # went with packsize before, but now we're running with doso, converting nans to 1
        "patientid": "patientid",
    },
    "SDS_indberetningmedpris": {
        "c_atc": "variable_code",
        "patientid": "patientid",
        "d_adm": "timestamp",
        "d_ord_slut": "end_date",
        "v_styrke_num": "value",
        "v_styrke_enhed": "unit",
    },
}

# Check if SDS_epikur normalizing is proper

# NOTE: For SDS_indberetningmedpris has to
# be preprocessed separately by having d_adm as a
# backup - the ordering often starts way before.

medicine_days_dict = {
    "SDS_indberetningmedpris": {
        "c_atc": "variable_code",
        "patientid": "patientid",
        "d_adm": "timestamp",
        "d_ord_start": "start_date",
        "d_ord_slut": "end_date",
        "v_styrke_num": "value",
        "v_styrke_enhed": "unit",
    },
    "adm_medicine": {
        "patientid": "patientid",
        "d_adm_date": "timestamp",
        "d_ord_start_date": "start_date",
        "d_ord_slut_date": "end_date",
        "c_atc": "variable_code",
        "v_adm_dosis": "value",
        "v_adm_dosis_enhed": "unit",
    },
    # consider adding number of days
    "SP_OrdineretMedicin": {
        "patientid": "patientid",
        "order_start_time": "timestamp",
        "order_end_time": "end_date",
        "atc": "variable_code",
        "hv_discrete_dose": "value",  # most likely wrong - doesn't match with strings
    },
}

medicine_days = {
    table_name: download_and_rename_data(
        table_name=table_name, config_dict=medicine_days_dict, cohort=lyfo_cohort
    )
    for table_name in medicine_days_dict
}

sds_indberetning = pd.concat([x for x in medicine_days.values()]).reset_index(drop=True)
sds_indberetning["data_source"] = "medicine_days"
sds_indberetning = sds_indberetning.drop_duplicates().reset_index(drop=True)

# fill missing startdates with d_adm

sds_indberetning.loc[
    sds_indberetning["start_date"].isna(), "start_date"
] = sds_indberetning[sds_indberetning["start_date"].isna()]["timestamp"]

# fill missing enddates with startdates

sds_indberetning.loc[
    sds_indberetning["end_date"].isna(), "end_date"
] = sds_indberetning[sds_indberetning["end_date"].isna()]["start_date"]

# end dates that are before start dates are set to be start date

sds_indberetning.loc[
    sds_indberetning["end_date"] < sds_indberetning["start_date"], "end_date"
] = sds_indberetning[sds_indberetning["end_date"] < sds_indberetning["start_date"]][
    "start_date"
]

WIDE_DATA_SUBSET = WIDE_DATA[["patientid", "date_treatment_1st_line"]]

sds_indberetning = sds_indberetning.merge(WIDE_DATA_SUBSET)

filtered_sds_indberetning = sds_indberetning[
    sds_indberetning["start_date"] < sds_indberetning["date_treatment_1st_line"]
].reset_index(drop=True)

filtered_sds_indberetning = filtered_sds_indberetning[
    filtered_sds_indberetning["variable_code"] != ""
].reset_index(drop=True)

filtered_sds_indberetning = filtered_sds_indberetning.drop_duplicates(
    subset=["patientid", "timestamp", "value", "variable_code"]
).reset_index(drop=True)

# clamp end days

filtered_sds_indberetning[
    "date_before_treatment"
] = filtered_sds_indberetning.progress_apply(
    lambda x: min((x["end_date"], x["date_treatment_1st_line"])), axis=1
)


filtered_sds_indberetning["90_days_before_treatment"] = filtered_sds_indberetning[
    "date_treatment_1st_line"
] - datetime.timedelta(days=90)
filtered_sds_indberetning["365_days_before_treatment"] = filtered_sds_indberetning[
    "date_treatment_1st_line"
] - datetime.timedelta(days=365)
filtered_sds_indberetning["1825_days_before_treatment"] = filtered_sds_indberetning[
    "date_treatment_1st_line"
] - datetime.timedelta(days=1825)

filtered_sds_indberetning["value"] = pd.to_numeric(
    filtered_sds_indberetning["value"], errors="coerce"
)
filtered_sds_indberetning.loc[filtered_sds_indberetning["value"].isna(), "value"] = 1


def calculate_days_and_cumulative_dosage(data, days):
    data[f"date_after_{days}_days"] = data.progress_apply(
        lambda x: max((x["start_date"], x[f"{days}_days_before_treatment"])), axis=1
    )
    data[f"days_of_medication_{days}_days"] = (
        data["date_before_treatment"] - data[f"date_after_{days}_days"]
    ).dt.days
    data.loc[
        data[f"days_of_medication_{days}_days"] == 0,
        f"days_of_medication_{days}_days",
    ] = 1
    data_days = data[data[f"days_of_medication_{days}_days"] > 0].reset_index(drop=True)
    data_days[f"cumulative_dosage_{days}_days"] = (
        data_days["value"] * data_days[f"days_of_medication_{days}_days"]
    )

    data_cumulative = (
        data_days[
            [
                "patientid",
                "start_date",
                "variable_code",
                "data_source",
                f"cumulative_dosage_{days}_days",
            ]
        ]
        .rename(
            columns={
                "start_date": "timestamp",
                f"cumulative_dosage_{days}_days": "value",
            }
        )
        .reset_index(drop=True)
    )
    data_cumulative["data_source"] = f"medicine_{days}_days_cumulative"

    data_days = (
        data_days[
            [
                "patientid",
                "start_date",
                "variable_code",
                "data_source",
                f"days_of_medication_{days}_days",
            ]
        ]
        .rename(
            columns={
                "start_date": "timestamp",
                f"days_of_medication_{days}_days": "value",
            }
        )
        .reset_index(drop=True)
    )

    return data_days, data_cumulative


(
    filtered_sds_indberetning_90_days,
    filtered_sds_indberetning_90_cumulative,
) = calculate_days_and_cumulative_dosage(filtered_sds_indberetning, days=90)
(
    filtered_sds_indberetning_365_days,
    filtered_sds_indberetning_365_cumulative,
) = calculate_days_and_cumulative_dosage(filtered_sds_indberetning, days=365)
(
    filtered_sds_indberetning_1825_days,
    filtered_sds_indberetning_1825_cumulative,
) = calculate_days_and_cumulative_dosage(filtered_sds_indberetning, days=365*5)
medicine_days = {
    "medicine_90_days_count": filtered_sds_indberetning_90_days,
    "medicine_365_days_count": filtered_sds_indberetning_365_days,
    "medicine_1825_days_count": filtered_sds_indberetning_1825_days,
    "medicine_90_days_cumulative": filtered_sds_indberetning_90_cumulative,
    "medicine_365_days_cumulative": filtered_sds_indberetning_365_cumulative,
    "medicine_1825_days_cumulative": filtered_sds_indberetning_1825_cumulative,
}

# NOTE: Need to fix dates before concatenating!

medicine_data = {
    table_name: download_and_rename_data(
        table_name=table_name, config_dict=medicine_dict, cohort=lyfo_cohort
    )
    for table_name in medicine_dict
}

medicine_data["RECEPTDATA_CLEAN"] = medicine_data["RECEPTDATA_CLEAN"][
    ["patientid", "timestamp", "variable_code", "value", "data_source"]
]
medicine_data["adm_medicine"] = medicine_data["adm_medicine"][
    ["patientid", "timestamp", "variable_code", "value", "data_source"]
]

medicine_data["SDS_epikur"].loc[
    medicine_data["SDS_epikur"]["value"].isna(), "value"
] = 1


medicine_data["SP_OrdineretMedicin"].loc[
    medicine_data["SP_OrdineretMedicin"]["value"] == "-", "value"
] = 1

# take the mean of ranges
medicine_data["SP_OrdineretMedicin"].loc[
    medicine_data["SP_OrdineretMedicin"]["value"].str.contains("-", na=False),
    "value",
] = medicine_data["SP_OrdineretMedicin"][
    medicine_data["SP_OrdineretMedicin"]["value"].str.contains("-", na=False)
][
    "value"
].apply(
    lambda x: sum([float(x) for x in x.split("-")]) / len(x.split("-"))
)
# medicine = pd.concat(medicine_data.values()).reset_index(drop=True)

# medicine["data_source"] = "medicine"

for medicine_days_key, medicine_days_data in medicine_days.items():
    medicine_data[medicine_days_key] = medicine_days_data

poly_pharmacy = pd.concat(
    [
        medicine_data["SDS_epikur"],
        medicine_data["RECEPTDATA_CLEAN"],
        medicine_data["adm_medicine"],
        medicine_data["SP_OrdineretMedicin"],
        medicine_data["SDS_indberetningmedpris"],
    ]
)

poly_pharmacy = poly_pharmacy.drop_duplicates(
    subset=["patientid", "variable_code", "timestamp", "value"]
).reset_index(drop=True)

poly_pharmacy = poly_pharmacy[poly_pharmacy["variable_code"].notna()].reset_index(
    drop=True
)

poly_pharmacy["atc_level_1"] = poly_pharmacy["variable_code"].apply(lambda x: x[0:1])

poly_pharmacy["atc_level_2"] = poly_pharmacy["variable_code"].apply(lambda x: x[0:3])

poly_pharmacy["atc_level_3"] = poly_pharmacy["variable_code"].apply(lambda x: x[0:4])

poly_pharmacy["atc_level_4"] = poly_pharmacy["variable_code"].apply(lambda x: x[0:5])

poly_pharmacy["atc_level_5"] = poly_pharmacy["variable_code"]

poly_pharmacy = poly_pharmacy[
    [
        x
        for x in poly_pharmacy.columns
        if x not in ["variable_code", "end_date", "unit", "data_source", "value"]
    ]
]

poly_pharmacy = poly_pharmacy.merge(WIDE_DATA[["patientid", "date_treatment_1st_line"]])

poly_pharmacy = poly_pharmacy[
    poly_pharmacy["timestamp"] <= poly_pharmacy["date_treatment_1st_line"]
].reset_index(drop=True)

poly_pharmacy = poly_pharmacy[
    [x for x in poly_pharmacy.columns if x != "date_treatment_1st_line"]
].reset_index(drop=True)

poly_pharmacy = poly_pharmacy.melt(
    id_vars=["timestamp", "patientid"], var_name="variable_code"
)

poly_pharmacy = (
    poly_pharmacy.merge(ATC_LOOKUP_TABLE, left_on="value", right_on="class_code")[
        ["patientid", "timestamp", "variable_code", "class_name"]
    ]
    .rename(columns={"class_name": "value"})
    .reset_index(drop=True)
)

poly_pharmacy = (
    poly_pharmacy.groupby(["patientid", "variable_code", "value"])
    .agg(timestamp=("timestamp", "max"))
    .reset_index()
)
# trick to get count aggregation to work
poly_pharmacy["value"] = 1

poly_pharmacy["data_source"] = "poly_pharmacy"


def melt_data(df: pd.DataFrame):
    df = df[[x for x in df.columns if x != "variable_code"]].reset_index(drop=True)
    return df.melt(
        id_vars=["patientid", "timestamp", "data_source", "value"],
        var_name="level",
        value_name="variable_code",
    ).reset_index(drop=True)[
        ["patientid", "timestamp", "data_source", "value", "variable_code"]
    ]


medicine_data_dict = {}

for dataset in tqdm(medicine_data):
    data = medicine_data[dataset][
        medicine_data[dataset]["variable_code"].notnull()
    ].reset_index(drop=True)

    data["atc_level_1"] = data["variable_code"].apply(lambda x: x[0:1])

    data["atc_level_2"] = data["variable_code"].apply(lambda x: x[0:3])

    data["atc_level_3"] = data["variable_code"].apply(lambda x: x[0:4])

    data["atc_level_4"] = data["variable_code"].apply(lambda x: x[0:5])

    data["atc_level_5"] = data["variable_code"]

    data = melt_data(data)

    medicine_data_dict[dataset] = (
        data.merge(ATC_LOOKUP_TABLE, left_on="variable_code", right_on="class_code")[
            ["patientid", "timestamp", "data_source", "value", "class_name"]
        ]
        .rename(columns={"class_name": "variable_code"})
        .reset_index(drop=True)
    )

# medicine in general
medicine_general = pd.concat(
    [
        medicine_data_dict["SDS_epikur"],
        medicine_data_dict["RECEPTDATA_CLEAN"],
        medicine_data_dict["adm_medicine"],
        medicine_data_dict["SP_OrdineretMedicin"],
        medicine_data_dict["SDS_indberetningmedpris"],
    ]
)
# NOTE: Probably need to remove duplicates here

medicine_general["data_source"] = "medicine_general"

medicine_general = medicine_general.drop_duplicates().reset_index(drop=True)

medicine_data = {
    "medicine_general": medicine_general,
}

for key in medicine_days:
    medicine_data[key] = medicine_data_dict[key]

# okaaay - so we need to
# (1) merge with wide data
# (2) get the treatment date
# (3) filter by treatment date
# (4) calculate first and last date of receiving that specific drug

medicine_general_merged = medicine_general.merge(WIDE_DATA_SUBSET)
medicine_general_merged = medicine_general_merged[
    medicine_general_merged["date_treatment_1st_line"]
    > medicine_general_merged["timestamp"]
].reset_index(drop=True)
medicine_general_merged["timestamp"] = pd.to_datetime(
    medicine_general_merged["timestamp"]
)

medicine_general_merged["days_from_treatment"] = (
    medicine_general_merged["date_treatment_1st_line"]
    - medicine_general_merged["timestamp"]
).dt.days

medicine_days_from_treatment = (
    medicine_general_merged.groupby(["patientid", "variable_code"])
    .agg(
        max_days=("days_from_treatment", "max"),
        min_days=("days_from_treatment", "min"),
    )
    .reset_index()
)

medicine_days_from_treatment["days_between_max_and_min"] = (
    medicine_days_from_treatment["max_days"] - medicine_days_from_treatment["min_days"]
)

n_patients_per_drug = (
    medicine_days_from_treatment.groupby("variable_code")
    .agg(counts=("patientid", "nunique"))
    .reset_index()
)

# here we filter by 10% of patients having received it

drugs_above_threshold = n_patients_per_drug[
    n_patients_per_drug["counts"] > 0.10 * len(WIDE_DATA_SUBSET)
]["variable_code"].values

medicine_days_from_treatment = medicine_days_from_treatment[
    medicine_days_from_treatment["variable_code"].isin(drugs_above_threshold)
].reset_index(drop=True)

medicine_days_from_treatment = medicine_days_from_treatment.melt(
    value_vars=["max_days", "min_days", "days_between_max_and_min"],
    id_vars=["patientid", "variable_code"],
)

medicine_days_from_treatment_pivot = medicine_days_from_treatment.pivot(
    index="patientid", columns=["variable_code", "variable"], values="value"
).reset_index()
medicine_days_from_treatment_pivot.columns = [
    "%s_%s" % (a, b) for a, b in medicine_days_from_treatment_pivot.columns
]
medicine_days_from_treatment_pivot = medicine_days_from_treatment_pivot.rename(
    columns={"patientid_": "patientid"}
)

# and there it is - static predictors on this and we're golden
# NOTE: fill nas with -1?
import numpy as np

# NOTE: needs to be added as static features

medicine_days_from_treatment_pivot = medicine_days_from_treatment_pivot.replace(
    np.nan, -1
)
