from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort, WIDE_DATA
from data_processing.lookup_tables import ATC_LOOKUP_TABLE
from tqdm import tqdm
import numpy as np

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

adm_medicine_dict = {
    "adm_medicine": {
        "patientid": "patientid",
        "d_adm_date": "timestamp",
        "d_ord_start_date": "start_date",
        "d_ord_slut_date": "end_date",
        "c_atc": "variable_code",
        "v_adm_dosis": "value",
        "v_adm_dosis_enhed": "unit",
    },
    "SDS_indberetningmedpris": {
        "c_atc": "variable_code",
        "patientid": "patientid",
        "d_adm": "timestamp",
        "d_ord_start": "start_date",
        "d_ord_slut": "end_date",
        "v_styrke_num": "value",
        "v_styrke_enhed": "unit",
    },
}
ord_medicine_dict = {
    "SP_OrdineretMedicin": {
        "patientid": "patientid",
        "order_start_time": "timestamp",
        "order_end_time": "end_date",
        "atc": "variable_code",
        "hv_discrete_dose": "value",
    },
    "SDS_epikur": {
        "atc": "variable_code",
        "eksd": "timestamp",
        "doso": "value",
        "patientid": "patientid",
    },
}

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
        "hv_discrete_dose": "value",
    },
}


def melt_data(df: pd.DataFrame):
    df = df[[x for x in df.columns if x != "variable_code"]].reset_index(drop=True)
    return df.melt(
        id_vars=["patientid", "timestamp", "data_source", "value"],
        var_name="level",
        value_name="variable_code",
    ).reset_index(drop=True)[
        ["patientid", "timestamp", "data_source", "value", "variable_code"]
    ]


def convert_to_atc_codes(data):
    data = data[data["variable_code"].notnull()].reset_index(drop=True)

    data["atc_level_1"] = data["variable_code"].apply(lambda x: x[0:1])

    data["atc_level_2"] = data["variable_code"].apply(lambda x: x[0:3])

    data["atc_level_3"] = data["variable_code"].apply(lambda x: x[0:4])

    data["atc_level_4"] = data["variable_code"].apply(lambda x: x[0:5])

    data["atc_level_5"] = data["variable_code"]

    data = melt_data(data)

    data = (
        data.merge(ATC_LOOKUP_TABLE, left_on="variable_code", right_on="class_code")[
            ["patientid", "timestamp", "data_source", "value", "class_name"]
        ]
        .rename(columns={"class_name": "variable_code"})
        .reset_index(drop=True)
    )
    return data


def calculate_days_from_data(data_dict, data_source_str):
    medicine_days = {
        table_name: download_and_rename_data(
            table_name=table_name, config_dict=data_dict, cohort=lyfo_cohort
        )
        for table_name in data_dict
    }

    sds_indberetning = pd.concat([x for x in medicine_days.values()]).reset_index(
        drop=True
    )
    sds_indberetning["data_source"] = data_source_str
    sds_indberetning = sds_indberetning.drop_duplicates().reset_index(drop=True)

    # fill missing startdates with d_adm

    if "start_date" not in sds_indberetning.columns:
        sds_indberetning["start_date"] = pd.NA

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
    filtered_sds_indberetning["1095_days_before_treatment"] = filtered_sds_indberetning[
        "date_treatment_1st_line"
    ] - datetime.timedelta(days=1095)

    filtered_sds_indberetning["value"] = pd.to_numeric(
        filtered_sds_indberetning["value"], errors="coerce"
    )
    filtered_sds_indberetning.loc[
        filtered_sds_indberetning["value"].isna(), "value"
    ] = 1

    def calculate_days_and_cumulative_dosage(data, days, data_source_str):
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
        data_days = data[data[f"days_of_medication_{days}_days"] > 0].reset_index(
            drop=True
        )
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
        data_cumulative["data_source"] = f"{data_source_str}_{days}_days_cumulative"
        data_days["data_source"] = f"{data_source_str}_{days}_days_count"
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

        data_days = convert_to_atc_codes(data_days)

        return data_days, data_cumulative

    (
        filtered_sds_indberetning_90_days,
        filtered_sds_indberetning_90_cumulative,
    ) = calculate_days_and_cumulative_dosage(
        filtered_sds_indberetning, days=90, data_source_str=data_source_str
    )
    (
        filtered_sds_indberetning_365_days,
        filtered_sds_indberetning_365_cumulative,
    ) = calculate_days_and_cumulative_dosage(
        filtered_sds_indberetning, days=365, data_source_str=data_source_str
    )
    (
        filtered_sds_indberetning_1095_days,
        filtered_sds_indberetning_1095_cumulative,
    ) = calculate_days_and_cumulative_dosage(
        filtered_sds_indberetning, days=365 * 3, data_source_str=data_source_str
    )

    return (
        filtered_sds_indberetning_90_days,
        filtered_sds_indberetning_365_days,
        filtered_sds_indberetning_1095_days,
    )


adm_medicine_days = calculate_days_from_data(adm_medicine_dict, "adm_medicine_days")
ord_medicine_days = calculate_days_from_data(ord_medicine_dict, "ord_medicine_days")

# need to merge with all the different ATC codes here

adm_medicine_data = {
    table_name: download_and_rename_data(
        table_name=table_name, config_dict=adm_medicine_dict, cohort=lyfo_cohort
    )
    for table_name in adm_medicine_dict
}

ord_medicine_data = {
    table_name: download_and_rename_data(
        table_name=table_name, config_dict=ord_medicine_dict, cohort=lyfo_cohort
    )
    for table_name in ord_medicine_dict
}

adm_medicine_data["adm_medicine"] = adm_medicine_data["adm_medicine"][
    ["patientid", "timestamp", "variable_code", "value", "data_source"]
]

ord_medicine_data["SDS_epikur"].loc[
    ord_medicine_data["SDS_epikur"]["value"].isna(), "value"
] = 1


ord_medicine_data["SP_OrdineretMedicin"].loc[
    ord_medicine_data["SP_OrdineretMedicin"]["value"] == "-", "value"
] = 1

# take the mean of ranges
ord_medicine_data["SP_OrdineretMedicin"].loc[
    ord_medicine_data["SP_OrdineretMedicin"]["value"].str.contains("-", na=False),
    "value",
] = ord_medicine_data["SP_OrdineretMedicin"][
    ord_medicine_data["SP_OrdineretMedicin"]["value"].str.contains("-", na=False)
][
    "value"
].apply(
    lambda x: sum([float(x) for x in x.split("-")]) / len(x.split("-"))
)

# for medicine_days_key, medicine_days_data in medicine_days.items():
#     medicine_data[medicine_days_key] = medicine_days_data


def get_poly_pharmacy_scores(data: pd.DataFrame, data_source: str):
    poly_pharmacy = data.drop_duplicates(
        subset=["patientid", "variable_code", "timestamp", "value"]
    ).reset_index(drop=True)

    poly_pharmacy = poly_pharmacy[poly_pharmacy["variable_code"].notna()].reset_index(
        drop=True
    )

    poly_overview = (
        poly_pharmacy.groupby(["variable_code", "data_source"])
        .agg(events=("patientid", "count"), n_patients=("patientid", "nunique"))
        .reset_index()
    )

    poly_overview = poly_overview.sort_values("events", ascending=False).reset_index(
        drop=True
    )

    poly_pharmacy["atc_level_1"] = poly_pharmacy["variable_code"].apply(
        lambda x: x[0:1]
    )

    poly_pharmacy["atc_level_2"] = poly_pharmacy["variable_code"].apply(
        lambda x: x[0:3]
    )

    poly_pharmacy["atc_level_3"] = poly_pharmacy["variable_code"].apply(
        lambda x: x[0:4]
    )

    poly_pharmacy["atc_level_4"] = poly_pharmacy["variable_code"].apply(
        lambda x: x[0:5]
    )

    poly_pharmacy["atc_level_5"] = poly_pharmacy["variable_code"]

    poly_pharmacy = poly_pharmacy[
        [
            x
            for x in poly_pharmacy.columns
            if x not in ["variable_code", "end_date", "unit", "data_source", "value"]
        ]
    ]

    poly_pharmacy = poly_pharmacy.merge(
        WIDE_DATA[["patientid", "date_treatment_1st_line", "date_diagnosis"]]
    )

    poly_pharmacy_since_diagnosis = poly_pharmacy.copy()

    poly_pharmacy = poly_pharmacy[
        poly_pharmacy["timestamp"] <= poly_pharmacy["date_treatment_1st_line"]
    ].reset_index(drop=True)

    poly_pharmacy_since_diagnosis = poly_pharmacy[
        poly_pharmacy["timestamp"] >= poly_pharmacy["date_diagnosis"]
    ].reset_index(drop=True)

    poly_pharmacy = poly_pharmacy[
        [x for x in poly_pharmacy.columns if x != "date_treatment_1st_line"]
    ].reset_index(drop=True)

    poly_pharmacy = poly_pharmacy[
        [x for x in poly_pharmacy.columns if x != "date_diagnosis"]
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

    poly_pharmacy["data_source"] = f"{data_source}_poly_pharmacy"

    poly_pharmacy_since_diagnosis = poly_pharmacy_since_diagnosis[
        [
            x
            for x in poly_pharmacy_since_diagnosis.columns
            if x != "date_treatment_1st_line"
        ]
    ].reset_index(drop=True)

    poly_pharmacy_since_diagnosis = poly_pharmacy_since_diagnosis[
        [x for x in poly_pharmacy_since_diagnosis.columns if x != "date_diagnosis"]
    ].reset_index(drop=True)

    poly_pharmacy_since_diagnosis = poly_pharmacy_since_diagnosis.melt(
        id_vars=["timestamp", "patientid"], var_name="variable_code"
    )

    poly_pharmacy_since_diagnosis = (
        poly_pharmacy_since_diagnosis.merge(
            ATC_LOOKUP_TABLE, left_on="value", right_on="class_code"
        )[["patientid", "timestamp", "variable_code", "class_name"]]
        .rename(columns={"class_name": "value"})
        .reset_index(drop=True)
    )

    poly_pharmacy_since_diagnosis = (
        poly_pharmacy_since_diagnosis.groupby(["patientid", "variable_code", "value"])
        .agg(timestamp=("timestamp", "max"))
        .reset_index()
    )
    # trick to get count aggregation to work
    poly_pharmacy_since_diagnosis["value"] = 1

    poly_pharmacy_since_diagnosis[
        "data_source"
    ] = f"{data_source}_poly_pharmacy_since_diagnosis"

    return poly_pharmacy, poly_pharmacy_since_diagnosis


adm_medicine_concat = pd.concat(adm_medicine_data).reset_index(drop=True)
ord_medicine_concat = pd.concat(ord_medicine_data).reset_index(drop=True)

adm_medicine_concat = adm_medicine_concat[
    ["patientid", "timestamp", "variable_code", "value", "data_source"]
]
ord_medicine_concat = ord_medicine_concat[
    ["patientid", "timestamp", "variable_code", "value", "data_source"]
]
(
    adm_medicine_poly_pharmacy,
    adm_medicine_poly_pharmacy_since_diagnosis,
) = get_poly_pharmacy_scores(adm_medicine_concat, "adm_medicine")
(
    ord_medicine_poly_pharmacy,
    ord_medicine_poly_pharmacy_since_diagnosis,
) = get_poly_pharmacy_scores(ord_medicine_concat, "ord_medicine")

adm_medicine_concat = convert_to_atc_codes(adm_medicine_concat)
ord_medicine_concat = convert_to_atc_codes(ord_medicine_concat)

adm_medicine_concat["data_source"] = "administered_medicine"
ord_medicine_concat["data_source"] = "ordered_medicine"

adm_medicine_concat.drop_duplicates().reset_index(drop=True)
ord_medicine_concat.drop_duplicates().reset_index(drop=True)

# okaaay - so we need to
# (1) merge with wide data
# (2) get the treatment date
# (3) filter by treatment date
# (4) calculate first and last date of receiving that specific drug

WIDE_DATA_SUBSET = WIDE_DATA[["patientid", "date_treatment_1st_line"]]


def get_days_from_treatment(data):
    medicine_general_merged = data.merge(WIDE_DATA_SUBSET)
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
        medicine_days_from_treatment["max_days"]
        - medicine_days_from_treatment["min_days"]
    )
    # exclude the test set patients
    test_patientids = pd.read_csv("data/test_patientids.csv")["patientid"]

    medicine_days_from_treatment = medicine_days_from_treatment[
        ~medicine_days_from_treatment["patientid"].isin(test_patientids)
    ].reset_index(drop=True)

    # here we filter by 10% of patients having received it
    n_patients_per_drug = (
        medicine_days_from_treatment.groupby("variable_code")
        .agg(counts=("patientid", "nunique"))
        .reset_index()
    )

    drugs_above_threshold = n_patients_per_drug[
        n_patients_per_drug["counts"]
        > 0.10
        * len(WIDE_DATA_SUBSET[~WIDE_DATA_SUBSET["patientid"].isin(test_patientids)])
    ]["variable_code"].values

    medicine_days_from_treatment_multi_disease = medicine_days_from_treatment[
        medicine_days_from_treatment["variable_code"].isin(drugs_above_threshold)
    ].reset_index(drop=True)

    medicine_days_from_treatment_multi_disease = (
        medicine_days_from_treatment_multi_disease.melt(
            value_vars=["max_days", "min_days", "days_between_max_and_min"],
            id_vars=["patientid", "variable_code"],
        )
    )

    medicine_days_from_treatment_pivot = (
        medicine_days_from_treatment_multi_disease.pivot(
            index="patientid", columns=["variable_code", "variable"], values="value"
        ).reset_index()
    )
    medicine_days_from_treatment_pivot.columns = [
        "%s_%s" % (a, b) for a, b in medicine_days_from_treatment_pivot.columns
    ]
    medicine_days_from_treatment_pivot = medicine_days_from_treatment_pivot.rename(
        columns={"patientid_": "patientid"}
    )

    medicine_days_from_treatment_pivot = medicine_days_from_treatment_pivot.replace(
        np.nan, -1
    )

    WIDE_DATA_DISEASE = WIDE_DATA[["patientid", "subtype"]]

    medicine_days_from_treatment = medicine_days_from_treatment.merge(WIDE_DATA_DISEASE)
    medicine_days_from_treatment = medicine_days_from_treatment[
        medicine_days_from_treatment["subtype"] == "DLBCL"
    ].reset_index(drop=True)

    # here we filter by 10% of patients having received it
    n_patients_per_drug = (
        medicine_days_from_treatment.groupby("variable_code")
        .agg(counts=("patientid", "nunique"))
        .reset_index()
    )

    drugs_above_threshold = n_patients_per_drug[
        n_patients_per_drug["counts"]
        > 0.10
        * len(WIDE_DATA_SUBSET[~WIDE_DATA_SUBSET["patientid"].isin(test_patientids)])
    ]["variable_code"].values

    medicine_days_from_treatment_single_disease = medicine_days_from_treatment[
        medicine_days_from_treatment["variable_code"].isin(drugs_above_threshold)
    ].reset_index(drop=True)

    medicine_days_from_treatment_single_disease = (
        medicine_days_from_treatment_single_disease.melt(
            value_vars=["max_days", "min_days", "days_between_max_and_min"],
            id_vars=["patientid", "variable_code"],
        )
    )

    medicine_days_from_treatment_pivot_single_disease = (
        medicine_days_from_treatment_single_disease.pivot(
            index="patientid", columns=["variable_code", "variable"], values="value"
        ).reset_index()
    )
    medicine_days_from_treatment_pivot_single_disease.columns = [
        "%s_%s" % (a, b)
        for a, b in medicine_days_from_treatment_pivot_single_disease.columns
    ]
    medicine_days_from_treatment_pivot_single_disease = (
        medicine_days_from_treatment_pivot_single_disease.rename(
            columns={"patientid_": "patientid"}
        )
    )

    medicine_days_from_treatment_pivot_single_disease = (
        medicine_days_from_treatment_pivot_single_disease.replace(np.nan, -1)
    )

    return (
        medicine_days_from_treatment_pivot,
        medicine_days_from_treatment_pivot_single_disease,
    )


(
    adm_medicine_days_from_treatment_pivot,
    adm_medicine_days_from_treatment_pivot_single_disease,
) = get_days_from_treatment(adm_medicine_concat)
(
    ord_medicine_days_from_treatment_pivot,
    ord_medicine_days_from_treatment_pivot_single_disease,
) = get_days_from_treatment(ord_medicine_concat)

adm_medicine_days_from_treatment_pivot = adm_medicine_days_from_treatment_pivot.rename(
    columns={
        x: f"administered_{x}"
        for x in adm_medicine_days_from_treatment_pivot.columns
        if x != "patientid"
    }
).reset_index(drop=True)
ord_medicine_days_from_treatment_pivot = ord_medicine_days_from_treatment_pivot.rename(
    columns={
        x: f"ordered_{x}"
        for x in ord_medicine_days_from_treatment_pivot.columns
        if x != "patientid"
    }
).reset_index(drop=True)


adm_medicine_days_from_treatment_pivot_single_disease = (
    adm_medicine_days_from_treatment_pivot_single_disease.rename(
        columns={
            x: f"administered_{x}"
            for x in adm_medicine_days_from_treatment_pivot_single_disease.columns
            if x != "patientid"
        }
    ).reset_index(drop=True)
)
ord_medicine_days_from_treatment_pivot_single_disease = (
    ord_medicine_days_from_treatment_pivot_single_disease.rename(
        columns={
            x: f"ordered_{x}"
            for x in ord_medicine_days_from_treatment_pivot_single_disease.columns
            if x != "patientid"
        }
    ).reset_index(drop=True)
)

# all the relevant feature datasets listed here
# static:
medicine_dict = {
    "adm_medicine_90_days_count": adm_medicine_days[0],
    "adm_medicine_365_days_count": adm_medicine_days[1],
    "adm_medicine_1095_days_count": adm_medicine_days[2],
    "ord_medicine_90_days_count": ord_medicine_days[0],
    "ord_medicine_365_days_count": ord_medicine_days[1],
    "ord_medicine_1095_days_count": ord_medicine_days[2],
    "adm_medicine_poly_pharmacy": adm_medicine_poly_pharmacy,
    "adm_medicine_poly_pharmacy_since_diagnosis": adm_medicine_poly_pharmacy_since_diagnosis,
    "ord_medicine_poly_pharmacy": ord_medicine_poly_pharmacy,
    "ord_medicine_poly_pharmacy_since_diagnosis": ord_medicine_poly_pharmacy_since_diagnosis,
    "administered_medicine": adm_medicine_concat,
    "ordered_medicine": ord_medicine_concat,
}
