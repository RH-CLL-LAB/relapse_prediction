from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort
from data_processing.lookup_tables import ATC_LOOKUP_TABLE
from tqdm import tqdm

tqdm.pandas()

# NOTE: adm_medicine is completely packed with
# absolute nonsense units, can't be standardized

# NOTE: SP_OrdineretMedicin only contains
# unit in weird text string, complete nonsense

normalizing_dict = {
    "RG": 0.001,
    "RGD": 0.001,
    "RGH": 0.001,
    "MG": 1,
    "MGH": 1,
    "MGD": 1,
    "MGF": 1,
    "MGG": 1,  # note that this doesn't make sense
    "MGM": 1,  # this doesn't make sense either
    "PC": 1,  # PC is percent, makes no sense
    "PW": 1,  # PW percent per w
    "UML": 1,  # makes no sense, enheder pr ml
    "MIU": 1,
    "IUM": 1,  # no idea what this is
    "RGM": 0.001,
    "RGT": 0.001,
    "G": 1000,
    "XAM": 1,  # what
    "MMO": 0.001,  # probably, but could be anything
    "EP": 0.001,  # anything goes
    "RGG": 0.001,
    "IU": 0.001,  # as is tradition, I don't know what is happening
    "XA": 0.001,  # as is tradition, I don't know what is happening
    "SQM": 0.001,  # as is tradition, I don't know what is happening
    "IUG": 0.001,  # as is tradition, I don't know what is happening
    "MGS": 1,
    "BHS": 1,  # behandlingssæt hahaha
    "INS": 1,  # initial sæt haha
    "SQ": 1,
    "EUM": 1,
    "GRAM": 1000,
    "MIKROGRAM": 0.001,
}


def standardize_units(unit, value):
    return normalizing_dict.get(unit, 1) * value


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
        "d_ord_start_date": "timestamp",
        "c_atc": "variable_code",
        "v_adm_dosis": "value",
        "v_adm_dosis_enhed": "unit",
    },
    # consider adding number of days
    "SP_OrdineretMedicin": {
        "patientid": "patientid",
        "order_start_time": "timestamp",
        "atc": "variable_code",
        "hv_discrete_dose": "value",  # most likely wrong - doesn't match with strings
    },
    "SDS_epikur": {
        "atc": "variable_code",
        "eksd": "timestamp",
        "doso": "value",  # went with packsize before, but now we're running with doso, converting nans to 1
        "patientid": "patientid",
    },
}

# NOTE: Need to fix dates before concatenating!

medicine_data = {
    table_name: download_and_rename_data(
        table_name=table_name, config_dict=medicine_dict, cohort=lyfo_cohort
    )
    for table_name in medicine_dict
}

medicine_data["RECEPTDATA_CLEAN"]["value"] = medicine_data[
    "RECEPTDATA_CLEAN"
].progress_apply(lambda x: standardize_units(x["unit"], x["value"]), axis=1)
medicine_data["adm_medicine"]["value"] = medicine_data["adm_medicine"].progress_apply(
    lambda x: standardize_units(x["unit"], x["value"]), axis=1
)
medicine_data["RECEPTDATA_CLEAN"]
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
medicine_data["SP_OrdineretMedicin"]
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

SP_OrdineretMedicin_days = download_and_rename_data(
    "SP_OrdineretMedicin",
    {
        "SP_OrdineretMedicin": {
            "patientid": "patientid",
            "order_start_time": "timestamp",
            "order_end_time": "end_date",
            "atc": "variable_code",
        }
    },
    cohort=lyfo_cohort,
)

adm_medicine_days = download_and_rename_data(
    "adm_medicine",
    {
        "adm_medicine": {
            "patientid": "patientid",
            "d_ord_start_date": "timestamp",
            "d_ord_slut_date": "end_date",
            "c_atc": "variable_code",
        }
    },
    cohort=lyfo_cohort,
)

for dataset in [SP_OrdineretMedicin_days, adm_medicine_days]:
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], errors="coerce")
    dataset["end_date"] = pd.to_datetime(dataset["end_date"], errors="coerce")
    dataset["value"] = (dataset["end_date"] - dataset["timestamp"]).dt.days

    dataset = dataset[["patientid", "timestamp", "variable_code", "value"]]

print("generate different atc levels")

SP_OrdineretMedicin_days["data_source"] = "SP_OrdineretMedicin_days"
adm_medicine_days["data_source"] = "adm_medicine_days"

medicine_data["SP_OrdineretMedicin_days"] = SP_OrdineretMedicin_days
medicine_data["adm_medicine_days"] = adm_medicine_days


def melt_data(df: pd.DataFrame):
    df = df[[x for x in df.columns if x != "variable_code"]].reset_index(drop=True)
    return df.melt(
        id_vars=["patientid", "timestamp", "data_source", "value"],
        var_name="level",
        value_name="variable_code",
    ).reset_index(drop=True)[
        ["patientid", "timestamp", "data_source", "value", "variable_code"]
    ]


for dataset in tqdm(medicine_data):
    medicine_data[dataset] = medicine_data[dataset][
        medicine_data[dataset]["variable_code"].notnull()
    ].reset_index(drop=True)
    medicine_data[dataset]["atc_level_1"] = medicine_data[dataset][
        "variable_code"
    ].apply(lambda x: x[0:1])

    medicine_data[dataset]["atc_level_2"] = medicine_data[dataset][
        "variable_code"
    ].apply(lambda x: x[0:3])

    medicine_data[dataset]["atc_level_3"] = medicine_data[dataset][
        "variable_code"
    ].apply(lambda x: x[0:4])

    medicine_data[dataset]["atc_level_4"] = medicine_data[dataset][
        "variable_code"
    ].apply(lambda x: x[0:5])

    medicine_data[dataset]["atc_level_5"] = medicine_data[dataset]["variable_code"]
    medicine_data[dataset] = melt_data(medicine_data[dataset])

    medicine_data[dataset] = (
        medicine_data[dataset]
        .merge(ATC_LOOKUP_TABLE, left_on="variable_code", right_on="class_code")[
            ["patientid", "timestamp", "data_source", "value", "class_name"]
        ]
        .rename(columns={"class_name": "variable_code"})
        .reset_index(drop=True)
    )
