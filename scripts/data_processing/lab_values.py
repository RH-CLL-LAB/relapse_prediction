from helpers.sql_helper import *
from helpers.processing_helper import *
from data_processing.wide_data import lyfo_cohort
from data_processing.lookup_tables import ATC_LOOKUP_TABLE
from tqdm import tqdm

tqdm.pandas()

lab_dict = {
    "LAB_IGHVIMGT": {
        "date_sample": "timestamp",
        "patientid": "patientid",
        "ighv": "value",
    },
    "LAB_BIOBANK_SAMPLES": {
        "patientid": "patientid",
        "date_samplecollection": "timestamp",
        "Type": "variable_code",
    },
    "LAB_Flowcytometry": {
        "patientid": "patientid",
        "DATE": "timestamp",
        "MATERIAL": "variable_code",
    },
}

lab_data = {
    table_name: download_and_rename_data(
        table_name=table_name, config_dict=lab_dict, cohort=lyfo_cohort
    )
    for table_name in lab_dict
}

lab_data["LAB_IGHVIMGT"].loc[:, "variable_code"] = "IGHV"
lab_data["LAB_IGHVIMGT"].loc[:, "value"] = pd.Categorical(
    lab_data["LAB_IGHVIMGT"]["value"], ordered=True, categories=["Unmutated", "Mutated"]
).codes

lab_data["LAB_BIOBANK_SAMPLES"].loc[:, "value"] = 1


lab_data["LAB_Flowcytometry"].loc[:, "value"] = 1
