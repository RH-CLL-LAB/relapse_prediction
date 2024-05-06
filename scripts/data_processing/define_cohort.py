from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from helpers.constants import LPR_THREE_TABLES

table_name = "t_dalycare_diagnoses"  # insert name of table to load
limit = 0  # insert limit of how many rows to include
subset_columns = []  # insert subset columns needed for the table
data = load_data_from_table(
    table_name=table_name, subset_columns=subset_columns, limit=limit
)

data = data[data["diagnosis"] != "DC92"].reset_index(drop=True)

COHORT = data["patientid"].unique()
