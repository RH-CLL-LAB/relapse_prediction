from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort
from data_processing.lookup_tables import SNOMED_LOOKUP_TABLE
from datetime import timedelta

pathology = load_data_from_table(
    "SDS_pato",
    cohort=lyfo_cohort,
    subset_columns=["patientid", "d_rekvdato", "c_snomedkode"],
)

pathology = pathology.merge(
    SNOMED_LOOKUP_TABLE, left_on="c_snomedkode", right_on="SKSkode"
)

pathology_codes = (
    pathology[pathology["c_snomedkode"].str.contains("^FE", regex=True)]["Kodetekst"]
    .value_counts()
    .reset_index()
)

negative_list = [
    "genstatus normal",
    "ikke påvist",
    "uden genamplikation",
    "negativ",
    "ikke hypermuteret",
]

"|".join(negative_list)

pathology_genes = pathology[
    pathology["c_snomedkode"].str.contains("^FE", regex=True)
].reset_index(drop=True)

pathology_genes.loc[
    pathology_genes["Kodetekst"].str.contains("|".join(negative_list), regex=True),
    "value",
] = 0
pathology_genes.loc[pathology_genes["value"].isna(), "value"] = 1

pathology_genes["variable_code"] = "all"
pathology_genes["data_source"] = "gene_alterations"

pathology_genes = pathology_genes[
    ["patientid", "d_rekvdato", "variable_code", "value", "data_source"]
].rename(columns={"d_rekvdato": "timestamp"})
