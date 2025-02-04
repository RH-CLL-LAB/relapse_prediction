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


pathologies = (
    pathology.groupby(["c_snomedkode", "Kodetekst"])
    .agg(patient_count=("patientid", "count"))
    .reset_index()
)

concatenated_pathology = pathology.copy()

concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T0X", regex=True),
    "variable_code",
] = "blood"

concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T02", regex=True),
    "variable_code",
] = "skin"

concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T06", regex=True),
    "variable_code",
] = "bone_marrow"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T08", regex=True),
    "variable_code",
] = "lymphs"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T28", regex=True),
    "variable_code",
] = "lungs"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T61", regex=True),
    "variable_code",
] = "palate"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T62", regex=True),
    "variable_code",
] = "esophagus"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T63", regex=True),
    "variable_code",
] = "ventricle"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T67", regex=True),
    "variable_code",
] = "colon"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T77", regex=True),
    "variable_code",
] = "prostate"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T6X", regex=True),
    "variable_code",
] = "cytology"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T7X", regex=True),
    "variable_code",
] = "cytology"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T9", regex=True),
    "variable_code",
] = "cytology"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^T8X", regex=True),
    "variable_code",
] = "cytology"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M00", regex=True),
    "variable_code",
] = "normal_cells"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M090", regex=True),
    "variable_code",
] = "unusable"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M091", regex=True),
    "variable_code",
] = "unusable"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M0945", regex=True),
    "variable_code",
] = "no_problem_cells"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M0946", regex=True),
    "variable_code",
] = "no_problem_cells"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M0947", regex=True),
    "variable_code",
] = "no_problem_cells"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M40", regex=True),
    "variable_code",
] = "inflamation"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M41", regex=True),
    "variable_code",
] = "acute_inflamation"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M42", regex=True),
    "variable_code",
] = "acute_inflamation"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M43", regex=True),
    "variable_code",
] = "chronic_inflamation"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M47", regex=True),
    "variable_code",
] = "lymphocyte_inflamation"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M54", regex=True),
    "variable_code",
] = "necrose"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M58", regex=True),
    "variable_code",
] = "athrophy"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M59", regex=True),
    "variable_code",
] = "cytopeny"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M6", regex=True),
    "variable_code",
] = "malign_cells"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M72", regex=True),
    "variable_code",
] = "hyperplasy"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M77", regex=True),
    "variable_code",
] = "proliferation"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M78", regex=True),
    "variable_code",
] = "lymphocyte"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M80", regex=True),
    "variable_code",
] = "tumor"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M81", regex=True),
    "variable_code",
] = "tumor"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M82", regex=True),
    "variable_code",
] = "adenoms"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M83", regex=True),
    "variable_code",
] = "adenoms"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M84", regex=True),
    "variable_code",
] = "adenoms"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M85", regex=True),
    "variable_code",
] = "adenoms"

concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M87", regex=True),
    "variable_code",
] = "naevus"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^MÆ", regex=True),
    "variable_code",
] = "text_specification"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^S4", regex=True),
    "variable_code",
] = "anaemi"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^P282", regex=True),
    "variable_code",
] = "bone_marrow"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^P306", regex=True),
    "variable_code",
] = "biopsy"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^P309", regex=True),
    "variable_code",
] = "biopsy"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^P308", regex=True),
    "variable_code",
] = "consultancy"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^P31", regex=True),
    "variable_code",
] = "cytology"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^P33", regex=True),
    "variable_code",
] = "extra_analyses"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^P35", regex=True),
    "variable_code",
] = "extra_analyses"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M95", regex=True),
    "variable_code",
] = "malign_lymphoma"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^M96", regex=True),
    "variable_code",
] = "malign_lymphoma"


concatenated_pathology.loc[
    concatenated_pathology["c_snomedkode"].str.contains("^ÆX", regex=True),
    "variable_code",
] = "unexpected_finding"


concatenated_pathology.loc[
    (concatenated_pathology["c_snomedkode"].str.contains("^ÆX", regex=True))
    & (~concatenated_pathology["c_snomedkode"].isin(["ÆYYY01", "ÆYYY05"])),
    "variable_code",
] = "progression"

concatenated_pathology = concatenated_pathology[
    concatenated_pathology["variable_code"].notna()
].reset_index(drop=True)

concatenated_pathology["value"] = 1
concatenated_pathology["data_source"] = "pathology_concat"

concatenated_pathology = concatenated_pathology[
    ["patientid", "d_rekvdato", "value", "data_source", "variable_code"]
].rename(columns={"d_rekvdato": "timestamp"})


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
