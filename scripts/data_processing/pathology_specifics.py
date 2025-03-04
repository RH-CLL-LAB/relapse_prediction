from helpers.sql_helper import *
from helpers.processing_helper import *
from data_processing.wide_data import lyfo_cohort
from data_processing.lookup_tables import SNOMED_LOOKUP_TABLE

# Load pathology data
pathology = load_data_from_table(
    "SDS_pato",
    cohort=lyfo_cohort,
    subset_columns=["patientid", "d_rekvdato", "c_snomedkode"],
)

# Merge with SNOMED lookup table
pathology = pathology.merge(
    SNOMED_LOOKUP_TABLE, left_on="c_snomedkode", right_on="SKSkode"
)

# Aggregate pathologies by code and description
pathologies = (
    pathology.groupby(["c_snomedkode", "Kodetekst"])
    .agg(patient_count=("patientid", "count"))
    .reset_index()
)

# Copy pathology data for further processing
concatenated_pathology = pathology.copy()

# Define mapping of SNOMED codes to variable codes
snomed_mapping = {
    "^T0X": "blood",
    "^T02": "skin",
    "^T06": "bone_marrow",
    "^T08": "lymphs",
    "^T28": "lungs",
    "^T61": "palate",
    "^T62": "esophagus",
    "^T63": "ventricle",
    "^T67": "colon",
    "^T77": "prostate",
    "^T6X": "cytology",
    "^T7X": "cytology",
    "^T8X": "cytology",
    "^T9": "cytology",
    "^M00": "normal_cells",
    "^M090": "unusable",
    "^M091": "unusable",
    "^M0945": "no_problem_cells",
    "^M0946": "no_problem_cells",
    "^M0947": "no_problem_cells",
    "^M40": "inflamation",
    "^M41": "acute_inflamation",
    "^M42": "acute_inflamation",
    "^M43": "chronic_inflamation",
    "^M47": "lymphocyte_inflamation",
    "^M54": "necrose",
    "^M58": "athrophy",
    "^M59": "cytopeny",
    "^M6": "malign_cells",
    "^M72": "hyperplasy",
    "^M77": "proliferation",
    "^M78": "lymphocyte",
    "^M80": "tumor",
    "^M81": "tumor",
    "^M82": "adenoms",
    "^M83": "adenoms",
    "^M84": "adenoms",
    "^M85": "adenoms",
    "^M87": "naevus",
    "^MÆ": "text_specification",
    "^S4": "anaemi",
    "^P282": "bone_marrow",
    "^P306": "biopsy",
    "^P309": "biopsy",
    "^P308": "consultancy",
    "^P31": "cytology",
    "^P33": "extra_analyses",
    "^P35": "extra_analyses",
    "^M95": "malign_lymphoma",
    "^M96": "malign_lymphoma",
    "^ÆX": "unexpected_finding",
}


# Apply mappings
for pattern, variable in snomed_mapping.items():
    concatenated_pathology.loc[
        concatenated_pathology["c_snomedkode"].str.contains(pattern, regex=True),
        "variable_code",
    ] = variable

# Special case for progression
concatenated_pathology.loc[
    (concatenated_pathology["c_snomedkode"].str.contains("^ÆX", regex=True))
    & (~concatenated_pathology["c_snomedkode"].isin(["ÆYYY01", "ÆYYY05"])),
    "variable_code",
] = "progression"

# Remove rows with missing variable_code
concatenated_pathology = concatenated_pathology.dropna(
    subset=["variable_code"]
).reset_index(drop=True)

# Add metadata columns
concatenated_pathology["value"] = 1
concatenated_pathology["data_source"] = "pathology_concat"

# Select and rename columns
concatenated_pathology = concatenated_pathology[
    ["patientid", "d_rekvdato", "value", "data_source", "variable_code"]
].rename(columns={"d_rekvdato": "timestamp"})

# Extract pathology codes containing "FE"
pathology_codes = (
    pathology[pathology["c_snomedkode"].str.contains("^FE", regex=True)]["Kodetekst"]
    .value_counts()
    .reset_index()
)

# Define negative gene alteration indicators
negative_list = [
    "genstatus normal",
    "ikke påvist",
    "uden genamplikation",
    "negativ",
    "ikke hypermuteret",
]

# Filter pathology genes containing "FE"
pathology_genes = pathology[
    pathology["c_snomedkode"].str.contains("^FE", regex=True)
].reset_index(drop=True)

# Assign values based on negative indicators
pathology_genes.loc[
    pathology_genes["Kodetekst"].str.contains("|".join(negative_list), regex=True),
    "value",
] = 0
pathology_genes.loc[pathology_genes["value"].isna(), "value"] = 1

# Add metadata columns
pathology_genes["variable_code"] = "all"
pathology_genes["data_source"] = "gene_alterations"

# Select and rename columns
pathology_genes = pathology_genes[
    ["patientid", "d_rekvdato", "variable_code", "value", "data_source"]
].rename(columns={"d_rekvdato": "timestamp"})
