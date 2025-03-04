from helpers.sql_helper import *
from helpers.processing_helper import *
from data_processing.wide_data import lyfo_cohort_strings

PERSIMUNE_MAPPING = {
    "PERSIMUNE_microbiology_analysis": {
        "patientid": "patientid",
        "samplingdatetime": "timestamp",
        "analysisshortname": "variable_code",
        "c_categoricalresult": "value",
    },
    "PERSIMUNE_microbiology_culture": {
        "patientid": "patientid",
        "samplingdatetime": "timestamp",
        "c_domain": "variable_code",
        "c_pm_categoricalresult": "value",
    },
}

# persimune patientid is in some weird object format :'()

persimune_dict = {
    table_name: download_and_rename_data(
        table_name, PERSIMUNE_MAPPING, cohort=lyfo_cohort_strings
    )
    for table_name in PERSIMUNE_MAPPING
}
# convert categorical results

value_convertion_dict = {
    "Negative": 0,
    "NULL": 0,
    "Positive": 1,
    "Possible Cont.": 1,
    "Not analyzed": 0,
}

persimune_dict["PERSIMUNE_microbiology_culture"]["value"] = persimune_dict[
    "PERSIMUNE_microbiology_culture"
]["value"].apply(lambda x: value_convertion_dict.get(x, 0))

persimune_dict["PERSIMUNE_microbiology_analysis"].loc[
    persimune_dict["PERSIMUNE_microbiology_analysis"]["value"].isna(), "value"
] = "Missing"

positive_values = ["Positive", "High"]

persimune_dict["PERSIMUNE_microbiology_analysis"]["value"] = persimune_dict[
    "PERSIMUNE_microbiology_analysis"
]["value"].apply(
    lambda x: 1 if any(positive_word in x for positive_word in positive_values) else 0
)

# convert categorical results to 1s and zeroes

overview = (
    persimune_dict["PERSIMUNE_microbiology_analysis"]["variable_code"]
    .value_counts()
    .reset_index()
)

# group 500 variables

persimune_dict["PERSIMUNE_microbiology_analysis"] = persimune_dict[
    "PERSIMUNE_microbiology_analysis"
][
    persimune_dict["PERSIMUNE_microbiology_analysis"]["variable_code"] != "NULL"
].reset_index(
    drop=True
)
persimune_dict["PERSIMUNE_microbiology_analysis"] = persimune_dict[
    "PERSIMUNE_microbiology_analysis"
][
    persimune_dict["PERSIMUNE_microbiology_analysis"]["variable_code"] != "_"
].reset_index(
    drop=True
)
persimune_dict["PERSIMUNE_microbiology_analysis"] = persimune_dict[
    "PERSIMUNE_microbiology_analysis"
][
    persimune_dict["PERSIMUNE_microbiology_analysis"]["variable_code"].notna()
].reset_index(
    drop=True
)

variable_dict = {
    "covid": ["SARS-CoV-2", "Coronavirus", "Corona", "coronavirus", "SARS-Cov"],
    "herpes": ["Herpes"],
    "cmv": ["CMV"],
    "influenza": ["Influenza", "Inf."],
    "parainfluenza": ["Parainfluenza"],
    "epstein": ["Epstein", "EBV", "EBNA IgG"],
    "cytomegalovirus": ["Cytomegalovirus"],
    "chlamydophila": ["Chlamydophila", "Chlamy."],
    "chlamydia": ["Chlamydia"],
    "adenovirus": ["Adeno"],
    "rotavirus": ["Rota"],
    "bocavirus": ["Boca"],
    "sapovirus": ["Sapo"],
    "noro": ["Norovirus"],
    "astro": ["Astro"],
    "entero": ["Entero"],
    "rhino": ["Rhino"],
    "toxoplasma": ["Toxoplasma", "Toxoplasmose", "Toxo"],
    # "toxoplasmose": ["Toxoplasmose"],
    "bartonella": ["Bartonella"],
    "hepatitis": ["Hepatitis"],
    "pneumokok": ["Pneumokok"],
    # "ebv": ["EBV"],
    "legionella": ["Legionella"],
    "aspergillus": ["Aspergillus"],
    "varicella": ["Varicella"],
}

# there are probably more groupings but for now lets stick to these

for group, search_terms in variable_dict.items():
    persimune_dict["PERSIMUNE_microbiology_analysis"]["variable_code"] = persimune_dict[
        "PERSIMUNE_microbiology_analysis"
    ]["variable_code"].apply(
        lambda x: group if any(search_term in x for search_term in search_terms) else x
    )


# we need to check for slopes if we can also get an intercept
# and not just the coefficient
# check with some dummy examples and make predictions before hand

# there might be more crazy variables (in biochemistry) - can you do the same trick?

persimune_dict["PERSIMUNE_leukocytes"] = download_and_rename_data(
    "PERSIMUNE_biochemistry",
    {
        "PERSIMUNE_biochemistry": {
            "patientid": "patientid",
            "samplingdatetime": "timestamp",
            "analysiscode": "variable_code",
            "c_associatedleukocytevalue": "value",
        }
    },
    cohort=lyfo_cohort_strings,
)

persimune_dict["PERSIMUNE_leukocytes"]["data_source"] = "PERSIMUNE_leukocytes"
# overall leukocytes
persimune_dict["PERSIMUNE_leukocytes"]["variable_code"] = "all"


for dataset in persimune_dict:
    persimune_dict[dataset]["patientid"] = persimune_dict[dataset]["patientid"].astype(
        int
    )
    persimune_dict[dataset]["timestamp"] = pd.to_datetime(
        persimune_dict[dataset]["timestamp"], errors="coerce", utc=True
    )

    persimune_dict[dataset]["timestamp"] = persimune_dict[dataset][
        "timestamp"
    ].dt.tz_localize(None)
