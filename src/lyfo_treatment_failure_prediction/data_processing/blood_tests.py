"""
blood_tests.py — Process blood culture test results into numeric features.

Outputs:
- blood_tests: per-test, numeric result (0/1 or numeric where possible).
- blood_tests_all: per-patient indicator 'all' for any blood test.

Behaviour is identical to the original script.
"""

import pandas as pd
from tqdm import tqdm

from lyfo_treatment_failure_prediction.helpers.sql_helper import download_and_rename_data
from lyfo_treatment_failure_prediction.data_processing.wide_data import lyfo_cohort

tqdm.pandas()

# ---------------------------------------------------------------------------
# 1. Load raw blood culture data
# ---------------------------------------------------------------------------

blood_tests = download_and_rename_data(
    "SP_Bloddyrkning_del1",
    {
        "SP_Bloddyrkning_del1": {
            "patientid": "patientid",
            "komponentnavn": "variable_code",
            "prøvetagningstidspunkt": "timestamp",
            "prøveresultat": "value",
        }
    },
    cohort=lyfo_cohort,
)

# ---------------------------------------------------------------------------
# 2. Map textual results to 0/1 where possible
# ---------------------------------------------------------------------------

# First: map obvious 'not detected' / negative phrases to 0
first_negative_words = ["Ikke", "IKKE", "ikke"]

for negative_word in first_negative_words:
    mask = blood_tests["value"].notnull() & blood_tests["value"].str.contains(
        negative_word, na=False
    )
    blood_tests.loc[mask, "value"] = 0

# Positive phrases → 1
positive_words = [
    "vist",
    "KMA",
    "Sendt",
    " Positiv",
    " POSITIV",
    "PÅVIST",
    "Erstattet",
    "Positiv",
    "Er taget",
    "Blandingsflora",
    "LIST",
    "negative stave",
    "negative satfy",
    "Ej KB",
    "Taget",
    "SSI",
    "POS",
    "Høj",
    "Gruppesvar",
    "Brevsvar",
    "BREVSVAR",
    "Udført",
    "Ubeskyttet",
    "Svælgflora",
    "positive kokker",
    "vækst af blanding",
    "Mucus med cylinder",
    "RH KIA",
    "Staphylococcus",
    "Enterococcus",
    "positive stave",
    "Talrige Kokker",
    "pos diplokok",
    "B.1.1",
    "B1617",
    "Intiminproducerende",
    "Campylobacter",
    "Trichophyton",
    "Skimmelsvamp",
    "gonorrhoeae",
    "flere end 2 slags bakterier",
    "Shigella flexneri",
    "Aeromonas",
    "Yersinia",
    "E.coli",
    "Escherichia coli",
]

for positive_word in positive_words:
    mask = blood_tests["value"].notnull() & blood_tests["value"].str.contains(
        positive_word, na=False
    )
    blood_tests.loc[mask, "value"] = 1

# Additional negative phrases → 0 (overrides positives if conflicting)
negative_words = [
    "NEGATIV",
    "Ingen speciel flora",
    "Ej udført",
    "Lav",
    "ANNUL",
    "Annulleret",
    "Bortkommet",
    "MISLYKKET",
    "UAFLÆSELIG",
    "Makuleret",
    "Slettet",
    " Negativ",
    "Ingen vækst",
    "NEG",
    "Negativ",
    "negativ",
    r"\*\*\*\*\*",
    "Komm",
    "Aflyst",
    "cancelled",
    "Ingen signifikant vækst",
    "Afbestilt",
    "Mislykket",
    "INKONKLUSIV",
    "Inkonklusiv",
    "Ubedømmelig",
    "Beskyttet",
    "Fejlglad",
    "Dublet",
    "Ingen mikroorganismer",
    "Udelukkende svælg",
    "Vækst af normal hud-",
    "vækst af normal hud-",
    "Uegnet materiale",
    "Ingen bakterier",
    "INGEN vækst",
    "Vækst af normal svælgflora",
    "Ny prøve udb",
    "normal svælgflora",
    "Ej modtaget",
    "normal tarmflora",
    "Cellefattigt materiale",
    "normal bakterieflora",
    "INGEN VÆKST",
    "vækst af normal",
    "Ej tilstede",
    "ingen æg",
    "Vækst af normal",
    "Ingen bakt",
    "For cellefattigt materiale",
    "Ingen Syre",
    "Ingen krystaller",
    "For gammel",
]

for negative_word in negative_words:
    mask = blood_tests["value"].notnull() & blood_tests["value"].str.contains(
        negative_word, na=False
    )
    blood_tests.loc[mask, "value"] = 0

# Missing values default to 0
blood_tests.loc[blood_tests["value"].isnull(), "value"] = 0
blood_tests.loc[blood_tests["value"].isna(), "value"] = 0

# ---------------------------------------------------------------------------
# 3. Clean up and convert to numeric
# ---------------------------------------------------------------------------

blood_tests["value"] = blood_tests["value"].astype(str)
blood_tests["value"] = blood_tests["value"].str.replace(",", ".")
blood_tests["value"] = blood_tests["value"].str.replace("=", "")
blood_tests["value"] = blood_tests["value"].str.replace("<", "")
blood_tests["value"] = blood_tests["value"].str.replace(">", "")

blood_tests["value_numeric"] = pd.to_numeric(
    blood_tests["value"], errors="coerce"
)

# Any remaining non-numeric (i.e. text-y) → 1
blood_tests.loc[blood_tests["value_numeric"].isna(), "value"] = 1
blood_tests["value"] = pd.to_numeric(blood_tests["value"], errors="coerce")

# Drop intermediate column
blood_tests = blood_tests[[c for c in blood_tests.columns if c != "value_numeric"]]

# ---------------------------------------------------------------------------
# 4. Aggregate “any blood test” indicator
# ---------------------------------------------------------------------------

blood_tests_all = blood_tests.copy()
blood_tests_all.loc[:, "variable_code"] = "all"
blood_tests_all.loc[:, "data_source"] = "blood_tests_all"

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = ["blood_tests", "blood_tests_all"]
