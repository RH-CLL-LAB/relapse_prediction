from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from data_processing.wide_data import lyfo_cohort

blood_tests = download_and_rename_data(
    "SP_Bloddyrkning_Del1",
    {
        "SP_Bloddyrkning_Del1": {
            "patientid": "patientid",
            "komponentnavn": "variable_code",
            "prøvetagningstidspunkt": "timestamp",
            "prøveresultat": "value",
        }
    },
    cohort=lyfo_cohort,
)

first_negative_words = ["Ikke", "IKKE", "ikke"]

for negative_word in first_negative_words:
    blood_tests.loc[
        (blood_tests["value"].str.contains(negative_word))
        & (blood_tests["value"].notnull()),
        "value",
    ] = 0

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
    blood_tests.loc[
        (blood_tests["value"].str.contains(positive_word))
        & (blood_tests["value"].notnull()),
        "value",
    ] = 1

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
    blood_tests.loc[
        (blood_tests["value"].str.contains(negative_word))
        & (blood_tests["value"].notnull()),
        "value",
    ] = 0

blood_tests.loc[blood_tests["value"].isnull(), "value"] = 0
blood_tests.loc[blood_tests["value"].isna(), "value"] = 0


blood_tests["value"] = blood_tests["value"].astype(str)
blood_tests["value"] = blood_tests["value"].str.replace(",", ".")
blood_tests["value"] = blood_tests["value"].str.replace("=", "")
blood_tests["value"] = blood_tests["value"].str.replace("<", "")
blood_tests["value"] = blood_tests["value"].str.replace(">", "")

blood_tests["value_numeric"] = pd.to_numeric(blood_tests["value"], errors="coerce")
blood_tests.loc[blood_tests["value_numeric"].isna(), "value"] = 1
blood_tests["value"] = pd.to_numeric(blood_tests["value"], errors="coerce")
blood_tests = blood_tests[[x for x in blood_tests.columns if x != "value_numeric"]]
