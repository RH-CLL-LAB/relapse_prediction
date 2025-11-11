# database connections
import json

# Read data from file:
with open(
    "/ngc/people/mikwer_r/db_access.json", "r"
) as f:  # CHANGE HERE FOR RELEVANT FILE PATH
    password_dict = json.load(f)


OPTIONS_DICTIONARY = {
    "database": "import",
    "host": "localhost",
    "port": 5432,
    "user": "mikwer_r",  # CHANGE HERE FOR RELEVANT USER NAME
    "password": password_dict["password"],
    "options": "-c search_path=public",
}

supplemental_columns = [
    "pred_RKKP_tumor_diameter_diagnosis_fallback_-1",
    "pred_RKKP_LDH_diagnosis_fallback_-1",
    "pred_RKKP_ALB_diagnosis_fallback_-1",
    "pred_RKKP_TRC_diagnosis_fallback_-1",
    "pred_RKKP_AA_stage_diagnosis_fallback_-1",
    "pred_RKKP_extranodal_disease_diagnosis_fallback_-1",
]

included_treatments = ["chop", "choep", "maxichop"]


colors = [
    "#D8AE9C",
    "#EE8D73",
    "#DC4649",
    "#CA0020",
    "#AEBBC3",
    "#7DB9D7",
    "#4195C3",
    "#0571B0",
]
