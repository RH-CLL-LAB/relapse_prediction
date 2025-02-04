# database connections
import json

# Read data from file:
with open("/ngc/people/mikwer_r/db_access.json", "r") as f:
    password_dict = json.load(f)


OPTIONS_DICTIONARY = {
    "database": "import",
    "host": "localhost",
    "port": 5432,
    "user": "mikwer_r",
    "password": password_dict["password"],
    "options": "-c search_path=public",
}
