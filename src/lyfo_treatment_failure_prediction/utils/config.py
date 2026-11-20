from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
REFERENCE_RESULTS = PROJECT_ROOT / "reference_results"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_IPI_PATH = Path(
    "../../../../../projects2/dalyca_r/clean_r/shared_projects/"
    "end_of_project_scripts_to_gihub/DALYCARE_methods/output/IPI_2.csv"
)

# Allow overriding via environment variable if you move the file:
#   export LYFO_IPI_PATH=/new/location/IPI_2.csv
IPI_PATH = Path(os.getenv("LYFO_IPI_PATH", str(_DEFAULT_IPI_PATH)))

SEED = 42
N_FOLDS = 5