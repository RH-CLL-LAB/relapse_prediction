import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from timeseriesflattener.aggregators import MaxAggregator, MinAggregator  # MinAggregator unused but kept

from lyfo_treatment_failure_prediction.feature_specification import feature_specs  # noqa: F401
from lyfo_treatment_failure_prediction.feature_maker.scripts.feature_maker_old import (
    FeatureMaker,
)
from lyfo_treatment_failure_prediction.data_processing.wide_data import (
    WIDE_DATA as BASE_WIDE_DATA,
)
from lyfo_treatment_failure_prediction.data_processing.preprocess import (
    preprocess_data,
)
from lyfo_treatment_failure_prediction.utils.config import DATA_DIR


# Defaults chosen to exactly match the original script
CACHED_DATA_DEFAULT = True
INCLUDE_PROXIES_DEFAULT = False  # currently unused, kept for backwards compat
SINGLE_DISEASE_DEFAULT = False   # currently unused, kept for backwards compat

# Note: tqdm import was only used for side-effects (progress bars) in the
# original script when iterating with .apply; here we don't need it explicitly,
# but we keep the import to avoid any surprises if you later add progress bars.
from tqdm import tqdm  # noqa: F401


def _load_wide_and_long(cached_data: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load WIDE_DATA and LONG_DATA either from cached pickles or by
    running the preprocessing pipeline.

    Parameters
    ----------
    cached_data :
        If True, load `data/WIDE_DATA.pkl` and `data/LONG_DATA.pkl`.
        If False, call `preprocess_data()` to recompute them.

    Returns
    -------
    (WIDE_DATA, LONG_DATA)
    """
    if cached_data:
        wide_path = DATA_DIR / "WIDE_DATA.pkl"
        long_path = DATA_DIR / "LONG_DATA.pkl"
        WIDE_DATA = pd.read_pickle(wide_path)
        LONG_DATA = pd.read_pickle(long_path)
    else:
        # Original script imported WIDE_DATA, LONG_DATA from preprocess_data
        # when CACHED_DATA was False. Here we call the same pipeline function.
        WIDE_DATA, LONG_DATA = preprocess_data()

    return WIDE_DATA, LONG_DATA


def _prepare_feature_maker(
    WIDE_DATA: pd.DataFrame,
    LONG_DATA: pd.DataFrame,
) -> FeatureMaker:
    """
    Prepare FeatureMaker with static predictors and outcome definitions.

    This function contains the body of the original script **before**
    the `if __name__ == "__main__":` block.
    """

    # Convert date columns in WIDE_DATA.
    # NOTE: The original script had a subtle bug/quirk:
    # inside the loop it used `WIDE_DATA[date_columns]` rather than
    # `WIDE_DATA[date_column]`. We keep this behaviour to avoid changing
    # anything in the data.
    date_columns = [x for x in WIDE_DATA.columns if "date" in x]
    for date_column in date_columns:
        WIDE_DATA[date_column] = pd.to_datetime(
            WIDE_DATA[date_columns], errors="coerce"
        )

    LONG_DATA["patientid"] = LONG_DATA["patientid"].astype(int)

    feature_maker = FeatureMaker(long_data=LONG_DATA, wide_data=WIDE_DATA)

    feature_maker._reset_all_features()
    feature_maker.specify_prediction_time_from_wide_format(
        "date_treatment_1st_line"
    )

    # Remove response evaluation - leakage
    list_of_exclusion_terms = [
        "date",
        "2nd_line",
        "relapse",
        "OS",
        "LYFO",
        "report_submitted",
        "dead",
        "FU",
        "patientid",
        "1st_line",
        "death",
        "treatment",
        # "age",
        "proxy",
        "dato",
        "_dt",
    ]

    static_predictors = [
        x
        for x in feature_maker.wide_data.columns
        if not any(
            [exclusion_term in x for exclusion_term in list_of_exclusion_terms]
        )
    ]

    for static_predictor in static_predictors:
        static_predictor_specification = {
            "data_source": "RKKP_LYFO",
            "value_column": static_predictor,
            "fallback": -1,
            "feature_base_name": f"RKKP_LYFO_{static_predictor}",
        }
        feature_maker.add_static_feature(static_predictor_specification)

    # Relapse label translation
    translation_dict = {9: 0, 1: 1}  # 0: np.NAN

    feature_maker.wide_data["relapse"] = feature_maker.wide_data[
        "relapse_label"
    ].apply(lambda x: translation_dict.get(x))

    # Define successful treatment (unchanged logic)
    def define_succesful_treatment(date_death, date_relapse):
        if pd.isnull(date_death) and pd.isnull(date_relapse):
            succesful_treatment_date = pd.NaT
        elif pd.isnull(date_death) and pd.notnull(date_relapse):
            succesful_treatment_date = date_relapse
        elif pd.notnull(date_death) and pd.isnull(date_relapse):
            succesful_treatment_date = date_death
        elif pd.notnull(date_death) and pd.notnull(date_relapse):
            succesful_treatment_date = min((date_death, date_relapse))
        if pd.isnull(succesful_treatment_date):
            succesful_treatment_label = 0
        else:
            succesful_treatment_label = 1
        return succesful_treatment_date, succesful_treatment_label

    (
        feature_maker.wide_data["succesful_treatment_date"],
        feature_maker.wide_data["succesful_treatment_label"],
    ) = zip(
        *feature_maker.wide_data.apply(
            lambda x: define_succesful_treatment(
                x["date_death"], x["relapse_date"]
            ),
            axis=1,
        )
    )

    # Outcomes (same as original)
    feature_maker.add_outcome_from_wide_format(
        "date_death",
        "dead_label",
        [dt.timedelta(730), dt.timedelta(365 * 5)],
        [MaxAggregator()],
    )

    feature_maker.add_outcome_from_wide_format(
        "relapse_date",
        "relapse",
        [dt.timedelta(730), dt.timedelta(365 * 5)],
        [MaxAggregator()],
    )

    feature_maker.add_outcome_from_wide_format(
        "succesful_treatment_date",
        "succesful_treatment_label",
        [dt.timedelta(730)],
        [MaxAggregator()],
    )

    return feature_maker


def define_and_save_test_patient_ids(
    cached_data: bool = CACHED_DATA_DEFAULT,
    include_proxies: bool = INCLUDE_PROXIES_DEFAULT,
    single_disease: bool = SINGLE_DISEASE_DEFAULT,
    seed: int = 46,
    test_size: float = 0.15,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Reproduce the original define_test_patient_ids.py behaviour:
    - Build features
    - Create feature matrix
    - Stratified train/test split
    - Save test patient IDs to CSV.

    Returns the test feature matrix.
    """
    # Currently these flags are not used in the original script; kept
    # here only for signature compatibility.
    _ = include_proxies, single_disease

    WIDE_DATA, LONG_DATA = _load_wide_and_long(cached_data=cached_data)
    feature_maker = _prepare_feature_maker(WIDE_DATA, LONG_DATA)

    # === ORIGINAL __main__ BLOCK LOGIC ===

    feature_maker.make_all_features()

    feature_matrix = feature_maker.create_feature_matrix(None)
    feature_matrix = feature_maker.feature_matrix

    sum_columns = [
        x
        for x in feature_matrix.columns
        if "sum_fallback_-1" in x or "count_fallback_-1" in x
    ]

    for column in sum_columns:
        feature_matrix.loc[feature_matrix[column] == 0, column] = -1

    feature_matrix.loc[
        feature_matrix[
            "outc_dead_label_within_0_to_730_days_max_fallback_0"
        ]
        == -1,
        "outc_dead_label_within_0_to_730_days_max_fallback_0",
    ] = 0

    feature_matrix = feature_matrix.replace(np.NAN, -1)

    feature_matrix.loc[
        feature_matrix[
            "outc_dead_label_within_0_to_730_days_max_fallback_0"
        ]
        == -1,
        "outc_dead_label_within_0_to_730_days_max_fallback_0",
    ] = 0

    feature_matrix.loc[
        feature_matrix[
            "outc_dead_label_within_0_to_1825_days_max_fallback_0"
        ]
        == -1,
        "outc_dead_label_within_0_to_1825_days_max_fallback_0",
    ] = 0

    # Column name sanitisation (unchanged)
    feature_matrix.columns = feature_matrix.columns.str.replace(
        "<", "less_than"
    )
    feature_matrix.columns = feature_matrix.columns.str.replace(
        ",", "_comma_"
    )
    feature_matrix.columns = feature_matrix.columns.str.replace(
        ">", "more_than"
    )
    feature_matrix.columns = feature_matrix.columns.str.replace(
        "[", "left_bracket"
    )
    feature_matrix.columns = feature_matrix.columns.str.replace(
        "]", "right_bracket"
    )

    feature_matrix["group"] = feature_matrix.apply(
        lambda x: (
            f'relapse_{str(x["outc_relapse_within_0_to_730_days_max_fallback_0"])}'
            f'_death_{str(x["outc_dead_label_within_0_to_730_days_max_fallback_0"])}'
            f'_disease_{str(x["pred_RKKP_subtype_fallback_-1"])}'
        ),
        axis=1,
    )

    feature_matrix, test = train_test_split(
        feature_matrix,
        test_size=test_size,
        stratify=feature_matrix["group"],
        random_state=seed,
    )

    # Where to save
    output_path = (
        Path(output_path)
        if output_path is not None
        else DATA_DIR / "test_patientids.csv"
    )

    test["patientid"].reset_index(drop=True).to_csv(
        output_path, index=False
    )

    print(f"Test patient IDs saved to {output_path}")

    return test


def main() -> None:
    """CLI entry point replicating the original script behaviour."""
    define_and_save_test_patient_ids()
