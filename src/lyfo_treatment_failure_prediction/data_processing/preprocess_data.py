from datetime import timedelta
from pathlib import Path

from tqdm import tqdm
import pandas as pd

from lyfo_treatment_failure_prediction.helpers.sql_helper import *
from lyfo_treatment_failure_prediction.helpers.processing_helper import *

from lyfo_treatment_failure_prediction.data_processing.medicine import *
from lyfo_treatment_failure_prediction.data_processing.persimune import persimune_dict
from lyfo_treatment_failure_prediction.data_processing.lookup_tables import *
from lyfo_treatment_failure_prediction.data_processing.social_history import *
from lyfo_treatment_failure_prediction.data_processing.blood_tests import *
from lyfo_treatment_failure_prediction.data_processing.lyfo_aki import *
from lyfo_treatment_failure_prediction.data_processing.wide_data import *
from lyfo_treatment_failure_prediction.data_processing.sks_opr import *
from lyfo_treatment_failure_prediction.data_processing.picture_diagnostics import *
from lyfo_treatment_failure_prediction.data_processing.lab_values import *
from lyfo_treatment_failure_prediction.data_processing.laboratory_measurements import (
    lab_measurements_data,
    lab_measurements_data_all,
)
from lyfo_treatment_failure_prediction.data_processing.pathology_specifics import (
    pathology_genes,
    concatenated_pathology,
)

from lyfo_treatment_failure_prediction.utils.config import (
    IPI_PATH,
    DATA_DIR,
)


def preprocess_data(
    ipi_path: Path | str | None = None,
    wide_output_path: Path | str | None = None,
    long_output_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the preprocessing pipeline that builds WIDE_DATA and LONG_DATA.

    Parameters
    ----------
    ipi_path :
        Path to the IPI_2.csv file. If None, uses config.IPI_PATH
        (which by default points to the historical location used in the
        original script, but can be overridden via the LYFO_IPI_PATH
        environment variable).
    wide_output_path :
        Where to write WIDE_DATA.pkl. Defaults to DATA_DIR / "WIDE_DATA.pkl".
    long_output_path :
        Where to write LONG_DATA.pkl. Defaults to DATA_DIR / "LONG_DATA.pkl".

    Returns
    -------
    (WIDE_DATA, LONG_DATA)
        The two DataFrames also written to disk.
    """

    # Resolve paths with backwards-compatible defaults
    ipi_path = Path(ipi_path) if ipi_path is not None else IPI_PATH
    wide_output_path = (
        Path(wide_output_path)
        if wide_output_path is not None
        else DATA_DIR / "WIDE_DATA.pkl"
    )
    long_output_path = (
        Path(long_output_path)
        if long_output_path is not None
        else DATA_DIR / "LONG_DATA.pkl"
    )

    # Load IPIs and WIDE DATA
    print("Loading IPIs and WIDE DATA...")
    IPIs = pd.read_csv(ipi_path, sep=";")

    # Convert categorical column to numerical codes
    IPIs["IPI"] = pd.Categorical(
        IPIs["IPI"],
        ordered=True,
        categories=["Low", "Intermediate", "High", "Very high"],
    ).codes

    # Aggregate performance status and IPI by patient
    IPIs_concat = (
        IPIs.groupby("patientid")
        .agg(PS=("PS", "mean"), IPI=("IPI", "mean"))
        .reset_index()
    )

    # Expand IPIs dataset with disease-specific columns
    diseases = IPIs["Disease"].unique()
    for disease in diseases:
        disease_mask = IPIs["Disease"] == disease
        IPIs.loc[disease_mask, f"{disease}_PS"] = IPIs.loc[disease_mask, "PS"]
        IPIs.loc[disease_mask, f"{disease}_IPI"] = IPIs.loc[disease_mask, "IPI"]

    # Forward fill missing values per patient
    IPIs_ffill = (
        IPIs.drop(columns=["Sex", "PS", "IPI", "Disease"])
        .groupby("patientid")
        .fillna(method="ffill")
    )
    IPIs_ffill["patientid"] = IPIs["patientid"]
    IPIs_ffill = IPIs_ffill.groupby("patientid").agg("last").reset_index()

    # Merge with aggregated data and WIDE_DATA (WIDE_DATA comes from imported wide_data module)
    IPIs_ffill = IPIs_ffill.merge(IPIs_concat, how="left")
    global WIDE_DATA  # make explicit that we mutate the imported global
    WIDE_DATA = WIDE_DATA.merge(IPIs_ffill, how="left")

    # Define table mapping for transformation
    TABLE_TO_LONG_FORMAT_MAPPING = {
        "SDS_pato": {
            "d_rekvdato": "timestamp",
            "patientid": "patientid",
            "c_snomedkode": "variable_code",
        },
        "diagnoses_all": {
            "patientid": "patientid",
            "date_diagnosis": "timestamp",
            "diagnosis": "variable_code",
        },
        "SP_VitaleVaerdier": {
            "patientid": "patientid",
            "recorded_time": "timestamp",
            "displayname": "variable_code",
            "meas_value_clean": "value",
        },
        "PERSIMUNE_radiology": {
            "patientid": "patientid",
            "bookingdatetime": "timestamp",
            "resultcode": "variable_code",
        },
    }

    print("Loading data from data dictionary...")
    # Download and rename data based on mapping
    data_dict = {
        table_name: download_and_rename_data(
            table_name,
            TABLE_TO_LONG_FORMAT_MAPPING,
            cohort=lyfo_cohort,
            cohort_column="patientid",
        )
        for table_name in tqdm(TABLE_TO_LONG_FORMAT_MAPPING)
    }

    # Add additional datasets to data dictionary
    data_dict.update(
        {
            "sks_at_the_hospital": sks_at_the_hospital,
            "sks_referals": sks_referals,
            "sks_at_the_hospital_unique": sks_at_the_hospital_unique,
            "sks_referals_unique": sks_referals_unique,
        }
    )

    data_dict.update(persimune_dict)
    data_dict.update(medicine_dict)
    data_dict.update(lab_data)

    # Free memory (same as before)
    del persimune_dict, medicine_dict

    # Merge diagnoses with lookup table
    data_dict["diagnoses_all"] = (
        data_dict["diagnoses_all"]
        .merge(
            DIAG_LOOKUP_TABLE, left_on="variable_code", right_on="Kode"
        )[["patientid", "timestamp", "data_source", "Tekst"]]
        .rename(columns={"Tekst": "variable_code"})
        .reset_index(drop=True)
    )

    # Filter diagnoses before treatment
    diagnosis_all_filtered = data_dict["diagnoses_all"].merge(
        WIDE_DATA[["patientid", "date_treatment_1st_line"]], how="left"
    )
    diagnosis_all_filtered = (
        diagnosis_all_filtered[
            diagnosis_all_filtered["timestamp"]
            < diagnosis_all_filtered["date_treatment_1st_line"]
        ]
        .reset_index(drop=True)[
            ["patientid", "timestamp", "variable_code", "data_source"]
        ]
    )

    # Assigning general comorbidity labels
    diagnosis_all_comorbidity = (
        diagnosis_all_filtered.groupby(["patientid", "variable_code"])
        .agg(timestamp=("timestamp", "max"))
        .reset_index()
    )
    diagnosis_all_comorbidity["variable_code"] = "all"
    diagnosis_all_comorbidity["value"] = 1
    diagnosis_all_comorbidity["data_source"] = "diagnoses_all_comorbidity"
    data_dict["diagnoses_all_comorbidity"] = diagnosis_all_comorbidity

    # Merge pathology codes
    data_dict["SDS_pato"] = (
        data_dict["SDS_pato"]
        .merge(
            SNOMED_LOOKUP_TABLE,
            left_on="variable_code",
            right_on="SKSkode",
        )[["patientid", "timestamp", "data_source", "Kodetekst"]]
        .rename(columns={"Kodetekst": "variable_code"})
        .reset_index(drop=True)
    )

    # Additional data integration
    data_dict.update(
        {
            "labmeasurements": lab_measurements_data,
            "lab_measurements_data_all": lab_measurements_data_all,
            "SP_Social_Hx": social_history_data,
            "SP_Bloddyrkning_del1": blood_tests,
            "blood_tests_all": blood_tests_all,
            "gene_alterations": pathology_genes,
            "pathology_concat": concatenated_pathology,
            "LYFO_AKI": LYFO_AKI.assign(
                timestamp=pd.to_datetime(LYFO_AKI["timestamp"])
            ),
            "SP_BilleddiagnostiskeUndersøgelser_Del1": picture_diagnostics,
        }
    )

    for data_source_name, data_source in data_dict.items():
        data_source = data_source.merge(
            WIDE_DATA[["patientid", "date_treatment_1st_line"]], how="left"
        )

        data_source = data_source[
            data_source["timestamp"] < data_source["date_treatment_1st_line"]
        ].reset_index(drop=True)[
            [x for x in data_source.columns if x != "date_treatment_1st_line"]
        ]
        data_dict[data_source_name] = data_source

    # Combine long data format
    LONG_DATA = pd.concat(data_dict.values()).reset_index(drop=True)
    LONG_DATA = LONG_DATA[LONG_DATA["timestamp"] != "NULL"].reset_index(
        drop=True
    )
    LONG_DATA["timestamp"] = pd.to_datetime(LONG_DATA["timestamp"])

    # Assign default values for certain categories
    LONG_DATA.loc[
        LONG_DATA["data_source"].isin(
            ["SDS_pato", "diagnoses_all", "PERSIMUNE_radiology"]
        ),
        "value",
    ] = 1
    LONG_DATA.loc[
        LONG_DATA["data_source"] == "LYFO_AKI", "variable_code"
    ] = "n_aki"
    LONG_DATA.loc[LONG_DATA["value"].isna(), "value"] = 1

    # Ensure patient IDs are integer
    LONG_DATA["patientid"] = LONG_DATA["patientid"].astype(int)

    # Convert date columns in WIDE_DATA
    for column in WIDE_DATA.columns:
        if "date" in column:
            WIDE_DATA[column] = pd.to_datetime(
                WIDE_DATA[column], errors="coerce"
            )

    # Save outputs (paths controlled via arguments / config)
    WIDE_DATA.to_pickle(wide_output_path)
    LONG_DATA.to_pickle(long_output_path)

    print(f"WIDE_DATA saved to {wide_output_path}")
    print(f"LONG_DATA saved to {long_output_path}")

    return WIDE_DATA, LONG_DATA


def main() -> None:
    """Entry point used by the script wrapper."""
    preprocess_data()