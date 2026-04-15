# LYFO Treatment Failure / Relapse Prediction

This repository contains the full ML pipeline for predicting **treatment failure or relapse within two years of first-line treatment** in DLBCL and related lymphoma subtypes. The pipeline covers data preprocessing, feature engineering, model training, evaluation, and survival analysis.

The project relies on the **DALY-CARE** data resource (clinical registry + electronic health records). For background on DALY-CARE, see the [data resource paper](https://pubmed.ncbi.nlm.nih.gov/39996155/).

> **Note:** Running this pipeline requires access to the DALY-CARE database. The code is published for transparency and reproducibility in related research settings — it cannot be run against publicly available data without adaptation.

---

## Repository structure

```
scripts/
├── preprocess_data.py              # Entry point: load and harmonize raw data
├── define_test_patient_ids.py      # Entry point: create train/test split
├── survival_TFS.R                  # Kaplan–Meier (treatment failure)
├── survival_OS.R                   # Kaplan–Meier (overall survival)
├── survival_TFS_concordance.R      # Concordance analysis
├── survival_TFS_concordance_LR.R   # Concordance analysis (logistic regression)
└── R/
    ├── helpers_survival.R
    └── helpers_concordance.R

src/lyfo_treatment_failure_prediction/
├── data_processing/
│   ├── wide_data.py                # LYFO cohort definition; builds WIDE_DATA
│   ├── preprocess_data.py          # Orchestrates all data loading and merging
│   ├── laboratory_measurements.py  # Lab measurements (labsystem + PERSIMUNE)
│   ├── lab_values.py               # Derived lab value features
│   ├── blood_tests.py              # Blood test features
│   ├── medicine.py                 # Medication data
│   ├── pathology_specifics.py      # Pathology codes
│   ├── picture_diagnostics.py      # Imaging/radiology data
│   ├── sks_opr.py                  # Surgical procedure codes
│   ├── social_history.py           # Socioeconomic variables
│   ├── persimune.py                # PERSIMUNE immunology data
│   ├── lyfo_aki.py                 # AKI flags
│   ├── lookup_tables.py            # SNOMED/SKS lookup tables
│   ├── define_test_patient_ids.py  # Test cohort logic
│   └── get_patient_characteristics.py  # Descriptive tables; IPI/CNS-IPI scores
├── feature_engineering/
│   ├── feature_specification.py   # Defines which features to build and how
│   ├── long_to_feature_matrix.py  # Converts long-format data to feature matrix
│   ├── feature_selection.py       # Lasso-based feature selection
│   └── feature_maker/             # Core aggregation machinery
├── modeling/
│   ├── stratify.py                # Main XGBoost model (DLBCL)
│   ├── stratify_only_IPI_variables.py       # IPI-only XGBoost baseline
│   ├── stratify_other_lymphomas.py          # Models for other lymphoma subtypes
│   ├── stratify_logistic.py                 # Logistic regression baseline
│   ├── stratify_tabular.py                  # TabPFN training + evaluation
│   ├── stratify_tabular_results.py          # TabPFN results and plots
│   ├── stratify_feature_minimization.py     # Feature ablation analysis
│   ├── stratify_performance_differences.py  # Subgroup performance analysis
│   ├── stratify_get_individual_outcomes_for_statistical_testing.py
│   ├── cross_validate.py          # Cross-validation
│   ├── make_confusion_matrices.py # Confusion matrix plots
│   └── benchmark_tabpfn.py        # TabPFN runtime benchmarks
├── analysis/
│   ├── compare_models.py                         # Compare XGBoost vs IPI vs TabPFN
│   ├── make_data_for_survival_plots.py           # Survival data (XGBoost)
│   ├── make_data_for_survival_plots_LR.py        # Survival data (logistic regression)
│   ├── make_data_for_survival_plots_OS.py        # Survival data (overall survival)
│   ├── stratify_IPI_only_variables_train.py      # Train IPI-only model
│   ├── stratify_only_IPI_variables_evaluate.py   # Evaluate IPI-only model
│   ├── stratify_only_IPI_variables_plot_calibration.py
│   ├── stratify_only_IPI_variables_plot_pr_roc.py
│   └── stratify_only_IPI_variables_plot_shap.py
├── helpers/
│   ├── constants.py               # Shared constants (supplemental columns, etc.)
│   ├── processing_helper.py       # Metric computation and feature utilities
│   └── sql_helper.py              # Database access helpers
├── visualization/
│   └── plot_age_reliance.py
└── utils/
    └── config.py                  # Database/environment configuration
```

---

## Installation

**Requirements:** Python ≥ 3.10, and optionally R ≥ 4.1 for survival analyses.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Verify:

```bash
python -c "import lyfo_treatment_failure_prediction; print('OK')"
```

---

## Running the pipeline

Scripts are designed to be run in order. Each step reads from `data/` and `results/` and writes its outputs there — those directories are created automatically on first run.

### 1. Data preprocessing

Loads raw data from the DALY-CARE database, harmonizes it, and produces `data/WIDE_DATA.pkl` along with the long-format data tables used by feature engineering.

```bash
python scripts/preprocess_data.py
```

### 2. Define the test cohort

Splits patients into train and test sets and saves `data/test_patientids.csv`.

```bash
python scripts/define_test_patient_ids.py
```

### 3. Build the feature matrix

Converts long-format time-series data into a flat feature matrix per prediction time.

```bash
python -m lyfo_treatment_failure_prediction.feature_engineering.long_to_feature_matrix
```

Optional: run Lasso-based feature selection to produce `results/feature_names_all.csv`.

```bash
python -m lyfo_treatment_failure_prediction.feature_engineering.feature_selection
```

### 4. Cross-validation

```bash
python -m lyfo_treatment_failure_prediction.modeling.cross_validate
```

### 5. Train and evaluate models

Main XGBoost model (DLBCL, trains on DLBCL only by default):

```bash
python -m lyfo_treatment_failure_prediction.modeling.stratify
```

IPI-only baseline:

```bash
python -m lyfo_treatment_failure_prediction.analysis.stratify_IPI_only_variables_train
python -m lyfo_treatment_failure_prediction.analysis.stratify_only_IPI_variables_evaluate
```

Other lymphoma subtypes:

```bash
python -m lyfo_treatment_failure_prediction.modeling.stratify_other_lymphomas
```

TabPFN comparison model:

```bash
python -m lyfo_treatment_failure_prediction.modeling.stratify_tabular
```

### 6. Survival analysis (R)

Requires the CSV outputs from step 5 and the analysis scripts.

```bash
python -m lyfo_treatment_failure_prediction.analysis.make_data_for_survival_plots
python -m lyfo_treatment_failure_prediction.analysis.make_data_for_survival_plots_OS

Rscript scripts/survival_TFS.R
Rscript scripts/survival_OS.R
Rscript scripts/survival_TFS_concordance.R
```

---

## Data requirements

All input data is accessed via the DALY-CARE data platform. The pipeline expects a configured database connection (see `src/lyfo_treatment_failure_prediction/utils/config.py`). The following external data sources are used:

- RKKP LYFO registry (baseline clinical variables, IPI scores, treatment data)
- Laboratory measurements (labsystem + PERSIMUNE)
- Diagnosis codes (ICD-10 / SKS)
- Medication (ATC codes)
- Pathology and radiology data
- Socioeconomic register data

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{lyfo_tfpipeline,
  title  = {LYFO Treatment Failure Prediction Pipeline},
  author = {Werling, M. and collaborators},
  year   = {2025},
  note   = {https://github.com/RH-CLL-LAB/relapse_prediction}
}
```

---

**Maintainer:** Mikkel Werling — please open an issue for questions or bug reports.
