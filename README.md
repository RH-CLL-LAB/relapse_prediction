# LYFO Treatment Failure / Relapse Prediction

This repository contains the full pipeline for predicting **treatment failure or relapse** in lymphoma (DLBCL) using multi‑modal clinical data and machine learning.  
It provides data preprocessing, feature engineering, model training, evaluation, and survival analysis scripts.

Critically, the project relies on the **DALY‑CARE** data resource for input and some preprocessing.  
For background, see the [paper detailing the DALY‑CARE data resource](https://pubmed.ncbi.nlm.nih.gov/39996155/).

The project is structured for **reproducibility**, **transparency**, and **ease of reuse** in related research settings.

---

## Overview

| Stage | Description |
|--------|--------------|
| **1. Data preprocessing** | Extracts and harmonizes raw clinical data (from SQL or CSV). |
| **2. Cohort definition** | Defines prediction windows and identifies test cohorts. |
| **3. Feature generation** | Converts time‑series and categorical data into feature matrices. |
| **4. Model training** | Performs cross‑validation and model selection across multiple algorithms. |
| **5. Evaluation & stratification** | Applies final models, stratifies patients, and prepares survival analyses. |
| **6. Survival analysis (R)** | Produces Kaplan–Meier and concordance plots using R scripts. |

---

## Repository structure

```
├─ scripts/
│  ├─ preprocess_data.py
│  ├─ define_test_patient_ids.py
│  ├─ long_to_feature_matrix.py
│  ├─ feature_selection.py
│  ├─ cross_validate.py
│  ├─ stratify.py
│  ├─ stratify_only_IPI_variables.py
│  ├─ stratify_other_lymphomas.py
│  └─ R/
│     ├─ survival_TFS.R
│     ├─ survival_OS.R
│     └─ survival_TFS_concordance*.R
├─ src/
│  └─ lyfo_treatment_failure_prediction/
│     ├─ data_processing/
│     ├─ feature_engineering/
│     ├─ modeling/
│     ├─ analysis/
│     ├─ helpers/
│     ├─ visualization/
│     └─ __init__.py
├─ configs/
├─ requirements.txt
├─ pyproject.toml
├─ ROADMAP.md
└─ README.md
```

---

## Installation

### Prerequisites
- **Python ≥ 3.10**
- (optional) **R ≥ 4.1** for survival analyses

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install dependencies and the package

To reproduce the exact research environment:

```bash
pip install -r requirements.txt
```

For development (editable install):

```bash
pip install -e .
```

Verify installation:

```bash
python -c "import lyfo_treatment_failure_prediction as m; print('OK:', m.__name__)"
```

---

## Usage

Each step of the workflow can be run independently.  
Default outputs are written to `data/` and `results/` folders.

### 1. Data preprocessing

```bash
python scripts/preprocess_data.py
```

Produces harmonized datasets such as `WIDE_DATA.pkl` and optionally `LONG_DATA.pkl`.

### 2. Define the test cohort

```bash
python scripts/define_test_patient_ids.py
```

### 3. Feature generation

```bash
python -m lyfo_treatment_failure_prediction.feature_engineering.long_to_feature_matrix
```

Optional feature selection:

```bash
python -m lyfo_treatment_failure_prediction.feature_engineering.feature_selection
```

### 4. Cross‑validation

```bash
python -m lyfo_treatment_failure_prediction.modeling.cross_validate
```

### 5. Final model training and stratification

```bash
python -m lyfo_treatment_failure_prediction.modeling.stratify
```

Additional models:

```bash
python -m lyfo_treatment_failure_prediction.modeling.stratify_only_IPI_variables
python -m lyfo_treatment_failure_prediction.modeling.stratify_other_lymphomas
```

### 6. Survival analysis (R)

```bash
Rscript scripts/R/survival_TFS.R
Rscript scripts/R/survival_OS.R
```

---


## Citation

If you use this repository in your research, please cite the corresponding manuscript once available.

```bibtex
@misc{lyfo_tfpipeline,
  title  = {LYFO Treatment Failure Prediction Pipeline},
  author = {Werling, M. and collaborators},
  year   = {2025},
  note   = {https://github.com/RH-CLL-LAB/relapse_prediction}
}
```

---

**Maintainer:** Mikkel Werling  
**Contact:** Please open issues or pull requests for questions, bug reports, or suggestions.
