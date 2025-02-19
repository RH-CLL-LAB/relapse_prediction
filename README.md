# Machine Learning Enhances Risk Stratification and Treatment Failure Prediction in Diffuse Large B-Cell Lymphoma
This repository contains the code used for the paper "Machine Learning Enhances Risk Stratification and Treatment Failure Prediction in Diffuse Large B-Cell Lymphoma" currently submitted to Blood. 

## Overview of the repository
*data_processing* contains individual preprocessing scripts for different data modalities. 
*feature_maker* contains the FeatureMaker class, which is the backbone of the feature generation process.
*helpers* contains key support functions for for instance loading the data from SQL.
*R* contains the R scripts used for survival analysis.

Each file in the main folder is explained individually.

## Main files
*cross-validate.py* performs the cross-validation on the training set for the different models. 
*define_test_patient_ids.py* generates the ids for the unseen testset. 
*feature_selection.py* performs feature selection via lasso regression.
*feature_specification.py* specifies the aggregation functions and data sources used for feature generation.
*get_patient_characteristics.py* generates the data for the patient overview table.
*long_to_feature_matrix.py* converts the WIDE and LONG data formats into a feature matrix.
*make_data_for_survival_plots.py* generates the data used for survival plots. 
*preprocess_data.py* calls on the individual scripts in **data_processing** and preprocesses the data.
*stratify_only_IPI_variables.py* provides the code for generating the $ML_{\text{IPI}}$ model as well as key figures.
*stratify_other_lymphomas.py* contains the code for testing performance on other lymphomas than DLBCL.
*stratify.py* contains the code for fitting $ML_{\text{DLBCL}}$ and $ML_{\text{All}}$