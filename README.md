# ehr4fl
EHR benchmarks for federated learning

## preprocessing
1. split.r:
   - input: df_eicu.csv
   - splits data into train, validation, and test (80/10/10) sets for each hospitalid (20 hospitalids included).
2. preprocess.py
   - input: output csvs from 1)
   - Performs missing value imputation: Mean imputation for numeric columns and mode imputation for binary columns. Then performs standardization on all feature columns.  

##  In-hospital mortality

In-hospital mortality prediction based on data collected during the first 24h of their ICU stay is a crucial task in clinical setting. When a patient is admitted to the ICU, predicting their mortality serves as an indicative measure of the patient's condition severity, thereby assisting healthcare providers in strategizing treatment pathways and optimizing resource allocation with greater efficacy.
