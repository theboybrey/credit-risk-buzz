# Loan Dataset README

## Overview
This dataset contains anonymized loan application records used for credit risk modeling and machine learning experiments. It is intended for academic research and educational purposes.

## Source and Ownership
- The dataset is sourced from public financial data repositories (e.g., Kaggle, LendingClub).
- Ownership remains with the original data provider; usage is permitted for non-commercial, educational research.

## Data Structure
- **Format:** CSV (Comma-Separated Values)
- **Location:**  
	- Raw data: loan.csv  
	- Processed data: loan_data_processed.csv

## Contents
Each row represents a loan application. The columns (features) typically include:

| loan_id | applicant_income | loan_amount | term | employment_type | credit_score | loan_purpose | gender | marital_status | education | property_area | loan_status |
|---------|------------------|-------------|------|-----------------|-------------|--------------|--------|---------------|-----------|---------------|-------------|
| 10001   | 45000            | 12000       | 36   | Salaried        | 720         | Home         | Male   | Married        | Graduate  | Urban         | Paid        |
| 10002   | 30000            | 8000        | 24   | Self-Employed   | 680         | Car          | Female | Single         | Graduate  | Rural         | Defaulted   |
| 10003   | 52000            | 15000       | 60   | Salaried        | 750         | Education    | Male   | Married        | Not Grad. | Semiurban     | Paid        |
| ...     | ...              | ...         | ...  | ...             | ...         | ...          | ...    | ...            | ...       | ...           | ...         |

## Data Size
- **Records:** ~10,000 (actual count may vary)
- **Features:** ~12 columns
- **Storage Size:** ~2.5 MB (loan.csv)

## Data Type
- Tabular data (structured rows and columns)
- Features include numerical, categorical, and binary types

## Big Data Characteristics
- **Volume:** Sufficient for scalable ML experiments
- **Variety:** Mixed feature types (numerical, categorical, binary)
- **Veracity:** Contains real-world noise, missing values, and inconsistencies
- **Velocity:** Static batch data, suitable for simulation of real-time scenarios

## Usage Notes
- Data is anonymized and should not be used for commercial purposes.
- Preprocessing (handling missing values, encoding categorical variables) is required before modeling.
- Class imbalance may be present in the target variable.

## Relevance
This dataset is used for developing and evaluating machine learning models for loan default prediction, supporting research in financial analytics and risk assessment.

## Citation
If you use this dataset, please cite the original data source (e.g., Kaggle, LendingClub) as appropriate.
