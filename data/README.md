
# Loan Dataset README

## Overview
This repository uses the Lending Club Loan Data for credit risk modeling and machine learning experiments. The dataset supports research in financial analytics, especially for loan default prediction.

## Source and Ownership
- **Source:** [Lending Club Loan Data on Kaggle](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)
- **Ownership:** Data is provided by Lending Club and hosted on Kaggle. Use is permitted for non-commercial, educational, and research purposes.

## Data Structure
- **Format:** CSV (Comma-Separated Values)
- **Location:**  
	- Raw data: `data/raw/loan.csv`  
	- Processed data: `data/processed/loan_data_processed.csv`
- **Note:** Due to the large storage size of the dataset, the actual data files are not included in this repository. Please download the data directly from Kaggle using the link above.

## Contents

### Raw Data (`loan.csv`)
- Contains the original Lending Club loan records with extensive features.
- Example columns:
	- `id`, `member_id`, `loan_amnt`, `funded_amnt`, `funded_amnt_inv`, `term`, `int_rate`, `installment`, `grade`, `sub_grade`, `emp_title`, `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `issue_d`, `loan_status`, `purpose`, `zip_code`, `addr_state`, `dti`, `delinq_2yrs`, `earliest_cr_line`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, and many more.

### Processed Data (`loan_data_processed.csv`)
- Cleaned and feature-engineered version for modeling.
- Example columns:
	- `loan_amnt`, `funded_amnt`, `int_rate`, `installment`, `term`, `grade`, `sub_grade`, `emp_length`, `annual_inc`, `verification_status`, `home_ownership`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `addr_state`, `purpose`, `loan_status`, `default`, `credit_history_years`

#### Simulated Tabular Example

| loan_amnt | funded_amnt | int_rate | installment | term      | grade | emp_length | annual_inc | home_ownership | dti  | purpose           | loan_status | default |
|-----------|-------------|----------|-------------|-----------|-------|------------|------------|----------------|------|-------------------|-------------|---------|
| 12000     | 12000       | 13.56    | 400.12      | 36 months | C     | 5 years    | 45000      | RENT           | 18.2 | debt_consolidation| Fully Paid  | 0       |
| 8000      | 8000        | 10.99    | 260.45      | 36 months | B     | 2 years    | 30000      | MORTGAGE       | 15.7 | car               | Charged Off | 1       |
| ...       | ...         | ...      | ...         | ...       | ...   | ...        | ...        | ...            | ...  | ...               | ...         | ...     |

## Data Size
- **Raw Data:** Hundreds of thousands of records, 100+ features, >50MB
- **Processed Data:** Reduced features, still large (>50MB)

## Data Type
- Tabular data (structured rows and columns)
- Features include numerical, categorical, and binary types

## Big Data Characteristics
- **Volume:** Large dataset, suitable for scalable ML experiments
- **Variety:** Mixed feature types (numerical, categorical, binary)
- **Veracity:** Contains real-world noise, missing values, and inconsistencies
- **Velocity:** Static batch data, but can be used to simulate real-time scenarios

## Usage Notes
- Data is anonymized and should not be used for commercial purposes.
- Preprocessing (handling missing values, encoding categorical variables) is required before modeling.
- Class imbalance may be present in the target variable.
- Download the data from Kaggle: https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv

## Relevance
This dataset is used for developing and evaluating machine learning models for loan default prediction, supporting research in financial analytics and risk assessment.

## Citation
If you use this dataset, please cite the original data source (Kaggle, Lending Club) as appropriate.
