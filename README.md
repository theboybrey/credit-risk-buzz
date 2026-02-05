# Loan Default Prediction: ML Model Comparison Under Class Imbalance

MSc Computer Science - Machine Learning with Big Data Analytics

## Overview
Comparative analysis of machine learning models for loan default prediction, 
focusing on performance under class imbalance conditions.

## Dataset
Lending Club Loan Data (Kaggle)
- Source: https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
- Size: [TO BE FILLED]
- Features: [TO BE FILLED]

## Research Questions
1. How do different ML models perform on imbalanced loan default data?
2. Which sampling strategies improve minority class detection?
3. What are the trade-offs between precision and recall?

## Models
- Logistic Regression
- Random Forest
- XGBoost

## Sampling Strategies
- Original (imbalanced)
- SMOTE (oversampling)
- Random undersampling
- Class weighting

## Setup
```bash
pip install -r requirements.txt
python src/data_loader.py
```

## Project Structure
```
├── data/              # Dataset files
├── src/               # Source code
├── notebooks/         # Jupyter notebooks
├── experiments/       # Results and outputs
└── requirements.txt   # Dependencies
```

## Author
@theboybrey - University of Ghana, Department of Computer Science