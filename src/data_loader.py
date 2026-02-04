# src/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
import kagglehub
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def download_lending_club_data():
    """Download Lending Club dataset from Kaggle"""
    print("Downloading Lending Club dataset...")
    path = kagglehub.dataset_download("adarshsng/lending-club-loan-data-csv")
    print(f"Dataset downloaded to: {path}")
    return path

def load_data(filepath=None):
    """Load the Lending Club dataset"""
    if filepath is None:
        # Try to find CSV in raw data directory
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if not csv_files:
            print("No CSV found. Downloading dataset...")
            download_path = download_lending_club_data()
            # Copy to raw data directory
            import shutil
            shutil.copy(Path(download_path) / "loan.csv", RAW_DATA_DIR / "loan.csv")
            filepath = RAW_DATA_DIR / "loan.csv"
        else:
            filepath = csv_files[0]
    
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def initial_exploration(df):
    """Quick data exploration"""
    print("\n" + "="*50)
    print("INITIAL DATA EXPLORATION")
    print("="*50)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    return df

if __name__ == "__main__":
    # Test the data loader
    df = load_data()
    initial_exploration(df)