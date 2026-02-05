# src/models.py
"""
Machine Learning Models for Loan Default Prediction
Section D: ML Model Design
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Using GradientBoosting as fallback.")


class ModelTrainer:
    """
    Handles training, evaluation, and comparison of ML models.
    Designed for scalability with large datasets.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.encoders = {}

    def prepare_data(self, df, target_col='default', test_size=0.2):
        """
        Prepare data for model training.
        Handles encoding and scaling.
        """
        print("="*60)
        print("PREPARING DATA FOR MODELING")
        print("="*60)

        # Separate features and target
        X = df.drop(columns=[target_col, 'loan_status'], errors='ignore')
        y = df[target_col]

        # Identify categorical and numerical columns
        cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        print(f"\nCategorical features ({len(cat_cols)}): {cat_cols}")
        print(f"Numerical features ({len(num_cols)}): {len(num_cols)} columns")

        # Encode categorical variables
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le

        # Handle missing values
        X = X.fillna(X.median())

        # Train-test split (stratified to maintain class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Class distribution in train: {dict(y_train.value_counts())}")
        print(f"Class distribution in test: {dict(y_test.value_counts())}")

        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

        self.scalers['standard'] = scaler
        self.feature_names = X.columns.tolist()

        return X_train_scaled, X_test_scaled, y_train, y_test

    def get_model_configs(self):
        """
        Define model configurations with hyperparameters.
        Includes scalable models suitable for big data.
        """
        configs = {
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    solver='saga',  # Scalable solver for large datasets
                    n_jobs=-1
                ),
                'param_grid': {
                    'C': [0.01, 0.1, 1.0],
                    'penalty': ['l1', 'l2']
                },
                'description': 'Linear model with regularization. Fast, interpretable, scalable.'
            },
            'RandomForest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,  # Parallel processing
                    class_weight='balanced'  # Handle imbalance
                ),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [5, 10]
                },
                'description': 'Tree-based ensemble. Handles non-linearity, feature importance.'
            }
        }

        if XGBOOST_AVAILABLE:
            configs['XGBoost'] = {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=-1,
                    tree_method='hist'  # Faster for large datasets
                ),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1],
                    'scale_pos_weight': [1, 4]  # Handle imbalance (ratio ~4:1)
                },
                'description': 'Gradient boosting. State-of-art for tabular data. Scalable.'
            }
        else:
            configs['GradientBoosting'] = {
                'model': GradientBoostingClassifier(
                    random_state=self.random_state
                ),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1]
                },
                'description': 'Gradient boosting (sklearn fallback). Good for imbalanced data.'
            }

        return configs

    def train_model(self, model_name, X_train, y_train, X_test, y_test,
                    tune_hyperparameters=False, cv_folds=3):
        """
        Train a single model with optional hyperparameter tuning.
        """
        print(f"\n{'='*60}")
        print(f"TRAINING: {model_name}")
        print("="*60)

        configs = self.get_model_configs()
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")

        config = configs[model_name]
        print(f"Description: {config['description']}")

        start_time = time.time()

        if tune_hyperparameters:
            print(f"\nPerforming GridSearchCV with {cv_folds} folds...")
            grid_search = GridSearchCV(
                config['model'],
                config['param_grid'],
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            model = config['model']
            model.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"\nTraining time: {training_time:.2f} seconds")

        # Store trained model
        self.models[model_name] = model

        # Evaluate
        results = self.evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
        results['training_time'] = training_time
        self.results[model_name] = results

        return model, results

    def evaluate_model(self, model, model_name, X_train, y_train, X_test, y_test):
        """
        Comprehensive model evaluation.
        """
        print(f"\n--- Evaluation Results ---")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilities (for AUC)
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_train_proba = y_train_pred
            y_test_proba = y_test_pred

        results = {
            'train': {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred),
                'recall': recall_score(y_train, y_train_pred),
                'f1': f1_score(y_train, y_train_pred),
                'roc_auc': roc_auc_score(y_train, y_train_proba)
            },
            'test': {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1': f1_score(y_test, y_test_pred),
                'roc_auc': roc_auc_score(y_test, y_test_proba)
            },
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }

        # Print results
        print(f"\n{'Metric':<15} {'Train':>10} {'Test':>10}")
        print("-" * 37)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            print(f"{metric:<15} {results['train'][metric]:>10.4f} {results['test'][metric]:>10.4f}")

        print(f"\nConfusion Matrix (Test):")
        print(results['confusion_matrix'])

        print(f"\nClassification Report (Test):")
        print(classification_report(y_test, y_test_pred, target_names=['Non-Default', 'Default']))

        return results

    def get_feature_importance(self, model_name):
        """
        Extract feature importance from trained model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet.")

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def compare_models(self):
        """
        Create comparison table of all trained models.
        """
        if not self.results:
            print("No models trained yet.")
            return None

        comparison = []
        for model_name, results in self.results.items():
            row = {'Model': model_name}
            row.update({f"Train_{k}": v for k, v in results['train'].items()})
            row.update({f"Test_{k}": v for k, v in results['test'].items()})
            row['Training_Time'] = results.get('training_time', 0)
            comparison.append(row)

        comparison_df = pd.DataFrame(comparison)
        return comparison_df


def train_all_models(df, models_to_train=['LogisticRegression', 'RandomForest', 'XGBoost'],
                     tune=False, sample_size=None):
    """
    Convenience function to train all specified models.

    Args:
        df: Preprocessed DataFrame
        models_to_train: List of model names
        tune: Whether to perform hyperparameter tuning
        sample_size: Optional sample size for faster experimentation
    """
    trainer = ModelTrainer()

    # Optional sampling for faster experiments
    if sample_size and len(df) > sample_size:
        print(f"\nSampling {sample_size:,} records for training...")
        df = df.sample(n=sample_size, random_state=42)

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)

    # Train each model
    for model_name in models_to_train:
        try:
            trainer.train_model(model_name, X_train, y_train, X_test, y_test,
                              tune_hyperparameters=tune)
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    # Compare results
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison = trainer.compare_models()
    print(comparison.to_string(index=False))

    return trainer, comparison


if __name__ == "__main__":
    # Test with processed data
    import sys
    sys.path.append('..')
    from config import PROCESSED_DATA_DIR

    df = pd.read_csv(PROCESSED_DATA_DIR / "loan_data_processed.csv")
    print(f"Loaded {len(df):,} records")

    # Train models (with sampling for quick test)
    trainer, comparison = train_all_models(
        df,
        models_to_train=['LogisticRegression', 'RandomForest'],
        tune=False,
        sample_size=100000  # Use subset for quick testing
    )
