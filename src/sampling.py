# src/sampling.py
"""
Sampling Strategies for Imbalanced Data
Section B & D: Handling Data Imbalance
"""

import numpy as np
import pandas as pd
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imbalanced-learn not installed. Only basic sampling available.")


class SamplingStrategy:
    """
    Various sampling strategies for handling class imbalance.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def get_class_distribution(self, y):
        """Show class distribution."""
        counter = Counter(y)
        total = len(y)
        print("\nClass Distribution:")
        for cls, count in sorted(counter.items()):
            print(f"  Class {cls}: {count:,} ({count/total*100:.2f}%)")
        return counter

    def random_undersample(self, X, y, ratio=1.0):
        """
        Random undersampling of majority class.
        ratio: desired ratio of majority to minority
        """
        print("\n--- Random Undersampling ---")
        self.get_class_distribution(y)

        if IMBLEARN_AVAILABLE:
            sampler = RandomUnderSampler(
                sampling_strategy=ratio,
                random_state=self.random_state
            )
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        else:
            # Manual implementation
            minority_class = y.value_counts().idxmin()
            majority_class = y.value_counts().idxmax()

            minority_idx = y[y == minority_class].index
            majority_idx = y[y == majority_class].index

            n_minority = len(minority_idx)
            n_majority_new = int(n_minority * ratio)

            np.random.seed(self.random_state)
            majority_sampled = np.random.choice(majority_idx, n_majority_new, replace=False)

            sampled_idx = np.concatenate([minority_idx, majority_sampled])
            X_resampled = X.loc[sampled_idx]
            y_resampled = y.loc[sampled_idx]

        print(f"After undersampling:")
        self.get_class_distribution(y_resampled)
        return X_resampled, y_resampled

    def smote_oversample(self, X, y, ratio='auto', k_neighbors=5):
        """
        SMOTE oversampling of minority class.
        Creates synthetic samples for minority class.
        """
        print("\n--- SMOTE Oversampling ---")
        self.get_class_distribution(y)

        if not IMBLEARN_AVAILABLE:
            print("SMOTE requires imbalanced-learn. Using random oversampling instead.")
            return self.random_oversample(X, y, ratio)

        sampler = SMOTE(
            sampling_strategy=ratio,
            k_neighbors=k_neighbors,
            random_state=self.random_state,
            n_jobs=-1
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        print(f"After SMOTE:")
        self.get_class_distribution(y_resampled)
        return X_resampled, y_resampled

    def random_oversample(self, X, y, ratio='auto'):
        """
        Random oversampling of minority class.
        Duplicates minority samples.
        """
        print("\n--- Random Oversampling ---")
        self.get_class_distribution(y)

        if IMBLEARN_AVAILABLE:
            sampler = RandomOverSampler(
                sampling_strategy=ratio,
                random_state=self.random_state
            )
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        else:
            # Manual implementation
            minority_class = y.value_counts().idxmin()
            majority_class = y.value_counts().idxmax()

            minority_idx = y[y == minority_class].index
            majority_idx = y[y == majority_class].index

            n_majority = len(majority_idx)
            n_to_add = n_majority - len(minority_idx)

            np.random.seed(self.random_state)
            oversample_idx = np.random.choice(minority_idx, n_to_add, replace=True)

            X_resampled = pd.concat([X, X.loc[oversample_idx]])
            y_resampled = pd.concat([y, y.loc[oversample_idx]])

        print(f"After oversampling:")
        self.get_class_distribution(y_resampled)
        return X_resampled, y_resampled

    def smote_tomek(self, X, y):
        """
        Combined SMOTE + Tomek links cleaning.
        Oversamples minority and removes noisy samples.
        """
        print("\n--- SMOTE + Tomek Links ---")
        self.get_class_distribution(y)

        if not IMBLEARN_AVAILABLE:
            print("SMOTETomek requires imbalanced-learn.")
            return X, y

        sampler = SMOTETomek(random_state=self.random_state, n_jobs=-1)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        print(f"After SMOTE-Tomek:")
        self.get_class_distribution(y_resampled)
        return X_resampled, y_resampled

    def adasyn_oversample(self, X, y):
        """
        ADASYN: Adaptive Synthetic Sampling.
        Generates more samples near decision boundary.
        """
        print("\n--- ADASYN Oversampling ---")
        self.get_class_distribution(y)

        if not IMBLEARN_AVAILABLE:
            print("ADASYN requires imbalanced-learn.")
            return self.random_oversample(X, y)

        try:
            sampler = ADASYN(random_state=self.random_state, n_jobs=-1)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        except ValueError as e:
            print(f"ADASYN failed: {e}")
            print("Falling back to SMOTE.")
            return self.smote_oversample(X, y)

        print(f"After ADASYN:")
        self.get_class_distribution(y_resampled)
        return X_resampled, y_resampled


def apply_sampling_strategy(X, y, strategy='smote', random_state=42, **kwargs):
    """
    Convenience function to apply sampling strategy.

    Args:
        X: Feature matrix
        y: Target vector
        strategy: One of 'undersample', 'oversample', 'smote', 'smote_tomek', 'adasyn'
        **kwargs: Additional arguments for the strategy
    """
    sampler = SamplingStrategy(random_state=random_state)

    strategies = {
        'undersample': sampler.random_undersample,
        'oversample': sampler.random_oversample,
        'smote': sampler.smote_oversample,
        'smote_tomek': sampler.smote_tomek,
        'adasyn': sampler.adasyn_oversample,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")

    return strategies[strategy](X, y, **kwargs)


def compute_class_weights(y):
    """
    Compute class weights for cost-sensitive learning.
    """
    counter = Counter(y)
    total = len(y)
    n_classes = len(counter)

    weights = {}
    for cls, count in counter.items():
        weights[cls] = total / (n_classes * count)

    print("\nComputed Class Weights:")
    for cls, weight in sorted(weights.items()):
        print(f"  Class {cls}: {weight:.4f}")

    return weights


if __name__ == "__main__":
    # Test sampling strategies
    print("Testing sampling strategies...")

    # Create imbalanced test data
    np.random.seed(42)
    n_majority = 1000
    n_minority = 100

    X = pd.DataFrame({
        'feature1': np.random.randn(n_majority + n_minority),
        'feature2': np.random.randn(n_majority + n_minority),
    })
    y = pd.Series([0]*n_majority + [1]*n_minority)

    print(f"\nOriginal data: {len(y)} samples")

    # Test each strategy
    sampler = SamplingStrategy()

    print("\n" + "="*50)
    X_under, y_under = sampler.random_undersample(X, y)

    print("\n" + "="*50)
    X_over, y_over = sampler.random_oversample(X, y)

    if IMBLEARN_AVAILABLE:
        print("\n" + "="*50)
        X_smote, y_smote = sampler.smote_oversample(X, y)

    print("\n" + "="*50)
    weights = compute_class_weights(y)



