# src/evaluation.py
"""
Model Evaluation and Robustness Analysis
Section E: Evaluation, Robustness & Scalability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, learning_curve
import time


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute comprehensive evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': None,  # Will compute from confusion matrix
    }

    # Compute specificity (true negative rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn

    # AUC-ROC (requires probabilities)
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

        # Precision-Recall AUC (better for imbalanced data)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        metrics['pr_auc'] = auc(recall_vals, precision_vals)

    return metrics


def evaluate_at_scales(model, X, y, scales=[0.1, 0.25, 0.5, 0.75, 1.0], random_state=42):
    """
    Evaluate model performance at different data scales.
    Demonstrates scalability analysis.

    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        scales: Fractions of data to use
    """
    results = []

    for scale in scales:
        n_samples = int(len(X) * scale)
        if n_samples < 100:
            continue

        # Sample data
        np.random.seed(random_state)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y_sample = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]

        # Time prediction
        start = time.time()
        y_pred = model.predict(X_sample)
        pred_time = time.time() - start

        # Compute metrics
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_sample)[:, 1]
        else:
            y_proba = None

        metrics = compute_metrics(y_sample, y_pred, y_proba)
        metrics['scale'] = scale
        metrics['n_samples'] = n_samples
        metrics['prediction_time'] = pred_time
        metrics['throughput'] = n_samples / pred_time  # samples per second

        results.append(metrics)

    return pd.DataFrame(results)


def evaluate_robustness_noise(model, X, y, noise_levels=[0, 0.05, 0.1, 0.2], random_state=42):
    """
    Test model robustness to noise in features.
    Adds Gaussian noise to numerical features.
    """
    results = []
    np.random.seed(random_state)

    # Get numerical columns
    if hasattr(X, 'select_dtypes'):
        num_cols = X.select_dtypes(include=[np.number]).columns
    else:
        num_cols = range(X.shape[1])

    X_array = X.values if hasattr(X, 'values') else X

    for noise_level in noise_levels:
        X_noisy = X_array.copy().astype(float)

        if noise_level > 0:
            # Add Gaussian noise proportional to feature std
            for col_idx, col in enumerate(num_cols):
                col_std = np.std(X_array[:, col_idx])
                noise = np.random.normal(0, noise_level * col_std, len(X_array))
                X_noisy[:, col_idx] += noise

        y_pred = model.predict(X_noisy)
        y_proba = model.predict_proba(X_noisy)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = compute_metrics(y, y_pred, y_proba)
        metrics['noise_level'] = noise_level

        results.append(metrics)

    return pd.DataFrame(results)


def evaluate_imbalance_sensitivity(model, X_train, y_train, X_test, y_test,
                                    imbalance_ratios=[(0.5, 0.5), (0.7, 0.3), (0.9, 0.1)],
                                    random_state=42):
    """
    Test model performance under different class imbalance ratios.
    Simulates varying imbalance by undersampling.
    """
    results = []
    np.random.seed(random_state)

    for ratio in imbalance_ratios:
        majority_ratio, minority_ratio = ratio

        # Get class indices
        minority_class = 1  # Default (loan default)
        majority_class = 0

        majority_idx = np.where(y_train == majority_class)[0]
        minority_idx = np.where(y_train == minority_class)[0]

        # Calculate target sizes
        total_minority = len(minority_idx)
        total_majority = int(total_minority * (majority_ratio / minority_ratio))

        if total_majority > len(majority_idx):
            total_majority = len(majority_idx)

        # Sample
        sampled_majority = np.random.choice(majority_idx, total_majority, replace=False)
        sampled_indices = np.concatenate([sampled_majority, minority_idx])
        np.random.shuffle(sampled_indices)

        X_sampled = X_train.iloc[sampled_indices] if hasattr(X_train, 'iloc') else X_train[sampled_indices]
        y_sampled = y_train.iloc[sampled_indices] if hasattr(y_train, 'iloc') else y_train[sampled_indices]

        # Retrain model on sampled data
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_sampled, y_sampled)

        # Evaluate on test set
        y_pred = model_copy.predict(X_test)
        y_proba = model_copy.predict_proba(X_test)[:, 1] if hasattr(model_copy, 'predict_proba') else None

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics['majority_ratio'] = majority_ratio
        metrics['minority_ratio'] = minority_ratio
        metrics['train_size'] = len(y_sampled)
        metrics['actual_imbalance'] = f"{y_sampled.value_counts()[0]/len(y_sampled):.2f}:{y_sampled.value_counts()[1]/len(y_sampled):.2f}"

        results.append(metrics)

    return pd.DataFrame(results)


def plot_roc_curves(models_dict, X_test, y_test, save_path=None):
    """
    Plot ROC curves for multiple models.

    Args:
        models_dict: Dictionary {model_name: trained_model}
        X_test: Test features
        y_test: Test labels
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))

    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_precision_recall_curves(models_dict, X_test, y_test, save_path=None):
    """
    Plot Precision-Recall curves (better for imbalanced data).
    """
    plt.figure(figsize=(10, 8))

    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, linewidth=2, label=f'{name} (AUC = {pr_auc:.3f})')

    # Baseline (random classifier)
    baseline = y_test.mean()
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                label=f'Baseline (prevalence = {baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve Comparison', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_learning_curves(model, X, y, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
                         cv=5, scoring='roc_auc', save_path=None):
    """
    Plot learning curves to analyze overfitting/underfitting.
    """
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))

    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')

    plt.plot(train_sizes_abs, test_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                     alpha=0.1, color='red')

    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel(f'{scoring.upper()}', fontsize=12)
    plt.title('Learning Curves', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf(), {'train_sizes': train_sizes_abs,
                       'train_mean': train_mean, 'train_std': train_std,
                       'test_mean': test_mean, 'test_std': test_std}


def generate_evaluation_report(model, model_name, X_train, y_train, X_test, y_test):
    """
    Generate comprehensive evaluation report for a model.
    """
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"EVALUATION REPORT: {model_name}")
    report.append("="*60)

    # Basic metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = compute_metrics(y_test, y_pred, y_proba)

    report.append("\n--- Performance Metrics (Test Set) ---")
    for metric, value in metrics.items():
        if isinstance(value, float):
            report.append(f"{metric}: {value:.4f}")
        else:
            report.append(f"{metric}: {value}")

    # Classification report
    report.append("\n--- Classification Report ---")
    report.append(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))

    # Confusion matrix
    report.append("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    report.append(f"TN={cm[0,0]:,}, FP={cm[0,1]:,}")
    report.append(f"FN={cm[1,0]:,}, TP={cm[1,1]:,}")

    # Scale evaluation
    report.append("\n--- Scalability Analysis ---")
    scale_results = evaluate_at_scales(model, X_test, y_test)
    report.append(scale_results.to_string(index=False))

    return "\n".join(report)


if __name__ == "__main__":
    print("Evaluation module loaded. Import and use with trained models.")
