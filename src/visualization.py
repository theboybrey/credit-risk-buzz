# src/visualization.py
"""
Visualization Functions for Analysis and Reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def set_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_class_distribution(y, title='Class Distribution', save_path=None):
    """Plot class distribution bar chart."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    counts = pd.Series(y).value_counts()
    bars = ax.bar(counts.index.astype(str), counts.values, color=['#2ecc71', '#e74c3c'])

    # Add value labels on bars
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{count:,}\n({count/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticklabels(['Non-Default (0)', 'Default (1)'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=['Non-Default', 'Default'],
                          title='Confusion Matrix', save_path=None):
    """Plot confusion matrix heatmap."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14})

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_feature_importance(importance_df, top_n=15, title='Feature Importance',
                           save_path=None):
    """Plot horizontal bar chart of feature importance."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = importance_df.head(top_n)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                   color=colors)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_roc_comparison(models_results, save_path=None):
    """
    Plot ROC curves for multiple models.

    Args:
        models_results: Dict of {model_name: (y_true, y_proba)}
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))

    for (name, (y_true, y_proba)), color in zip(models_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_pr_comparison(models_results, save_path=None):
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        models_results: Dict of {model_name: (y_true, y_proba)}
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))

    for (name, (y_true, y_proba)), color in zip(models_results.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f'{name} (AUC = {pr_auc:.3f})')

    # Baseline
    baseline = list(models_results.values())[0][0].mean()
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
               label=f'Baseline ({baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_metrics_comparison(comparison_df, metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                           save_path=None):
    """
    Plot bar chart comparing metrics across models.

    Args:
        comparison_df: DataFrame with model comparison results
    """
    set_style()

    # Extract test metrics
    test_metrics = comparison_df[['Model'] + [f'Test_{m}' for m in metrics]].copy()
    test_metrics.columns = ['Model'] + metrics

    # Melt for plotting
    df_melted = test_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score')

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', ax=ax)

    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison (Test Set)')
    ax.legend(title='Model', loc='lower right')

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_correlation_matrix(df, numerical_only=True, save_path=None):
    """Plot correlation matrix heatmap."""
    set_style()

    if numerical_only:
        df_num = df.select_dtypes(include=[np.number])
    else:
        df_num = df

    corr = df_num.corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, annot_kws={'size': 8})

    ax.set_title('Feature Correlation Matrix')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_scale_performance(scale_results_df, metric='roc_auc', save_path=None):
    """
    Plot model performance at different data scales.

    Args:
        scale_results_df: DataFrame from evaluate_at_scales
        metric: Metric to plot
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Performance vs scale
    ax1.plot(scale_results_df['n_samples'], scale_results_df[metric],
             'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel(metric.upper())
    ax1.set_title(f'{metric.upper()} vs Data Scale')
    ax1.grid(True, alpha=0.3)

    # Throughput vs scale
    ax2.plot(scale_results_df['n_samples'], scale_results_df['throughput'],
             'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Prediction Throughput vs Data Scale')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def create_summary_dashboard(trainer, X_test, y_test, save_path=None):
    """
    Create a multi-panel summary dashboard.

    Args:
        trainer: ModelTrainer instance with trained models
        X_test: Test features
        y_test: Test labels
    """
    set_style()
    fig = plt.figure(figsize=(16, 12))

    # 1. Class distribution
    ax1 = fig.add_subplot(2, 3, 1)
    counts = pd.Series(y_test).value_counts()
    ax1.bar(counts.index.astype(str), counts.values, color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Class Distribution (Test Set)')
    ax1.set_xticklabels(['Non-Default', 'Default'])

    # 2. ROC curves
    ax2 = fig.add_subplot(2, 3, 2)
    for name, model in trainer.models.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, linewidth=2, label=f'{name} ({roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_title('ROC Curves')
    ax2.legend(fontsize=9)

    # 3. Metrics comparison
    ax3 = fig.add_subplot(2, 3, 3)
    comparison = trainer.compare_models()
    metrics = ['Test_accuracy', 'Test_precision', 'Test_recall', 'Test_f1']
    x = np.arange(len(metrics))
    width = 0.25
    for i, (_, row) in enumerate(comparison.iterrows()):
        values = [row[m] for m in metrics]
        ax3.bar(x + i*width, values, width, label=row['Model'])
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([m.replace('Test_', '') for m in metrics])
    ax3.set_title('Metrics Comparison')
    ax3.legend(fontsize=9)
    ax3.set_ylim([0, 1])

    # 4-6. Confusion matrices for each model
    for i, (name, model) in enumerate(trainer.models.items()):
        if i >= 3:
            break
        ax = fig.add_subplot(2, 3, 4+i)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    print("Visualization module loaded. Import and use functions for plotting.")
