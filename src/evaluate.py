"""
evaluate.py — Evaluation metrics and visualization.

Generates:
  • Confusion matrices (classification + anomaly)
  • Model comparison bar chart
  • Feature importance plot
  • Attack distribution chart
  • Anomaly score distribution
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def _save_fig(fig, name: str, save_dir: str = 'reports/figures'):
    """Save a matplotlib figure to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', filename='confusion_matrix.png'):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    _save_fig(fig, filename)


def plot_model_comparison(results: dict, filename='model_comparison.png'):
    """Bar chart comparing accuracy and F1 of all models."""
    model_names = [k for k in results if k != 'best_model_name']
    accuracies = [results[k]['accuracy'] for k in model_names]
    f1_scores  = [results[k]['f1_macro'] for k in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#2196F3')
    bars2 = ax.bar(x + width/2, f1_scores,  width, label='F1 (macro)', color='#FF9800')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    _save_fig(fig, filename)


def plot_feature_importance(model, feature_names, top_n=15, filename='feature_importance.png'):
    """Horizontal bar chart of top-N most important features.

    Supports tree-based models (feature_importances_) and linear models
    (coef_ — uses mean absolute coefficient across classes).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For multi-class linear models, coef_ is (n_classes, n_features)
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print("  ⚠ Model has no feature_importances_ or coef_, skipping.")
        return

    top_n = min(top_n, len(importances))
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices], color='#4CAF50')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    _save_fig(fig, filename)


def plot_attack_distribution(y_train, label_encoder, filename='attack_distribution.png'):
    """Pie + bar chart of attack category distribution."""
    labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    counts = np.bincount(y_train, minlength=len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    colors = ['#4CAF50', '#F44336', '#FF9800', '#2196F3', '#9C27B0']
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Attack Distribution (Train)')

    # Bar chart
    ax2.bar(labels, counts, color=colors)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Count')
    ax2.set_title('Attack Category Counts')
    for i, v in enumerate(counts):
        ax2.text(i, v + 200, f'{v:,}', ha='center', fontsize=9)

    plt.tight_layout()
    _save_fig(fig, filename)


def plot_anomaly_scores(anomaly_scores, y_true_binary, filename='anomaly_scores.png'):
    """Distribution of anomaly scores for normal vs attack samples."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(anomaly_scores[y_true_binary == 0], bins=50, alpha=0.6, label='Normal', color='#4CAF50')
    ax.hist(anomaly_scores[y_true_binary == 1], bins=50, alpha=0.6, label='Attack', color='#F44336')
    ax.axvline(x=0, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Anomaly Score Distribution')
    ax.legend()
    _save_fig(fig, filename)


def generate_all_plots(
    clf_results: dict,
    anomaly_results: dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    feature_names: list[str],
):
    """Generate all evaluation plots."""
    print("\n📊 Generating Evaluation Plots")
    print("=" * 40)

    label_names = list(label_encoder.classes_)
    best_name = clf_results['best_model_name']

    # 1. Classification confusion matrix (best model)
    plot_confusion_matrix(
        y_test, clf_results[best_name]['y_pred'],
        label_names,
        title=f'{best_name} — Confusion Matrix',
        filename='clf_confusion_matrix.png'
    )

    # 2. Model comparison
    plot_model_comparison(clf_results)

    # 3. Feature importance — prefer Random Forest (always has feature_importances_)
    #    Fall back to best model if RF isn't available.
    if 'Random Forest' in clf_results:
        fi_model = clf_results['Random Forest']['model']
    else:
        fi_model = clf_results[best_name]['model']
    plot_feature_importance(fi_model, feature_names)

    # 4. Attack distribution
    plot_attack_distribution(y_train, label_encoder)

    # 5. Anomaly detection confusion matrix
    plot_confusion_matrix(
        anomaly_results['y_true_binary'],
        anomaly_results['anomaly_preds'],
        ['Normal', 'Anomaly'],
        title='Anomaly Detection — Confusion Matrix',
        filename='anomaly_confusion_matrix.png'
    )

    # 6. Anomaly score distribution
    plot_anomaly_scores(
        anomaly_results['anomaly_scores'],
        anomaly_results['y_true_binary']
    )

    print("\n  ✅ All plots saved to reports/figures/")
