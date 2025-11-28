import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


def plot_target_distribution(y, title='Target Variable Distribution'):
    """
    Plot the distribution of the target variable.

    Parameters:
    -----------
    y : array-like
        Target variable
    title : str
        Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    value_counts = pd.Series(y).value_counts()
    axes[0].bar(value_counts.index, value_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_xlabel('Disease Status', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    for i, v in enumerate(value_counts.values):
        axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

    axes[1].pie(value_counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Disease Status Proportion', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"Class Distribution:\n{value_counts}")
    print(f"Class Ratio: {value_counts[1]/value_counts[0]:.3f}")


def plot_numerical_distribution(df, column, bins=30, color='skyblue'):
    """
    Plot distribution of a numerical variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    bins : int
        Number of histogram bins
    color : str
        Color for the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    data = df[column].dropna()

    axes[0].hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)
    axes[0].axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {data.mean():.2f}')
    axes[0].axvline(data.median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {data.median():.2f}')
    axes[0].set_xlabel(column, fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{column} Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel(column, fontsize=12, fontweight='bold')
    axes[1].set_title(f'{column} Boxplot', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    stats.probplot(data, dist="norm", plot=axes[2])
    axes[2].set_title(f'Q-Q Plot for {column}', fontsize=14, fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(df, column, figsize=(14, 5)):
    """
    Plot distribution of a categorical variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    value_counts = df[column].value_counts()

    axes[0].bar(value_counts.index, value_counts.values, alpha=0.7)
    axes[0].set_xlabel(column, fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{column} Distribution', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(value_counts.values):
        axes[0].text(i, v + max(value_counts.values)*0.01, str(v),
                     ha='center', fontweight='bold')

    axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    axes[1].set_title(f'{column} Proportion', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_target_vs_numerical(df, numerical_col, target_col='disease_status'):
    """
    Plot relationship between numerical feature and target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_col : str
        Numerical column name
    target_col : str
        Target column name
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_clean = df.dropna(subset=[numerical_col])

    axes[0].boxplot([df_clean[df_clean[target_col]==0][numerical_col],
                     df_clean[df_clean[target_col]==1][numerical_col]],
                    labels=['No Disease', 'Disease'],
                    patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))
    axes[0].set_ylabel(numerical_col, fontsize=12, fontweight='bold')
    axes[0].set_title(f'{numerical_col} by Disease Status', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].hist([df_clean[df_clean[target_col]==0][numerical_col],
                  df_clean[df_clean[target_col]==1][numerical_col]],
                 bins=20, label=['No Disease', 'Disease'], alpha=0.7,
                 color=['#2ecc71', '#e74c3c'])
    axes[1].set_xlabel(numerical_col, fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{numerical_col} Distribution by Disease Status',
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(
        df_clean[df_clean[target_col]==0][numerical_col].dropna(),
        df_clean[df_clean[target_col]==1][numerical_col].dropna()
    )
    print(f"\nT-test Results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Conclusion: Statistically significant difference (p < 0.05)")
    else:
        print(f"  Conclusion: No statistically significant difference (p >= 0.05)")


def plot_correlation_matrix(df, figsize=(12, 10), annot=True):
    """
    Plot correlation matrix heatmap.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    figsize : tuple
        Figure size
    annot : bool
        Whether to annotate cells
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    return corr_matrix


def plot_feature_importance(feature_names, importances, top_n=20, title='Feature Importance'):
    """
    Plot feature importance.

    Parameters:
    -----------
    feature_names : list
        List of feature names
    importances : array-like
        Feature importances
    top_n : int
        Number of top features to display
    title : str
        Plot title
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'], color='teal', alpha=0.7)
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return importance_df


def plot_confusion_matrix(cm, labels=['No Disease', 'Disease'], normalize=False):
    """
    Plot confusion matrix.

    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    labels : list
        Class labels
    normalize : bool
        Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels,
                yticklabels=labels, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_roc_curves(model_results, figsize=(10, 8)):
    """
    Plot ROC curves for multiple models.

    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and tuples (fpr, tpr, auc) as values
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)

    for model_name, (fpr, tpr, auc_score) in model_results.items():
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(model_results, figsize=(10, 8)):
    """
    Plot Precision-Recall curves for multiple models.

    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and tuples (precision, recall, ap) as values
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)

    for model_name, (precision, recall, ap_score) in model_results.items():
        plt.plot(recall, precision, linewidth=2,
                 label=f'{model_name} (AP = {ap_score:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(train_sizes, train_scores, val_scores, title='Learning Curves'):
    """
    Plot learning curves.

    Parameters:
    -----------
    train_sizes : array-like
        Training set sizes
    train_scores : array-like
        Training scores
    val_scores : array-like
        Validation scores
    title : str
        Plot title
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='r')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='g')

    plt.xlabel('Training Examples', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
