from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import pandas as pd
import numpy as np


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Comprehensive model evaluation.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities

    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    results = {}

    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision'] = precision_score(y_true, y_pred, zero_division=0)
    results['recall'] = recall_score(y_true, y_pred, zero_division=0)
    results['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    if y_pred_proba is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        results['average_precision'] = average_precision_score(y_true, y_pred_proba)

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        results['fpr'] = fpr
        results['tpr'] = tpr

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        results['precision_curve'] = precision
        results['recall_curve'] = recall

    tn, fp, fn, tp = results['confusion_matrix'].ravel()
    results['true_negatives'] = tn
    results['false_positives'] = fp
    results['false_negatives'] = fn
    results['true_positives'] = tp

    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['balanced_accuracy'] = (results['sensitivity'] + results['specificity']) / 2

    return results


def print_evaluation_report(results, model_name='Model'):
    """
    Print formatted evaluation report.

    Parameters:
    -----------
    results : dict
        Evaluation results from evaluate_model
    model_name : str
        Name of the model
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*80}\n")

    print(f"{'Classification Metrics':-^80}")
    print(f"  Accuracy:           {results['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {results['balanced_accuracy']:.4f}")
    print(f"  Precision:          {results['precision']:.4f}")
    print(f"  Recall (Sensitivity): {results['recall']:.4f}")
    print(f"  Specificity:        {results['specificity']:.4f}")
    print(f"  F1 Score:           {results['f1_score']:.4f}")

    if 'roc_auc' in results:
        print(f"\n{'Probability-Based Metrics':-^80}")
        print(f"  ROC AUC:            {results['roc_auc']:.4f}")
        print(f"  Average Precision:  {results['average_precision']:.4f}")

    print(f"\n{'Confusion Matrix':-^80}")
    print(f"  True Negatives:     {results['true_negatives']}")
    print(f"  False Positives:    {results['false_positives']}")
    print(f"  False Negatives:    {results['false_negatives']}")
    print(f"  True Positives:     {results['true_positives']}")

    print(f"\n{'='*80}\n")


def compare_models(model_results):
    """
    Compare multiple models and return ranked comparison.

    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and evaluation results as values

    Returns:
    --------
    pd.DataFrame
        Comparison dataframe sorted by F1 score
    """
    comparison_data = []

    for model_name, results in model_results.items():
        row = {
            'Model': model_name,
            'Accuracy': results.get('accuracy', 0),
            'Balanced Accuracy': results.get('balanced_accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'Specificity': results.get('specificity', 0),
            'F1 Score': results.get('f1_score', 0),
        }

        if 'roc_auc' in results:
            row['ROC AUC'] = results['roc_auc']

        if 'average_precision' in results:
            row['Avg Precision'] = results['average_precision']

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False)

    return comparison_df


def calculate_cost_sensitive_metrics(results, cost_fp=1, cost_fn=5):
    """
    Calculate cost-sensitive metrics for medical context.

    Parameters:
    -----------
    results : dict
        Evaluation results
    cost_fp : float
        Cost of false positive (unnecessary treatment)
    cost_fn : float
        Cost of false negative (missed disease)

    Returns:
    --------
    dict
        Cost-related metrics
    """
    fp = results['false_positives']
    fn = results['false_negatives']

    total_cost = (fp * cost_fp) + (fn * cost_fn)
    avg_cost_per_prediction = total_cost / (results['true_positives'] +
                                            results['true_negatives'] +
                                            fp + fn)

    cost_metrics = {
        'total_cost': total_cost,
        'avg_cost_per_prediction': avg_cost_per_prediction,
        'false_positive_cost': fp * cost_fp,
        'false_negative_cost': fn * cost_fn,
        'cost_fp': cost_fp,
        'cost_fn': cost_fn
    }

    return cost_metrics


def get_classification_report_dict(y_true, y_pred, target_names=['No Disease', 'Disease']):
    """
    Get classification report as dictionary.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : list
        Class names

    Returns:
    --------
    dict
        Classification report as dictionary
    """
    return classification_report(y_true, y_pred, target_names=target_names, output_dict=True)


def calculate_clinical_metrics(results):
    """
    Calculate clinical interpretation metrics.

    Parameters:
    -----------
    results : dict
        Evaluation results

    Returns:
    --------
    dict
        Clinical metrics
    """
    tp = results['true_positives']
    tn = results['true_negatives']
    fp = results['false_positives']
    fn = results['false_negatives']

    total = tp + tn + fp + fn

    clinical_metrics = {
        'prevalence': (tp + fn) / total if total > 0 else 0,
        'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'false_discovery_rate': fp / (fp + tp) if (fp + tp) > 0 else 0,
        'diagnostic_odds_ratio': (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
    }

    return clinical_metrics


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal classification threshold.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    metric : str
        Metric to optimize: 'f1', 'balanced_accuracy', 'precision', 'recall'

    Returns:
    --------
    tuple
        (optimal_threshold, best_score)
    """
    thresholds = np.arange(0.0, 1.0, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    return optimal_threshold, best_score


def generate_model_summary(model_results, model_name='Model'):
    """
    Generate comprehensive model summary.

    Parameters:
    -----------
    model_results : dict
        Evaluation results
    model_name : str
        Name of the model

    Returns:
    --------
    str
        Formatted summary string
    """
    clinical_metrics = calculate_clinical_metrics(model_results)

    summary = f"""
{'-'*80}
MODEL SUMMARY: {model_name}
{'-'*80}

PERFORMANCE METRICS:
  • Accuracy: {model_results['accuracy']:.2%}
  • Balanced Accuracy: {model_results['balanced_accuracy']:.2%}
  • F1 Score: {model_results['f1_score']:.2%}
  • Precision (PPV): {model_results['precision']:.2%}
  • Recall (Sensitivity): {model_results['recall']:.2%}
  • Specificity: {model_results['specificity']:.2%}

CLINICAL INTERPRETATION:
  • Positive Predictive Value: {clinical_metrics['positive_predictive_value']:.2%}
  • Negative Predictive Value: {clinical_metrics['negative_predictive_value']:.2%}
  • False Positive Rate: {clinical_metrics['false_positive_rate']:.2%}
  • False Negative Rate: {clinical_metrics['false_negative_rate']:.2%}

CONFUSION MATRIX:
                    Predicted: No Disease    Predicted: Disease
  Actual: No Disease        {model_results['true_negatives']:>6}              {model_results['false_positives']:>6}
  Actual: Disease           {model_results['false_negatives']:>6}              {model_results['true_positives']:>6}

{'-'*80}
"""

    return summary
