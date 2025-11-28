import pandas as pd
import numpy as np
from datetime import datetime
import json


def generate_html_report(model_results, comparison_df, dataset_info, output_path='model_report.html'):
    """
    Generate comprehensive HTML report.

    Parameters:
    -----------
    model_results : dict
        Dictionary of model evaluation results
    comparison_df : pd.DataFrame
        Model comparison dataframe
    dataset_info : dict
        Information about the dataset
    output_path : str
        Path to save HTML report
    """
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Disease Status Prediction - Model Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metric {{
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .best-model {{
            background-color: #2ecc71;
            color: white;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Disease Status Prediction - Comprehensive Model Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Dataset Overview</h2>
        <div class="metric">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{dataset_info.get('total_records', 'N/A'):,}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Disease Cases</div>
            <div class="metric-value">{dataset_info.get('disease_cases', 'N/A'):,} ({dataset_info.get('disease_percentage', 0):.1f}%)</div>
        </div>

        <h2>Model Comparison</h2>
        {comparison_df.to_html(index=False, classes='comparison-table')}

        <h2>Best Performing Model</h2>
        <div class="metric best-model">
            <div class="metric-label">Model Name</div>
            <div class="metric-value">{comparison_df.iloc[0]['Model']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{comparison_df.iloc[0]['F1 Score']:.4f}</div>
        </div>

        <h2>Detailed Model Results</h2>
"""

    for model_name, results in model_results.items():
        html_content += f"""
        <h3>{model_name}</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td>Accuracy</td><td>{results['accuracy']:.4f}</td></tr>
            <tr><td>Balanced Accuracy</td><td>{results['balanced_accuracy']:.4f}</td></tr>
            <tr><td>Precision</td><td>{results['precision']:.4f}</td></tr>
            <tr><td>Recall</td><td>{results['recall']:.4f}</td></tr>
            <tr><td>F1 Score</td><td>{results['f1_score']:.4f}</td></tr>
            <tr><td>Specificity</td><td>{results['specificity']:.4f}</td></tr>
"""
        if 'roc_auc' in results:
            html_content += f"            <tr><td>ROC AUC</td><td>{results['roc_auc']:.4f}</td></tr>\n"

        html_content += "        </table>\n"

    html_content += """
        <div class="footer">
            <p>Disease Status Prediction Analysis | Machine Learning Report</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"HTML report saved to: {output_path}")


def save_results_to_json(model_results, comparison_df, output_path='model_results.json'):
    """
    Save model results to JSON file.

    Parameters:
    -----------
    model_results : dict
        Dictionary of model evaluation results
    comparison_df : pd.DataFrame
        Model comparison dataframe
    output_path : str
        Path to save JSON file
    """
    results_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': {}
    }

    for model_name, results in model_results.items():
        model_dict = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                model_dict[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                model_dict[key] = float(value)
            else:
                model_dict[key] = value

        results_dict['models'][model_name] = model_dict

    results_dict['comparison'] = comparison_df.to_dict(orient='records')

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved to JSON: {output_path}")


def save_results_to_excel(model_results, comparison_df, output_path='model_results.xlsx'):
    """
    Save model results to Excel file with multiple sheets.

    Parameters:
    -----------
    model_results : dict
        Dictionary of model evaluation results
    comparison_df : pd.DataFrame
        Model comparison dataframe
    output_path : str
        Path to save Excel file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)

        for model_name, results in model_results.items():
            results_df = pd.DataFrame([
                {'Metric': 'Accuracy', 'Value': results['accuracy']},
                {'Metric': 'Balanced Accuracy', 'Value': results['balanced_accuracy']},
                {'Metric': 'Precision', 'Value': results['precision']},
                {'Metric': 'Recall', 'Value': results['recall']},
                {'Metric': 'F1 Score', 'Value': results['f1_score']},
                {'Metric': 'Specificity', 'Value': results['specificity']},
            ])

            if 'roc_auc' in results:
                results_df = pd.concat([results_df, pd.DataFrame([
                    {'Metric': 'ROC AUC', 'Value': results['roc_auc']}
                ])], ignore_index=True)

            cm_df = pd.DataFrame(
                results['confusion_matrix'],
                columns=['Predicted: No Disease', 'Predicted: Disease'],
                index=['Actual: No Disease', 'Actual: Disease']
            )

            sheet_name = model_name[:31]
            results_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

            cm_df.to_excel(writer, sheet_name=sheet_name, startrow=len(results_df) + 3)

    print(f"Results saved to Excel: {output_path}")


def generate_summary_report(model_results, comparison_df, dataset_info):
    """
    Generate text summary report.

    Parameters:
    -----------
    model_results : dict
        Dictionary of model evaluation results
    comparison_df : pd.DataFrame
        Model comparison dataframe
    dataset_info : dict
        Information about the dataset

    Returns:
    --------
    str
        Formatted summary report
    """
    report = f"""
{'='*80}
DISEASE STATUS PREDICTION - MODEL EVALUATION SUMMARY
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION:
  • Total Records: {dataset_info.get('total_records', 'N/A'):,}
  • Disease Cases: {dataset_info.get('disease_cases', 'N/A'):,} ({dataset_info.get('disease_percentage', 0):.1f}%)
  • Features Used: {dataset_info.get('num_features', 'N/A')}

{'='*80}
MODEL COMPARISON SUMMARY
{'='*80}

Top 5 Models by F1 Score:
"""

    for idx, row in comparison_df.head(5).iterrows():
        report += f"\n{idx+1}. {row['Model']}"
        report += f"\n   F1 Score: {row['F1 Score']:.4f} | "
        report += f"Accuracy: {row['Accuracy']:.4f} | "
        report += f"Precision: {row['Precision']:.4f} | "
        report += f"Recall: {row['Recall']:.4f}"

    best_model = comparison_df.iloc[0]['Model']
    report += f"\n\n{'='*80}\n"
    report += f"BEST PERFORMING MODEL: {best_model}\n"
    report += f"{'='*80}\n"

    best_results = model_results[best_model]
    report += f"""
Performance Metrics:
  • Accuracy: {best_results['accuracy']:.2%}
  • Balanced Accuracy: {best_results['balanced_accuracy']:.2%}
  • Precision: {best_results['precision']:.2%}
  • Recall (Sensitivity): {best_results['recall']:.2%}
  • Specificity: {best_results['specificity']:.2%}
  • F1 Score: {best_results['f1_score']:.2%}
"""

    if 'roc_auc' in best_results:
        report += f"  • ROC AUC: {best_results['roc_auc']:.2%}\n"

    report += f"\nConfusion Matrix:\n"
    report += f"  True Negatives: {best_results['true_negatives']}\n"
    report += f"  False Positives: {best_results['false_positives']}\n"
    report += f"  False Negatives: {best_results['false_negatives']}\n"
    report += f"  True Positives: {best_results['true_positives']}\n"

    report += f"\n{'='*80}\n"

    return report
