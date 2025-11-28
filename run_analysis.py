#!/usr/bin/env python
"""
Run the complete machine learning analysis pipeline.
This script executes the main steps from the notebooks.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DISEASE STATUS PREDICTION - AUTOMATED ANALYSIS")
print("="*80)

# Import required libraries
print("\n[1/3] Loading libraries...")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import joblib
import os

# Import custom modules
sys.path.append('scripts')
from data_loader import load_hospital_data, data_quality_report
from preprocessing import (
    preprocess_admin_data, preprocess_lab_data, handle_lab_duplicates,
    impute_lab_values, pivot_lab_to_wide, merge_admin_lab_data,
    create_temporal_features, create_lab_aggregates, create_risk_indicators,
    handle_missing_values
)
from model_training import get_all_models, create_model_pipeline
from model_evaluation import evaluate_model, compare_models
from visualization import plot_target_distribution, plot_roc_curves

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
os.makedirs('output/models', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)

print("✓ Libraries loaded successfully")

# STEP 1: Load and preprocess data
print("\n" + "="*80)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("="*80)

print("\nLoading data...")
admin, lab = load_hospital_data()
print(f"✓ Administrative data: {admin.shape}")
print(f"✓ Laboratory data: {lab.shape}")

print("\nPreprocessing administrative data...")
admin_processed = preprocess_admin_data(admin)
print(f"✓ Created {len(admin_processed.columns) - len(admin.columns)} new features")

print("\nPreprocessing laboratory data...")
lab_processed = preprocess_lab_data(lab)
lab_cleaned = handle_lab_duplicates(lab_processed, strategy='lowest')
lab_imputed = impute_lab_values(lab_cleaned, impute_year=2002, strategy='mean')
lab_wide = pivot_lab_to_wide(lab_imputed)
print(f"✓ Laboratory data reshaped: {lab_wide.shape}")

print("\nMerging datasets...")
merged_data = merge_admin_lab_data(admin_processed, lab_wide, how='left')
print(f"✓ Merged dataset: {merged_data.shape}")

print("\nEngineering features...")
merged_data = create_temporal_features(merged_data)

lab_columns = [
    'Bicarbonate plasma', 'Chloride plasma', 'Creatinine plasma',
    'Potassium plasma', 'Sodium plasma', 'Urea plasma'
]
merged_data = create_lab_aggregates(merged_data, lab_columns)
merged_data = create_risk_indicators(merged_data)
merged_data = handle_missing_values(merged_data, numerical_strategy='median', categorical_strategy='mode')

print(f"✓ Total features created: {merged_data.shape[1]}")

# Save processed data
merged_data.to_csv('output/processed_data.csv', index=False)
print("✓ Processed data saved to: output/processed_data.csv")

# STEP 2: Prepare for modeling
print("\n" + "="*80)
print("STEP 2: MODEL PREPARATION")
print("="*80)

exclude_columns = ['ID', 'admission_date', 'discharge_date', 'admission_time', 'discharge_time', 'disease_status']
feature_columns = [col for col in merged_data.columns if col not in exclude_columns]

numerical_features = merged_data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = merged_data[feature_columns].select_dtypes(include=['object']).columns.tolist()

print(f"\nTotal features: {len(feature_columns)}")
print(f"  - Numerical: {len(numerical_features)}")
print(f"  - Categorical: {len(categorical_features)}")

X = merged_data[feature_columns]
y = merged_data['disease_status']

print(f"\nTarget distribution:")
print(y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# STEP 3: Train models
print("\n" + "="*80)
print("STEP 3: MODEL TRAINING")
print("="*80)

models = get_all_models(random_state=RANDOM_STATE)
print(f"\nTraining {len(models)} models...\n")

model_results = {}
trained_models = {}
training_times = {}

for idx, (model_name, model) in enumerate(models.items(), 1):
    print(f"[{idx}/{len(models)}] Training {model_name}...", end=" ")

    pipeline = create_model_pipeline(model, numerical_features, categorical_features)

    start_time = time()
    pipeline.fit(X_train, y_train)
    training_time = time() - start_time
    training_times[model_name] = training_time

    y_pred_test = pipeline.predict(X_test)

    try:
        y_pred_proba_test = pipeline.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_pred_proba_test = None

    test_results = evaluate_model(y_test, y_pred_test, y_pred_proba_test)
    model_results[model_name] = test_results
    trained_models[model_name] = pipeline

    print(f"✓ (F1: {test_results['f1_score']:.4f}, Time: {training_time:.2f}s)")

# STEP 4: Compare and save results
print("\n" + "="*80)
print("STEP 4: RESULTS AND COMPARISON")
print("="*80)

comparison_df = compare_models(model_results)
print("\nModel Performance Comparison (sorted by F1 Score):")
print(comparison_df.to_string(index=False))

comparison_df.to_csv('output/model_comparison.csv', index=False)
print("\n✓ Model comparison saved to: output/model_comparison.csv")

# Save trained models
print("\nSaving trained models...")
for model_name, pipeline in trained_models.items():
    filename = f'output/models/{model_name.replace(" ", "_").lower()}_model.pkl'
    joblib.dump(pipeline, filename)
print(f"✓ {len(trained_models)} models saved to: output/models/")

# Generate visualizations
print("\nGenerating visualizations...")

# ROC curves
roc_data = {}
for model_name in model_results.keys():
    results = model_results[model_name]
    if 'fpr' in results and 'tpr' in results:
        roc_data[model_name] = (results['fpr'], results['tpr'], results['roc_auc'])

if roc_data:
    plot_roc_curves(roc_data, figsize=(12, 8))
    plt.savefig('output/figures/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC curves saved to: output/figures/roc_curves.png")

# Best model summary
best_model_name = comparison_df.iloc[0]['Model']
best_model_results = model_results[best_model_name]

summary = f"""
{'='*80}
BEST MODEL SUMMARY
{'='*80}

Model: {best_model_name}

Performance Metrics:
  • Accuracy: {best_model_results['accuracy']:.2%}
  • Balanced Accuracy: {best_model_results['balanced_accuracy']:.2%}
  • Precision: {best_model_results['precision']:.2%}
  • Recall (Sensitivity): {best_model_results['recall']:.2%}
  • Specificity: {best_model_results['specificity']:.2%}
  • F1 Score: {best_model_results['f1_score']:.2%}

Confusion Matrix:
  True Negatives: {best_model_results['true_negatives']}
  False Positives: {best_model_results['false_positives']}
  False Negatives: {best_model_results['false_negatives']}
  True Positives: {best_model_results['true_positives']}

{'='*80}
"""

with open('output/best_model_summary.txt', 'w') as f:
    f.write(summary)

print(summary)

# Final summary
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print(f"\nResults saved to:")
print(f"  • Processed data: output/processed_data.csv")
print(f"  • Model comparison: output/model_comparison.csv")
print(f"  • Best model summary: output/best_model_summary.txt")
print(f"  • Trained models: output/models/ ({len(trained_models)} files)")
print(f"  • Visualizations: output/figures/")

total_time = sum(training_times.values())
print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Best model: {best_model_name} (F1 Score: {comparison_df.iloc[0]['F1 Score']:.4f})")

print("\n✓ All done!")
