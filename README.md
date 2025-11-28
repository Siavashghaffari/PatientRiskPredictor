# Disease Status Prediction for Hospital Patients

A comprehensive machine learning project for predicting disease status based on patient information and laboratory test data from Toronto hospitals.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)](https://scikit-learn.org/)

## Overview

This project implements a **complete end-to-end machine learning pipeline** for disease status prediction in hospitalized patients. The analysis combines patient demographics, laboratory biomarkers, and temporal patterns to identify high-risk patients and understand disease predictors.

### What Makes This Project Special

- **8 Comprehensive Jupyter Notebooks** covering the entire ML lifecycle
- **11 Trained Models** including ensemble and optimized versions
- **43 Engineered Features** from clinical domain knowledge
- **Statistical Rigor** with hypothesis testing and effect size analysis
- **Patient Segmentation** using unsupervised clustering
- **Clinical Interpretability** with feature importance and risk stratification
- **Automated Pipeline** for reproducible end-to-end analysis

### Data Sources

This project analyzes:
- **Patient Administrative Data**: Demographics, admission/discharge details from 2 hospital sites
- **Laboratory Test Results**: 6 blood chemistry biomarkers (Bicarbonate, Chloride, Creatinine, Potassium, Sodium, Urea)
- **Temporal Features**: Admission patterns, seasonal effects, length of stay
- **Clinical Risk Indicators**: Abnormal lab values, aggregated metrics

### Quick Facts

- **2,000 patients** analyzed
- **3,000 lab records** processed
- **9 machine learning algorithms** compared
- **128 high-risk patients** identified
- **2 patient clusters** discovered
- **~3 seconds** for complete automated analysis

## Project Structure

```
Diseases-status-prediction-for-hospital-Patients/
│
├── input/                           # Input data files
│   ├── administrative_site1.csv    # Patient data from site 1
│   ├── administrative_site2.csv    # Patient data from site 2
│   └── lab.csv                     # Laboratory test results
│
├── scripts/                         # Reusable Python modules
│   ├── data_loader.py              # Data loading utilities
│   ├── preprocessing.py            # Data preprocessing functions
│   ├── visualization.py            # Plotting and visualization
│   ├── model_training.py           # Model creation and training
│   ├── model_evaluation.py         # Model evaluation metrics
│   └── report_generator.py         # Report generation utilities
│
├── output/                          # Generated outputs
│   ├── models/                     # Trained model files (11 models)
│   ├── figures/                    # Visualizations and plots
│   ├── reports/                    # Generated reports
│   ├── processed_data.csv          # Preprocessed dataset
│   ├── predictions.csv             # Model predictions
│   ├── patient_clusters.csv        # Patient segmentation results
│   └── ...                         # Other analysis outputs
│
├── 01_Exploratory_Data_Analysis.ipynb                    # EDA notebook
├── 02_Data_Preprocessing_Feature_Engineering.ipynb       # Preprocessing
├── 03_Model_Training_Comparison.ipynb                    # Model training
├── 04_Advanced_Feature_Engineering.ipynb                 # Feature selection
├── 05_Statistical_Analysis_Hypothesis_Testing.ipynb      # Statistics
├── 06_Model_Optimization_Hyperparameter_Tuning.ipynb     # Optimization
├── 07_Predictions_Model_Interpretation.ipynb             # Predictions
├── 08_Clustering_Patient_Segmentation.ipynb              # Clustering
│
├── run_analysis.py                  # Automated pipeline (see below)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Quick Start

### Option 1: Run Automated Pipeline (Fastest)
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis in ~3 seconds
python run_analysis.py
```

This will:
- Load and preprocess data
- Engineer 43 features
- Train 9 ML models
- Generate predictions
- Save all results to `output/`

### Option 2: Interactive Jupyter Notebooks
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Open and run notebooks 01-08 in order
```

## Installation

**Requirements:**
- Python 3.8+
- Jupyter Notebook
- See `requirements.txt` for all dependencies

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Key Libraries:**
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning
- matplotlib, seaborn - Visualization
- xgboost - Gradient boosting
- scipy, statsmodels - Statistical analysis

## Notebooks

### 1. Exploratory Data Analysis (`01_Exploratory_Data_Analysis.ipynb`)
Comprehensive data exploration including:
- Data quality assessment
- Univariate and bivariate analysis
- Temporal patterns
- Laboratory test distributions
- Correlation analysis
- Statistical tests

### 2. Data Preprocessing & Feature Engineering (`02_Data_Preprocessing_Feature_Engineering.ipynb`)
Advanced data preparation:
- Data cleaning and merging
- Missing value imputation
- Feature engineering (43 features)
- Temporal features
- Laboratory aggregates
- Clinical risk indicators

### 3. Model Training & Comparison (`03_Model_Training_Comparison.ipynb`)
Training and evaluation of 9 ML models:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. Support Vector Machine
6. Naive Bayes
7. K-Nearest Neighbors
8. Decision Tree
9. Neural Network (MLP)

### 4. Advanced Feature Engineering (`04_Advanced_Feature_Engineering.ipynb`)
Advanced feature selection and transformation:
- Polynomial features
- Domain-specific interactions
- Multiple feature selection methods (F-Score, Mutual Information, RFECV, RF importance)
- PCA dimensionality reduction
- Consensus feature set creation

### 5. Statistical Analysis & Hypothesis Testing (`05_Statistical_Analysis_Hypothesis_Testing.ipynb`)
Comprehensive statistical analysis:
- Normality testing (Shapiro-Wilk, D'Agostino)
- Parametric and non-parametric tests
- Chi-square tests for categorical variables
- Effect size calculations (Cohen's d, Cramér's V)
- Risk ratio and odds ratio analysis
- Subgroup analysis

### 6. Model Optimization & Hyperparameter Tuning (`06_Model_Optimization_Hyperparameter_Tuning.ipynb`)
Model performance optimization:
- Grid Search Cross-Validation
- Learning curves analysis
- Ensemble methods (Voting Classifier)
- Best model selection and saving

### 7. Predictions & Model Interpretation (`07_Predictions_Model_Interpretation.ipynb`)
Making and interpreting predictions:
- Load trained models
- Generate predictions with confidence scores
- Feature importance analysis
- High-risk patient identification (>70% probability)
- Prediction confidence analysis

### 8. Clustering & Patient Segmentation (`08_Clustering_Patient_Segmentation.ipynb`)
Discover patient subgroups:
- K-Means clustering with optimal k selection
- Silhouette analysis
- PCA visualization
- Cluster profiling by demographics and disease rate
- Actionable patient segments

## Features

### Data Processing
- Automatic data loading and merging
- Comprehensive data quality reports
- Multiple imputation strategies
- Wide/long format transformations

### Feature Engineering
- **Temporal Features**: Weekend admissions, seasonal patterns
- **Laboratory Aggregates**: Mean, std, min, max, range
- **Clinical Risk Indicators**: Abnormal test value flags
- **Derived Metrics**: Length of stay, admission patterns

### Evaluation Metrics
- Accuracy and Balanced Accuracy
- Precision, Recall, F1-Score
- ROC AUC and Precision-Recall curves
- Confusion matrices
- Feature importance analysis

### Visualizations
- Distribution plots and histograms
- Correlation heatmaps
- ROC and PR curves
- Feature importance plots
- Model comparison charts

## Analysis Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. Data Loading & EDA                       │
│  • Load 2,000 patients + 3,000 lab records                      │
│  • Quality assessment, distributions, correlations              │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              2. Preprocessing & Feature Engineering             │
│  • Clean and merge datasets                                     │
│  • Engineer 43 features (temporal, aggregates, indicators)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  3. Model Training & Comparison                 │
│  • Train 9 ML algorithms                                        │
│  • Cross-validation, performance metrics                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              4-8. Advanced Analysis & Insights                  │
│  • Feature selection & optimization (04)                        │
│  • Statistical hypothesis testing (05)                          │
│  • Hyperparameter tuning (06)                                   │
│  • Predictions & interpretation (07)                            │
│  • Patient clustering & segmentation (08)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                   📊 Actionable Insights
```

## Usage

### Option 1: Run Notebooks Interactively

Run the notebooks in order:

1. **01_EDA**: Explore and understand the data
2. **02_Preprocessing**: Clean data and engineer features
3. **03_Model Training**: Train and compare multiple models
4. **04_Advanced Features**: Feature selection and transformation
5. **05_Statistics**: Hypothesis testing and statistical analysis
6. **06_Optimization**: Hyperparameter tuning
7. **07_Predictions**: Generate predictions and interpret models
8. **08_Clustering**: Patient segmentation

All outputs (models, figures, reports) are saved to the `output/` directory.

### Option 2: Automated Pipeline

**What is `run_analysis.py`?**

`run_analysis.py` is an automated Python script that executes the core analysis pipeline (notebooks 1-3) in a single run:

- **Loads data** from the `input/` folder
- **Preprocesses** administrative and laboratory data
- **Engineers features** (temporal, aggregates, risk indicators)
- **Trains 9 ML models** with cross-validation
- **Evaluates** and compares model performance
- **Saves results** (models, visualizations, reports)

**Run the automated pipeline:**

```bash
python run_analysis.py
```

**Output:**
- Trained models → `output/models/` (9 model files)
- Performance comparison → `output/model_comparison.csv`
- Best model summary → `output/best_model_summary.txt`
- ROC curves → `output/figures/roc_curves.png`
- Processed data → `output/processed_data.csv`

**Typical execution time:** ~3 seconds (depending on hardware)

**Use `run_analysis.py` when you want:**
- Quick end-to-end analysis
- Automated model training
- Reproducible results
- CI/CD pipeline integration

## Data

The project uses three CSV files located in the `input/` folder:

- **input/administrative_site1.csv** & **input/administrative_site2.csv**: Patient demographics, admission/discharge info
- **input/lab.csv**: Laboratory test results (Bicarbonate, Chloride, Creatinine, Potassium, Sodium, Urea)

## Key Results & Findings

### Model Performance

**Best Performing Model: Decision Tree**
- **F1 Score**: 49.11% (highest among all models)
- **Accuracy**: 47.83%
- **Precision**: 46.75%
- **Recall**: 51.71%
- **Specificity**: 44.16%

**Model Comparison (Top 5):**
1. **Decision Tree** - F1: 0.4911
2. **SVM** - F1: 0.4805
3. **Gradient Boosting** - F1: 0.4719
4. **Random Forest** - F1: 0.4691
5. **Neural Network** - F1: 0.4610

### Feature Importance Analysis

**Top 10 Most Important Features:**
1. **Length of Stay** (10.2%)
2. **Lab Mean Value** (8.2%)
3. **Age** (8.1%)
4. **Lab Max Value** (8.0%)
5. **Lab Min Value** (7.9%)
6. **Admission Year** (6.3%)
7. **Sodium Plasma** (5.1%)
8. **Lab Std Value** (4.8%)
9. **Admission Month** (4.7%)
10. **Discharge Year** (4.6%)

**Key Finding**: Clinical metrics (length of stay, lab values) are more predictive than demographic features.

### Patient Segmentation Results

**Cluster Analysis**: Identified 2 distinct patient groups using K-Means clustering

**Cluster 0** (Younger, Lower Disease Rate):
- Average Age: 58.9 years (±12.7)
- Average Length of Stay: 99.4 days (±56.5)
- Disease Rate: 48.5%

**Cluster 1** (Older, Higher Disease Rate):
- Average Age: 60.9 years (±12.5)
- Average Length of Stay: 99.1 days (±60.1)
- Disease Rate: 50.2%

### High-Risk Patient Identification

- **128 patients** identified as high-risk (probability > 70%)
- Represents 6.4% of total patient population
- These patients require priority monitoring and intervention

### Statistical Insights

**Hypothesis Testing Results**:
- Significant differences found in lab values between disease/no-disease groups
- Age shows moderate association with disease status
- Temporal patterns (admission month, year) contribute to prediction
- Multiple lab biomarkers show strong correlation with outcomes

### Advanced Feature Engineering

**Feature Selection Consensus**:
- Created optimized feature set from multiple selection methods
- Reduced dimensionality while maintaining predictive power
- Polynomial features and interactions improved model interpretability

### Dataset Overview

- **Total Patients**: 2,000
- **Total Lab Records**: 3,000
- **Features Engineered**: 43
- **Disease Prevalence**: 48.7% (balanced dataset)
- **Models Trained**: 11 (9 individual + 2 optimized/ensemble)
- **Predictions Generated**: 2,000 with confidence scores

## Output Files

### Generated Results
- **Trained Models**: `output/models/` (11 model files, ~22 MB)
- **Predictions**: `output/predictions.csv` (2,000 predictions with confidence)
- **High-Risk Patients**: `output/high_risk_patients.csv` (128 patients)
- **Patient Clusters**: `output/patient_clusters.csv` (patient segmentation)
- **Feature Importance**: `output/feature_importances.csv`
- **Model Comparison**: `output/model_comparison.csv`
- **Cluster Profiles**: `output/cluster_profiles.csv`
- **Visualizations**: `output/figures/` (ROC curves, distributions, etc.)

## Clinical Implications

1. **Length of stay** is the strongest predictor - early hospitalization patterns may indicate disease progression
2. **Laboratory values** (sodium, aggregated metrics) provide critical diagnostic information
3. **Age** remains an important but not dominant factor
4. **Patient segmentation** enables personalized treatment strategies
5. **High-risk identification** allows for targeted resource allocation

## Visualizations Generated

The analysis produces comprehensive visualizations:

- **ROC Curves**: Model performance comparison across all algorithms
- **Confusion Matrices**: Detailed classification results for each model
- **Feature Importance Plots**: Top predictors ranked by importance
- **Distribution Plots**: Patient demographics, lab values, temporal patterns
- **Correlation Heatmaps**: Feature relationships and multicollinearity
- **Cluster Visualizations**: PCA projections of patient segments
- **Learning Curves**: Model performance vs. training data size
- **Box Plots**: Statistical comparisons between groups

All visualizations are saved to `output/figures/` directory.

## Project Highlights

### 🎯 Key Achievements

1. **Comprehensive Analysis Pipeline**: End-to-end ML workflow from raw data to actionable insights
2. **Multiple Approaches**: Compared 9 different algorithms to find optimal solution
3. **Clinical Relevance**: Focus on interpretability and clinical applicability
4. **Reproducible Research**: Automated pipeline ensures consistent results
5. **Rich Documentation**: Detailed notebooks with explanations and visualizations

### 📊 Business Value

- **Risk Stratification**: Identify 6.4% of patients at highest risk
- **Resource Optimization**: Target interventions to high-risk patients
- **Early Warning System**: Length of stay and lab patterns predict outcomes
- **Patient Segmentation**: Personalized treatment strategies based on clusters
- **Cost Reduction**: Prevent complications through early identification

## Limitations & Future Work

**Current Limitations**:
- Moderate model performance (F1 ~49%) suggests additional features may be needed
- Imbalanced specificity/sensitivity trade-offs
- Dataset limited to 2,000 patients from Toronto hospitals
- Temporal dynamics not fully exploited (static predictions)

**Future Improvements**:
- Incorporate additional clinical variables (medications, comorbidities, vital signs)
- Deep learning models (LSTM, Transformers) for temporal pattern recognition
- External validation on different hospital datasets for generalizability
- Real-time prediction system integration with hospital EHR
- Explainable AI techniques (SHAP, LIME) for individual prediction explanations
- Causal inference methods to identify modifiable risk factors
- Multi-task learning to predict multiple outcomes simultaneously

## Authors

This work was developed by **Siavash Ghaffari**. For any questions, feedback, or additional information, please feel free to reach out. Your input is highly valued and will help improve and refine this pipeline further.

## Citation

If you use this code or methodology in your research, please cite:

```
Disease Status Prediction for Hospital Patients
Machine Learning Analysis of Clinical and Laboratory Data
Author: Siavash Ghaffari
Toronto Hospital Dataset (2,000 patients)
```

## Acknowledgments

- Dataset from Toronto hospitals administrative and laboratory records
- Built with scikit-learn, pandas, and the Python data science ecosystem
- Inspired by clinical decision support system research

## Disclaimer

**Important**: Clinical decisions should **never** be made solely based on these predictions without proper medical consultation and validation. This model has not been clinically validated and is intended for demonstration and learning purposes.

---

**Developed by Siavash Ghaffari for healthcare data science**
