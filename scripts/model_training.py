from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except (ImportError, AttributeError):
    LIGHTGBM_AVAILABLE = False
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


def get_logistic_regression_model(random_state=42):
    """
    Get Logistic Regression model with best practice hyperparameters.

    Parameters:
    -----------
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    LogisticRegression
        Configured logistic regression model
    """
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced',
        solver='liblinear'
    )


def get_random_forest_model(random_state=42):
    """
    Get Random Forest model with best practice hyperparameters.

    Parameters:
    -----------
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    RandomForestClassifier
        Configured random forest model
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )


def get_gradient_boosting_model(random_state=42):
    """
    Get Gradient Boosting model with best practice hyperparameters.

    Parameters:
    -----------
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    GradientBoostingClassifier
        Configured gradient boosting model
    """
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state
    )


def get_xgboost_model(random_state=42):
    """
    Get XGBoost model with best practice hyperparameters.

    Parameters:
    -----------
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    XGBClassifier or None
        Configured XGBoost model or None if not available
    """
    if not XGBOOST_AVAILABLE:
        return None

    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        scale_pos_weight=1,
        n_jobs=-1,
        eval_metric='logloss'
    )


def get_lightgbm_model(random_state=42):
    """
    Get LightGBM model with best practice hyperparameters.

    Parameters:
    -----------
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    LGBMClassifier or None
        Configured LightGBM model or None if not available
    """
    if not LIGHTGBM_AVAILABLE:
        return None

    return LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
        verbose=-1
    )


def get_svm_model(random_state=42):
    """
    Get SVM model with best practice hyperparameters.

    Parameters:
    -----------
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    SVC
        Configured SVM model
    """
    return SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=random_state,
        class_weight='balanced',
        probability=True
    )


def get_naive_bayes_model():
    """
    Get Naive Bayes model.

    Returns:
    --------
    GaussianNB
        Configured Naive Bayes model
    """
    return GaussianNB()


def get_knn_model():
    """
    Get K-Nearest Neighbors model.

    Returns:
    --------
    KNeighborsClassifier
        Configured KNN model
    """
    return KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )


def get_decision_tree_model(random_state=42):
    """
    Get Decision Tree model.

    Parameters:
    -----------
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    DecisionTreeClassifier
        Configured decision tree model
    """
    return DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=random_state,
        class_weight='balanced'
    )


def get_neural_network_model(random_state=42):
    """
    Get Neural Network (MLP) model.

    Parameters:
    -----------
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    MLPClassifier
        Configured neural network model
    """
    return MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1
    )


def get_all_models(random_state=42):
    """
    Get dictionary of all configured models.

    Parameters:
    -----------
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    dict
        Dictionary of model names and configured models
    """
    models = {
        'Logistic Regression': get_logistic_regression_model(random_state),
        'Random Forest': get_random_forest_model(random_state),
        'Gradient Boosting': get_gradient_boosting_model(random_state),
        'SVM': get_svm_model(random_state),
        'Naive Bayes': get_naive_bayes_model(),
        'K-Nearest Neighbors': get_knn_model(),
        'Decision Tree': get_decision_tree_model(random_state),
        'Neural Network': get_neural_network_model(random_state)
    }

    # Add optional models if available
    xgb_model = get_xgboost_model(random_state)
    if xgb_model is not None:
        models['XGBoost'] = xgb_model

    lgbm_model = get_lightgbm_model(random_state)
    if lgbm_model is not None:
        models['LightGBM'] = lgbm_model

    return models


def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create preprocessing pipeline for numerical and categorical features.

    Parameters:
    -----------
    numerical_features : list
        List of numerical feature names
    categorical_features : list
        List of categorical feature names

    Returns:
    --------
    ColumnTransformer
        Preprocessing pipeline
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def create_model_pipeline(model, numerical_features, categorical_features):
    """
    Create end-to-end pipeline with preprocessing and model.

    Parameters:
    -----------
    model : sklearn estimator
        Machine learning model
    numerical_features : list
        List of numerical feature names
    categorical_features : list
        List of categorical feature names

    Returns:
    --------
    Pipeline
        Complete pipeline with preprocessing and model
    """
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline


def perform_cross_validation(pipeline, X, y, cv=10, scoring='accuracy'):
    """
    Perform cross-validation on a pipeline.

    Parameters:
    -----------
    pipeline : Pipeline
        Model pipeline
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric

    Returns:
    --------
    dict
        Cross-validation results
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

    results = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max()
    }

    return results


def hyperparameter_tuning(model, param_grid, X, y, cv=5, scoring='accuracy', n_jobs=-1):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
    -----------
    model : sklearn estimator
        Machine learning model
    param_grid : dict
        Hyperparameter grid
    X : pd.DataFrame or array-like
        Feature matrix
    y : pd.Series or array-like
        Target vector
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    n_jobs : int
        Number of parallel jobs

    Returns:
    --------
    GridSearchCV
        Fitted grid search object
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )

    grid_search.fit(X, y)

    return grid_search
