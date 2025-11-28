import pandas as pd
import numpy as np
from typing import Tuple, List


def preprocess_admin_data(admin: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess administrative data: convert dates, create length_of_stay, temporal features.

    Parameters:
    -----------
    admin : pd.DataFrame
        Raw administrative data

    Returns:
    --------
    pd.DataFrame
        Preprocessed administrative data
    """
    admin = admin.copy()

    admin['discharge_date'] = pd.to_datetime(admin['discharge_date'], format='mixed', dayfirst=False)
    admin['admission_date'] = pd.to_datetime(admin['admission_date'], format='mixed', dayfirst=False)
    admin['length_of_stay'] = (admin['discharge_date'] - admin['admission_date']).dt.days

    admin['admission_year'] = admin['admission_date'].dt.year
    admin['admission_month'] = admin['admission_date'].dt.month
    admin['admission_day_of_week'] = admin['admission_date'].dt.dayofweek
    admin['admission_quarter'] = admin['admission_date'].dt.quarter

    return admin


def preprocess_lab_data(lab: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess laboratory data: convert dates, extract temporal features.

    Parameters:
    -----------
    lab : pd.DataFrame
        Raw laboratory data

    Returns:
    --------
    pd.DataFrame
        Preprocessed laboratory data
    """
    lab = lab.copy()

    lab['result_date'] = pd.to_datetime(lab['result_date'], format='mixed', dayfirst=False)
    lab['result_year'] = lab['result_date'].dt.year
    lab['result_month'] = lab['result_date'].dt.month

    return lab


def handle_lab_duplicates(lab: pd.DataFrame, strategy='lowest') -> pd.DataFrame:
    """
    Handle duplicate lab tests for the same patient.

    Parameters:
    -----------
    lab : pd.DataFrame
        Laboratory data
    strategy : str
        Strategy for handling duplicates: 'lowest', 'highest', 'mean', 'first', 'last'

    Returns:
    --------
    pd.DataFrame
        Cleaned laboratory data
    """
    if strategy == 'lowest':
        lab_cleaned = lab.sort_values('result_value', ascending=True)
        lab_cleaned = lab_cleaned.drop_duplicates(['ID', 'test_name'], keep='first')
    elif strategy == 'highest':
        lab_cleaned = lab.sort_values('result_value', ascending=False)
        lab_cleaned = lab_cleaned.drop_duplicates(['ID', 'test_name'], keep='first')
    elif strategy == 'mean':
        lab_cleaned = lab.groupby(['ID', 'test_name'])['result_value'].mean().reset_index()
    elif strategy == 'first':
        lab_cleaned = lab.drop_duplicates(['ID', 'test_name'], keep='first')
    elif strategy == 'last':
        lab_cleaned = lab.drop_duplicates(['ID', 'test_name'], keep='last')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return lab_cleaned.sort_index()


def impute_lab_values(lab: pd.DataFrame, impute_year: int = 2002,
                     strategy: str = 'mean') -> pd.DataFrame:
    """
    Impute missing laboratory values.

    Parameters:
    -----------
    lab : pd.DataFrame
        Laboratory data
    impute_year : int
        Year to mark values as missing
    strategy : str
        Imputation strategy: 'mean', 'median', 'mode'

    Returns:
    --------
    pd.DataFrame
        Laboratory data with imputed values
    """
    lab_imputed = lab.copy()

    if impute_year is not None:
        lab_imputed.loc[lab_imputed['result_year'] == impute_year, 'result_value'] = np.nan

    for test_name in lab_imputed['test_name'].unique():
        mask = (lab_imputed['test_name'] == test_name) & (lab_imputed['result_value'].isnull())

        if strategy == 'mean':
            fill_value = lab_imputed[lab_imputed['test_name'] == test_name]['result_value'].mean()
        elif strategy == 'median':
            fill_value = lab_imputed[lab_imputed['test_name'] == test_name]['result_value'].median()
        elif strategy == 'mode':
            fill_value = lab_imputed[lab_imputed['test_name'] == test_name]['result_value'].mode()[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        lab_imputed.loc[mask, 'result_value'] = fill_value

    return lab_imputed


def pivot_lab_to_wide(lab: pd.DataFrame) -> pd.DataFrame:
    """
    Convert laboratory data from long to wide format.

    Parameters:
    -----------
    lab : pd.DataFrame
        Laboratory data in long format

    Returns:
    --------
    pd.DataFrame
        Laboratory data in wide format
    """
    lab_wide = lab.pivot_table(
        index='ID',
        columns='test_name',
        values='result_value',
        aggfunc='mean'
    )

    return lab_wide


def merge_admin_lab_data(admin: pd.DataFrame, lab_wide: pd.DataFrame,
                         how: str = 'left') -> pd.DataFrame:
    """
    Merge administrative and laboratory data.

    Parameters:
    -----------
    admin : pd.DataFrame
        Administrative data
    lab_wide : pd.DataFrame
        Laboratory data in wide format
    how : str
        Type of merge: 'left', 'right', 'inner', 'outer'

    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    merged = admin.merge(lab_wide, left_on='ID', right_index=True, how=how)

    return merged


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional temporal features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with admission_date

    Returns:
    --------
    pd.DataFrame
        Dataframe with additional temporal features
    """
    df = df.copy()

    df['is_weekend_admission'] = df['admission_day_of_week'].isin([5, 6]).astype(int)
    df['is_winter'] = df['admission_month'].isin([12, 1, 2]).astype(int)
    df['is_spring'] = df['admission_month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['admission_month'].isin([6, 7, 8]).astype(int)
    df['is_fall'] = df['admission_month'].isin([9, 10, 11]).astype(int)

    return df


def create_lab_aggregates(df: pd.DataFrame, lab_columns: List[str]) -> pd.DataFrame:
    """
    Create aggregate features from laboratory test results.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with laboratory test columns
    lab_columns : List[str]
        List of laboratory test column names

    Returns:
    --------
    pd.DataFrame
        Dataframe with additional aggregate features
    """
    df = df.copy()

    df['lab_test_count'] = df[lab_columns].notna().sum(axis=1)
    df['lab_mean_value'] = df[lab_columns].mean(axis=1)
    df['lab_std_value'] = df[lab_columns].std(axis=1)
    df['lab_max_value'] = df[lab_columns].max(axis=1)
    df['lab_min_value'] = df[lab_columns].min(axis=1)
    df['lab_range'] = df['lab_max_value'] - df['lab_min_value']

    return df


def create_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create clinical risk indicator features based on lab values.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with laboratory test columns

    Returns:
    --------
    pd.DataFrame
        Dataframe with risk indicator features
    """
    df = df.copy()

    if 'Creatinine plasma' in df.columns:
        df['creatinine_high'] = (df['Creatinine plasma'] > 120).astype(int)

    if 'Sodium plasma' in df.columns:
        df['sodium_low'] = (df['Sodium plasma'] < 135).astype(int)
        df['sodium_high'] = (df['Sodium plasma'] > 145).astype(int)
        df['sodium_abnormal'] = ((df['Sodium plasma'] < 135) | (df['Sodium plasma'] > 145)).astype(int)

    if 'Potassium plasma' in df.columns:
        df['potassium_low'] = (df['Potassium plasma'] < 3.5).astype(int)
        df['potassium_high'] = (df['Potassium plasma'] > 5.0).astype(int)
        df['potassium_abnormal'] = ((df['Potassium plasma'] < 3.5) | (df['Potassium plasma'] > 5.0)).astype(int)

    if 'Urea plasma' in df.columns:
        df['urea_high'] = (df['Urea plasma'] > 50).astype(int)

    if 'Chloride plasma' in df.columns:
        df['chloride_low'] = (df['Chloride plasma'] < 95).astype(int)
        df['chloride_high'] = (df['Chloride plasma'] > 105).astype(int)

    if 'Bicarbonate plasma' in df.columns:
        df['bicarbonate_low'] = (df['Bicarbonate plasma'] < 22).astype(int)
        df['bicarbonate_high'] = (df['Bicarbonate plasma'] > 29).astype(int)

    return df


def handle_missing_values(df: pd.DataFrame, numerical_strategy: str = 'median',
                          categorical_strategy: str = 'mode') -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_strategy : str
        Strategy for numerical columns: 'mean', 'median', 'drop'
    categorical_strategy : str
        Strategy for categorical columns: 'mode', 'drop', 'unknown'

    Returns:
    --------
    pd.DataFrame
        Dataframe with handled missing values
    """
    df = df.copy()

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numerical_cols:
        if df[col].isna().any():
            if numerical_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif numerical_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif numerical_strategy == 'drop':
                df.dropna(subset=[col], inplace=True)

    for col in categorical_cols:
        if df[col].isna().any():
            if categorical_strategy == 'mode' and len(df[col].mode()) > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif categorical_strategy == 'unknown':
                df[col].fillna('Unknown', inplace=True)
            elif categorical_strategy == 'drop':
                df.dropna(subset=[col], inplace=True)

    return df
