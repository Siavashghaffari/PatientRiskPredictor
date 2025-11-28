import pandas as pd
import numpy as np
from typing import Tuple


def load_hospital_data(admin_site1_path='input/administrative_site1.csv',
                       admin_site2_path='input/administrative_site2.csv',
                       lab_path='input/lab.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine hospital administrative data from two sites and lab data.

    Parameters:
    -----------
    admin_site1_path : str
        Path to administrative site 1 CSV file
    admin_site2_path : str
        Path to administrative site 2 CSV file
    lab_path : str
        Path to laboratory data CSV file

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Combined administrative data and laboratory data
    """
    admin_site1 = pd.read_csv(admin_site1_path)
    admin_site2 = pd.read_csv(admin_site2_path)
    lab = pd.read_csv(lab_path)

    admin = pd.concat([
        admin_site1,
        admin_site2.rename(columns=dict(zip(admin_site2.columns, admin_site1.columns)))
    ], ignore_index=True)

    return admin, lab


def data_quality_report(df, name="Dataset"):
    """
    Generate a comprehensive data quality report.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    name : str
        Name of the dataset for reporting

    Returns:
    --------
    dict
        Dictionary containing data quality metrics
    """
    report = {}

    print(f"\n{'='*80}")
    print(f"DATA QUALITY REPORT: {name}")
    print(f"{'='*80}\n")

    report['total_records'] = len(df)
    report['total_features'] = len(df.columns)
    report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2

    print(f"Total Records: {report['total_records']:,}")
    print(f"Total Features: {report['total_features']}")
    print(f"Memory Usage: {report['memory_usage_mb']:.2f} MB")

    print(f"\n{'Missing Values':-^80}")
    missing = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

    if len(missing) > 0:
        print(missing)
        report['missing_values'] = missing.to_dict()
    else:
        print("No missing values found!")
        report['missing_values'] = {}

    print(f"\n{'Duplicate Records':-^80}")
    duplicates = df.duplicated().sum()
    report['duplicates'] = duplicates
    print(f"Number of duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")

    print(f"\n{'Data Types':-^80}")
    print(df.dtypes.value_counts())
    report['data_types'] = df.dtypes.value_counts().to_dict()

    return report
