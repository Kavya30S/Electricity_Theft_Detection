import pandas as pd
import os

def load_data(data_file):
    """Load the dataset from a single CSV file."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    data = pd.read_csv(data_file)
    # Rename 'FLAG' or 'flag' to 'label' for consistency
    if 'FLAG' in data.columns:
        data = data.rename(columns={'FLAG': 'label'})
    elif 'flag' in data.columns:
        data = data.rename(columns={'flag': 'label'})
    else:
        raise ValueError("Label column 'FLAG' or 'flag' not found in dataset")
    return data

def clean_data(df):
    """Clean the dataset by handling missing values and outliers."""
    consumption_cols = [col for col in df.columns if col.startswith('20')]
    df[consumption_cols] = df[consumption_cols].fillna(df[consumption_cols].mean())
    df[consumption_cols] = df[consumption_cols].clip(lower=0)
    return df

def normalize_data(df):
    """Normalize consumption data to zero mean and unit variance."""
    consumption_cols = [col for col in df.columns if col.startswith('20')]
    df[consumption_cols] = (df[consumption_cols] - df[consumption_cols].mean()) / df[consumption_cols].std()
    return df