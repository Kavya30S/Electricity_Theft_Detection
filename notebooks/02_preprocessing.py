import os
import pandas as pd
from scipy.stats import iqr

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_preprocessing():
    # Load raw data
    raw_data_path = os.path.join(base_path, 'data', 'raw', 'data.csv')
    data = pd.read_csv(raw_data_path)
    print("Raw data loaded successfully")

    # Handle missing values
    data = data.fillna(data.mean(numeric_only=True))

    # Remove outliers
    def remove_outliers(df, cols):
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        return df
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data = remove_outliers(data, numeric_cols)

    # Save processed data
    processed_data_path = os.path.join(base_path, 'data', 'processed', 'processed_data.csv')
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    run_preprocessing()