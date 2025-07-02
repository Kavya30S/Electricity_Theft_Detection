import os
import pandas as pd
import numpy as np

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_feature_engineering():
    # Load processed data
    processed_data_path = os.path.join(base_path, 'data', 'processed', 'processed_data.csv')
    data = pd.read_csv(processed_data_path)
    print("Processed data loaded successfully")

    # Feature engineering: compute statistical features
    consumption_cols = [col for col in data.columns if col.startswith('20')]
    features = pd.DataFrame(index=data.index)
    features['mean_consumption'] = data[consumption_cols].mean(axis=1)
    features['std_dev_consumption'] = data[consumption_cols].std(axis=1)
    features['max_consumption'] = data[consumption_cols].max(axis=1)
    features['min_consumption'] = data[consumption_cols].min(axis=1)

    # Save features
    features_path = os.path.join(base_path, 'data', 'processed', 'features.csv')
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")

if __name__ == "__main__":
    run_feature_engineering()