import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

def train_anomaly_model(features):
    """Train Isolation Forest for anomaly detection."""
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features)
    return model

def predict_anomalies(model, features):
    """Predict anomalies using Isolation Forest."""
    preds = model.predict(features)
    return np.where(preds == -1, 1, 0)

def train_autoencoder_model(features, epochs=10):
    """Train Autoencoder for anomaly detection."""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(features.shape[1],)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(features.shape[1], activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(features, features, epochs=epochs, batch_size=32, verbose=0)
    return model

def predict_autoencoder_anomalies(model, features):
    """Predict anomalies using Autoencoder."""
    reconstructions = model.predict(features)
    mse = np.mean(np.power(features - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 90)
    return (mse > threshold).astype(int)

def train_supervised_model(features, labels):
    """Train XGBoost with SMOTE for supervised learning, handling single-class case."""
    from collections import Counter
    class_counts = Counter(labels)
    if len(class_counts) < 2:
        print("Warning: Only one class found in labels. Training without SMOTE.")
        model = XGBClassifier(random_state=42)
        model.fit(features, labels)
        return model, features, labels
    else:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(features, labels)
        model = XGBClassifier(random_state=42)
        model.fit(X_resampled, y_resampled)
        return model, X_resampled, y_resampled

def predict_supervised(model, features):
    """Predict using trained XGBoost model."""
    return model.predict(features)

def run_model_training():
    # Load data and features
    data_path = os.path.join(base_path, 'data', 'processed', 'processed_data.csv')
    features_path = os.path.join(base_path, 'data', 'processed', 'features.csv')
    data = pd.read_csv(data_path)
    features = pd.read_csv(features_path)
    print("Data and features loaded successfully")

    # Extract labels (assuming 'FLAG' is the label column in data.csv)
    labels = data['FLAG'] if 'FLAG' in data.columns else None

    # Train anomaly model
    anomaly_model = train_anomaly_model(features)
    anomaly_preds = predict_anomalies(anomaly_model, features)
    print("Anomaly model trained and predictions made:", anomaly_preds[:5])

    # Train autoencoder model
    autoencoder = train_autoencoder_model(features)
    autoencoder_preds = predict_autoencoder_anomalies(autoencoder, features)
    print("Autoencoder model trained and predictions made:", autoencoder_preds[:5])

    # Train supervised models
    if labels is not None:
        # XGBoost
        xgb_model, X_resampled, y_resampled = train_supervised_model(features, labels)
        xgb_preds = predict_supervised(xgb_model, features)
        print("XGBoost model trained and predictions made:", xgb_preds[:5])

        # Random Forest (enhancement)
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_resampled, y_resampled)
        rf_preds = rf_model.predict(features)
        print("Random Forest model trained and predictions made:", rf_preds[:5])
    else:
        xgb_preds = None
        rf_preds = None

    # Save predictions
    results = pd.DataFrame({
        'CONS_NO': data['CONS_NO'] if 'CONS_NO' in data.columns else range(len(features)),
        'anomaly_preds': anomaly_preds,
        'autoencoder_preds': autoencoder_preds
    })
    if xgb_preds is not None:
        results['xgb_preds'] = xgb_preds
    if rf_preds is not None:
        results['rf_preds'] = rf_preds
    if labels is not None:
        results['true_labels'] = labels

    results_path = os.path.join(base_path, 'data', 'processed', 'predictions.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")

if __name__ == "__main__":
    run_model_training()