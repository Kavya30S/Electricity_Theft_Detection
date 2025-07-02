import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_anomaly_model(features):
    """Train Isolation Forest for anomaly detection."""
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features)
    return model

def predict_anomalies(model, features):
    """Predict anomalies using Isolation Forest."""
    preds = model.predict(features)
    return np.where(preds == -1, 1, 0)  # Convert -1 (anomaly) to 1, 1 (normal) to 0

def train_autoencoder_model(features, epochs=20):
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
    threshold = np.percentile(mse, 90)  # Top 10% as anomalies
    return (mse > threshold).astype(int)

def train_supervised_model(features, labels):
    """Train XGBoost with SMOTE for supervised learning."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(features, labels)
    model = XGBClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    return model

def predict_supervised(model, features):
    """Predict using trained XGBoost model."""
    return model.predict(features)

if __name__ == "__main__":
    print("Testing models.py")
    df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'label': [0, 1, 0]})
    model = train_supervised_model(df[['feature1', 'feature2']], df['label'])
    preds = predict_supervised(model, df[['feature1', 'feature2']])
    print("Sample XGBoost predictions:", preds)