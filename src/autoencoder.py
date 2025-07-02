import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_autoencoder(input_dim, encoding_dim):
    """Build an autoencoder model."""
    input_layer = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoder = layers.Dense(input_dim, activation='sigmoid')(encoder)
    autoencoder = models.Model(input_layer, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(autoencoder, features, epochs=50, batch_size=32):
    """Train the autoencoder."""
    autoencoder.fit(features, features, epochs=epochs, batch_size=batch_size, verbose=0)
    return autoencoder

def predict_anomalies_autoencoder(autoencoder, features, threshold=0.1):
    """Predict anomalies using autoencoder."""
    reconstructions = autoencoder.predict(features)
    mse = np.mean(np.power(features - reconstructions, 2), axis=1)
    return [1 if error > threshold else 0 for error in mse]

if __name__ == "__main__":
    print("Testing autoencoder.py")
    features = np.array([[1, 2], [3, 4], [5, 6]])
    autoencoder = build_autoencoder(features.shape[1], 32)
    print("Autoencoder built")
