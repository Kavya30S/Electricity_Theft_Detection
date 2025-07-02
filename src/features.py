import pandas as pd
import numpy as np
from scipy.fft import fft

def multi_resolution_features(df):
    """Extract multi-resolution features from consumption data."""
    consumption_cols = [col for col in df.columns if col.startswith('20')]
    features = pd.DataFrame(index=df.index)
    features['mean'] = df[consumption_cols].mean(axis=1)
    features['max'] = df[consumption_cols].max(axis=1)
    features['min'] = df[consumption_cols].min(axis=1)
    return features

def statistical_features(df):
    """Extract statistical features from consumption data."""
    consumption_cols = [col for col in df.columns if col.startswith('20')]
    stats = pd.DataFrame(index=df.index)
    stats['std_dev'] = df[consumption_cols].std(axis=1)
    stats['skewness'] = df[consumption_cols].skew(axis=1)
    stats['kurtosis'] = df[consumption_cols].kurtosis(axis=1)
    return stats

def frequency_features(df, n_components=5):
    """Extract frequency domain features using FFT."""
    consumption_cols = [col for col in df.columns if col.startswith('20')]
    freq_features = []
    for i in df.index:
        signal = df.loc[i, consumption_cols].values
        fft_vals = np.abs(fft(signal))[:n_components]
        freq_features.append(fft_vals)
    return pd.DataFrame(freq_features, index=df.index, columns=[f'fft_{i}' for i in range(n_components)])

def enhanced_features(df):
    """Combine all feature extraction methods."""
    multi_res = multi_resolution_features(df)
    stats = statistical_features(df)
    freq = frequency_features(df)
    return pd.concat([multi_res, stats, freq], axis=1)