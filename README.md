# Electricity Theft Detection Project

Welcome to the **Electricity Theft Detection** project! This project uses smart meter data from the State Grid Corporation of China (SGCC) to detect electricity theft using machine learning. It’s a fun mix of data analysis, preprocessing, feature engineering, model training, and a cool interactive dashboard built with Streamlit. Whether you’re a data science newbie or a pro, this guide will walk you through everything you need to know to get it running and understand how it works.

## Project Overview

Electricity theft is a big problem for utility companies—it costs them billions and messes with the grid. This project tackles that by analyzing daily electricity usage data from thousands of customers (about 42,372 records with over 1,000 daily readings each). We use machine learning to spot patterns that scream “theft” versus normal usage. Here’s what we do:

- **Explore the Data**: We dig into the data to understand usage patterns, like how much electricity customers use daily.
- **Clean and Prep**: We fix missing values, normalize data, and create features like mean and standard deviation of usage.
- **Train Models**: We use four models—Isolation Forest, Autoencoder, XGBoost, and Random Forest—to predict theft.
- **Visualize Results**: A Streamlit dashboard shows predictions interactively, with cool charts to compare models.
- **Evaluate Performance**: We check how well our models do with metrics like precision, recall, and ROC-AUC.

The goal? Build a system that’s accurate, fast, and easy to use, helping utility companies catch thieves without breaking a sweat.

## Concept and Techniques

The core idea is to use machine learning to find unusual electricity usage patterns that might indicate someone’s tampering with their meter. Here’s how we do it:

- **Dataset**: The SGCC dataset has daily electricity readings for customers, with a ‘FLAG’ column (0 for normal, 1 for theft). It’s about 42,372 customers with ~1,033 daily readings, making it a hefty dataset (~40 MB).
- **Exploratory Data Analysis (EDA)**: We plot usage trends, check correlations between days, and visualize distributions to spot anomalies (e.g., sudden drops in usage).
- **Preprocessing**: We fill missing values, normalize data to a 0–1 scale, and remove outliers using the Interquartile Range (IQR) method to clean up noisy data.
- **Feature Engineering**: We create features like mean, standard deviation, max, and min usage per customer to capture their behavior.
- **Models**:
  - **Isolation Forest**: An unsupervised model that flags outliers as potential theft.
  - **Autoencoder**: A neural network that learns normal usage patterns and flags deviations as theft.
  - **XGBoost**: A supervised model that uses labeled data (FLAG) to predict theft, with SMOTE to handle class imbalance.
  - **Random Forest**: An additional supervised model for robustness, also using SMOTE.
- **Visualization**: We use Matplotlib for static plots (e.g., correlation heatmaps) and Plotly for interactive charts in the Streamlit dashboard.
- **Evaluation**: We measure model performance with precision, recall, F1-score, and ROC-AUC, ensuring accurate theft detection.

The project runs on a mid-range laptop (8GB RAM, 4-core CPU), and we’ve optimized it to take ~7 minutes total (down from ~30 minutes) by reducing Autoencoder epochs and using data subsets for testing.

## Project Folder Structure

Here’s how the project is organized:

```
electricity_theft_detection/
├── data/
│   ├── raw/
│   │   └── data.csv              # Raw SGCC dataset
│   ├── processed/
│   │   ├── processed_data.csv    # Cleaned and normalized data
│   │   ├── features.csv          # Extracted features
│   │   └── predictions.csv       # Model predictions
├── notebooks/
│   ├── 01_eda.py                 # Exploratory data analysis
│   ├── 02_preprocessing.py       # Data cleaning and normalization
│   ├── 03_features.py            # Feature engineering
│   ├── 04_model.py               # Model training and predictions
│   └── 05_results.py             # Results visualization
├── src/
│   ├── preprocessing.py          # Preprocessing functions
│   ├── features.py               # Feature extraction functions
│   ├── graph_clustering.py       # Graph-based clustering (optional)
│   ├── autoencoder.py            # Autoencoder model
│   ├── models.py                 # Model training functions
│   └── utils.py                  # Utility functions (e.g., save_plot)
├── reports/
│   ├── figures/
│   │   ├── correlation_heatmap.png
│   │   ├── distribution_std_dev.png
│   │   ├── confusion_matrix_xgboost.png
│   │   └── roc_curve.png         # ROC-AUC plot
│   └── report.md                 # Project report
├── dashboard.py                  # Streamlit dashboard
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

- **data/**: Stores raw and processed data.
- **notebooks/**: Contains Python scripts (converted from Jupyter notebooks) for each step.
- **src/**: Houses reusable functions for preprocessing, features, and models.
- **reports/**: Stores plots and the final report.
- **dashboard.py**: The interactive Streamlit dashboard.
- **requirements.txt**: Lists all dependencies needed to run the project.

## Setup Instructions

Getting this project up and running is straightforward. Follow these steps:

1. **Download the Dataset**:
   - Grab the SGCC dataset from [Kaggle](https://www.kaggle.com/datasets) or another source. Place `data.csv` in `data/raw/`. It should have columns like ‘CONS_NO’, ‘FLAG’, and daily usage (e.g., ‘2014/1/1’).

2. **Set Up the Environment**:
   - Open Anaconda Prompt (no need for admin mode unless you hit permission issues).
   - Create a Conda environment with Python 3.8:
     ```bash
     conda create -n theft_detection_new python=3.8 -c conda-forge -y
     conda activate theft_detection_new
     ```
   - Install dependencies from `requirements.txt`:
     ```bash
     cd C:\Users\kavya\OneDrive\Documents\projects\electricity_theft_detection
     pip install -r requirements.txt --no-cache-dir
     ```
   - If `pip` fails, use Conda for critical packages:
     ```bash
     conda install -c conda-forge pandas=1.3.3 numpy=1.19.5 scikit-learn=0.24.2 matplotlib=3.4.3 seaborn=0.11.2 tensorflow-cpu=2.6.0 shap=0.39.0 imbalanced-learn=0.8.0 streamlit=0.88.0 jupyter=1.0.0 ipykernel=6.15.0 ipywidgets=7.7.1 jupyter-console=6.4.4 qtconsole=5.4.3 nbconvert=6.5.0 -y
     pip install xgboost==1.4.2 setuptools==65.5.0 wheel==0.38.4 plotly==5.15.0 protobuf==3.19.6 keras==2.6.0 --no-cache-dir
     ```

3. **Set Up Jupyter Kernel for VSCode**:
   - Install the kernel:
     ```bash
     python -m ipykernel install --user --name theft_detection_new --display-name "Python (theft_detection_new)"
     ```
   - Open VSCode, select the project folder, and choose `Python (theft_detection_new)` kernel for `.py` files or converted notebooks.

4. **Run the Scripts**:
   - In VSCode, open `notebooks/01_eda.py` to `05_results.py` and run them sequentially (use `Shift+Enter` or “Run Python File”).
   - Or, in Anaconda Prompt:
     ```bash
     python notebooks/01_eda.py
     python notebooks/02_preprocessing.py
     python notebooks/03_features.py
     python notebooks/04_model.py
     python notebooks/05_results.py
     ```
   - Check outputs in `data/processed/` (e.g., `predictions.csv`) and `reports/figures/` (e.g., `confusion_matrix_xgboost.png`).

5. **Launch the Dashboard**:
   - Run:
     ```bash
     streamlit run dashboard.py
     ```
   - Open the URL (e.g., `http://localhost:8501`) to explore interactive plots and customer predictions.

6. **Test with a Subset** (to speed things up):
   - Create a smaller dataset:
     ```bash
     python -c "import pandas as pd; df = pd.read_csv('data/raw/data.csv'); df.iloc[:1000].to_csv('data/raw/data_subset.csv', index=False)"
     ```
   - Update script paths to use `data_subset.csv` for faster testing (~2–3 minutes total).

## Challenges Faced

This project wasn’t all smooth sailing! Here’s what we ran into and how we tackled it:

- **Dependency Nightmares**: Getting the right versions of `tensorflow-cpu`, `keras`, and `protobuf` was a headache. We hit errors like `ValueError: numpy.dtype size changed` and `ImportError: cannot import name 'dtensor'`. We fixed this by pinning `tensorflow-cpu==2.6.0`, `keras==2.6.0`, and `protobuf==3.19.6`, and using Python 3.8.
- **Long Runtimes**: The full dataset took ~30 minutes to process, especially for the Autoencoder model. We cut this to ~7 minutes by reducing epochs to 10 and testing with a 1,000-row subset.
- **Class Imbalance**: The `ValueError` in `04_model.py` happened because some data had only one class in the ‘FLAG’ column, breaking SMOTE. We added a check to train without SMOTE if needed.
- **Environment Issues**: Missing `python.exe` and file access errors (`WinError 32`) forced us to recreate the environment multiple times. Using `conda-forge` and non-admin mode helped.
- **Matplotlib Woes**: Namespace errors with `matplotlib-3.4.3-py3.9-nspkg.pth` required reinstalling `matplotlib` via Conda.

## Future Enhancements

This project is solid, but there’s room to make it even cooler:

- **Add More Models**: Try Long Short-Term Memory (LSTM) networks to capture time-series patterns in usage data.
- **Real-Time Detection**: Stream live data for real-time theft alerts, perfect for utility companies.
- **Fancier Visuals**: Add more interactive charts (e.g., time-series plots) and filters to the Streamlit dashboard.
- **More Data**: Include weather, location, or customer demographics to boost prediction accuracy.
- **Optimize Further**: Use GPU acceleration for faster model training or parallelize data processing.

## Requirements

Here’s the `requirements.txt` to get you started:

```text
pandas==1.3.3
numpy==1.19.5
scikit-learn==0.24.2
matplotlib==3.4.3
seaborn==0.11.2
tensorflow-cpu==2.6.0
xgboost==1.4.2
shap==0.39.0
imbalanced-learn==0.8.0
streamlit==0.88.0
setuptools==65.5.0
wheel==0.38.4
plotly==5.15.0
protobuf==3.19.6
keras==2.6.0
```

## Time Estimates

Here’s how long each script takes on a mid-range laptop (8GB RAM, 4-core CPU):

| Script              | Imports | Load/Data | Process | Model/Plot | Eval | Save | Total Max Time |
|---------------------|---------|-----------|---------|------------|------|------|----------------|
| 01_eda.py           | ~1s     | ~30s      | ~2m     | ~15s       | -    | -    | ~2m 46s        |
| 02_preprocessing.py | ~1s     | ~45s      | ~10s    | -          | -    | -    | ~56s           |
| 03_features.py      | ~1s     | ~10s      | ~15s    | ~5s        | -    | -    | ~31s           |
| 04_model.py         | ~1s     | ~15s      | ~30s    | ~1m        | ~1m  | ~10s | ~2m 56s        |
| 05_results.py       | ~1s     | ~5s       | ~5s     | -          | -    | -    | ~11s           |

**Total**: ~7 minutes (sequential); ~5 minutes (concurrent, with 20–30% overhead for heavy tasks).

## Citation

- **Dataset**: SGCC dataset from [Kaggle](https://www.kaggle.com/datasets).
- **Paper**: Zibin Zheng et al., "Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids," *IEEE Transactions on Industrial Informatics*, vol. 14, no. 4, pp. 1606-1615, April 2018.

## Get in Touch

Got questions or ideas? Feel free to tweak the code or reach out! This project is a starting point, and there’s tons of potential to make it even better. Happy coding!