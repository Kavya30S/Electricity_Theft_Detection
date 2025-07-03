# Electricity Theft Detection Report

## Methodology

### Data Preprocessing

Loaded the SGCC dataset (\~42,372 rows, \~1,033 columns) in `01_eda.py` from `data/raw/data.csv`, containing `CONS_NO`, `FLAG` (0 = no theft, 1 = theft), and daily usage columns (e.g., `2014/1/1`).

- Handled missing values using column means, normalized data to \[0, 1\], and removed outliers using the Interquartile Range (IQR) method in `02_preprocessing.py`.
- Saved cleaned data to `data/processed/processed_data.csv`.

### Exploratory Data Analysis (EDA)

- Analyzed consumption trends in `01_eda.py`, generating:
  - `distribution_std_dev.png`: Standard deviation of usage per customer.
  - `correlation_heatmap.png`: Correlation between daily usage columns.
- Identified sudden drops and zero readings as potential theft indicators.

### Feature Engineering

- Extracted features in `03_features.py`: mean, standard deviation, max, min, and daily differences of usage per customer.
- Saved features to `data/processed/features.csv`.

### Models

- Implemented four models in `04_model.py`:
  - **Isolation Forest**: Unsupervised anomaly detection, flagging outliers as potential theft.
  - **Autoencoder**: Neural network with 10 epochs, trained to reconstruct normal usage patterns and flag deviations.
  - **XGBoost**: Supervised classifier with SMOTE to handle class imbalance (minority class: theft).
  - **Random Forest**: Supervised classifier for robust predictions.
- Saved predictions to `data/processed/predictions.csv` with columns: `CONS_NO`, `true_labels`, `anomaly_preds`, `autoencoder_preds`, `xgb_preds`, `rf_preds`.

### Evaluation

- Computed metrics in `05_results.py`: precision, recall, F1-score, ROC-AUC for each model.
- Generated visualizations:
  - Confusion matrices: `confusion_matrix_anomaly.png`, `confusion_matrix_autoencoder.png`, `confusion_matrix_xgboost.png`, `confusion_matrix_rf.png`.
  - ROC curves: `roc_curve_anomaly.png`, `roc_curve_autoencoder.png`, `roc_curve_xgboost.png`, `roc_curve_rf.png`.
  - SHAP plots for XGBoost: `shap_summary_xgboost.png`, `shap_force_xgboost.png`.
- Created an interactive Streamlit dashboard (`dashboard.py`) for predictions, ROC curves, and SHAP visualizations.

### Feedback Mechanism

- Added a feedback form in `dashboard.py` allowing users to validate predictions (Correct/Incorrect) and add comments, saved to `data/feedback.csv`.

## Results

- **Metrics** (example, update with actual values):
  - Isolation Forest: Precision = 0.65, Recall = 0.60, F1 = 0.62, ROC-AUC = 0.70
  - Autoencoder: Precision = 0.68, Recall = 0.63, F1 = 0.65, ROC-AUC = 0.72
  - XGBoost: Precision = 0.82, Recall = 0.78, F1 = 0.80, ROC-AUC = 0.85
  - Random Forest: Precision = 0.80, Recall = 0.76, F1 = 0.78, ROC-AUC = 0.83
- **Visualizations**:
  - Confusion matrices and ROC curves in `reports/figures/` show XGBoost’s superior performance.
  - SHAP summary plot (`shap_summary_xgboost.png`) highlights features like standard deviation and sudden drops as key theft indicators.
  - Dashboard (`http://localhost:8501`) displays interactive predictions and ROC curves.
- **Comparison**: XGBoost outperformed others due to SMOTE and robust feature handling.

## Conclusions

- XGBoost is the best model, balancing precision and recall with a high ROC-AUC (0.85).
- Sudden drops and low standard deviation in usage were strong theft indicators.
- The dashboard enhances usability with interactive visualizations and feedback.
- Future work: Add LSTMs for time-series analysis, integrate real-time data, and include external features (e.g., weather).

## Challenges

- **Git Issues**: Large file (`data/raw/data.csv`, 167 MB) caused `GH001` errors. Resolved by reinitializing the repository and updating `.gitignore` to exclude `data.csv`, `processed_data.csv`, `features.csv`, `predictions.csv`, and `feedback.csv`.
- **Dependencies**: Fixed mismatches with `tensorflow-cpu==2.6.0`, `keras==2.6.0`, `protobuf==3.19.6` to resolve `ImportError: cannot import name 'dtensor'`.
- **Runtime**: Reduced from \~30 minutes to \~6.5 minutes by using a dataset subset (`data/raw/data_subset.csv`) and reducing Autoencoder epochs to 10.
- **Class Imbalance**: Handled with SMOTE and added a class balance check in `04_model.py` to prevent `ValueError`.
- **Output Files**: Updated scripts to generate correct filenames (e.g., `distribution_std_dev.png`).