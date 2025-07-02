import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page configuration
st.set_page_config(page_title="Electricity Theft Detection Dashboard", layout="wide")

# Define base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Load predictions
results_path = os.path.join(base_path, 'data', 'processed', 'predictions.csv')
try:
    results = pd.read_csv(results_path)
except FileNotFoundError:
    st.error("Predictions file not found. Please run 04_model.py first.")
    st.stop()

# Title and description
st.title("Electricity Theft Detection Dashboard")
st.markdown("Explore predictions from Isolation Forest, Autoencoder, XGBoost, and Random Forest models with interactive visualizations.")

# Customer selection
customer_ids = results['CONS_NO'].unique()
selected_customer = st.selectbox("Select Customer ID", customer_ids)

# Filter data for selected customer
customer_data = results[results['CONS_NO'] == selected_customer]

# Display predictions
st.subheader(f"Predictions for Customer {selected_customer}")
col1, col2, col3, col4 = st.columns(4)
if 'anomaly_preds' in customer_data.columns:
    col1.metric("Isolation Forest", "Theft" if customer_data['anomaly_preds'].iloc[0] == 1 else "No Theft")
if 'autoencoder_preds' in customer_data.columns:
    col2.metric("Autoencoder", "Theft" if customer_data['autoencoder_preds'].iloc[0] == 1 else "No Theft")
if 'xgb_preds' in customer_data.columns:
    col3.metric("XGBoost", "Theft" if customer_data['xgb_preds'].iloc[0] == 1 else "No Theft")
if 'rf_preds' in customer_data.columns:
    col4.metric("Random Forest", "Theft" if customer_data['rf_preds'].iloc[0] == 1 else "No Theft")

# Interactive Plotly visualization
st.subheader("Prediction Comparison")
fig = px.scatter(results, x='CONS_NO', y='xgb_preds', color='true_labels',
                labels={'xgb_preds': 'XGBoost Prediction', 'true_labels': 'True Label'},
                title="XGBoost Predictions vs True Labels",
                color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'})
fig.update_traces(marker=dict(size=8))
fig.update_layout(xaxis_title="Customer ID", yaxis_title="Prediction", showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ROC Curve for XGBoost
if 'true_labels' in results.columns and 'xgb_preds' in results.columns and len(set(results['true_labels'])) == 2:
    from sklearn.metrics import roc_curve, roc_auc_score
    st.subheader("ROC Curve - XGBoost")
    fpr, tpr, _ = roc_curve(results['true_labels'], results['xgb_preds'])
    auc = roc_auc_score(results['true_labels'], results['xgb_preds'])
    roc_fig = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                      title=f'ROC Curve - XGBoost (AUC = {auc:.2f})')
    roc_fig.update_layout(showlegend=False)
    st.plotly_chart(roc_fig, use_container_width=True)

# Additional visualization: Prediction distribution
st.subheader("Prediction Distribution Across Models")
pred_cols = [col for col in ['anomaly_preds', 'autoencoder_preds', 'xgb_preds', 'rf_preds'] if col in results.columns]
if pred_cols:
    pred_counts = pd.DataFrame({col: results[col].value_counts() for col in pred_cols}).fillna(0)
    pred_counts = pred_counts.reset_index().melt(id_vars='index', value_vars=pred_cols)
    fig_dist = px.bar(pred_counts, x='variable', y='value', color='index',
                      title='Prediction Distribution (0 = No Theft, 1 = Theft)',
                      labels={'variable': 'Model', 'value': 'Count', 'index': 'Prediction'})
    st.plotly_chart(fig_dist, use_container_width=True)