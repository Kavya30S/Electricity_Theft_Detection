import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(page_title="Electricity Theft Detection Dashboard", layout="wide")
base_path = os.path.dirname(os.path.abspath(__file__))

# Load predictions
results_path = os.path.join(base_path, 'data', 'processed', 'predictions.csv')
try:
    results = pd.read_csv(results_path)
except FileNotFoundError:
    st.error("Predictions file not found. Please run 04_model.py first.")
    st.stop()

st.title("Electricity Theft Detection Dashboard")
st.markdown("Explore predictions from Isolation Forest, Autoencoder, XGBoost, and Random Forest models with interactive visualizations.")

# Customer selection and predictions
customer_ids = results['CONS_NO'].unique()
selected_customer = st.selectbox("Select Customer ID", customer_ids)
customer_data = results[results['CONS_NO'] == selected_customer]

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

# Feedback form
st.subheader("Validate Anomalies")
with st.form(key="feedback_form"):
    feedback_customer = st.selectbox("Select Customer ID for Feedback", customer_ids)
    feedback = st.radio("Is this prediction correct?", ("Correct", "Incorrect"))
    comment = st.text_area("Additional Comments (optional)")
    submit = st.form_submit_button("Submit Feedback")
    if submit:
        feedback_path = os.path.join(base_path, 'data', 'feedback.csv')
        with open(feedback_path, "a") as f:
            f.write(f"{feedback_customer},{feedback},{comment}\n")
        st.success("Feedback submitted!")

# SHAP visualizations
st.subheader("XGBoost Model Explainability (SHAP)")
shap_summary_path = os.path.join(base_path, 'reports', 'figures', 'shap_summary_xgboost.png')
shap_force_path = os.path.join(base_path, 'reports', 'figures', 'shap_force_xgboost.png')
if os.path.exists(shap_summary_path):
    st.image(shap_summary_path, caption="SHAP Summary Plot for XGBoost")
else:
    st.warning("SHAP summary plot not found. Run 05_results.py to generate.")
if os.path.exists(shap_force_path):
    st.image(shap_force_path, caption="SHAP Force Plot for XGBoost")
else:
    st.warning("SHAP force plot not found. Run 05_results.py to generate.")

# Prediction comparison
st.subheader("Prediction Comparison")
if 'CONS_NO' in results.columns and 'true_labels' in results.columns and 'xgb_preds' in results.columns:
    fig = px.scatter(results, x='CONS_NO', y='xgb_preds', color='true_labels',
                     labels={'xgb_preds': 'XGBoost Prediction', 'true_labels': 'True Label'},
                     title="XGBoost Predictions vs True Labels",
                     color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'})
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(xaxis_title="Customer ID", yaxis_title="Prediction", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ROC curve for XGBoost
if 'true_labels' in results.columns and 'xgb_preds' in results.columns and len(set(results['true_labels'])) == 2:
    st.subheader("ROC Curve - XGBoost")
    fpr, tpr, _ = roc_curve(results['true_labels'], results['xgb_preds'])
    auc = roc_auc_score(results['true_labels'], results['xgb_preds'])
    roc_fig = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                      title=f'ROC Curve - XGBoost (AUC = {auc:.2f})')
    roc_fig.update_layout(showlegend=False)
    st.plotly_chart(roc_fig, use_container_width=True)

# Prediction distribution
st.subheader("Prediction Distribution Across Models")
pred_cols = [col for col in ['anomaly_preds', 'autoencoder_preds', 'xgb_preds', 'rf_preds'] if col in results.columns]
if pred_cols:
    pred_counts = pd.DataFrame({col: results[col].value_counts() for col in pred_cols}).fillna(0)
    pred_counts = pred_counts.reset_index().melt(id_vars='index', value_vars=pred_cols)
    fig_dist = px.bar(pred_counts, x='variable', y='value', color='index',
                      title='Prediction Distribution (0 = No Theft, 1 = Theft)',
                      labels={'variable': 'Model', 'value': 'Count', 'index': 'Prediction'})
    st.plotly_chart(fig_dist, use_container_width=True)