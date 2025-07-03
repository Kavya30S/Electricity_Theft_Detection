import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap
from xgboost import XGBClassifier

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_evaluation():
    # Load predictions
    predictions_path = os.path.join(base_path, 'data', 'processed', 'predictions.csv')
    try:
        results = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print("Error: predictions.csv not found. Run 04_model.py first.")
        return

    print("Predictions loaded successfully")
    # Select consumption columns (starting with '20', e.g., dates)
    consumption_cols = [col for col in results.columns if col.startswith('20')]

    # Evaluate models if true labels exist
    if 'true_labels' in results.columns:
        true_labels = results['true_labels']
        for pred_col in ['anomaly_preds', 'autoencoder_preds', 'xgb_preds', 'rf_preds']:
            if pred_col in results.columns:
                print(f"\n{pred_col.replace('_preds', '').capitalize()} Model Classification Report:")
                print(classification_report(true_labels, results[pred_col]))

                # Confusion matrix
                cm = confusion_matrix(true_labels, results[pred_col])
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {pred_col.replace("_preds", "")}')
                cm_filename = f'confusion_matrix_{pred_col.replace("_preds", "")}.png'
                plt.savefig(os.path.join(base_path, 'reports', 'figures', cm_filename))
                plt.close()

                # ROC-AUC and curve (for binary classification)
                if len(set(true_labels)) == 2:
                    auc = roc_auc_score(true_labels, results[pred_col])
                    print(f'ROC-AUC Score for {pred_col}: {auc:.2f}')
                    fpr, tpr, _ = roc_curve(true_labels, results[pred_col])
                    fig = px.line(x=fpr, y=tpr, title=f'ROC Curve - {pred_col.replace("_preds", "")} (AUC = {auc:.2f})')
                    fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                    roc_filename = f'roc_curve_{pred_col.replace("_preds", "")}.png'
                    fig.write_image(os.path.join(base_path, 'reports', 'figures', roc_filename))

                # SHAP for XGBoost
                if pred_col == 'xgb_preds':
                    try:
                        # Load or retrain XGBoost model (replace with actual model loading if saved)
                        xgb_model = XGBClassifier()
                        xgb_model.fit(results[consumption_cols], true_labels)  # Simplified; use saved model if available
                        explainer = shap.TreeExplainer(xgb_model)
                        shap_values = explainer.shap_values(results[consumption_cols])

                        # SHAP summary plot
                        shap.summary_plot(shap_values, results[consumption_cols], show=False)
                        plt.savefig(os.path.join(base_path, 'reports', 'figures', 'shap_summary_xgboost.png'))
                        plt.close()

                        # SHAP force plot for first instance
                        shap.initjs()
                        shap.force_plot(explainer.expected_value, shap_values[0, :], results[consumption_cols].iloc[0, :], matplotlib=True, show=False)
                        plt.savefig(os.path.join(base_path, 'reports', 'figures', 'shap_force_xgboost.png'))
                        plt.close()
                    except Exception as e:
                        print(f"Error generating SHAP plots: {e}")

    # Plotly scatter plot for XGBoost predictions
    if 'CONS_NO' in results.columns and 'true_labels' in results.columns and 'xgb_preds' in results.columns:
        fig = px.scatter(results, x='CONS_NO', y='xgb_preds', color='true_labels', title='XGBoost Predictions')
        fig.write_image(os.path.join(base_path, 'reports', 'figures', 'xgb_predictions.png'))

    print("Evaluation completed. Figures saved in reports/figures/")

if __name__ == "__main__":
    run_evaluation()