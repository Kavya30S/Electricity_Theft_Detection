import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_evaluation():
    # Load predictions
    predictions_path = os.path.join(base_path, 'data', 'processed', 'predictions.csv')
    results = pd.read_csv(predictions_path)
    print("Predictions loaded successfully")

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
                plt.savefig(os.path.join(base_path, 'reports', 'figures', f'confusion_matrix_{pred_col.replace("_preds", "")}.png'))
                plt.close()

                # ROC-AUC and curve (for binary classification)
                if len(set(true_labels)) == 2:  # Ensure binary classification
                    auc = roc_auc_score(true_labels, results[pred_col])
                    print(f'ROC-AUC Score for {pred_col}: {auc:.2f}')
                    fpr, tpr, _ = roc_curve(true_labels, results[pred_col])
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {pred_col.replace("_preds", "")}')
                    plt.legend()
                    plt.savefig(os.path.join(base_path, 'reports', 'figures', f'roc_curve_{pred_col.replace("_preds", "")}.png'))
                    plt.close()

    # Plotly scatter plot for XGBoost predictions
    if 'CONS_NO' in results.columns and 'true_labels' in results.columns:
        fig = px.scatter(results, x='CONS_NO', y='xgb_preds', color='true_labels', title='XGBoost Predictions')
        fig.write_html(os.path.join(base_path, 'reports', 'figures', 'xgb_predictions.html'))

    print("Evaluation completed. Figures and HTML files saved in reports/figures/")

if __name__ == "__main__":
    run_evaluation()