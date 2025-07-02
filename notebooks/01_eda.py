import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sanitize_filename(name):
    """Replace invalid characters in filenames with underscores."""
    return name.replace('/', '_').replace('\\', '_')

def run_eda():
    # Load data
    data_path = os.path.join(base_path, 'data', 'raw', 'data.csv')
    data = pd.read_csv(data_path)
    print("Data loaded successfully")

    # Summary statistics
    print("Summary Statistics:")
    print(data.describe())

    # Select top 10 numeric columns by standard deviation for plotting
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 10:
        std_devs = data[numeric_cols].std()
        top_10_cols = std_devs.nlargest(10).index
    else:
        top_10_cols = numeric_cols

    # Plot distributions for top 10 numeric columns
    for column in top_10_cols:
        plt.figure(figsize=(10, 6))
        data[column].hist()
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        safe_column = sanitize_filename(column)
        plt.savefig(os.path.join(base_path, 'reports', 'figures', f'distribution_{safe_column}.png'))
        plt.close()

    # Plot standard deviation distribution
    consumption_cols = [col for col in data.columns if col.startswith('20')]
    std_dev = data[consumption_cols].std(axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(std_dev, bins=50)
    plt.title('Distribution of Standard Deviation')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(base_path, 'reports', 'figures', 'distribution_std_dev.png'))
    plt.close()

    # Correlation heatmap for top 10 columns
    corr = data[top_10_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (Top 10 Numeric Columns)')
    plt.savefig(os.path.join(base_path, 'reports', 'figures', 'correlation_heatmap_top10.png'))
    plt.close()

    print("EDA completed. Figures saved in reports/figures/")

if __name__ == "__main__":
    run_eda()