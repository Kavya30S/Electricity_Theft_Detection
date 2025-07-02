import pandas as pd
import matplotlib.pyplot as plt
import os

def save_results(df, filepath):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def save_plot(fig, filepath):
    """Save matplotlib figure."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)