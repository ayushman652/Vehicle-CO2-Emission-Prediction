#eda = exploratory data analysis

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OUTPUTS_DIR

def analyze_correlation(dataframe):
    """calculate and visualize the correlation between numericle fetaures"""
    
    numeric_df = dataframe.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr()
    
    print("\ncorrelation_matrix")
    print(correlation_matrix.round(2))
    
    plt.figure(figsize=(11, 8))
    sns.heatmap(
        correlation_matrix,
        annot = True,
        cmap = "coolwarm",
        fmt = ".2f"
    )    
    
    plt.title("vehicle feature - correlation Heatmap")
    plt.tight_layout()
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(
        OUTPUTS_DIR/"correlation_heatmap.png",
        dpi = 300
    )
    
    plt.show()
    