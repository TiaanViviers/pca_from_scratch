import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from svd_pca import SVDPCA
from eigendecomp_pca import EIGHPCA


def main():
    """Run a simple PCA workflow on the YM volatility dataset."""
    raw_data_matrix = read_data()
    
    # Transpose raw data to allign with module standards
    # rows: features, cols: observations
    X = raw_data_matrix.T
    print(f"Data shape: {X.shape[0]} rows, {X.shape[1]} cols")
    
    eigh_pca = EIGHPCA()
    eigh_pca.fit(X)


def explained_variance_report(var_ratios):
    """Print explained variance ratios for each principal component."""
    total_variance = np.sum(var_ratios)
    print("Explained Variance Ratios:")
    for i, ratio in enumerate(var_ratios):
        print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    print(f"Total Variance Explained: {total_variance:.4f}")


def read_data():
    """Load the input CSV and return the raw numpy data matrix."""
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "../data/YM_vol.csv"
        
    df = pd.read_csv(data_path)
    X_raw = df.to_numpy()
    
    return X_raw
        


if __name__ == "__main__":
    main()
    
