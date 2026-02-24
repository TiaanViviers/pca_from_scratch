import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tiaan_pca import PCA


def main():
    """Run a simple PCA workflow on the YM volatility dataset."""
    raw_data_matrix = read_data()
    
    # Transpose raw data to allign with module standards
    # rows: features, cols: observations
    X = raw_data_matrix.T
    print(f"Data shape: {X.shape[0]} rows, {X.shape[1]} cols")
    
    #Test PCA without whitening
    pca = PCA()
    Z = pca.fit_transform(X)
    X_hat = pca.inverse_transform(Z)
    frobenius_norm = pca.frobenius_norm(X_hat, X)
    rel_frobenius_norm = pca.relative_frobenius_norm(X_hat, X)
    mse = pca.mse(X_hat, X)
    print(f"NON-Whitened PCA reconstruction error:")
    print(f"Frobenius norm           {frobenius_norm}")
    print(f"Relative frobenius norm: {rel_frobenius_norm}")
    print(f"MSE:                     {mse}")
    print()
    
    #Test PCA with whitening
    pca = PCA(whiten=True)
    Z = pca.fit_transform(X)
    X_hat = pca.inverse_transform(Z)
    frobenius_norm = pca.frobenius_norm(X_hat, X)
    rel_frobenius_norm = pca.relative_frobenius_norm(X_hat, X)
    mse = pca.mse(X_hat, X)
    print(f"Whitened PCA reconstruction error:")
    print(f"Frobenius norm           {frobenius_norm}")
    print(f"Relative frobenius norm: {rel_frobenius_norm}")
    print(f"MSE:                     {mse}")
    print()

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
    
