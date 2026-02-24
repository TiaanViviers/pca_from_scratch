import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main():
    #read YM_vol data
    df = pd.read_csv("../data/YM_vol.csv")
    columns = df.columns
    X = df.to_numpy()
    rows, cols = X.shape
    print(f"dataset has {rows} rows, {cols} columns")

    #fit the PCA model
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_new)
    
    explained_variance_report(pca.explained_variance_ratio_)
    plot_reconstruction_comparison(X, X_reconstructed, columns)
    
    
def explained_variance_report(var_ratios):
    total_variance = np.sum(var_ratios)
    print("Explained Variance Ratios:")
    for i, ratio in enumerate(var_ratios):
        print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    print(f"Total Variance Explained: {total_variance:.4f}")


def plot_reconstruction_comparison(X, X_reconstructed, feature_names):
    obs_idx = np.arange(X.shape[0])

    for i, feature_name in enumerate(feature_names):
        residual = X[:, i] - X_reconstructed[:, i]

        fig, axes = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        axes[0].plot(obs_idx, X[:, i], label="Original", linewidth=1.8)
        axes[0].plot(
            obs_idx,
            X_reconstructed[:, i],
            label="PCA Reconstructed",
            linewidth=1.4,
            alpha=0.9,
        )
        axes[0].set_title(f"{feature_name}: Original vs PCA Reconstruction")
        axes[0].set_ylabel("Value")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        axes[1].plot(obs_idx, residual, color="tab:red", linewidth=1.2)
        axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.6)
        axes[1].set_xlabel("Observation Count")
        axes[1].set_ylabel("Error")
        axes[1].grid(alpha=0.2)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
