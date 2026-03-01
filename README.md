# PCA From Scratch

This repository documents a from-scratch PCA implementation built for the CS315 machine learning module.

## Scope

This project implements:
- PCA via eigendecomposition of the covariance matrix.
- PCA via SVD of the centred data matrix.
- Projection to a lower-dimensional principal subspace.
- Reconstruction from projected coordinates.
- Optional whitening transform.
- Utility metrics for reconstruction comparison.

Main source files:
- `tiaan_pca/pca.py`
- `tiaan_pca/pca_utils.py`
- `comparison/sklearn_comparison.ipynb`

## Data Convention Used In This Project

The implementation uses the CS315 convention:

- Features on rows
- Observations on columns

So the data matrix is:
$$
X = [x_1 \; x_2 \; \cdots \; x_N] \in \mathbb{R}^{d \times N},
$$
where each observation vector satisfies:
$$
x_n \in \mathbb{R}^d, \quad n=1,\dots,N.
$$

## Core Notation

- $d$: number of features.
- $N$: number of observations.
- $\nu$: number of retained principal components (`n_components` in code).
- $\bar{x}$: sample mean vector.
- $D$: centred data matrix.
- $S$: sample covariance matrix.
- $Q$: full eigenvector matrix of $S$.
- $Q_\nu$: first $\nu$ eigenvectors (principal directions/loadings).
- $\Lambda$: diagonal matrix of eigenvalues.
- $\Lambda_\nu$: first $\nu$ eigenvalues on a diagonal matrix.
- $\lambda_i$: variance captured by principal direction $i$.
- $\sigma_i$: singular value $i$ of $D$.

## Mathematical Pipeline Implemented

### 1. Centering

Sample mean:
$$
\bar{x} = \frac{1}{N}\sum_{n=1}^N x_n.
$$

Centred columns:
$$
d_n = x_n - \bar{x}.
$$

Centred matrix:
$$
D = [d_1 \; \cdots \; d_N] \in \mathbb{R}^{d \times N}.
$$

### 2. Covariance (EIGH route)

In this implementation, covariance scaling uses `ddof`:
$$
S = \frac{1}{N-\text{ddof}}DD^T.
$$

For the module default in code, `ddof=0`, this is:
$$
S=\frac{1}{N}DD^T.
$$

### 3. Eigendecomposition route

Compute:
$$
SQ = Q\Lambda, \quad Q^TQ=I.
$$

Sort eigenvalues in non-increasing order:
$$
\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d \ge 0.
$$

Retain first $\nu$ directions:
$$
Q_\nu = [q_1,\dots,q_\nu], \quad \Lambda_\nu=\mathrm{diag}(\lambda_1,\dots,\lambda_\nu).
$$

### 4. SVD route

Compute:
$$
D = U\Sigma V^T.
$$

Then principal directions are columns of $U$, and eigenvalues are:
$$
\lambda_i = \frac{\sigma_i^2}{N-\text{ddof}}.
$$

### 5. Projection (Dimensionality Reduction)

Projected coordinates (scores) for each observation:
$$
y_n = Q_\nu^T(x_n-\bar{x}) \in \mathbb{R}^{\nu}.
$$

Matrix form (all observations):
$$
Z = Q_\nu^T D \in \mathbb{R}^{\nu \times N}.
$$

### 6. Reconstruction

From projected coordinates:
$$
\hat{x}_n = \bar{x} + Q_\nu y_n.
$$

Matrix form:
$$
\hat{X} = Q_\nu Z + \bar{x}\mathbf{1}^T.
$$

### 7. Whitening (Optional)

With whitening enabled, transformed coordinates are:
$$
z_n = \Lambda_\nu^{-1/2}Q_\nu^T(x_n-\bar{x}).
$$

Inverse transform reverses whitening with $\Lambda_\nu^{1/2}$ before reconstruction.
Whitening requires strictly positive retained eigenvalues.

## Explained Variance Quantities

Explained variance per retained component:
$$
\text{explained\_variance}_i = \lambda_i.
$$

Total variance:
$$
\sum_{j}\lambda_j.
$$

Explained variance ratio:
$$
\text{explained\_variance\_ratio}_i = \frac{\lambda_i}{\sum_j \lambda_j}.
$$

Cumulative explained variance for first $\nu$ components:
$$
\frac{\sum_{i=1}^{\nu}\lambda_i}{\sum_{j}\lambda_j}.
$$

## Mapping Math To Code

In `tiaan_pca/pca.py`:

- `_centre_train(X)`: computes $\bar{x}$ and $D$.
- `_build_cov(D)`: computes $S = \frac{1}{N-\text{ddof}}DD^T$.
- `fit_eigh(X)`: eigendecomposition path to obtain $Q_\nu$ and $\lambda_i$.
- `fit_svd(X)`: SVD path to obtain $Q_\nu$ (via $U$) and $\lambda_i$ (via $\sigma_i^2/(N-\text{ddof})$).
- `transform(X_new)`: computes projected coordinates ($Z$ or whitened $Z$).
- `inverse_transform(Z)`: reconstructs $\hat{X}$ (and un-whitens if needed).

In `tiaan_pca/pca_utils.py`:

- `frobenius_norm(A,B)`: $\|A-B\|_F$
- `relative_frobenius_norm(A,B)`: $\|A-B\|_F / \|B\|_F$
- `mse(A,B)`: $\text{mean}((A-B)^2)$
- `scree_plot(...)`: explained variance ratio plot

## Notes On Comparison With scikit-learn

Notebook:
- `comparison/sklearn_comparison.ipynb`

Comparison setup:
- Custom PCA uses $X \in \mathbb{R}^{d \times N}$.
- scikit-learn PCA uses $X \in \mathbb{R}^{N \times d}$.
- The notebook transposes appropriately and compares:
  - explained variance
  - explained variance ratio
  - component directions (with sign alignment)
  - score differences
  - reconstruction errors

Small numerical differences are expected across implementations due to solver and numerical details.
