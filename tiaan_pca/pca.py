import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """Simple PCA implementation using SVD."""
    
    def __init__(self, n_components=None, whiten=False, ddof=None):
        """Initialise PCA with an optional component count and variance ddof."""
        self.n_components = n_components
        self.whiten = whiten
        self.ddof = 0 if ddof is None else ddof
        
        # Mean vector of features calculated on 'training' data (used for centering)
        self.x_bar = None
        # Columns are principal directions (eigenvectors of covariance matrix)
        self.principle_components = None
        # Eigenvalues of covariance matrix (variance captured per principal component)
        self.explained_variance = None
        # Fraction of total variance explained by each principal component
        self.explained_variance_ratio = None
        
    #===========================================================================
    # External API
    #===========================================================================

    
    
    def fit_svd(self, X):
        """
        Perform SVD to derive the principal components and singular values.
        
        This method also sets the explained variance and explained variance
        ratios.
        
        Parameters:
        -----------
        X : ndarray of shape (d, N)
            The input matrix where `d` represents the feature dimension 
            and `N` is the number of observations.
            
        Returns:
        --------
        None
        """
        self._set_n_components(X, method="svd")
        
        # Perform Singular Value Decomposition
        D = self._centre_train(X)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        # store 'n_component' first principle components
        self.principle_components = U[:, :self.n_components]
        # store 'n_component' first singular values
        # Related to eigenvalues via lambda_i = sigma_i^2 / N
        singular_values = S[:self.n_components]
        
        # compute total varaince based on all singular values
        total_variance = self._expl_var(S, D.shape[1])
        # compute explained variance on n_component truncated singular values
        self.explained_variance = self._expl_var(singular_values, D.shape[1])
        # compute explained varaince ratio of each principle component based on total variance
        self._set_expl_var_ratios(total_variance)
        

    def fit_eigh(self, X):
        """
        """
        self._set_n_components(X, method="eigh")
            
        # centre data and compute Covariance Matrix
        D = self._centre_train(X)
        S = self._build_cov(D)
        # Eigendecompose the Covariance Matrix
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        #sort eigenvalues and eigenvectors in non-increasing order
        eigenvalues = np.flip(eigenvalues)
        eigenvectors = np.fliplr(eigenvectors)
        
        self.principle_components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        total_variance = sum(eigenvalues)
        self._set_expl_var_ratios(total_variance)
            
    
    def transform(self, X_new):
        """Project new data onto the fitted principal component basis."""
        if self.principle_components is None:
            raise Exception("Please fit the PCA First")
        
        D_new = self._centre_oos(X_new)
        if self.whiten:
            if np.any(self.explained_variance <= 0):
                raise ValueError(
                    "Whitening requires strictly positive explained variance "
                    "for all retained components. Reduce n_components or disable whiten."
                )
            A = np.diag(np.ones(shape=len(self.explained_variance)) / np.sqrt(self.explained_variance))
            Z = A @ self.principle_components.T @ D_new
        else: 
            Z = self.principle_components.T @ D_new
        
        return Z
    
    
    def fit_svd_transform(self, X):
        """Fit the PCA model and return the transformed data."""
        self.fit_svd(X)
        return self.transform(X)
    
    
    def fit_eigh_transform(self, X):
        """Fit the PCA model and return the transformed data."""
        self.fit_eigh(X)
        return self.transform(X)
    
    def inverse_transform(self, Z):
        """Map transformed data back into the original feature space."""
        if self.principle_components is None:
            raise Exception("Please fit the PCA First")
        
        if self.whiten:
            A = np.diag(np.sqrt(self.explained_variance))
            Z = A @ Z
        
        D_hat = self.principle_components @ Z
        x_bar_matrix = np.outer(self.x_bar, np.ones(shape=D_hat.shape[1]).T)
        X_hat = D_hat +  x_bar_matrix
        return X_hat
    
    
    def scree_plot(self, kind=None, threshold=None, color=None):
        """
        """
        if self.explained_variance_ratio is None:
            raise Exception("Please fit PCA first")
        
        ax = plt.subplot()
        color = 'blue' if color is None else color
        pc_names = []
        for i in range(self.n_components):
            pc_names.append("PC" + str(i+1))
        
        if kind is None or kind.lower() == "bar":
            ax.bar(pc_names, self.explained_variance_ratio[:self.n_components], color=color)
        elif kind.lower() == "line":
            ax.plot(pc_names, self.explained_variance_ratio[:self.n_components], color=color)
        else:
            raise ValueError(f"kind={kind}, not supported, only bar|line")
        
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            
        ax.set_xlabel("Principle Components")
        ax.set_ylabel("Proportion of explained variance")
        ax.set_title("SCREE Plot")
            
        return ax


    #===========================================================================
    # Internal Helpers
    #===========================================================================
    
    def _set_n_components(self, X, method):
        """
        Validate and set the number of principal components.

        For SVD with X in shape (d, N), at most min(d, N) non-zero singular
        directions can be recovered. For covariance eigendecomposition, the
        maximum is d.
        """
        d, N = X.shape
        max_components = min(d, N) if method == "svd" else d

        if self.n_components is None:
            self.n_components = max_components
            return

        if not isinstance(self.n_components, (int, np.integer)):
            raise TypeError("n_components must be an integer or None")
        if self.n_components <= 0:
            raise ValueError("n_components must be >= 1")
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds maximum "
                f"allowed ({max_components}) for method='{method}' and "
                f"input shape {X.shape}"
            )

    
    def _centre_train(self, X):
        """
        Centre the data by subtracting the mean vector x_bar from each column.
        
        This method also sets the x_bar class variable
        
        Parameters:
        ----------
        X : ndarray of shape (d, N)
            The input matrix where `d` represents the feature dimension 
            and `N` is the number of observations.
            
        Returns:
        -------
        D : ndarray of shape (d, N)
            The Centred data matrix of the "training" observations
        """
        self.x_bar = np.mean(X, axis=1)
        x_bar_matrix = np.outer(self.x_bar, np.ones(shape=X.shape[1]).T)
        D = X - x_bar_matrix
        return D
    
    
    def _centre_oos(self, X_new):
        """Centre out-of-sample data using the training-set mean vector."""
        x_bar_matrix = np.outer(self.x_bar, np.ones(shape=X_new.shape[1]).T)
        D_new = X_new - x_bar_matrix
        return D_new


    def _build_cov(self, D):
        """
        """
        if D.shape[1] - self.ddof <= 0:
            raise ValueError(
                f"Invalid ddof={self.ddof} for N={D.shape[1]}. "
                "Require N - ddof > 0."
            )
        S = (1 / (D.shape[1] - self.ddof)) * (D @ D.T)
        return S
    
    
    def _expl_var(self, sing_vals, N):
        """
        Calculate explained variance as squared singular values divided by (N-ddof).
        Singular values are related to eigenvalues via lambda_i = (sing_vals)_i^2 / N
        
        Parameters:
        -----------
        sing_vals: np.ndarray

        N : int
            The number of observations in the "training" set
        
        Returns:
        --------
        The explained variance (eigenvalues) based on the provided singular values
        """
        if N - self.ddof <= 0:
            raise ValueError(
                f"Invalid ddof={self.ddof} for N={N}. Require N - ddof > 0."
            )
        explained_variance = np.empty(len(sing_vals))
        for i, s in enumerate(sing_vals):
            explained_variance[i] = s**2 / (N - self.ddof)
        
        return explained_variance
        
        
    def _set_expl_var_ratios(self, total_variance):
        """
        Set explained variance ratios as:
            explained_variance[i] / sum(explained_variance)
        
        Parameters:
        -----------
        N : int
            The number of observations in the "training" set
        
        Returns:
        --------
        None, just sets the class variable 'explained_variance_ratio'
        """
        self.explained_variance_ratio = np.empty(len(self.explained_variance))
        
        for i, lambda_ in enumerate(self.explained_variance):
            self.explained_variance_ratio[i] = lambda_ / total_variance
