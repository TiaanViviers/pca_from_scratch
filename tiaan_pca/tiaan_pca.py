import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """Simple PCA implementation using SVD on a centred data matrix."""
    
    def __init__(self, n_components=None, whiten=False, ddof=None):
        """Initialise PCA with an optional component count and variance ddof."""
        self.n_components = n_components
        self.whiten = whiten
        self.ddof = 0 if ddof is None else ddof
        
        self.x_bar = None
        self.principle_components = None
        self.singular_values = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        
    #===========================================================================
    # External API
    #===========================================================================
    
    def fit(self, X):
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
        # set n_components if not already set
        if self.n_components is None:
            self.n_components = X.shape[0]
        
        # Perform Singular Value Decomposition
        D = self._centre_train(X)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        
        # store 'n_component' first principle components
        self.principle_components = U[:, :self.n_components]
        
        # store 'n_component' first singular values
        self.singular_values = S[:self.n_components]
        
        self._set_expl_var(D.shape[1])
        self._set_expl_var_ratios(D.shape[1])
        
    
    def transform(self, X_new):
        """Project new data onto the fitted principal component basis."""
        if self.principle_components is None:
            raise Exception("Please fit the PCA First")
        
        D_new = self._centre_oos(X_new)
        if self.whiten:
            A = np.diag(np.ones(shape=len(self.explained_variance)) / np.sqrt(self.explained_variance))
            Z = A @ self.principle_components.T @ D_new
        else: 
            Z = self.principle_components.T @ D_new
        
        return Z
    
    
    def fit_transform(self, X):
        """Fit the PCA model and return the transformed data."""
        self.fit(X)
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
    
    def scree_plot(self, n_components=None, kind=None, threshold=None, color=None):
        """
        """
        if self.explained_variance_ratio is None:
            raise Exception("Please fit PCA first")
        
        ax = plt.subplot()
        color = 'blue' if color is None else color
        n_components = len(self.principle_components) if n_components is None else n_components
        pc_names = []
        for i in range(n_components):
            pc_names.append("PC" + str(i+1))
        
        if kind is None or kind.lower() == "bar":
            ax.bar(pc_names, self.explained_variance_ratio[:n_components], color=color)
        elif kind.lower() == "line":
            ax.plot(pc_names, self.explained_variance_ratio[:n_components], color=color)
        else:
            raise ValueError(f"kind={kind}, not supported, only bar|line")
        
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            
        ax.set_xlabel("Principle Components")
        ax.set_ylabel("Proportion of explained variance")
        ax.set_title("SCREE Plot")
            
        return ax
    
    
    def frobenius_norm(self, A, B):
        """
        Calculates the frobenius norm between two matrices
        
        0 means perfect reconstruction / no error
        
        Parameters:
        -----------
            A: The first matrix (an approximation).
            B: The second matrix (the reference or true value).

        Returns:
        --------
            The Frobenius norm as a float.
        """
        E = A - B
        return np.linalg.norm(E, ord="fro")
            
            
    def relative_frobenius_norm(self, A, B):
        """
        Calculates the relative Frobenius norm between two matrices.

        Parameters:
        -----------
            A: The first matrix (an approximation).
            B: The second matrix (the reference or true value).

        Returns:
        --------
            The relative Frobenius norm as a float.
        """
        diff = A - B
        norm_diff = np.linalg.norm(diff, ord="fro")
        norm_ref = np.linalg.norm(B, ord="fro")
        
        if norm_ref == 0:
            return float('inf') if norm_diff != 0 else 0.0
        
        return norm_diff / norm_ref
    
    
    def mse(self, A, B):
        """
        Calculate the mean squared error between 2 matrices
        
        Parameters:
        -----------
            A: The first matrix (an approximation).
            B: The second matrix (the reference or true value).

        Returns:
        --------
            The mse as a float.
        """
        E = A - B
        E_squared = np.square(E)
        mse = np.mean(E_squared)
        return mse
        

    #===========================================================================
    # Internal Helpers
    #===========================================================================
    
    def _centre_train(self, X):
        """
        Centre the data by subtracting the mean vector x_bar from each column.
        
        This method also sets the x_bar(mean) class variable
        
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

    
    def _set_expl_var(self, N):
        """
        Set explained variance as squared singular values divided by (N-ddof).
        
        Parameters:
        -----------
        N : int
            The number of observations in the "training" set
        
        Returns:
        --------
        None, just sets the class variable 'explained_variance'
        """
        self.explained_variance = np.empty(len(self.singular_values))
        for i, s in enumerate(self.singular_values):
            self.explained_variance[i] = s**2 / (N - self.ddof)
        
        
    def _set_expl_var_ratios(self, N):
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
        self.explained_variance_ratio = np.empty(len(self.singular_values))
        
        sum_variances = sum(self.explained_variance)
        for i, y in enumerate(self.explained_variance):
            self.explained_variance_ratio[i] = y / sum_variances
    
            
        

    
