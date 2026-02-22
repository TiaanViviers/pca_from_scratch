import numpy as np


class PCA:
    
    def __init__(self, n_components=None, ddof=None):
        self.n_components = n_components
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
        Perform Singular value decomposition to derive the principle 
        components (Eigenvectors of centred Covariance matrix), and the singular
        values (Eigenvalues of centred Covariance matrix)
        
        This method also sets the explained variance and explained variance
        ratios
        
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
        print(f"Fitting PCA for first {self.n_components}")
        
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
        """
        """
        D_new = self._centre_oos(X_new)
        Z = self.principle_components.T @ D_new
        return Z
    
    def fit_transform(self, X):
        """
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, Z):
        """
        """
        D_hat = self.principle_components.T @ Z
        x_bar_matrix = np.outer(self.x_bar, np.ones(shape=D_hat.shape[1]).T)
        X_hat = D_hat +  x_bar_matrix
        return X_hat
        

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
        """
        """
        x_bar_matrix = np.outer(self.x_bar, np.ones(shape=X_new.shape[1]).T)
        D_new = X_new - x_bar_matrix
        return D_new

    
    def _set_expl_var(self, N):
        """
        Set explained variance as squared singular values devided by (N-ddof)
        
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
    
            
        

    

