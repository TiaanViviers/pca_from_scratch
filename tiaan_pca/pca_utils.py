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