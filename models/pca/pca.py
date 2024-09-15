import numpy as np
from tqdm import tqdm

class PCA:
    """
    Class to perform Principal Component Analysis
    
    Parameters
    ----------
    n_components: int
        Number of components the data should be reduced to
    """
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.principal_components = None
    
    def fit(self, X):
        """
        Find the principal components of the data
        
        Parameters
        ----------
        X: np.ndarray
            The data to be reduced
        """
        # center the data
        X = X - np.mean(X, axis=0)
        # Find the principal components of the data using SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.principal_components = Vt[:self.n_components]
    
    def transform(self, X):
        """
        Transforms the data to the reduced dimension
        
        Parameters
        ----------
        X: np.ndarray
            The data to be transformed
        """
        X = X - np.mean(X, axis=0)
        return X @ self.principal_components.T
    
    def checkPCA(self, X):
        """
        Check the PCA implementation
        
        Parameters
        ----------
        X: np.ndarray
            The data whose PCA implementation has to be checked
        """
        # check the MSE between the reconstructed data and the original data
        X = X - np.mean(X, axis=0)
        transformed_data = self.transform(X)
        reconstructed_data = transformed_data @ self.principal_components
        reconstruction_error = np.mean((X - reconstructed_data)**2)
        print(f"Reconstruction error in checkPCA(): {reconstruction_error}")
        
        if reconstruction_error < 0.1:
            return True
        else:
            return False