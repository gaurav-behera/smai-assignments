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
        U, S, Vt = np.linalg.svd(X)
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
        try:
            X = X - np.mean(X, axis=0)
            transformedX = self.transform(X)
            if transformedX.shape[1] != self.n_components:
                return False
            return True
        except:
            return False 