import numpy as np
import pandas as pd

from ..MLP.MLP import MLP_regression

class AutoEncoder_MLP(MLP_regression):
    """
    AutoEncoder class for dimensionality reduction
    """
    def __init__(self, input_dim, reduced_dims, learning_rate=0.01, activation_function="sigmoid", optimizer='sgd', num_hidden_layers=1, num_neurons=[32], batch_size=32, epochs=1000):
        """
        Initialize the AutoEncoder

        Parameters
        ----------
        input_dim : int
            The input dimension
        reduced_dims : int
            The reduced dimension
        num_hidden_layers : int, optional
            The number of hidden layers for the encoder or decoder. The default is 1.
        num_neurons : list, optional
            The number of neurons in each hidden layer of the encoder. The decoder is reverse of this. The default is [32].
        """
        super().__init__(learning_rate, activation_function, optimizer, 2*num_hidden_layers+1, num_neurons + [reduced_dims] + num_neurons[::-1], batch_size, epochs, input_dim, input_dim)
        
    def fit(self, X):
        """
        Fit the AutoEncoder on the data
        
        Parameters
        ----------
        X : numpy.ndarray
            The input data
        """
        super().fit(X, X)
    
    def get_latent(self, X):
        """
        Get the latent representation of the data
        
        Parameters
        ----------
        X : numpy.ndarray
            The input data
        
        Returns
        -------
        numpy.ndarray
            The latent representation of the data
        """
        self.predict(X)
        return self.activations[self.num_hidden_layers//2 + 1]
    
    