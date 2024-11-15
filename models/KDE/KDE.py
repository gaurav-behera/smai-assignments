import numpy as np
import plotly.graph_objects as go

class KDE:
    """
    Kernel Density Estimation (KDE) class
    
    Parameters
    ----------
    kernel : str
        Kernel to be used for KDE. Default is 'box'. Available kernels are 'box', 'gaussian', 'triangular'
    bandwidth : float
        Bandwidth for KDE. Default is 0.5
    """
    def __init__(self, kernel='box', bandwidth=0.5):
        self.kernel = kernel
        self.bandwidth = bandwidth
        match self.kernel:
            case 'box':
                self.kernel_func = self.box_kernel
            case 'gaussian':
                self.kernel_func = self.gaussian_kernel
            case 'triangular':
                self.kernel_func = self.triangular_kernel
            case _:
                raise ValueError('Invalid kernel')
    
    def box_kernel(self, x, d):
        return np.all(np.abs(x) < 0.5, axis=1)
    
    def gaussian_kernel(self, x, d):
        return (2*np.pi)**(-d/2) * np.exp(-0.5*np.sum(x**2, axis=1))
    
    def triangular_kernel(self, x, d):
        return np.maximum(1 - np.sum(np.abs(x), axis=1), 0)
    
    def fit(self, X):
        """
        Fit the KDE model
        
        Parameters
        ----------
        X : numpy.ndarray
            Data to be fitted (N x d) where N is the number of samples and d is the number of features
        """
        self.X = X
        
    def predict(self, x):
        """
        Predict the KDE model
        
        Parameters
        ----------
        x : numpy.ndarray
            Density to be predicted for one sample (d,)
        """
        N = self.X.shape[0]
        d = self.X.shape[1]
        p = np.sum(self.kernel_func((x-self.X) / self.bandwidth, d)) / (N*self.bandwidth**d)
        
        return p
    
    def plot(self, title='', return_fig=False):
        """
        Plot the KDE model
        
        Parameters
        ----------
        title : str
            Title of the plot
        """
        assert (self.X.shape[1] == 2) , 'Only 2D data is supported'
        X = self.X[:,0]
        Y = self.X[:,1]
        Z = np.array([self.predict(np.array([x, y])) for x, y in zip(X, Y)])
        plt = go.Scatter(x=X, y=Y, mode='markers', marker=dict(size=5, color=Z, colorscale='Viridis', showscale=True))
        if return_fig:
            return plt
        fig = go.Figure()
        fig.add_trace(plt)
        fig.update_layout(title=title, width=600, height=600)
        fig.show()