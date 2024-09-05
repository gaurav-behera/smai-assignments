import numpy as np
from tqdm import tqdm

class KMeans:
    """
    Parameters
    ----------
    k: int
        Number of clusters
    cost_threshold: float
        The threshold for the cost function to stop the centroid calculation. The default value is 1e-6
    """
    
    def __init__(self, k, cost_threshold=1e-6):
        """
        Initialize the model
        """
        self.k = k
        self.cost_threshold = cost_threshold
        self.centroids = None
        self.wcss = None
        
    def fit(self, X):
        """
        Finds the centroids of the clusters
        
        Parameters
        ----------
        X: np.ndarray
            The data that has to be classfied into clusters
        """
        # initialize the centroids and the cost
        centroid_idx = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[centroid_idx]
        
        costs = [min([np.sum((x - c)**2) for c in self.centroids]) for x in X]
        self.wcss = np.sum(costs)
        
        pbar = tqdm(desc="Finding centroids", position=0, leave=True)
        # update centroids until the cost converges
        while True:
            clusters = self.predict(X)
            cluster_means = [np.mean(X[clusters == i], axis=0) for i in range(self.k)]
            new_costs = [min([np.sum((x - c)**2) for c in cluster_means]) for x in X]
            new_wcss = np.sum(new_costs)
            if self.wcss - new_wcss < self.cost_threshold:
                break
            else:
                self.centroids = cluster_means
                self.wcss = new_wcss
            
            pbar.update(1)
        
    
    def predict(self, X):
        """
        Predicts the cluster of the data
        
        Parameters
        ----------
        X: np.ndarray
            The data that has to be classfied into clusters
        """
        return np.array([np.argmin([np.sum((x - c)**2) for c in self.centroids]) for x in X])
    
    def getCost(self):
        """
        Returns the within cluster sum of squares
        """
        return self.wcss
        
        
    