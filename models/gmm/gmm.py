import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.linalg import solve
from scipy.special import logsumexp
class GMM:
    """
    Parameters
    ----------
    k: int
        Number of clusters
    cost_threshold: float
        The threshold for the cost function to stop the cluster calculation. The default value is 1e-3
    max_iter: int
        The maximum number of iterations for the cluster calculation. The default value is 1000
    """

    def __init__(self, k, cost_threshold=1e-3, max_iter=1000):
        """
        Initialize the model
        """
        self.k = k
        self.cost_threshold = cost_threshold
        self.max_iter = max_iter
        self._means = None
        self._covs = None
        self._pis = None
        self.log_likelihood = None

    def fit(self, X):
        """
        Uses Expectation Maximization to find the means, covariances and the mixing coefficients

        Parameters
        ----------
        X: np.ndarray
            The data that has to be classfied into clusters
        """
        # initialize the parameters
        means_idx = np.random.choice(X.shape[0], self.k, replace=False)
        self._means = X[means_idx]
        self._covs = np.eye(X.shape[1])[np.newaxis, :, :].repeat(self.k, axis=0)
        pis = np.random.rand(self.k)
        self._pis = pis / np.sum(pis)
        self.log_likelihood = self.getLogLikelihood(X)
        pbar = tqdm(desc="Finding clusters", position=0, leave=True)

        curr_iter = 0
        while True:
            # E step: evaluate the responsibilities using the current parameters
            resp = self.getMembership(X)
            # M step: update the parameters using the current responsibilities
            weights_sum = np.sum(resp, axis=0)
            new_means = (resp.T @ X) / weights_sum[:, np.newaxis]
            
            X_centered = (X[:, np.newaxis, :] - new_means) * np.sqrt(resp)[:, :, np.newaxis]
            new_covs = np.zeros((self.k, X.shape[1], X.shape[1]))
            for k in range(self.k):
                new_covs[k] = X_centered[:, k, :].T @ X_centered[:, k, :] / weights_sum[k] 
                
            new_pis = weights_sum / weights_sum.sum()

            prev_parameters = (self._means, self._covs, self._pis)
            self._means = new_means.copy()
            self._covs = new_covs.copy()
            self._pis = new_pis.copy()

            # evaluate log likelihood and update the parameters until convergence
            log_likelihood = self.getLogLikelihood(X)
            curr_iter += 1
            if curr_iter >= self.max_iter:
                break
            if log_likelihood < self.log_likelihood:
                self._means, self._covs, self._pis = prev_parameters
                break
            if log_likelihood - self.log_likelihood < self.cost_threshold:
                break
            else:
                self.log_likelihood = log_likelihood.copy()
            pbar.update(1)

    def _multivariate_normal(self, x, mean, cov, return_log=False):
        """
        Returns the probability of the data point x given the mean and covariance

        Parameters
        ----------
        x: np.ndarray
            The data point
        mean: np.ndarray
            The mean of the cluster
        cov: np.ndarray
            The covariance of the cluster
        return_log: bool
            If True, returns the log of the probability. The default value is False
        """
        if return_log:
            val = multivariate_normal(mean=mean, cov=cov, allow_singular=True).logpdf(x)
        else:
            val = multivariate_normal(mean=mean, cov=cov, allow_singular=True).pdf(x)
        return val
        

    def getParams(self):
        """
        Returns the means, covariances and the mixing coefficients
        """
        return self._means, self._covs, self._pis

    def getMembership(self, X):
        """
        Returns the membership of the data points

        Parameters
        ----------
        X: np.ndarray
            The data that has to be classfied into clusters
        """
        cluster_probs = np.zeros((X.shape[0], self.k))
        for j in range(self.k):
            cluster_probs[:, j] = self._pis[j] * self._multivariate_normal(X, self._means[j], self._covs[j])
        cluster_probs_sum = np.sum(cluster_probs, axis=1)
        cluster_probs_sum = np.where(cluster_probs_sum == 0, 1, cluster_probs_sum)
        return cluster_probs / cluster_probs_sum[:, np.newaxis]
        
    def getLikelihood(self, X):
        """
        Returns the likelihood of the data given the parameters
        """
        likelihood = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            likelihood[:, k] = self._pis[k] * self._multivariate_normal(X, self._means[k], self._covs[k])
        return np.prod(np.sum(likelihood, axis=1))

    def getLogLikelihood(self, X):
        """
        Returns the log likelihood of the data given the parameters
        """
        likelihood = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            likelihood[:, k] = self._pis[k] * self._multivariate_normal(X, self._means[k], self._covs[k])
        likelihood = np.sum(likelihood, axis=1)
        return np.sum(np.log(likelihood))
    
    def num_parameters(self):
        """
        Returns the number of parameters in the model
        """
        dims = self._means.shape[1]
        return self.k*dims + self.k*(dims*dims + dims)//2 + self.k - 1
    
    def aic(self, X):
        """
        Returns the Akaike Information Criterion of the model
        """
        return 2*self.num_parameters() - 2*self.getLogLikelihood(X)
    
    def bic(self, X):
        """
        Returns the Bayesian Information Criterion of the model
        """
        return self.num_parameters()*np.log(X.shape[0]) - 2*self.getLogLikelihood(X)

# formatted sklearn GMM Model based on fuctionalities required
class GMM_sklearn:
    """
    Parameters
    ----------
    k: int
        Number of clusters
    """

    def __init__(self, k):
        """
        Initialize the model
        """
        self.model = GaussianMixture(n_components=k)

    def fit(self, X):
        """
        Uses Expectation Maximization to find the means, covariances and the mixing coefficients

        Parameters
        ----------
        X: np.ndarray
            The data that has to be classfied into clusters
        """
        self.model.fit(X)


    def getParams(self):
        """
        Returns the means, covariances and the mixing coefficients
        """
        return self.model.means_, self.model.covariances_, self.model.weights_

    def getMembership(self, X):
        """
        Returns the membership of the data points

        Parameters
        ----------
        X: np.ndarray
            The data that has to be classfied into clusters
        """
        return self.model.predict_proba(X)
        
    def getLikelihood(self, X):
        """
        Returns the likelihood of the data given the parameters
        """
        return np.exp(self.getLogLikelihood(X))

    def getLogLikelihood(self, X):
        """
        Returns the log likelihood of the data given the parameters
        """
        return np.sum(self.model.score_samples(X))
    
    def aic(self, X):
        """
        Returns the Akaike Information Criterion of the model
        """
        return self.model.aic(X)
    
    def bic(self, X):
        """
        Returns the Bayesian Information Criterion of the model
        """
        return self.model.bic(X)

    