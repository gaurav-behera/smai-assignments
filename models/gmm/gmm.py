import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

class GMM:
    """
    Parameters
    ----------
    k: int
        Number of clusters
    cost_threshold: float
        The threshold for the cost function to stop the cluster calculation. The default value is 1e-6
    """

    def __init__(self, k, cost_threshold=1e-6):
        """
        Initialize the model
        """
        self.k = k
        self.cost_threshold = cost_threshold
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
        self._covs = [np.cov(X.T) for _ in range(self.k)]
        pis = np.random.rand(self.k)
        self._pis = pis / np.sum(pis)
        self.log_likelihood = self.getLogLikelihood(X)

        pbar = tqdm(desc="Finding clusters", position=0, leave=True)

        while True:
            # E step: evaluate the responsibilities using the current parameters
            resp = self.getMembership(X)

            # M step: update the parameters using the current responsibilities
            new_means = np.array(
                [
                    np.sum(resp[:, k, np.newaxis] * X, axis=0) / np.sum(resp[:, k])
                    for k in range(self.k)
                ]
            )
            new_covs = []
            for k in range(self.k):
                cov_matrix = np.zeros((X.shape[1], X.shape[1]))
                for i in range(X.shape[0]):
                    diff = (X[i] - new_means[k]).reshape(-1, 1)
                    cov_matrix += resp[i, k] * np.dot(diff, diff.T)
                cov_matrix /= np.sum(resp[:, k])
                new_covs.append(cov_matrix)
                
            if np.nan in np.array(new_covs).flatten():
                print("nan in new_covs")

            new_pis = np.sum(resp, axis=0) / X.shape[0]

            self._means = new_means
            self._covs = new_covs
            self._pis = new_pis

            # evaluate log likelihood and update the parameters until convergence
            log_likelihood = self.getLogLikelihood(X)
            print(log_likelihood)
            if log_likelihood - self.log_likelihood < self.cost_threshold:
                break
            else:
                self.log_likelihood = log_likelihood
            pbar.update(1)

    def _multivariate_normal(self, x, mean, cov):
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
        """
        
        val = multivariate_normal(mean=mean, cov=cov, allow_singular=True).pdf(x)
        if val < -10e-6:
            return -10e6
        if val > 10e6:
            return 10e6
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
        resp = np.zeros([X.shape[0], self.k])
        for i in range(X.shape[0]):
            cluster_probs = [
                self._pis[j]
                * self._multivariate_normal(X[i], self._means[j], self._covs[j])
                for j in range(self.k)
            ]
            if np.inf in cluster_probs:
                print("inf in cluster_probs")
            if -np.inf in cluster_probs:
                print("-inf in cluster_probs")
            cluster_probs_sum = np.sum(cluster_probs)
            resp[i, :] = cluster_probs / cluster_probs_sum
        return resp
        
    def getLikelihood(self, X):
        """
        Returns the likelihood of the data given the parameters
        """
        return np.prod(
            [
                np.sum(
                    [
                        self._pis[k]
                        * self._multivariate_normal(X[n], self._means[k], self._covs[k])
                        for k in range(self.k)
                    ]
                )
                for n in range(X.shape[0])
            ]
        )

    def getLogLikelihood(self, X):
        """
        Returns the log likelihood of the data given the parameters
        """
        return np.sum(
            [
                np.log(
                    np.sum(
                        [
                            self._pis[k]
                            * self._multivariate_normal(
                                X[n], self._means[k], self._covs[k]
                            )
                            for k in range(self.k)
                        ]
                    )
                )
                for n in range(X.shape[0])
            ]
        )

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


# Original GMM class that does not handle singular covariance matrices
class GMM_old:
    """
    Parameters
    ----------
    k: int
        Number of clusters
    cost_threshold: float
        The threshold for the cost function to stop the cluster calculation. The default value is 1e-6
    """

    def __init__(self, k, cost_threshold=1e-6):
        """
        Initialize the model
        """
        self.k = k
        self.cost_threshold = cost_threshold
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
        self._covs = [np.cov(X.T) for _ in range(self.k)]
        pis = np.random.rand(self.k)
        self._pis = pis / np.sum(pis)
        self.log_likelihood = self.getLogLikelihood(X)

        pbar = tqdm(desc="Finding clusters", position=0, leave=True)

        while True:
            # E step: evaluate the responsibilities using the current parameters
            resp = self.getMembership(X)

            # M step: update the parameters using the current responsibilities
            new_means = np.array(
                [
                    np.sum(resp[:, k, np.newaxis] * X, axis=0) / np.sum(resp[:, k])
                    for k in range(self.k)
                ]
            )
            new_covs = []
            for k in range(self.k):
                cov_matrix = np.zeros((X.shape[1], X.shape[1]))
                for i in range(X.shape[0]):
                    diff = (X[i] - new_means[k]).reshape(-1, 1)
                    cov_matrix += resp[i, k] * np.dot(diff, diff.T)
                cov_matrix /= np.sum(resp[:, k])
                new_covs.append(cov_matrix)

            new_pis = np.sum(resp, axis=0) / X.shape[0]

            self._means = new_means
            self._covs = new_covs
            self._pis = new_pis

            # evaluate log likelihood and update the parameters until convergence
            log_likelihood = self.getLogLikelihood(X)
            if log_likelihood - self.log_likelihood < self.cost_threshold:
                break
            else:
                self.log_likelihood = log_likelihood
            pbar.update(1)

    def _multivariate_normal(self, x, mean, cov):
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
        """
        return np.exp(-0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)) / np.sqrt(
            ((2 * np.pi) ** x.shape[0]) * np.linalg.det(cov)
        )

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
        resp = np.zeros([X.shape[0], self.k])
        for i in range(X.shape[0]):
            cluster_probs = [
                self._pis[j]
                * self._multivariate_normal(X[i], self._means[j], self._covs[j])
                for j in range(self.k)
            ]
            cluster_probs_sum = np.sum(cluster_probs)
            resp[i, :] = cluster_probs / cluster_probs_sum
        return resp
        
    def getLikelihood(self, X):
        """
        Returns the likelihood of the data given the parameters
        """
        return np.prod(
            [
                np.sum(
                    [
                        self._pis[k]
                        * self._multivariate_normal(X[n], self._means[k], self._covs[k])
                        for k in range(self.k)
                    ]
                )
                for n in range(X.shape[0])
            ]
        )

    def getLogLikelihood(self, X):
        """
        Returns the log likelihood of the data given the parameters
        """
        return np.sum(
            [
                np.log(
                    np.sum(
                        [
                            self._pis[k]
                            * self._multivariate_normal(
                                X[n], self._means[k], self._covs[k]
                            )
                            for k in range(self.k)
                        ]
                    )
                )
                for n in range(X.shape[0])
            ]
        )
