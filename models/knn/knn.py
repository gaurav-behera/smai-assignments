import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# Initial knn implementation that uses for loops for computation
class KNNInitial:
    """
    Parameters
    ----------
    k : int, optional
        Number of neighbors to consider. The default is 5.
    metric : str, optional
        Distance metric to use. The default is 'euclidean'. Possible values: 'euclidean', 'manhattan', 'cosine'.
    weights : str, optional
        Weight function used in prediction. The default is 'uniform'. Possible values: 'uniform', 'distance'.
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        """
        Initialize the KNN model
        """
        self.k = k
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the training data points

        Parameters
        ----------
        X : pandas.DataFrame
            The training features
        y : pandas.Series
            The training target values
        """
        with tqdm(total=1, desc="Fitting", unit="point") as pbar:
            self.X = X.to_numpy()
            self.y = y.to_numpy()
            pbar.update(1)

    def predict(self, X):
        """
        Predict the class labels for the provided data points

        Parameters
        ----------
        X : pandas.DataFrame
            The data points for which to predict the class labels

        Returns
        -------
        numpy.ndarray
            The predicted class labels
        """

        X = X.to_numpy()
        # precompute the norms of the data points
        if self.metric == "cosine":
            self.X_norms = np.linalg.norm(self.X, axis=1)

        y_pred = []
        datapoints_count = X.shape[0]
        for i in tqdm(range(0, datapoints_count), desc="Predicting", unit="points"):
            # get the distances of the current data point from all the training data points
            distances = []
            for j in range(self.X.shape[0]):
                match self.metric:
                    case "euclidean":
                        dist = np.sqrt(np.sum((X[i] - self.X[j]) ** 2))
                    case "manhattan":
                        dist = np.sum(np.abs(X[i] - self.X[j]))
                    case "cosine":
                        dist = 1 - np.dot(X[i], self.X[j]) / (
                            np.linalg.norm(X[i]) * self.X_norms[j]
                        )
                    case _:
                        raise ValueError("Invalid metric")
                distances.append(dist)

            distances = np.array(distances)
            # get the indices of the k nearest neighbors
            idx = np.argsort(distances)[: self.k]
            # get the classes of the k nearest neighbors
            y_pool = self.y[idx]
            # label encode y pool
            y_strings, y_pool_int = np.unique(y_pool, return_inverse=True)

            match self.weights:
                case "uniform":
                    y = np.argmax(np.bincount(y_pool_int))
                case "distance":
                    weights = 1 / (1 + distances[idx])
                    y = np.argmax(np.bincount(y_pool_int, weights=weights))
                case _:
                    raise ValueError("Invalid weights")
            y_pred.append(y_strings[y])

        y_pred = np.array(y_pred)
        return y_pred


# Best knn implementation that uses numpy default vectorization for faster computation
class KNNBest:
    """
    Parameters
    ----------
    k : int, optional
        Number of neighbors to consider. The default is 5.
    metric : str, optional
        Distance metric to use. The default is 'euclidean'. Possible values: 'euclidean', 'manhattan', 'cosine'.
    weights : str, optional
        Weight function used in prediction. The default is 'uniform'. Possible values: 'uniform', 'distance'.
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        self.k = k
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the training data points

        Parameters
        ----------
        X : pandas.DataFrame
            The training features
        y : pandas.Series
            The training target values
        """
        with tqdm(total=1, desc="Fitting", unit="point") as pbar:
            self.X = X.to_numpy()
            self.y = y.to_numpy()
            pbar.update(1)

    def predict(self, X):
        """
        Predict the class labels for the provided data points

        Parameters
        ----------
        X : pandas.DataFrame
            The data points for which to predict the class labels

        Returns
        -------
        numpy.ndarray
            The predicted class labels
        """

        X = X.to_numpy()
        # precompute the norms of the data points
        if self.metric == "cosine":
            self.X_norms = np.linalg.norm(self.X, axis=1)

        y_pred = []
        datapoints_count = X.shape[0]
        chunk_size = 50
        for start in tqdm(
            range(0, datapoints_count, chunk_size), desc="Predicting", unit="points"
        ):
            end = min(start + chunk_size, datapoints_count)
            chunk = X[start:end]
            # compute the distances of the chunk from all the training data points
            distances = self._compute_distances(chunk)

            for i in range(end - start):
                # get the top k nearest neighbors
                idx = np.argsort(distances[i])[: self.k]
                y_pool = self.y[idx]
                _, y_numeric = np.unique(y_pool, return_inverse=True)

                # predict class label based on the weights
                match self.weights:
                    case "uniform":
                        y = np.argmax(np.bincount(y_numeric))
                    case "distance":
                        weights = 1 / (1 + distances[i][idx])
                        y = np.argmax(np.bincount(y_numeric, weights=weights))
                    case _:
                        raise ValueError("Invalid weights")

                y_pred.append(np.unique(y_pool)[y])

        return np.array(y_pred)

    def _compute_distances(self, X):
        if self.metric == "euclidean":
            return np.sqrt(((np.expand_dims(X, axis=1) - self.X) ** 2).sum(axis=2))
        elif self.metric == "manhattan":
            return np.abs(np.expand_dims(X, axis=1) - self.X).sum(axis=2)
        elif self.metric == "cosine":
            dot_product = X @ self.X.T
            norms = np.outer(np.linalg.norm(X, axis=1), self.X_norms)
            return 1 - dot_product / norms
        else:
            raise ValueError("Invalid metric")


# Most optimized knn implementation that uses parallel processing for faster computation
class KNN:
    """
    Parameters
    ----------
    k : int, optional
        Number of neighbors to consider. The default is 5.
    metric : str, optional
        Distance metric to use. The default is 'euclidean'. Possible values: 'euclidean', 'manhattan', 'cosine'.
    weights : str, optional
        Weight function used in prediction. The default is 'uniform'. Possible values: 'uniform', 'distance'.
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        self.k = k
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the training data points

        Parameters
        ----------
        X : pandas.DataFrame
            The training features
        y : pandas.Series
            The training target values
        """
        with tqdm(total=1, desc="Fitting", unit="point") as pbar:
            self.X = X.to_numpy()
            self.y = y.to_numpy()
            pbar.update(1)

    def predict(self, X):
        """
        Predict the class labels for the provided data points

        Parameters
        ----------
        X : pandas.DataFrame
            The data points for which to predict the class labels

        Returns
        -------
        numpy.ndarray
            The predicted class labels
        """

        X = X.to_numpy()
        # precompute the norms of the data points
        if self.metric == "cosine":
            self.X_norms = np.linalg.norm(self.X, axis=1)

        datapoints_count = X.shape[0]
        chunk_size = 50
        num_chunks = (datapoints_count + chunk_size - 1) // chunk_size
        futures = []
        y_pred = [None] * datapoints_count

        def process_chunk(start, end):
            chunk = X[start:end]
            distances = self._compute_distances(chunk)
            chunk_predictions = []
            for i in range(end - start):
                # get the top k nearest neighbors
                idx = np.argsort(distances[i])[: self.k]
                y_pool = self.y[idx]
                _, y_numeric = np.unique(y_pool, return_inverse=True)

                # predict class label based on the weights
                match self.weights:
                    case "uniform":
                        y = np.argmax(np.bincount(y_numeric))
                    case "distance":
                        weights = 1 / (1 + distances[i][idx])
                        y = np.argmax(np.bincount(y_numeric, weights=weights**2))
                    case _:
                        raise ValueError("Invalid weights")

                chunk_predictions.append(np.unique(y_pool)[y])
            return start, end, chunk_predictions

        with ThreadPoolExecutor() as executor:
            with tqdm(total=num_chunks, desc="Predicting", unit="chunk") as pbar:
                for start in range(0, datapoints_count, chunk_size):
                    end = min(start + chunk_size, datapoints_count)
                    futures.append(executor.submit(process_chunk, start, end))

                for future in futures:
                    start, end, chunk_predictions = future.result()
                    y_pred[start:end] = chunk_predictions
                    pbar.update(1)

        return np.array(y_pred)

    def _compute_distances(self, X):
        """
        Compute the distances between the input data points and the training data points
        """
        if self.metric == "euclidean":
            return np.sqrt(((np.expand_dims(X, axis=1) - self.X) ** 2).sum(axis=2))
        elif self.metric == "manhattan":
            return np.abs(np.expand_dims(X, axis=1) - self.X).sum(axis=2)
        elif self.metric == "cosine":
            dot_product = X @ self.X.T
            norms = np.outer(np.linalg.norm(X, axis=1), self.X_norms)
            return 1 - dot_product / norms
        else:
            raise ValueError("Invalid metric")
