import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class LinearRegression:
    """
    Parameters
    ----------
    reg_lambda : float, optional
        Regularization parameter. The default is 0.0.
    degree : int, optional
        The degree of the polynomial. The default is 1.
    """

    def __init__(self, reg_lambda=0.0, degree=1):
        self.reg_lambda = reg_lambda
        self.degree = degree
        self.weights = None

    def fit(
        self,
        X,
        y,
        type="gradient_descent",
        learning_rate=0.01,
        epochs=1000,
        regularizer="ridge",
    ):
        """
        Fit the training data points

        Parameters
        ----------
        X : numpy.ndarray
            The training features
        y : numpy.ndarray
            The training target values
        type : str, optional
            The type of optimization to use. The default is "gradient_descent". Possible values: "gradient_descent", "closed_form"
        learning_rate : float, optional
            The learning rate for gradiant descent. The default is 0.01.
        epochs : int, optional
            The number of epochs for gradient descent. The default is 100.
        regularizer : str, optional
            The type of regularizer to use. The default is "ridge". Possible values: "ridge", "lasso"
        """
        # add polynomial features
        X = self._add_polynomial_features(X)

        match type:
            case "gradient_descent":
                # initialize weights
                self.weights = np.random.rand(X.shape[1])
                self._gradient_descent(X, y, learning_rate, epochs, regularizer)
            case "closed_form":
                if regularizer != "ridge":
                    raise ValueError(
                        "Closed form solution is only available for ridge regression"
                    )
                with tqdm(total=1, desc="Fitting", unit="point") as pbar:
                    self.weights = (
                        np.linalg.inv(X.T @ X + self.reg_lambda * np.eye(X.shape[1]))
                        @ X.T
                        @ y
                    )
                    pbar.update(1)
            case _:
                raise ValueError("Invalid type")

    def predict(self, X):
        """
        Predict the target values

        Parameters
        ----------
        X : numpy.ndarray
            The input features

        Returns
        -------
        numpy.ndarray
            The predicted target values
        """
        X = self._add_polynomial_features(X)
        return X @ self.weights

    def get_weights(self):
        """
        Get the weights of the model

        Returns
        -------
        numpy.ndarray
            The weights of the model
        """
        return self.weights

    def _add_polynomial_features(self, X):
        """
        Add polynomial features to the input features

        Parameters
        ----------
        X : numpy.ndarray
            The input features

        Returns
        -------
        numpy.ndarray
            The input features with polynomial features
        """
        return np.hstack([X**i for i in range(0, self.degree + 1)])

    def _gradient_descent(self, X, y, learning_rate, epochs, regularizer="ridge"):
        """
        Perform gradient descent optimization
        """

        # perform gradient descent
        for _ in tqdm(range(epochs)):
            datapoints_count = X.shape[0]
            preds = X @ self.weights
            gradient = (1 / datapoints_count) * X.T @ (preds - y)
            # regularization
            if self.reg_lambda != 0:
                match regularizer:
                    case "ridge":
                        gradient += 2 * self.reg_lambda * self.weights / datapoints_count
                    case "lasso":
                        gradient += self.reg_lambda * np.sign(self.weights) / datapoints_count
                    case _:
                        raise ValueError("Invalid regularizer")
            # update weights
            self.weights -= learning_rate * gradient
            # print(f"Loss: {np.mean((preds - y)**2)}")
            if np.abs(gradient).mean() < 1e-5:
                print("Converged")
                break
    