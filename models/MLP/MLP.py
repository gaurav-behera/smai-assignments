import numpy as np
import pandas as pd
from performance_measures.metrics import Metrics
import wandb


class MLP:
    """
    Multi-layer perceptron class for classification and regression tasks.

    Parameters
    ----------
    learning_rate : float
        The learning rate for the model. Default is 0.1.
    activation_function : str
        The activation function to use in the hidden layers. Default is "relu". Available options are "relu", "sigmoid", "tanh", "linear".
    optimizer : str
        The optimizer to use for training the model. Default is "sgd". Available options are "sgd", "batch-gd", "minibatch-gd.
    num_hidden_layers : int
        The number of hidden layers in the model. Default is 1.
    num_neurons : list
        The number of neurons in each hidden layer. Default is [64].
    batch_size : int
        The batch size for training the model. Default is 32.
    epochs : int
        The number of epochs to train the model. Default is 1000.
    input_layer_size : int
        The size of the input layer. Default is 1.
    output_layer_size : int
        The size of the output layer. Default is 1.
    log_wandb : bool
        Whether to log the metrics to Weights & Biases. Default is False.
    log_local : bool
        Whether to log the metrics to a local variable. Default is False.
    task : str
        The task to perform. Default is "regression". Available options are "regression", "classification-single-label", "classification-multi-label".
    """

    def __init__(
        self,
        learning_rate=0.1,
        activation_function="relu",
        optimizer="sgd",
        num_hidden_layers=1,
        num_neurons=[64],
        batch_size=32,
        epochs=1000,
        input_layer_size=1,
        output_layer_size=1,
        log_wandb=False,
        log_local=False,
        task="regression",
    ):
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.log_wandb = log_wandb
        self.log_local = log_local
        self.task = task
        self.metrics = Metrics()

        if self.optimizer not in ["sgd", "batch-gd", "minibatch-gd"]:
            raise ValueError(
                "Invalid optimizer specified. Available options are 'sgd', 'batch-gd', 'minibatch-gd'."
            )

        match self.activation_function:
            case "relu":
                self._activation = self._relu
                self._activation_derivative = self._relu_derivative
            case "sigmoid":
                self._activation = self._sigmoid
                self._activation_derivative = self._sigmoid_derivative
            case "tanh":
                self._activation = self._tanh
                self._activation_derivative = self._tanh_derivative
            case "linear":
                self._activation = self._linear
                self._activation_derivative = self._linear_derivative
            case _:
                raise ValueError(
                    "Invalid activation function specified. Available options are 'relu', 'sigmoid', 'tanh', 'linear'."
                )

        match self.task:
            case "regression":
                self._cost_function = self._mean_squared_error
                self._activation_output = self._linear
                self._activation_derivative_output = self._linear_derivative
            case "classification-single-label":
                self._cost_function = self._cross_entropy
                self._activation_output = self._softmax
                self._activation_derivative_output = self._softmax_derivative
            case "classification-multi-label":
                self._cost_function = self._binary_cross_entropy
                self._activation_output = self._sigmoid
                self._activation_derivative_output = self._sigmoid_derivative
            case _:
                raise ValueError(
                    "Invalid task specified. Available options are 'regression', 'classification-single-label', 'classification-multi-label'."
                )

        if log_local:
            self.logs = []

        self._initialize_model()

    # activation functions
    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)  # for numerical stability
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return np.tanh(x)

    def _linear(self, x):
        return x

    def _softmax(self, x):
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    # activation function derivative
    def _relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)

    def _sigmoid_derivative(self, x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def _tanh_derivative(self, x):
        return np.ones_like(x) - np.tanh(x) ** 2

    def _linear_derivative(self, x):
        return np.ones_like(x)

    def _softmax_derivative(self, x):
        s = self._softmax(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    # cost functions
    def _mean_squared_error(self, y_pred, y):
        diff = np.subtract(y_pred, y, dtype=np.float64)
        return np.mean(np.square(diff, dtype=np.float64))

    def _cross_entropy(self, y_pred, y):
        epsilon = 1e-4
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.sum(y * np.log(y_pred))

    def _binary_cross_entropy(self, y_pred, y):
        epsilon = 1e-4
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # model initialization
    def _initialize_model(self):
        """
        Initialize the model with the random parameters.
        """
        self.weights = []
        self.biases = []

        # set weights as random values and biases as zeros
        self.weights.append(
            np.random.randn(self.input_layer_size, self.num_neurons[0])
            / np.sqrt(self.input_layer_size)
        )
        self.biases.append(np.zeros(self.num_neurons[0]))

        for i in range(self.num_hidden_layers - 1):
            self.weights.append(
                np.random.randn(self.num_neurons[i], self.num_neurons[i + 1])
                / np.sqrt(self.num_neurons[i])
            )
            self.biases.append(np.zeros(self.num_neurons[i + 1]))

        self.weights.append(
            np.random.randn(self.num_neurons[-1], self.output_layer_size)
            / np.sqrt(self.num_neurons[-1])
        )
        self.biases.append(np.zeros(self.output_layer_size))

    def forward(self, X):
        """
        Forward pass of the model.

        Parameters
        ----------
        X : numpy array
            The input to the model.

        Returns
        -------
        numpy array
            The output of the model.
        """
        self.activations = [X]

        # perform forward pass
        for i in range(self.num_hidden_layers):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.activations.append(self._activation(z))

        # last layer is either linear or softmax based on the task
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.activations.append(z)

        return self.activations[-1]

    def cost_function(self, X, y):
        """
        Compute the cost function for the model.

        Parameters
        ----------
        X : numpy array
            The input to the model.
        y : numpy array
            The target output.

        Returns
        -------
        float
            The cost of the model.
        """
        predictions = self.predict(X)
        return self._cost_function(predictions, y)

    def cost_function_derivative(self, X, y):
        """
        Compute the derivative of the cost function.

        Parameters
        ----------
        X : numpy array
            The input to the model.
        y : numpy array
            The target output.

        Returns
        -------
        tuple of lists
        The derivatives of the cost function with respect to the weights and biases.
        w_grads: list of numpy arrays
            The gradients of the weights.
        b_grads: list of numpy arrays
            The gradients of the biases.
        """
        predictions = self.forward(X)
        delta = -(y - predictions)
        # delta = -(y - predictions) * self._activation_derivative_output(predictions)
        deltas = [delta]

        # compute deltas which are the propagated errors
        for i in range(self.num_hidden_layers, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self._activation_derivative(
                self.activations[i]
            )
            deltas.append(delta)

        deltas = deltas[::-1]
        w_grads = []
        b_grads = []

        # compute gradients for weights and biases based on deltas
        for i in range(len(deltas)):
            w_grads.append(np.dot(self.activations[i].T, deltas[i]) / X.shape[0])
            b_grads.append(np.mean(deltas[i], axis=0))

        return w_grads, b_grads

    def numerical_cost_function_derivative(self, X, y, e=1e-4):
        w_grads = []
        b_grads = []

        # numerical weights gradient
        for i in range(len(self.weights)):
            w_grad = np.zeros_like(self.weights[i])
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    self.weights[i][j, k] += e
                    c1 = self.cost_function(X, y)
                    self.weights[i][j, k] -= 2 * e
                    c2 = self.cost_function(X, y)
                    w_grad[j, k] = (c1 - c2) / (2 * e)
                    self.weights[i][j, k] += e
            w_grads.append(w_grad)

        # numerical bias gradient
        for i in range(len(self.biases)):
            b_grad = np.zeros_like(self.biases[i])
            for j in range(self.biases[i].shape[0]):
                self.biases[i][j] += e
                c1 = self.cost_function(X, y)
                self.biases[i][j] -= 2 * e
                c2 = self.cost_function(X, y)
                b_grad[j] = (c1 - c2) / (2 * e)
                self.biases[i][j] += e
            b_grads.append(b_grad)

        return w_grads, b_grads

    def check_gradients(self, X, y):
        w_grads, b_grads = self.cost_function_derivative(X, y)
        w_grads_num, b_grads_num = self.numerical_cost_function_derivative(X, y)

        # flatten out the gradients
        grads = np.array([])
        for i in range(len(w_grads)):
            grads = np.concatenate((grads, w_grads[i].flatten()))

        for i in range(len(b_grads)):
            grads = np.concatenate((grads, b_grads[i].flatten()))

        num_grads = np.array([])
        for i in range(len(w_grads_num)):
            num_grads = np.concatenate((num_grads, w_grads_num[i].flatten()))

        for i in range(len(b_grads_num)):
            num_grads = np.concatenate((num_grads, b_grads_num[i].flatten()))

        # compute score
        return np.linalg.norm(grads - num_grads) / (np.linalg.norm(grads + num_grads))

    def fit(self, X, y, early_stop=False, X_val=None, y_val=None, patience=5):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : numpy array
            The input to the model.
        y : numpy array
            The target output.
        early_stop : bool
            Whether to use early stopping. Default is False.
        X_val : numpy array
            The input to the model for validation. Default is None.
        y_val : numpy array
            The target output for validation. Default is None.
        """

        match self.optimizer:
            case "sgd":
                self._gradient_descent_loop(
                    X, y, X_val, y_val, early_stop, patience, batch_size=1
                )
            case "batch-gd":
                self._gradient_descent_loop(X, y, X_val, y_val, early_stop, patience)
            case "minibatch-gd":
                self._gradient_descent_loop(
                    X, y, X_val, y_val, early_stop, patience, batch_size=self.batch_size
                )
            case _:
                raise ValueError(
                    "Invalid optimizer specified. Available options are 'sgd', 'batch-gd', 'minibatch-gd'."
                )

    def _update_weights_and_biases(self, w_grads, b_grads):
        for j in range(len(self.weights)):
            self.weights[j] -= self.learning_rate * w_grads[j]
            self.biases[j] -= self.learning_rate * b_grads[j]

    def _print_metrics(self, X, y, predictions, phase):
        cost = self.cost_function(X, y)
        print(f"{phase} Cost: {cost}")
        match self.task:
            case "regression":
                print(f"{phase} MSE: {self.metrics.mean_squared_error(y, predictions)}")
            case "classification-single-label":
                print(f"{phase} Accuracy: {self.metrics.accuracy(y, predictions)}")
            case "classification-multi-label":
                print(
                    f"{phase} Accuracy: {self.metrics.multi_label_accuracy(y, predictions)}"
                )
            case _:
                raise ValueError(
                    "Invalid task specified. Available options are 'regression', 'classification-single-label', 'classification-multi-label'."
                )
        if self.log_wandb:
            match self.task:
                case "regression":
                    wandb.log(
                        {
                            f"{phase}_cost": cost,
                            f"{phase}_mse": self.metrics.mean_squared_error(
                                y, predictions
                            ),
                            f"{phase}_rmse": self.metrics.root_mean_squared_error(
                                y, predictions
                            ),
                            f"{phase}_r2": self.metrics.r2_score(y, predictions),
                        }
                    )
                case "classification-single-label":
                    wandb.log(
                        {
                            f"{phase}_cost": cost,
                            f"{phase}_accuracy": self.metrics.accuracy(y, predictions),
                            f"{phase}_f1": self.metrics.f1_score(y, predictions),
                            f"{phase}_precision": self.metrics.precision(
                                y, predictions
                            ),
                            f"{phase}_recall": self.metrics.recall(y, predictions),
                        }
                    )
                case "classification-multi-label":
                    wandb.log(
                        {
                            f"{phase}_cost": cost,
                            f"{phase}_accuracy": self.metrics.multi_label_accuracy(
                                y, predictions
                            ),
                            f"{phase}_f1": self.metrics.f1_score(y, predictions),
                            f"{phase}_precision": self.metrics.precision(
                                y, predictions
                            ),
                            f"{phase}_recall": self.metrics.recall(y, predictions),
                            f"{phase}_hamming_loss": self.metrics.hamming_loss(
                                y, predictions
                            ),
                        }
                    )
                case _:
                    raise ValueError(
                        "Invalid task specified. Available options are 'regression', 'classification-single-label', 'classification-multi-label'."
                    )

        if self.log_local and phase == "Validation":
            match self.task:
                case "regression":
                    self.logs.append(
                        {
                            "loss": cost,
                            "mse": self.metrics.mean_squared_error(y, predictions),
                        }
                    )
                case "classification-single-label":
                    self.logs.append(
                        {
                            "loss": cost,
                            "accuracy": self.metrics.accuracy(y, predictions),
                        }
                    )
                case "classification-multi-label":
                    self.logs.append(
                        {
                            "loss": cost,
                            "accuracy": self.metrics.multi_label_accuracy(
                                y, predictions
                            ),
                        }
                    )
                case _:
                    raise ValueError(
                        "Invalid task specified. Available options are 'regression', 'classification-single-label', 'classification-multi-label'."
                    )

        return cost

    def _check_early_stop(self, val_cost, last_val_cost):
        if val_cost > last_val_cost + 1e-4:
            return True, last_val_cost
        return False, min(val_cost, last_val_cost)

    def _gradient_descent_step(self, X, y, batch_indices=None):
        if batch_indices is None:
            w_grads, b_grads = self.cost_function_derivative(X, y)
        else:
            w_grads, b_grads = self.cost_function_derivative(
                X[batch_indices], y[batch_indices]
            )
        self._update_weights_and_biases(w_grads, b_grads)

    def _gradient_descent_loop(
        self, X, y, X_val, y_val, early_stop, patience, batch_size=None
    ):
        last_val_cost = np.inf
        best_params = (self.weights.copy(), self.biases.copy())
        for e in range(self.epochs):
            if batch_size is not None:
                # Shuffle data
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                X, y = X[indices], y[indices]

            for i in range(0, X.shape[0], batch_size or X.shape[0]):
                batch_indices = slice(i, i + batch_size) if batch_size else None
                self._gradient_descent_step(X, y, batch_indices)

            train_predictions = self.predict(X)
            print(f"Epoch {e+1}/{self.epochs}")
            train_cost = self._print_metrics(X, y, train_predictions, "Train")

            if early_stop:
                val_predictions = self.predict(X_val)
                val_cost = self._print_metrics(
                    X_val, y_val, val_predictions, "Validation"
                )

                stop, last_val_cost = self._check_early_stop(val_cost, last_val_cost)
                if stop:
                    patience -= 1
                else:
                    best_params = self.weights, self.biases
                if patience == 0:
                    break
            else:
                best_params = self.weights, self.biases
        self.weights, self.biases = best_params

    def predict(self, X, threshold=0.5):
        """
        Predict the output for the given input.

        Parameters
        ----------
        X : numpy array
            The input to the model.

        Returns
        -------
        numpy array
            The predicted output.
        """
        ypred = self.forward(X)
        ypred = self._activation_output(ypred)
        match self.task:
            case "classification-single-label":
                argmaxs = np.argmax(ypred, axis=1)
                res = np.zeros_like(ypred)
                res[np.arange(ypred.shape[0]), argmaxs] = 1
                return res
            case "classification-multi-label":
                return (ypred > threshold).astype(int)
            case "regression":
                return ypred
            case _:
                raise ValueError(
                    "Invalid task specified. Available options are 'regression', 'classification-single-label', 'classification-multi-label'."
                )


class MLP_classification_single_label(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, task="classification-single-label")


class MLP_classification_multi_label(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, task="classification-multi-label")


class MLP_regression(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, task="regression")
