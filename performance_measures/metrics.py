import numpy as np


class Metrics:
    """
    A class to calculate performance metrics
    """

    def __init__(self):
        pass

    def accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The accuracy of the model
        """
        # return np.mean(y_true == y_pred)
        if len(y_pred.shape) == 1 or len(y_true.shape) == 1:
            return np.mean(y_true == y_pred)
        return np.mean(np.all(y_true == y_pred, axis=1))
    
    def multi_label_accuracy(self, y_true, y_pred):
        """
        Calculate the multi-label accuracy of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The multi-label accuracy of the model
        """
        return np.mean([np.sum(t == p) / len(t) for t, p in zip(y_true, y_pred)])

    def precision(self, y_true, y_pred, type="micro"):
        """
        Calculate the precision of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values
        type : str, optional
            The type of precision to calculate. The default is 'micro'. Possible values: 'micro', 'macro'

        Returns
        -------
        float
            The precision of the model
        """
        if type == "micro":
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        elif type == "macro":
            precisions = []
            for i in range(y_true.shape[1]):  # Loop over each label
                tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
                fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            return np.mean(precisions)
        else:
            raise ValueError("Invalid type")

    def recall(self, y_true, y_pred, type="micro"):
        """
        Calculate the recall of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values
        type : str, optional
            The type of recall to calculate. The default is 'micro'. Possible values: 'micro', 'macro'

        Returns
        -------
        float
            The recall of the model
        """
        if type == "micro":
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        elif type == "macro":
            recalls = []
            for i in range(y_true.shape[1]):  # Loop over each label
                tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
                fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
                recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            return np.mean(recalls)
        else:
            raise ValueError("Invalid type")

    def f1_score(self, y_true, y_pred, type="micro"):
        """
        Calculate the F1 score of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values
        type : str, optional
            The type of precision to calculate. The default is 'micro'. Possible values: 'micro', 'macro

        Returns
        -------
        float : The F1 score of the model
        """
        if type == "micro":
            precision = self.precision(y_true, y_pred, type="micro")
            recall = self.recall(y_true, y_pred, type="micro")
            return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        elif type == "macro":
            f1_list = []
            for i in range(y_true.shape[1]):  # Loop over each label
                tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
                fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
                fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_list.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            return np.mean(f1_list)
        else:
            raise ValueError("Invalid type")

    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate the mean squared error of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The mean squared error of the model
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def mean_absolute_error(self, y_true, y_pred):
        """
        Calculate the mean absolute error of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The mean absolute error of the model
        """
        return np.mean(np.abs(y_true - y_pred))

    def standard_deviation(self, y_true, y_pred):
        """
        Calculate the standard deviation of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The standard deviation of the model
        """
        return np.std(y_true - y_pred)

    def variance(self, y_true, y_pred):
        """
        Calculate the variance of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The variance of the model
        """
        return np.var(y_true - y_pred)
    
    def root_mean_squared_error(self, y_true, y_pred):
        """
        Calculate the root mean squared error of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The root mean squared error of the model
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def r2_score(self, y_true, y_pred):
        """
        Calculate the R^2 score of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The R^2 score of the model
        """
        mean_y = np.mean(y_true)
        ss_tot = np.sum((y_true - mean_y) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def hamming_loss(self, y_true, y_pred):
        """
        Calculate the hamming loss of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The hamming loss of the model
        """
        return np.sum(y_true != y_pred) / (y_true.shape[0] * y_true.shape[1])
    

    def multi_label_precision(self, y_true, y_pred):
        """
        Calculate the multi-label precision of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The multi-label precision of the model
        """
        precisions = []
        for t, p in zip(y_true, y_pred):
            tp = np.sum((t == 1) & (p == 1))
            fp = np.sum((t == 0) & (p == 1))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return np.mean(precisions)

    def multi_label_recall(self, y_true, y_pred):
        """
        Calculate the multi-label recall of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The multi-label recall of the model
        """
        recalls = []
        for t, p in zip(y_true, y_pred):
            tp = np.sum((t == 1) & (p == 1))
            fn = np.sum((t == 1) & (p == 0))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return np.mean(recalls)

    def multi_label_f1(self, y_true, y_pred):
        """
        Calculate the multi-label F1 score of the model

        Parameters
        ----------
        y_true : numpy.ndarray
            The true target values
        y_pred : numpy.ndarray
            The predicted target values

        Returns
        -------
        float
            The multi-label F1 score of the model
        """
        precisions = [self.multi_label_precision([t], [p]) for t, p in zip(y_true, y_pred)]
        recalls = [self.multi_label_recall([t], [p]) for t, p in zip(y_true, y_pred)]
        f1_scores = [
            (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            for precision, recall in zip(precisions, recalls)
        ]
        return np.mean(f1_scores)
