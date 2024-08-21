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
        return np.mean(y_true == y_pred)

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
        classes = np.unique(np.concatenate((y_true, y_pred)))
        if type == "micro":
            tp = np.sum([np.sum((y_true == c) & (y_pred == c)) for c in classes])
            fp = np.sum([np.sum((y_true != c) & (y_pred == c)) for c in classes])
            return (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        elif type == "macro":
            precision_list = []
            for c in classes:
                tp = np.sum((y_true == c) & (y_pred == c))
                fp = np.sum((y_true != c) & (y_pred == c))
                precision_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            return np.mean(precision_list)
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
        classes = np.unique(np.concatenate((y_true, y_pred)))
        if type == "micro":
            tp = np.sum([np.sum((y_true == c) & (y_pred == c)) for c in classes])
            fn = np.sum([np.sum((y_true == c) & (y_pred != c)) for c in classes])
            return (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        elif type == "macro":
            recall_list = []
            for c in classes:
                tp = np.sum((y_true == c) & (y_pred == c))
                fn = np.sum((y_true == c) & (y_pred != c))
                recall_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            return np.mean(recall_list)
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
        classes = np.unique(np.concatenate((y_true, y_pred)))
        if type == "micro":
            precision = self.precision(y_true, y_pred, type)
            recall = self.recall(y_true, y_pred, type)
            return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        elif type == "macro":
            precision_list = []
            recall_list = []
            f1_list = []
            for c in classes:
                tp = np.sum((y_true == c) & (y_pred == c))
                fp = np.sum((y_true != c) & (y_pred == c))
                fn = np.sum((y_true == c) & (y_pred != c))
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_list.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            return np.mean(f1_list)
        else:
            raise ValueError("Invalid type")
