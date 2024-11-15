import numpy as np
from hmmlearn import hmm

class HMM:
    """
    Hidden Markov Model (HMM) class
    """
    def __init__(self, num_digits=10):
        self.models = {}
        self.num_digits = num_digits
    
    def fit(self, train_data):
        for i in range(self.num_digits):
            X = np.concatenate(train_data[i])
            model = hmm.GaussianHMM(n_components=5, covariance_type='diag')
            model.fit(X)
            self.models[i] = model
            
    def predict(self, test_data):
        predictions = []
        for mfccs in test_data:
            scores = [model.score(mfccs) for model in self.models.values()]
            predictions.append(np.argmax(scores))
        return predictions
        