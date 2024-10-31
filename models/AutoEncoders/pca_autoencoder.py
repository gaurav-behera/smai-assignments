import torch

class PCA_AutoEncoder():
    def __init__(self, input_size=784, hidden_size=12):
        self.input_size = input_size
        self.hidden_size = hidden_size
    
    def loader_to_matrix(self, loader):
        images = []
        for i, (image, _) in enumerate(loader):
            images.append(image.view(image.size(0), -1))
        images = torch.vstack(images)
        return images
    
    def fit(self, X):
        # flatten images
        self.mean = X.mean(dim=0)
        
        X = X - self.mean
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        self.principal_components = Vt[:self.hidden_size]
        
    def encode(self, X):
        X = X - self.mean
        return X @ self.principal_components.T
    
    def decode(self, encoded):
        return encoded @ self.principal_components + self.mean
    
    def forward(self, X):
        encoded = self.encode(X)
        return self.decode(encoded)
        