import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

class CNN(nn.Module):
    """
    Class for a Convolutional Neural Network
    """
    def __init__(
        self,
        learning_rate=0.01,
        dropout_rate=0.5,
        kernel_sizes=[3,3,3],
        channel_sizes=[32, 64, 128],
        activation_functions=[nn.ReLU(), nn.ReLU(), nn.ReLU()],
        input_channels=1,
        input_size=128, # assuming square images
        output_size=1,
        epochs=10,
        optimizer=optim.Adam,
        task="classification",
        log_wandb=False
    ):
        super(CNN, self).__init__()
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.input_channels = input_channels
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.task = task
        self.log_wandb = log_wandb

        # check the hidden layers
        assert len(kernel_sizes) == len(channel_sizes) == len(activation_functions)
        self.kernel_sizes = kernel_sizes
        self.channel_sizes = channel_sizes
        self.activation_functions = activation_functions
        self.num_layers = len(kernel_sizes)
        
        # determine loss function based on task
        match task:
            case "classification":
                if self.output_size == 1:
                    self.criterion = nn.BCELoss() # Binary classification
                else:
                    self.criterion = nn.CrossEntropyLoss()  # Multi-class classification
            case "regression":
                self.criterion = nn.MSELoss()
            case "multilabel-classification":
                self.criterion = nn.CrossEntropyLoss()
            case _:
                raise ValueError("Invalid Task")

        self.make_model()

        # determine device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")
        self.device = device
        self.to(device)
        
        # adding the optimizer
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)

    def make_model(self):
        """Construct the CNN architecture"""
        self.model = nn.Sequential()
        feat_size = self.input_size
        
        for i in range(self.num_layers):
            # convolutional layer
            self.model.add_module(
                    f"conv{i}",
                    nn.Conv2d(
                        in_channels=self.input_channels if i==0 else self.channel_sizes[i-1],
                        out_channels=self.channel_sizes[i],
                        kernel_size=self.kernel_sizes[i], 
                    )
                )
            # update the feature size
            feat_size = feat_size - self.kernel_sizes[i] + 1
            # activation function
            self.model.add_module(f"activation{i}", self.activation_functions[i])
            # max pooling layer
            self.model.add_module(f"pooling{i}", nn.MaxPool2d(kernel_size=2, stride=2))
            # update feature size
            feat_size = feat_size // 2
            
        
        # add dropout layer
        self.model.add_module("dropout", nn.Dropout(self.dropout_rate))
        # add flatten, fully connected, and output layers
        self.model.add_module("flatten", nn.Flatten())
        self.model.add_module("fc1", nn.Linear(self.channel_sizes[-1]*feat_size*feat_size, 32))
        self.model.add_module("activation_linear", nn.ReLU())
        self.model.add_module("fc2", nn.Linear(32, self.output_size))
        
            
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    

    def fit(self, train_loader, val_loader=None):
        """Train the model"""

        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss}")
            if val_loader is not None:
                self.evaluate(train_loader, type='Train', print_output=False)
                self.evaluate(val_loader, type='Validation')

    def evaluate(self, loader, type, print_output=True, return_output=False):
        """Evaluate the model"""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        correct_ham = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)
                # outputs = torch.clamp(outputs, min=-1e6, max=1e6)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                total += labels.size(0)

                match self.task:
                    case "classification":
                        _, predicted = torch.max(outputs.data, 1)
                        _, true_labels = torch.max(labels.data, 1)
                        correct += (predicted == true_labels).sum().item()
                    case "regression":
                        correct += (torch.round(outputs) == labels).sum().item()
                    case "multilabel-classification":
                        # Split outputs and labels, get predictions, and reformat as needed
                        outputs1, outputs2, outputs3 = torch.split(outputs, 11, dim=1)
                        predicted = torch.cat([torch.max(out, 1)[1].unsqueeze(1) for out in (outputs1, outputs2, outputs3)], dim=1)

                        labels1, labels2, labels3 = torch.split(labels, 11, dim=1)
                        labels = torch.cat([torch.argmax(lbl, dim=1).unsqueeze(1) for lbl in (labels1, labels2, labels3)], dim=1)

                        # print(predicted.shape)
                        correct += (predicted == labels).all(dim=1).sum().item()
                        correct_ham += (predicted == labels).sum().item()

        avg_loss = total_loss / len(loader)        
        accuracy = 100 * correct / total
        hamming_acc = 100 * (correct_ham / total) / 3
        
        if return_output:
            return avg_loss, accuracy, hamming_acc
        
        if print_output:
            print(f"\t{type} Loss: {avg_loss:.4f}", f"{type} Accuracy: {accuracy:.2f}%" if accuracy !=0 else '', f"Hamming Accuracy: {hamming_acc:.2f}%" if hamming_acc !=0 else '')

        if self.log_wandb:
            match self.task:
                case "classification":
                    wandb.log({f"{type} Loss": avg_loss, f"{type} Accuracy": accuracy})
                case "regression":
                    wandb.log({f"{type} Loss": avg_loss, f"{type} Accuracy": accuracy})
                case "multilabel-classification":
                    wandb.log({f"{type} Loss": avg_loss, f"{type} Accuracy": accuracy, f"{type} Hamming Accuracy": hamming_acc})
        
    def get_feature_maps(self, x):
        """Get the feature maps for the input x"""
        feature_maps = []
        x = x.to(self.device)
        for i, layer in enumerate(self.model):
            x = layer(x)
            if "Pool" in layer._get_name():
                feature_maps.append(x.sum(dim=1, keepdim=True))
        return feature_maps
