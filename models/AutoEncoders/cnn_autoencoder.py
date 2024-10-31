import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class CNNAutoencoder(nn.Module):

    def __init__(
        self,
        learning_rate=0.01,
        kernel_sizes=[3, 5],
        channel_sizes=[16, 32],
        activation_functions=[nn.Sigmoid(), nn.Sigmoid()],
        input_channels=1,
        input_size=28,
        epochs=10,
        optimizer=optim.Adam,
        log_wandb=False,
    ):
        super(CNNAutoencoder, self).__init__()
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.input_size = input_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.log_wandb = log_wandb

        # check the hidden layers
        assert len(kernel_sizes) == len(channel_sizes) == len(activation_functions)
        self.kernel_sizes = kernel_sizes
        self.channel_sizes = channel_sizes
        self.activation_functions = activation_functions
        self.num_layers = len(kernel_sizes)

        self.criterion = nn.MSELoss()

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

    def make_model(self):
        """Construct the encoder and decoder architecture"""
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        latent_dims = [self.input_size]

        # encoder
        for i in range(self.num_layers):
            self.encoder.add_module(
                f"enc_conv{i}",
                nn.Conv2d(
                    in_channels=(
                        self.input_channels if i == 0 else self.channel_sizes[i - 1]
                    ),
                    out_channels=self.channel_sizes[i],
                    kernel_size=self.kernel_sizes[i],
                    padding="same",
                ),
            )
            self.encoder.add_module(
                f"enc_pooling{i}", nn.MaxPool2d(kernel_size=2, stride=2)
            )
            latent_dims.append(
                latent_dims[-1] // 2
            )
            self.encoder.add_module(f"enc_activation{i}", self.activation_functions[i])

        # decoder
        for i in range(self.num_layers - 1, -1, -1):
            self.decoder.add_module(
                f"dec_conv{i}",
                nn.Conv2d(
                    in_channels=self.channel_sizes[i],
                    out_channels=(
                        self.input_channels if i == 0 else self.channel_sizes[i - 1]
                    ),
                    kernel_size=self.kernel_sizes[i],
                    padding="same",
                ),
            )
            self.decoder.add_module(f"dec_upsampling{i}", nn.Upsample(size=latent_dims[i]))
            if i > 0:
                self.decoder.add_module(
                    f"dec_activation{i}", self.activation_functions[i]
                )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def fit(self, train_loader, val_loader):
        """Train the model"""
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            for i, (images, _) in enumerate(train_loader):
                images = images.to(self.device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, images)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            print(f"Epoch: {epoch+1}/{self.epochs}, Train Loss: {train_loss}")
            if val_loader is not None:
                val_loss = self.evaluate(
                    val_loader, type="Validation", return_output=True
                )

                if self.log_wandb:
                    wandb.log({"train_loss": train_loss, "val_loss": val_loss})

    def evaluate(self, loader, type, print_output=True, return_output=False):
        """Evaluate the model"""
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                images = images.to(self.device)
                outputs = self(images)
                loss = self.criterion(outputs, images)
                total_loss += loss.item()
            total_loss /= len(loader)

        if print_output:
            print(f"\t{type} Loss: {total_loss}")
            
        if return_output:
            return total_loss


    def predict(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            return self(x)
        
    def get_latent_representation(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            return self.encode(x)