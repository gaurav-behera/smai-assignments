import torch
import torch.nn as nn
from tqdm import tqdm

class Binary_RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=1, dropout=0, batch_norm=False, epochs=10):
        super(Binary_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.epochs = epochs
        
        # define layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.batch_norm = nn.LayerNorm(hidden_size) if batch_norm else None
        self.fc = nn.Linear(hidden_size, output_size)
        
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
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        if self.batch_norm:
            out = self.batch_norm(out)
        out = self.fc(out)
        
        return out
    
    def fit(self, train_loader, val_loader):
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            for x, y in tqdm(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self(x.unsqueeze(-1))
                loss = criterion(outputs, y.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss/len(train_loader)}')
            self.evaluate(val_loader, 'Validation')
    
    def evaluate(self, loader, type):
        criterion = nn.L1Loss()
        self.eval()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self(x.unsqueeze(-1))
                loss += criterion(outputs, y.unsqueeze(-1))
                preds = outputs.round()
                correct += (preds == y.unsqueeze(-1)).sum().item()
                total += y.size(0)
                
        accuracy = correct/total * 100
        print(f'{type} Loss: {loss/len(loader)}')
        print(f'{type} Accuracy: {accuracy:.2f}%')
        return loss/len(loader), accuracy
    
    
class OCR_RNN(nn.Module):
    def __init__(self, cnn_layers=2, cnn_channels=[16, 16], rnn_hidden_size=16, rnn_num_layers=1, dropout=0, vocab_size=27, epochs=10):
        super(OCR_RNN, self).__init__()
        self.cnn_layers = cnn_layers
        self.cnn_channels = cnn_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.epochs = epochs

        self.make_model()
    
    def make_model(self):
        img_size = [64, 256]
        # encoder
        self.cnn = nn.Sequential()
        for i in range(self.cnn_layers):
            in_channels = 1 if i == 0 else self.cnn_channels[i-1]
            out_channels = self.cnn_channels[i]
            self.cnn.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size=5, padding='same'))
            self.cnn.add_module(f'relu{i}', nn.ReLU())
            self.cnn.add_module(f'maxpool{i}', nn.MaxPool2d(kernel_size=2, stride=2))
            img_size[0] = img_size[0] // 2
            img_size[1] = img_size[1] // 2
        
        # decoder
        self.rnn = nn.RNN(img_size[0]*self.cnn_channels[-1], self.rnn_hidden_size, self.rnn_num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.rnn_hidden_size, self.vocab_size)
        
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
        
    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        h0 = torch.zeros(self.rnn_num_layers, x.size(0), self.rnn_hidden_size, device=self.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        
        return out
    
    def fit(self, train_loader, val_loader):
        criterion = nn.CTCLoss(blank=0)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            for x, y in tqdm(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self(x)
                
                # calculate loss
                outputs = outputs.permute(1, 0, 2)
                log_softmax_out = nn.functional.log_softmax(outputs, dim=2)
                input_lengths = torch.full((x.size(0),), log_softmax_out.size(0), dtype=torch.int32)
                target_lengths = torch.full((x.size(0),), y.size(1), dtype=torch.int32)
                loss = criterion(log_softmax_out, y, input_lengths, target_lengths)   
                         
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss/len(train_loader)}')
            self.evaluate(val_loader, 'Validation')
            
    def decode_predictions(self, predictions):
        preds = predictions.softmax(2)
        preds = torch.argmax(preds, dim=2)
        # print(preds)
        pred_str = []
        prev_char = None
        for i in preds[0]:
            v = i.item()
            if v == 0:
                prev_char = 0
                continue
            elif v == prev_char:
                continue
            prev_char = v
            pred_str.append(v)
            
        return torch.tensor(pred_str)
        
    
    def evaluate(self, loader, type):
        criterion = nn.CTCLoss(blank=0)
        self.eval()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self(x)
                preds = self.decode_predictions(outputs).to(self.device)
                # calculate loss
                outputs = outputs.permute(1, 0, 2)
                log_softmax_out = nn.functional.log_softmax(outputs, dim=2)
                input_lengths = torch.full((x.size(0),), log_softmax_out.size(0), dtype=torch.int32)
                target_lengths = torch.full((x.size(0),), y.size(1), dtype=torch.int32)
                loss += criterion(log_softmax_out, y, input_lengths, target_lengths)
                # avg number of correct characters
                total += y.size(1)
                min_len = min(y.size(1), preds.size(0))
                if y.size(1) <= min_len:
                    y_ = torch.zeros_like(preds)
                    y_[:y.size(1)] = y
                else:
                    y_ = y[:,:min_len]
                y_ = y_.squeeze(0).to(self.device)
                correct += (preds == y_).sum().item()
                
        accuracy = correct/total * 100
        print(f'{type} Loss: {loss/len(loader)}')
        print(f'{type} Accuracy: {accuracy:.2f}%')
        return loss/len(loader), accuracy
