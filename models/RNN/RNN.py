import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import matplotlib.pyplot as plt
import librosa
from hmmlearn import hmm
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

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