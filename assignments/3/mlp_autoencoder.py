import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
from plotly.subplots import make_subplots
from utils import setup_base_dir
import wandb
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


base_dir = setup_base_dir(levels=2)

from data.operations.preprocess import process_data
from data.operations.split_data import split_data
from models.AutoEncoders.AutoEncoders import AutoEncoder_MLP
from models.knn.knn import KNN
from models.MLP.MLP import MLP_classification_single_label

data = pd.read_csv(os.path.join(base_dir, "data", "processed", "spotify.csv"))
data = process_data(data, linear_norm=[col for col in data.columns if col != 'track_genre'])
split = split_data(data, 'track_genre')

trainX, trainY = split['trainX'].values, split['trainY']
valX, valY = split['valX'].values, split['valY']
testX, testY = split['testX'].values, split['testY']

def train_autoencoder():
    model = AutoEncoder_MLP(input_dim=trainX.shape[1], reduced_dims=6, num_hidden_layers=2, num_neurons=[8, 8], batch_size=256, epochs=50, optimizer='minibatch-gd', learning_rate=0.1)
    
    model.fit(trainX)
    
    return model

def get_reduced_dataset(model, data):
    reduced_data = model.get_latent(data)
    return reduced_data

def get_knn_model(trainX, trainY):
    knn = KNN(k=28, metric='manhattan')
    df = pd.DataFrame(trainX, columns=[f'f{i}' for i in range(trainX.shape[1])])
    knn.fit(df, trainY)
    return knn

def apply_knn(model, X, y):
    X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    y_pred = model.predict(X_df)
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred, average='micro'))
    print("Recall:", recall_score(y, y_pred, average='micro'))
    print("F1 Score:", f1_score(y, y_pred, average='micro'))
    
def get_inference_time(model, X):
    X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    start = time.time()
    model.predict(X_df)
    end = time.time()
    print("Inference time:", end-start)
    
def task_autoencoder():
    # task 4.3
    model = train_autoencoder()

    # task 4.4
    reduced_trainX = get_reduced_dataset(model, trainX)
    reduced_valX = get_reduced_dataset(model, valX)
    reduced_testX = get_reduced_dataset(model, testX)
    knn = get_knn_model(reduced_trainX, trainY)

    print("Training set metrics")
    apply_knn(knn, reduced_trainX, trainY)
    print("Validation set metrics")
    apply_knn(knn, reduced_valX, valY)
    print("Test set metrics")
    apply_knn(knn, reduced_testX, testY)
    get_inference_time(knn, reduced_valX)


# mlp classifier
def task_4_4():
    # data processing
    data = pd.read_csv(os.path.join(base_dir, "data", "processed", "spotify.csv"), index_col=0)
    data = process_data(data, hash_encode={'track_genre':10000}, linear_norm=[col for col in data.columns])
    classes = data["track_genre"].unique()
    classes.sort()
    classes = classes.tolist()

    def hot_encoding(data, classes):
        y = pd.get_dummies(data, dtype=int)
        for c in classes:
            if c not in y.columns:
                y[c] = 0

        return y.reindex(sorted(y.columns), axis=1).values
    
    split = split_data(data, 'track_genre')
    trainX, trainY = split['trainX'].values, hot_encoding(split['trainY'], classes)
    valX, valY = split['valX'].values, hot_encoding(split['valY'], classes)
    testX, testY = split['testX'].values, hot_encoding(split['testY'], classes)

    # MLP classification
    model = MLP_classification_single_label(learning_rate=0.1,activation_function='tanh', optimizer='minibatch-gd', batch_size=256, epochs=50, num_hidden_layers=3, num_neurons=[16, 32, 64], input_layer_size=trainX.shape[1], output_layer_size=trainY.shape[1])
    model.fit(trainX, trainY, True, valX, valY)
    preds = model.predict(testX)
    print("Test Set Metrics:")
    print("Accuracy: ", model.metrics.accuracy(testY, preds))
    print("Precision: ", model.metrics.precision(testY, preds))
    print("Recall: ", model.metrics.recall(testY, preds))
    print("F1 Score: ", model.metrics.f1_score(testY, preds))
    

    
    

