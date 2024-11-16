import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import librosa
import random
from utils import setup_base_dir


base_dir = setup_base_dir(levels=2)
from models.HMM.HMM import HMM

def get_mfccs(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.T

# plot mfccs heatmap
def task_3_2():
    # george
    fig = make_subplots(rows=5, cols=2, subplot_titles=[f'Digit {i}' for i in range(10)])
    for i in range(5):
        for j in range(2):
            mfccs = get_mfccs(f"../../data/external/recordings/{i*2+j}_george_0.wav")
            fig.add_trace(
                go.Heatmap(z=mfccs, zmax=300, zmin=-700, showscale=(i == 0 and j == 0)),
                row=i + 1,
                col=j + 1,
            )
            fig.update_xaxes(title_text='MFCC', row=i + 1, col=j + 1)
            fig.update_yaxes(title_text='Time', row=i + 1, col=j + 1)
            
    fig.update_layout(width=800, height=1200, title="MFCC for all digits (george_0)")
    fig.show()
    
    for i in range(5):
        for j in range(2):
            mfccs = get_mfccs(f"../../data/external/recordings/{i*2+j}_jackson_0.wav")
            fig.add_trace(
                go.Heatmap(z=mfccs, zmax=300, zmin=-700, showscale=(i == 0 and j == 0)),
                row=i + 1,
                col=j + 1,
            )
            fig.update_xaxes(title_text='MFCC', row=i + 1, col=j + 1)
            fig.update_yaxes(title_text='Time', row=i + 1, col=j + 1)
            
    fig.update_layout(width=800, height=1200, title="MFCC for all digits (jackson_0)")
    fig.show()
    
def task_3_4():
    # store all the data
    all_data = []
    for file in os.listdir("../../data/external/recordings"):
        if file.endswith(".wav"):
            label = int(file.split("_")[0])
            mfccs = get_mfccs(f"../../data/external/recordings/{file}")
            all_data.append((label,mfccs))
    
    # create train and test data as list of labels and mfccs
    train_data = {}
    test_data = {}
    self_data = {}

    for i in range(10):
        train_data[i] = []
        test_data[i] = []
        self_data[i] = []

    random.shuffle(all_data)
    for i, (label, mfccs) in enumerate(all_data):
        if i < 0.8*len(all_data):
            train_data[label].append(mfccs)
        else:
            test_data[label].append(mfccs)
            
    # load self data
    for file in os.listdir("../../data/external/self_recordings"):
        if file.endswith(".wav"):
            label = int(file.split("_")[0])
            mfccs = get_mfccs(f"../../data/external/self_recordings/{file}")
            self_data[label].append(mfccs)
            
    # train HMM models
    h = HMM()
    h.fit(train_data)
    
    def evaluate_hmm(data):
        mfccs_test = []
        labels_test = []
        for i in range(10):
            mfccs_test.extend(data[i])
            labels_test.extend([i]*len(data[i]))
            
        predictions = h.predict(mfccs_test)
        accuracy = np.mean(np.array(predictions) == np.array(labels_test))
        return accuracy * 100
    
    train_accuracy = evaluate_hmm(train_data)
    test_accuracy = evaluate_hmm(test_data)
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # self data
    self_accuracy = evaluate_hmm(self_data)
    print(f"Self data accuracy: {self_accuracy:.2f}%")