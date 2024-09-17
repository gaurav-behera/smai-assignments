import numpy as np
import pandas as pd
import os
from utils import setup_base_dir
import plotly.express as px
import json
import time

base_dir = setup_base_dir(levels=2)

from data.operations.split_data import split_data
from models.knn.knn import KNNInitial, KNNBest, KNN
from models.pca.pca import PCA
from data.operations.preprocess import process_data
from performance_measures.metrics import Metrics

metrics = Metrics()


def scree_plot_spotify():
    data = pd.read_csv(
        os.path.join(base_dir, "data", "external", "spotify.csv"), index_col=0
    )
    data = process_data(
        data=data.copy(),
        null_cols={
            "artists": "unknown",
            "album_name": "unknown",
            "track_name": "unknown",
        },
        boolean_encode=["explicit"],
        # hash_encode={"artists": 10000, "album_name": 10000, "track_name": 10000},
        drop_columns=["track_id", "artists", "album_name", "track_name", "artists"],
    )
    data = data.drop(columns=["track_genre"])
    # print(data)
    data = data.to_numpy()

    data = data - np.mean(data, axis=0)
    data = np.matrix(data)
    u, s, _ = np.linalg.svd(data, full_matrices=False)
    s = s**2
    normalised_s = s / np.sum(s)
    df = pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(len(s))],
            "eigenvalues": s,
            "normalised_eigenvalues": normalised_s,
        }
    )
    df = df[1:]
    fig = px.line(
        df,
        x="PC",
        y="eigenvalues",
        title="Scree Plot for Spotify",
        width=800,
        height=600,
    )
    fig.show()
    df.to_csv("scree-plot-spotify.csv")


def save_reduced_dataset(n_components):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify.csv"), index_col=0
    )
    genre = data["track_genre"]
    data = data.drop(columns=["track_genre"])
    data = data.to_numpy()

    pca = PCA(n_components)
    pca.fit(data)
    transformed_data = pca.transform(data)
    print("Check PCA:", pca.checkPCA(data))
    df = pd.DataFrame(
        transformed_data, columns=[f"PC{i+1}" for i in range(n_components)]
    )
    df["track_genre"] = genre
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(os.path.join(base_dir, "data", "processed", f"spotify-reduced.csv"))


def sample_knn_test(k, metric):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify-reduced.csv"), index_col=0
    )
    split = split_data(data, target_column="track_genre", ratio=[0.8, 0.0, 0.2])
    x_train, y_train = split["trainX"], split["trainY"]
    x_test, y_test = split["testX"], split["testY"]

    model = KNN(k=k, metric=metric)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("KNN Model with k =", k, "and metric =", metric)
    print("\tAccuracy:", metrics.accuracy(y_test, y_pred))
    print("\tMicro Precision:", metrics.precision(y_test, y_pred, type="micro"))
    print("\tMicro Recall:", metrics.recall(y_test, y_pred, type="micro"))
    print("\tMicro F1:", metrics.f1_score(y_test, y_pred, type="micro"))
    print("\tMacro Precision:", metrics.precision(y_test, y_pred, type="macro"))
    print("\tMacro Recall:", metrics.recall(y_test, y_pred, type="macro"))
    print("\tMacro F1:", metrics.f1_score(y_test, y_pred, type="macro"))


results = []


def sample_knn(k, metric, reduced=False):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify.csv"), index_col=0
    )
    if reduced:
        data = pd.read_csv(
            os.path.join(base_dir, "data", "processed", "spotify-reduced.csv"),
            index_col=0,
        )
    split = split_data(data, target_column="track_genre", ratio=[0.8, 0.1, 0.1])
    x_train, y_train = split["trainX"], split["trainY"]
    x_val, y_val = split["valX"], split["valY"]
    x_test, y_test = split["testX"], split["testY"]

    model = KNN(k=k, metric=metric)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    print(
        "KNN Model with k =",
        k,
        "and metric =",
        metric,
        f"{'on reduced data' if reduced else ''}",
    )
    print("Validation set metrics")
    print("\tAccuracy:", metrics.accuracy(y_val, y_pred))
    print("\tMicro Precision:", metrics.precision(y_val, y_pred, type="micro"))
    print("\tMicro Recall:", metrics.recall(y_val, y_pred, type="micro"))
    print("\tMicro F1:", metrics.f1_score(y_val, y_pred, type="micro"))
    print("\tMacro Precision:", metrics.precision(y_val, y_pred, type="macro"))
    print("\tMacro Recall:", metrics.recall(y_val, y_pred, type="macro"))
    print("\tMacro F1:", metrics.f1_score(y_val, y_pred, type="macro"))
    print()
    print("Test set metrics")

    start = time.time()
    y_pred = model.predict(x_test)
    end = time.time()
    print("\tAccuracy:", metrics.accuracy(y_test, y_pred))
    print("\tMicro Precision:", metrics.precision(y_test, y_pred, type="micro"))
    print("\tMicro Recall:", metrics.recall(y_test, y_pred, type="micro"))
    print("\tMicro F1:", metrics.f1_score(y_test, y_pred, type="micro"))
    print("\tMacro Precision:", metrics.precision(y_test, y_pred, type="macro"))
    print("\tMacro Recall:", metrics.recall(y_test, y_pred, type="macro"))
    print("\tMacro F1:", metrics.f1_score(y_test, y_pred, type="macro"))

    print("Inference time:", end - start)

    result = {
        "accuracy": metrics.accuracy(y_test, y_pred),
        "inference_time": end - start,
    }
    results.append(result)


def plot_results():
    results = pd.read_json("knn_results.json")
    runs = ["Original", "Reduced"]
    fig = px.bar(
        results,
        x=runs,
        y="inference_time",
        title="Inference Time vs. Dataset",
        width=800,
        height=600,
    )
    fig.update_layout(yaxis_title="Inference Time", xaxis_title="Dataset")
    fig.show()


def task_9_1():
    scree_plot_spotify()
    save_reduced_dataset(6)
    sample_knn_test(k=28, metric="manhattan")


def task_9_2():
    sample_knn(k=28, metric="manhattan")
    sample_knn(k=28, metric="manhattan", reduced=True)
    with open("knn_results.json", "w") as f:
        json.dump(results, f)
    plot_results()
