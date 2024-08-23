import numpy as np
import pandas as pd
import os
from utils import setup_base_dir
import wandb
import plotly.express as px
import json
import time

base_dir = setup_base_dir(levels=2)

from data.operations.split_data import split_data
from models.knn.knn import KNNInitial, KNNBest, KNN
from sklearn.neighbors import KNeighborsClassifier
from performance_measures.metrics import Metrics

# read the data
data = pd.read_csv(
    os.path.join(base_dir, "data", "processed", "spotify.csv"), index_col=0
)

# split the data
split = split_data(data, target_column="track_genre", ratio=[0.8, 0.1, 0.1])
x_train, y_train = split["trainX"], split["trainY"]
x_val, y_val = split["valX"], split["valY"]
x_test, y_test = split["testX"], split["testY"]

metrics = Metrics()

# mod is used to indicate that it is the best training set model


def report_metrics(model, k, metric):
    model.fit(x_train, y_train)
    print(f"K: {k}, Metric: {metric}")

    print("Valdiation set metrics")
    y_pred = model.predict(x_val)
    print(f"\tAccuracy: {metrics.accuracy(y_val, y_pred)}")
    print(f"\tMicro Precision: {metrics.precision(y_val, y_pred, type='micro')}")
    print(f"\tMicro Recall: {metrics.recall(y_val, y_pred, type='micro')}")
    print(f"\tMicro F1: {metrics.f1_score(y_val, y_pred, type='micro')}")
    print(f"\tMacro Precision: {metrics.precision(y_val, y_pred, type='macro')}")
    print(f"\tMacro Recall: {metrics.recall(y_val, y_pred, type='macro')}")
    print(f"\tMacro F1: {metrics.f1_score(y_val, y_pred, type='macro')}")

    print("Test set metrics")
    y_pred = model.predict(x_test)
    print(f"\tAccuracy: {metrics.accuracy(y_test, y_pred)}")
    print(f"\tMicro Precision: {metrics.precision(y_test, y_pred, type='micro')}")
    print(f"\tMicro Recall: {metrics.recall(y_test, y_pred, type='micro')}")
    print(f"\tMicro F1: {metrics.f1_score(y_test, y_pred, type='micro')}")
    print(f"\tMacro Precision: {metrics.precision(y_test, y_pred, type='macro')}")
    print(f"\tMacro Recall: {metrics.recall(y_test, y_pred, type='macro')}")
    print(f"\tMacro F1: {metrics.f1_score(y_test, y_pred, type='macro')}")
    print("-----------------------------------")


def wandb_tune(mod=None):
    # wandb
    sweep_config = {
        "method": "grid",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "k": {"values": list(range(1, 31))},
            "metric": {"values": ["euclidean", "manhattan", "cosine"]},
        },
    }

    tuning_results = []

    def tune_model():
        with wandb.init() as run:
            config = wandb.config
            run_name = f"k-{config.k}_metric-{config.metric}"
            wandb.run.name = run_name
            # train the model
            model = KNN(k=config.k, metric=config.metric)
            model.fit(x_train, y_train)

            # evaluate the model
            y_pred = model.predict(x_val)
            accuracy = metrics.accuracy(y_val, y_pred)

            # log the accuracy
            wandb.log({"accuracy": accuracy})

            tuning_results.append(
                {"k": config.k, "metric": config.metric, "accuracy": accuracy}
            )

    sweep_id = wandb.sweep(
        sweep_config,
        project="spotify-knn-hyperparameter-tuning" + ("-best" if mod else ""),
    )
    wandb.agent(sweep_id, function=tune_model, count=90)
    print("Hyperparameter tuning completed")
    # save dict as json in a file
    if mod:
        with open("tuning_results_best.json", "w") as f:
            json.dump(tuning_results, f)
    else:
        with open("tuning_results.json", "w") as f:
            json.dump(tuning_results, f)


def print_top_results(mod=None):
    # print ordered list of best accuracy for each k and metric
    if mod:
        with open("tuning_results_best.json", "r") as f:
            tuning_results = json.load(f)
    else:
        with open("tuning_results.json", "r") as f:
            tuning_results = json.load(f)
    tuning_results = sorted(tuning_results, key=lambda x: x["accuracy"], reverse=True)
    print("Top 10 best pairs of k and metric")
    for i in range(10):
        print(
            f"K: {tuning_results[i]['k']}, Metric: {tuning_results[i]['metric']}, Accuracy: {tuning_results[i]['accuracy']}"
        )


def plot_accuracy_graph(mod=None):
    # graph of k vs accuracy for each metric
    if mod:
        with open("tuning_results_best.json", "r") as f:
            tuning_results = json.load(f)
    else:
        with open("tuning_results.json", "r") as f:
            tuning_results = json.load(f)
    tuning_results = sorted(tuning_results, key=lambda x: x["k"], reverse=True)
    df = pd.DataFrame(tuning_results)
    fig = px.line(df, x="k", y="accuracy", color="metric")
    # fig.update_xaxes(range=[0, 31])
    # fig.update_yaxes(range=[0, 1])
    fig.update_layout(title="K vs Accuracy for each metric", height=600, width=800)
    fig.show()


def best_features():
    # to plot the details for the best processed features
    global data, x_train, y_train, x_val, y_val, x_test, y_test
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify_best.csv"), index_col=0
    )

    # split the data
    split = split_data(data, target_column="track_genre", ratio=[0.8, 0.1, 0.1])
    x_train, y_train = split["trainX"], split["trainY"]
    x_val, y_val = split["valX"], split["valY"]
    x_test, y_test = split["testX"], split["testY"]

    wandb_tune(mod=True)
    print_top_results(mod=True)
    plot_accuracy_graph(mod=True)


def inferencetime_vs_model():
    # plotting inference time vs model
    k, metric = 7, "euclidean"
    models = [
        KNNInitial(k=k, metric=metric),
        KNNBest(k=k, metric=metric),
        KNN(k=k, metric=metric),
        KNeighborsClassifier(n_neighbors=k, metric=metric),
    ]
    inference_times = []
    for model in models:
        model.fit(x_train, y_train)
        start = time.time()
        model.predict(x_val)
        end = time.time()
        inference_times.append(end - start)

    df = pd.DataFrame(
        {
            "Model": ["KNNInitial", "KNNBest", "KNN", "KNeighborsClassifier"],
            "Inference Time": inference_times,
        }
    )
    print(df)
    fig = px.bar(df, x="Model", y="Inference Time")
    fig.update_layout(title="Model vs Inference Time", height=600, width=800)
    fig.show()


def inferecetime_vs_trainsize():
    # plotting the inference time vs training size for the 4 models
    k, metric = 7, "euclidean"
    sizes = [10, 100, 500, 1000, 5000, 10000, 50000, 91200]
    models = [
        KNNInitial(k=k, metric=metric),
        KNNBest(k=k, metric=metric),
        KNN(k=k, metric=metric),
        KNeighborsClassifier(n_neighbors=k, metric=metric),
    ]
    inference_times = []
    for model in models:
        for size in sizes:
            model.fit(x_train[:size], y_train[:size])
            start = time.time()
            model.predict(x_val)
            end = time.time()
            inference_times.append(
                {
                    "Model": model.__class__.__name__,
                    "Training Size": size,
                    "Inference Time": end - start,
                }
            )

    df = pd.DataFrame(inference_times)
    print(df)
    fig = px.line(df, x="Training Size", y="Inference Time", color="Model")
    fig.update_layout(title="Inference Time vs Training Size", height=600, width=800)
    fig.show()


def evaluate_spotify2():
    global x_train, y_train, x_val, y_val, x_test, y_test
    train_data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify-2", "train.csv"),
        index_col=0,
    )
    val_data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify-2", "validate.csv"),
        index_col=0,
    )
    test_data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify-2", "test.csv"),
        index_col=0,
    )
    x_train, y_train = (
        train_data.drop(columns=["track_genre"], inplace=False),
        train_data["track_genre"],
    )
    x_val, y_val = (
        val_data.drop(columns=["track_genre"], inplace=False),
        val_data["track_genre"],
    )
    x_test, y_test = (
        test_data.drop(columns=["track_genre"], inplace=False),
        test_data["track_genre"],
    )
    k, metric = 28, "manhattan"
    report_metrics(KNN(k, metric), k, metric)


def evaluate_spotify2_best():

    global x_train, y_train, x_val, y_val, x_test, y_test
    train_data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify-2", "train_best.csv"),
        index_col=0,
    )
    val_data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify-2", "validate_best.csv"),
        index_col=0,
    )
    test_data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "spotify-2", "test_best.csv"),
        index_col=0,
    )
    x_train, y_train = (
        train_data.drop(columns=["track_genre"], inplace=False),
        train_data["track_genre"],
    )
    x_val, y_val = (
        val_data.drop(columns=["track_genre"], inplace=False),
        val_data["track_genre"],
    )
    x_test, y_test = (
        test_data.drop(columns=["track_genre"], inplace=False),
        test_data["track_genre"],
    )
    k, metric = 1, "manhattan"
    report_metrics(KNN(k, metric), k, metric)


# report_metrics(KNN(28, "manhattan"), 7, "manhattan")
# wandb_tune()
# print_top_results()
# plot_accuracy_graph()
# best_features()
# inferencetime_vs_model()
# inferecetime_vs_trainsize()
# evaluate_spotify2()
# evaluate_spotify2_best()
