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

base_dir = setup_base_dir(levels=2)

from data.operations.preprocess import process_data
from data.operations.split_data import split_data
from models.MLP.MLP import (
    MLP_classification_single_label,
    MLP_classification_multi_label,
)


# Dataset Ananlysis and Preprocessing
def task_2_1():
    data = pd.read_csv(os.path.join(base_dir, "data", "external", "WineQT.csv"))
    data = process_data(data, drop_columns=["Id"])

    # data statistics
    data_stats = data.describe().loc[["mean", "std", "min", "max"]].T
    print(data_stats)

    # make plots for each feature in the dataset using subplots
    fig = make_subplots(rows=4, cols=3, subplot_titles=data.columns)
    for i, col in enumerate(data.columns):
        fig.add_trace(
            go.Histogram(x=data[col], name=col), row=i // 3 + 1, col=i % 3 + 1
        )
        fig.update_xaxes(title_text="Value", row=i // 3 + 1, col=i % 3 + 1)
        fig.update_yaxes(title_text="Count", row=i // 3 + 1, col=i % 3 + 1)
    fig.update_layout(showlegend=False, title_text="Feature Distributions")
    fig.show()

    # handle missing or inconsistent data
    data = process_data(data, null_cols={col: data[col].mean() for col in data.columns})
    # normalize the data
    data = process_data(data, linear_norm=[col for col in data.columns])

    data.to_csv(os.path.join(base_dir, "data", "processed", "WineQT.csv"), index=False)


# read dataset
def read_dataset(name):
    match name:
        case "wineqt":
            data = pd.read_csv(
                os.path.join(base_dir, "data", "processed", "WineQT.csv")
            )
            classes = data["quality"].unique()
            classes.sort()
            classes = classes.tolist()

            def hot_encoding(data, classes):
                y = pd.get_dummies(data, dtype=int)
                for c in classes:
                    if c not in y.columns:
                        y[c] = 0

                return y.reindex(sorted(y.columns), axis=1).values

            split = split_data(data, target_column="quality")
            trainX, trainY = split["trainX"].values, hot_encoding(
                split["trainY"], classes
            )
            valX, valY = split["valX"].values, hot_encoding(split["valY"], classes)
            testX, testY = split["testX"].values, hot_encoding(split["testY"], classes)
            return trainX, trainY, valX, valY, testX, testY
        case "advertisement":
            data = pd.read_csv(
                os.path.join(base_dir, "data", "processed", "advertisement.csv")
            )
            all_labels = [word for label in data["labels"] for word in label.split(" ")]
            all_labels = list(set(all_labels))
            all_labels.sort()
            split = split_data(data, target_column="labels")
            trainX, trainY = (
                split["trainX"].values,
                split["trainY"]
                .str.get_dummies(sep=" ")
                .reindex(columns=all_labels, fill_value=0)
                .values,
            )
            valX, valY = (
                split["valX"].values,
                split["valY"]
                .str.get_dummies(sep=" ")
                .reindex(columns=all_labels, fill_value=0)
                .values,
            )
            testX, testY = (
                split["testX"].values,
                split["testY"]
                .str.get_dummies(sep=" ")
                .reindex(columns=all_labels, fill_value=0)
                .values,
            )
            return trainX, trainY, valX, valY, testX, testY

        case _:
            return None


# Hyperparameter Tuning for Single Label Classification
def task_2_3():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"values": [0.01, 0.1]},
            "activation_function": {"values": ["relu", "sigmoid", "tanh"]},
            "optimizer": {"values": ["sgd", "batch-gd", "minibatch-gd"]},
            "num_neurons": {
                "values": [
                    [32],
                    [32, 64],
                    [32, 64, 32],
                ]
            },
            "batch_size": {"values": [16, 256]},
            "epochs": {"values": [10, 100]},
        },
    }

    # sweep
    sweep_id = wandb.sweep(sweep_config, project="mlp-single-label-classification")

    trainX, trainY, valX, valY, testX, testY = read_dataset("wineqt")
    run_id = 0

    def train():
        nonlocal run_id
        wandb.init(project="mlp-single-label-classification", name=f"run-{run_id}")
        run_id += 1
        config = wandb.config
        model = MLP_classification_single_label(
            learning_rate=config.learning_rate,
            activation_function=config.activation_function,
            optimizer=config.optimizer,
            num_hidden_layers=len(config.num_neurons),
            num_neurons=config.num_neurons,
            batch_size=config.batch_size,
            epochs=config.epochs,
            input_layer_size=trainX.shape[1],
            output_layer_size=trainY.shape[1],
            log_wandb=True,
        )

        model.fit(trainX, trainY, True, valX, valY)
        preds = model.predict(valX)
        wandb.log(
            {
                "accuracy": model.metrics.accuracy(valY, preds),
                "f1": model.metrics.f1_score(valY, preds),
                "precision": model.metrics.precision(valY, preds),
                "recall": model.metrics.recall(valY, preds),
            }
        )

        wandb.finish()

    wandb.agent(sweep_id, train)


# best model results and saving the model
def task_2_4():
    trainX, trainY, valX, valY, testX, testY = read_dataset("wineqt")
    best_params = {
        "activation_function": "tanh",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.01,
        "num_neurons": [32],
        "num_hidden_layers": 1,
        "optimizer": "minibatch-gd",
    }
    model = MLP_classification_single_label(
        learning_rate=best_params["learning_rate"],
        activation_function=best_params["activation_function"],
        optimizer=best_params["optimizer"],
        num_hidden_layers=best_params["num_hidden_layers"],
        num_neurons=best_params["num_neurons"],
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        input_layer_size=trainX.shape[1],
        output_layer_size=trainY.shape[1],
    )
    model.fit(trainX, trainY, early_stop=True, X_val=valX, y_val=valY)
    with open("single_classification_model.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "params": best_params,
                    "weights": [w.tolist() for w in model.weights],
                    "biases": [b.tolist() for b in model.biases],
                }
            )
        )
    # print results on test data
    preds = model.predict(testX)
    print("Accuracy: ", model.metrics.accuracy(testY, preds))
    print("F1 Score: ", model.metrics.f1_score(testY, preds))
    print("Precision: ", model.metrics.precision(testY, preds))
    print("Recall: ", model.metrics.recall(testY, preds))


# ananlysing hyperparameters effects
def task_2_5():
    trainX, trainY, valX, valY, testX, testY = read_dataset("wineqt")

    # effect of non linearlity in activation function
    def effect_of_activation_function():
        results_df = pd.DataFrame()
        for activation_function in ["relu", "sigmoid", "tanh", "linear"]:
            best_params = {
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 0.01,
                "num_neurons": [32],
                "num_hidden_layers": 1,
                "optimizer": "minibatch-gd",
            }
            model = MLP_classification_single_label(
                learning_rate=best_params["learning_rate"],
                activation_function=activation_function,
                optimizer=best_params["optimizer"],
                num_hidden_layers=best_params["num_hidden_layers"],
                num_neurons=best_params["num_neurons"],
                batch_size=best_params["batch_size"],
                epochs=best_params["epochs"],
                input_layer_size=trainX.shape[1],
                output_layer_size=trainY.shape[1],
                log_local=True,
            )
            model.fit(trainX, trainY, early_stop=True, X_val=valX, y_val=valY)
            logs = model.logs
            logs_df = pd.DataFrame(logs)
            logs_df["epoch"] = logs_df.index
            logs_df["activation_function"] = activation_function
            results_df = pd.concat([results_df, logs_df])
        fig = px.line(
            results_df,
            x="epoch",
            y="loss",
            color="activation_function",
            title="Effect of Activation Function on Loss",
        )
        fig.update_layout(
            xaxis_title="Epoch", yaxis_title="Loss", width=800, height=600
        )
        fig.show()

    # effect of learning rate
    def effect_of_learning_rate():
        results_df = pd.DataFrame()
        for learning_rate in [0.001, 0.05, 0.01, 0.1]:
            best_params = {
                "activation_function": "tanh",
                "epochs": 10,
                "batch_size": 16,
                "num_neurons": [32],
                "num_hidden_layers": 1,
                "optimizer": "minibatch-gd",
            }
            model = MLP_classification_single_label(
                learning_rate=learning_rate,
                activation_function=best_params["activation_function"],
                optimizer=best_params["optimizer"],
                num_hidden_layers=best_params["num_hidden_layers"],
                num_neurons=best_params["num_neurons"],
                batch_size=best_params["batch_size"],
                epochs=best_params["epochs"],
                input_layer_size=trainX.shape[1],
                output_layer_size=trainY.shape[1],
                log_local=True,
            )
            model.fit(trainX, trainY, early_stop=True, X_val=valX, y_val=valY)
            logs = model.logs
            logs_df = pd.DataFrame(logs)
            logs_df["epoch"] = logs_df.index
            logs_df["learning_rate"] = learning_rate
            results_df = pd.concat([results_df, logs_df])
        fig = px.line(
            results_df,
            x="epoch",
            y="loss",
            color="learning_rate",
            title="Effect of Learning Rate on Loss",
        )
        fig.update_layout(
            xaxis_title="Epoch", yaxis_title="Loss", width=800, height=600
        )
        fig.show()

    # effect of batch size
    def effect_of_batch_size():
        results_df = pd.DataFrame()
        for batch_size in [32, 64, 128, 512]:
            best_params = {
                "activation_function": "tanh",
                "epochs": 10,
                "learning_rate": 0.01,
                "num_neurons": [32],
                "num_hidden_layers": 1,
                "optimizer": "minibatch-gd",
            }
            model = MLP_classification_single_label(
                learning_rate=0.01,
                activation_function=best_params["activation_function"],
                optimizer=best_params["optimizer"],
                num_hidden_layers=best_params["num_hidden_layers"],
                num_neurons=best_params["num_neurons"],
                epochs=best_params["epochs"],
                batch_size=batch_size,
                input_layer_size=trainX.shape[1],
                output_layer_size=trainY.shape[1],
                log_local=True,
            )
            model.fit(trainX, trainY, early_stop=True, X_val=valX, y_val=valY)
            logs = model.logs
            logs_df = pd.DataFrame(logs)
            logs_df["epoch"] = logs_df.index
            logs_df["batch_size"] = batch_size
            results_df = pd.concat([results_df, logs_df])
        fig = px.line(
            results_df,
            x="epoch",
            y="loss",
            color="batch_size",
            title="Effect of Batch Size on Loss",
        )
        fig.update_layout(
            xaxis_title="Epoch", yaxis_title="Loss", width=800, height=600
        )
        fig.show()

    effect_of_activation_function()
    effect_of_learning_rate()
    effect_of_batch_size()


# process advertisement.csv
def process_advertisement():
    data = pd.read_csv(os.path.join(base_dir, "data", "external", "advertisement.csv"))
    df = process_data(
        data,
        label_encode=["gender", "education", "city", "occupation", "most bought item"],
        boolean_encode=["married"],
    )
    all_labels = [word for label in df["labels"] for word in label.split(" ")]
    all_labels = list(set(all_labels))
    all_labels.sort()
    label_dict = {label: i / len(all_labels) for i, label in enumerate(all_labels)}
    df["labels"] = df["labels"].apply(
        lambda x: " ".join([str(label_dict[label]) for label in x.split(" ")])
    )
    data = process_data(
        df, linear_norm=[col for col in df.columns if col not in ["labels"]]
    )
    data.to_csv(
        os.path.join(base_dir, "data", "processed", "advertisement.csv"), index=False
    )


# best model results and saving the model
def multi_results():
    trainX, trainY, valX, valY, testX, testY = read_dataset("advertisement")
    best_params = {
        "activation_function": "relu",
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.01,
        "num_neurons": [32, 64],
        "optimizer": "batch-gd",
        "num_hidden_layers": 2,
    }
    model = MLP_classification_multi_label(
        learning_rate=best_params["learning_rate"],
        activation_function=best_params["activation_function"],
        optimizer=best_params["optimizer"],
        num_hidden_layers=best_params["num_hidden_layers"],
        num_neurons=best_params["num_neurons"],
        epochs=best_params["epochs"],
        input_layer_size=trainX.shape[1],
        output_layer_size=trainY.shape[1],
    )
    model.fit(trainX, trainY, early_stop=True, X_val=valX, y_val=valY)
    with open("multi_classification_model.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "params": best_params,
                    "weights": [w.tolist() for w in model.weights],
                    "biases": [b.tolist() for b in model.biases],
                }
            )
        )
    # print results on test data
    preds = model.predict(testX)
    print("Accuracy: ", model.metrics.multi_label_accuracy(testY, preds))
    print("F1 Score: ", model.metrics.f1_score(testY, preds))
    print("Precision: ", model.metrics.precision(testY, preds))
    print("Recall: ", model.metrics.recall(testY, preds))
    print("Hamming Loss: ", model.metrics.hamming_loss(testY, preds))


# hyperparameter tuning for multi label classification
def task_2_6():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"values": [0.01, 0.1]},
            "activation_function": {"values": ["relu", "sigmoid", "tanh"]},
            "optimizer": {"values": ["sgd", "batch-gd", "minibatch-gd"]},
            "num_neurons": {
                "values": [
                    [32],
                    [32, 64],
                    [32, 64, 32],
                ]
            },
            "batch_size": {"values": [16, 256]},
            "epochs": {"values": [10, 100]},
        },
    }

    # sweep
    sweep_id = wandb.sweep(sweep_config, project="mlp-multi-label-classification")

    trainX, trainY, valX, valY, testX, testY = read_dataset("advertisement")
    run_id = 0

    def train():
        nonlocal run_id
        wandb.init(project="mlp-multi-label-classification", name=f"run-{run_id}")
        run_id += 1
        config = wandb.config
        model = MLP_classification_multi_label(
            learning_rate=config.learning_rate,
            activation_function=config.activation_function,
            optimizer=config.optimizer,
            num_hidden_layers=len(config.num_neurons),
            num_neurons=config.num_neurons,
            batch_size=config.batch_size,
            epochs=config.epochs,
            input_layer_size=trainX.shape[1],
            output_layer_size=trainY.shape[1],
            log_wandb=True,
        )

        model.fit(trainX, trainY, True, valX, valY)
        preds = model.predict(valX)
        wandb.log(
            {
                "accuracy": model.metrics.multi_label_accuracy(valY, preds),
                "f1": model.metrics.f1_score(valY, preds),
                "precision": model.metrics.precision(valY, preds),
                "recall": model.metrics.recall(valY, preds),
                "hamming_loss": model.metrics.hamming_loss(valY, preds),
            }
        )

        wandb.finish()

    wandb.agent(sweep_id, train)
    
    multi_results()

def task_2_7():
    # Single label classification
    trainX, trainY, valX, valY, testX, testY = read_dataset("wineqt")
    best_params = {
        "activation_function": "tanh",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.01,
        "num_neurons": [32],
        "num_hidden_layers": 1,
        "optimizer": "minibatch-gd",
    }
    model = MLP_classification_single_label(
        learning_rate=best_params["learning_rate"],
        activation_function=best_params["activation_function"],
        optimizer=best_params["optimizer"],
        num_hidden_layers=best_params["num_hidden_layers"],
        num_neurons=best_params["num_neurons"],
        epochs=best_params["epochs"],
        input_layer_size=trainX.shape[1],
        output_layer_size=trainY.shape[1],
    )
    model.fit(trainX, trainY, early_stop=True, X_val=valX, y_val=valY)
    preds = model.predict(testX)

    # distribution of true and predicted classes
    fig = make_subplots(rows=1, cols=2, subplot_titles=["True Classes", "Predicted Classes"])
    fig.add_trace(go.Histogram(x=np.argmax(testY, axis=1), name="True Classes"), row=1, col=1)
    fig.add_trace(go.Histogram(x=np.argmax(preds, axis=1), name="Predicted Classes"), row=1, col=2)
    fig.update_xaxes(title_text="Class", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Class", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(showlegend=False, title_text="Class Distribution", width=800, height=400)
    fig.show()
    
    # multi label classification
    trainX, trainY, valX, valY, testX, testY = read_dataset("advertisement")
    best_params = {
        "activation_function": "relu",
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.01,
        "num_neurons": [32, 64],
        "optimizer": "batch-gd",
        "num_hidden_layers": 2,
    }
    model = MLP_classification_multi_label(
        learning_rate=best_params["learning_rate"],
        activation_function=best_params["activation_function"],
        optimizer=best_params["optimizer"],
        num_hidden_layers=best_params["num_hidden_layers"],
        num_neurons=best_params["num_neurons"],
        epochs=best_params["epochs"],
        input_layer_size=trainX.shape[1],
        output_layer_size=trainY.shape[1],
    )
    model.fit(trainX, trainY, early_stop=True, X_val=valX, y_val=valY)
    preds = model.predict(testX)
    
    # make confustion matrix for multi label classification
    confusion_matrix = np.zeros((testY.shape[1], testY.shape[1]))
    for i in range(testY.shape[1]):
        for j in range(testY.shape[1]):
            confusion_matrix[i, j] = np.sum((testY[:, i] == 1) & (preds[:, j] == 1))
    fig = ff.create_annotated_heatmap(
        z=confusion_matrix,
        x=[f"Predicted {i}" for i in range(testY.shape[1])],
        y=[f"True {i}" for i in range(testY.shape[1])],
        colorscale="Viridis",
    )
    fig.update_layout(title_text="Confusion Matrix", width=800, height=800)
    fig.show()
    
    # df of each class vs count of true and predicted classes
    true_counts = np.sum(testY, axis=0)
    pred_counts = np.sum(preds, axis=0)
    df = pd.DataFrame({"True": true_counts, "Predicted": pred_counts})
    # plot the distribution of true and predicted classes
    fig = make_subplots(rows=1, cols=2, subplot_titles=["True Classes", "Predicted Classes"])
    fig.add_trace(go.Bar(x=df.index, y=df["True"], name="True Classes"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Predicted"], name="Predicted Classes"), row=1, col=2)
    fig.update_xaxes(title_text="Class", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Class", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(showlegend=False, title_text="Class Distribution", width=800, height=400)
    fig.show()
    
    
    