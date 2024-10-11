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
from models.MLP.MLP import MLP_regression, MLP


# Dataset Analysis and Preprocessing
def task_3_1():
    data = pd.read_csv(os.path.join(base_dir, "data", "external", "HousingData.csv"))
    # data statistics
    data_stats = data.describe().loc[["mean", "std", "min", "max"]].T
    print(data_stats)

    # make plots for each feature
    fig = make_subplots(rows=5, cols=3, subplot_titles=data.columns)
    for i, col in enumerate(data.columns):
        fig.add_trace(
            go.Histogram(x=data[col], name=col), row=i // 3 + 1, col=i % 3 + 1
        )
        fig.update_xaxes(title_text="Value", row=i // 3 + 1, col=i % 3 + 1)
        fig.update_yaxes(title_text="Count", row=i // 3 + 1, col=i % 3 + 1)
    fig.update_layout(
        title_text="Feature Distribution", showlegend=False, width=1000, height=1500
    )
    fig.show()

    # handle missing values
    data = process_data(data, null_cols={col: data[col].mean() for col in data.columns})
    # normalize data
    data = process_data(data, linear_norm=[col for col in data.columns])

    data.to_csv(
        os.path.join(base_dir, "data", "processed", "HousingData.csv"), index=False
    )


# read dataset
def read_dataset(name):
    match name:
        case "HousingData":
            data = pd.read_csv(
                os.path.join(base_dir, "data", "processed", "HousingData.csv")
            )
            split = split_data(data, target_column="MEDV")
            trainX, trainY = split["trainX"].values, split["trainY"].values.reshape(
                -1, 1
            )
            valX, valY = split["valX"].values, split["valY"].values.reshape(-1, 1)
            testX, testY = split["testX"].values, split["testY"].values.reshape(-1, 1)
            return trainX, trainY, valX, valY, testX, testY
        case "diabetes":
            data = pd.read_csv(
                os.path.join(base_dir, "data", "processed", "diabetes.csv")
            )
            split = split_data(data, target_column="Outcome")
            trainX, trainY = split["trainX"].values, split["trainY"].values.reshape(
                -1, 1
            )
            valX, valY = split["valX"].values, split["valY"].values.reshape(-1, 1)
            testX, testY = split["testX"].values, split["testY"].values.reshape(-1, 1)
            return trainX, trainY, valX, valY, testX, testY
        case _:
            return None


# hyperparameter tuning
def task_3_3():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "mse", "goal": "minimize"},
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
    sweep_id = wandb.sweep(sweep_config, project="mlp-linear-regression")

    trainX, trainY, valX, valY, testX, testY = read_dataset("HousingData")
    run_id = 0

    def train():
        nonlocal run_id
        wandb.init(project="mlp-linear-regresion", name=f"run-{run_id}")
        run_id += 1
        config = wandb.config
        model = MLP_regression(
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
                "mse": model.metrics.mean_squared_error(valY, preds),
                "rmse": model.metrics.root_mean_squared_error(valY, preds),
                "r2": model.metrics.r2_score(valY, preds),
            }
        )

        wandb.finish()

    wandb.agent(sweep_id, train)


def task_3_4():
    trainX, trainY, valX, valY, testX, testY = read_dataset("HousingData")
    best_params = {
        "learning_rate": 0.1,
        "activation_function": "relu",
        "optimizer": "sgd",
        "num_neurons": [32, 64],
        "num_hidden_layers": 2,
        "epochs": 100,
    }
    model = MLP_regression(
        learning_rate=best_params["learning_rate"],
        activation_function=best_params["activation_function"],
        optimizer=best_params["optimizer"],
        num_hidden_layers=best_params["num_hidden_layers"],
        num_neurons=best_params["num_neurons"],
        epochs=best_params["epochs"],
        input_layer_size=trainX.shape[1],
        output_layer_size=trainY.shape[1],
    )
    model.fit(trainX, trainY, True, valX, valY)
    with open("regression_model.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "params": best_params,
                    "weights": [w.tolist() for w in model.weights],
                    "biases": [b.tolist() for b in model.biases],
                }
            )
        )
    preds = model.predict(testX)
    print("MSE:", model.metrics.mean_squared_error(testY, preds))
    print("RMSE:", model.metrics.root_mean_squared_error(testY, preds))
    print("R2:", model.metrics.r2_score(testY, preds))
    print("MAE:", model.metrics.mean_absolute_error(testY, preds))


def process_diabetes_data():
    data = pd.read_csv(os.path.join(base_dir, "data", "external", "diabetes.csv"))
    data = process_data(
        data,
        null_cols={col: data[col].mean() for col in data.columns},
        linear_norm=[col for col in data.columns],
    )
    data.to_csv(
        os.path.join(base_dir, "data", "processed", "diabetes.csv"), index=False
    )

# MSE VS BCE
def task_3_5():
    X_train, y_train, X_val, y_val, X_test, y_test = read_dataset("diabetes")
    # make models
    model1 = MLP(
        learning_rate=0.01,
        activation_function="relu",
        optimizer="batch-gd",
        num_hidden_layers=2,
        num_neurons=[32, 64],
        epochs=100,
        input_layer_size=X_train.shape[1],
        output_layer_size=1,
        log_local=True,
        task="logistic-bce",
    )
    model2 = MLP(
        learning_rate=0.01,
        activation_function="relu",
        optimizer="batch-gd",
        num_hidden_layers=2,
        num_neurons=[32, 64],
        epochs=100,
        input_layer_size=X_train.shape[1],
        output_layer_size=1,
        log_local=True,
        task="logistic-mse",
    )
    # fit models
    model1.fit(X_train, y_train, True, X_val, y_val)
    model2.fit(X_train, y_train, True, X_val, y_val)
    # predict
    preds1 = model1.predict(X_test)
    preds2 = model2.predict(X_test)
    # print metrics
    print("Model 1 BCE")
    print("MSE:", model1.metrics.mean_squared_error(y_test, preds1))
    print("RMSE:", model1.metrics.root_mean_squared_error(y_test, preds1))
    print("R2:", model1.metrics.r2_score(y_test, preds1))
    print("MAE:", model1.metrics.mean_absolute_error(y_test, preds1))
    print("Model 2 MSE")
    print("MSE:", model2.metrics.mean_squared_error(y_test, preds2))
    print("RMSE:", model2.metrics.root_mean_squared_error(y_test, preds2))
    print("R2:", model2.metrics.r2_score(y_test, preds2))
    print("MAE:", model2.metrics.mean_absolute_error(y_test, preds2))

    logs1 = model1.logs
    logs2 = model2.logs
    logs_df1 = pd.DataFrame(logs1)
    logs_df1["epoch"] = logs_df1.index
    logs_df2 = pd.DataFrame(logs2)
    logs_df2["epoch"] = logs_df2.index

    # plot loss vs epochs
    fig = make_subplots(rows=1, cols=2, subplot_titles=["BCE", "MSE"])
    fig.add_trace(
        go.Scatter(x=logs_df1["epoch"], y=logs_df1["loss"], mode="lines"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=logs_df2["epoch"], y=logs_df2["loss"], mode="lines"), row=1, col=2
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_layout(title_text="Loss vs Epochs", showlegend=False, width=800, height=400)
    fig.show()

# analysis
def task_3_6():
    # For every datapoint in the test dataset, observe the MSE Loss. Do you notice if there is a pattern in the datapoints for which it gives a high MSE Loss or a low MSE Loss
    trainX, trainY, valX, valY, testX, testY = read_dataset("HousingData")
    best_params = {
        "learning_rate": 0.1,
        "activation_function": "relu",
        "optimizer": "sgd",
        "num_neurons": [32, 64],
        "num_hidden_layers": 2,
        "epochs": 10,
    }
    model = MLP_regression(
        learning_rate=best_params["learning_rate"],
        activation_function=best_params["activation_function"],
        optimizer=best_params["optimizer"],
        num_hidden_layers=best_params["num_hidden_layers"],
        num_neurons=best_params["num_neurons"],
        epochs=best_params["epochs"],
        input_layer_size=trainX.shape[1],
        output_layer_size=trainY.shape[1],
    )
    model.fit(trainX, trainY, True, valX, valY)
    preds = model.predict(testX)
    mses = np.mean((preds - testY) ** 2, axis=1)
    # plot testY  and mse values for each data point
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(testY)), y=testY.flatten(), mode="lines", name="testY"))
    fig.add_trace(go.Scatter(x=np.arange(len(testY)), y=mses, mode="lines", name="MSE"))
    fig.update_xaxes(title_text="Data Point")
    fig.update_yaxes(title_text="Value")
    fig.update_layout(title_text="TestY and MSE values for each data point", showlegend=True, width=800, height=600)
    fig.show()
    
    
    


    