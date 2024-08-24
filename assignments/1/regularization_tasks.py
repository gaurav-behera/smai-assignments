import numpy as np
import pandas as pd
import os
from utils import setup_base_dir
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json

base_dir = setup_base_dir(levels=2)

from data.operations.split_data import split_data
from performance_measures.metrics import Metrics
from models.linear_regression.linreg import LinearRegression

# read linreg.csv and split the data
data = pd.read_csv(os.path.join(base_dir, "data", "processed", "regularisation.csv"))
split = split_data(data, target_column="y", ratio=[0.8, 0.1, 0.1])
trainX, trainY = split["trainX"], split["trainY"]
valX, valY = split["valX"], split["valY"]
testX, testY = split["testX"], split["testY"]

metrics = Metrics()


def all_degree_plots(reg_lambda=0, reg_type="ridge", plt_title="", file_name=""):
    # create subplot for plotting the graphs
    fig = make_subplots(
        rows=5, cols=4, subplot_titles=tuple(["Degree" + str(i) for i in range(1, 21)])
    )
    results = []
    for degree in range(1, 21):
        model = LinearRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(trainX, trainY, regularizer=reg_type)
        y_pred = model.predict(valX)
        mse = metrics.mean_squared_error(valY, y_pred)
        std = metrics.standard_deviation(valY, y_pred)
        var = metrics.variance(valY, y_pred)
        results.append({"degree": degree, "mse": mse, "std": std, "var": var})

        # plot the datapoints and the regression line
        fig.add_trace(
            go.Scatter(
                x=valX["x"],
                y=valY,
                mode="markers",
                name="Data",
                line=dict(color="blue"),
            ),
            row=(degree - 1) // 4 + 1,
            col=(degree - 1) % 4 + 1,
        )
        sorted_zip = sorted(zip(valX["x"], y_pred))
        # plot an orange line)
        fig.add_trace(
            go.Scatter(
                x=[x[0] for x in sorted_zip],
                y=[x[1] for x in sorted_zip],
                mode="lines",
                name="Regression Line",
                line=dict(color="orange"),
            ),
            row=(degree - 1) // 4 + 1,
            col=(degree - 1) % 4 + 1,
        )
        fig.update_xaxes(
            title_text="x", row=(degree - 1) // 4 + 1, col=(degree - 1) % 4 + 1
        )
        fig.update_yaxes(
            title_text="y", row=(degree - 1) // 4 + 1, col=(degree - 1) % 4 + 1
        )

    fig.update_layout(showlegend=False)
    fig.update_layout(title_text=plt_title, width=1000, height=1250)
    fig.show()

    with open(file_name, "w") as f:
        json.dump(results, f)

def compare():
    # read json to df
    with open("no_regularization_results.json", "r") as f:
        no_reg_results = json.load(f)
    with open("L1_regularization_results.json", "r") as f:
        l1_reg_results = json.load(f)
    with open("L2_regularization_results.json", "r") as f:
        l2_reg_results = json.load(f)
    no_reg_df = pd.DataFrame(no_reg_results)
    l1_reg_df = pd.DataFrame(l1_reg_results)
    l2_reg_df = pd.DataFrame(l2_reg_results)
    
    # plot the degree vs mse for the three regularization types
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=no_reg_df["degree"], y=no_reg_df["mse"], mode="lines", name="No Regularization"))
    fig.add_trace(go.Scatter(x=l1_reg_df["degree"], y=l1_reg_df["mse"], mode="lines", name="L1 Regularization"))
    fig.add_trace(go.Scatter(x=l2_reg_df["degree"], y=l2_reg_df["mse"], mode="lines", name="L2 Regularization"))
    fig.update_layout(title="Degree vs MSE for the three regularization types", xaxis_title="Degree", yaxis_title="MSE", height=600, width=800)
    fig.show()
    
    # plot the degree vs std for the three regularization types
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=no_reg_df["degree"], y=no_reg_df["std"], mode="lines", name="No Regularization"))
    fig.add_trace(go.Scatter(x=l1_reg_df["degree"], y=l1_reg_df["std"], mode="lines", name="L1 Regularization"))
    fig.add_trace(go.Scatter(x=l2_reg_df["degree"], y=l2_reg_df["std"], mode="lines", name="L2 Regularization"))
    fig.update_layout(title="Degree vs STD for the three regularization types", xaxis_title="Degree", yaxis_title="STD", height=600, width=800)
    fig.show()
    
    # plot the degree vs var for the three regularization types
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=no_reg_df["degree"], y=no_reg_df["var"], mode="lines", name="No Regularization"))
    fig.add_trace(go.Scatter(x=l1_reg_df["degree"], y=l1_reg_df["var"], mode="lines", name="L1 Regularization"))
    fig.add_trace(go.Scatter(x=l2_reg_df["degree"], y=l2_reg_df["var"], mode="lines", name="L2 Regularization"))
    fig.update_layout(title="Degree vs VAR for the three regularization types", xaxis_title="Degree", yaxis_title="VAR", height=600, width=800)
    fig.show() 
    
    
    
# all_degree_plots(
#     reg_lambda=0,
#     plt_title="No Regularization Plots",
#     file_name="no_regularization_results.json",
# )
# all_degree_plots(
#     reg_lambda=1,
#     reg_type="lasso",
#     plt_title="L1 Regularization Plots",
#     file_name="L1_regularization_results.json",
# )
# all_degree_plots(
#     reg_lambda=1,
#     reg_type="ridge",
#     plt_title="L2 Regularization Plots",
#     file_name="L2_regularization_results.json",
# )
# compare()