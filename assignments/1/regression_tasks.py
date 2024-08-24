import numpy as np
import pandas as pd
import os
from utils import setup_base_dir
import plotly.express as px
import matplotlib.pyplot as plt
import json
from PIL import Image
from matplotlib.gridspec import GridSpec

base_dir = setup_base_dir(levels=2)

from data.operations.split_data import split_data
from performance_measures.metrics import Metrics
from models.linear_regression.linreg import LinearRegression

# read linreg.csv and split the data
data = pd.read_csv(os.path.join(base_dir, "data", "processed", "linreg.csv"))
split = split_data(data, target_column="y", ratio=[0.8, 0.1, 0.1])
trainX, trainY = split["trainX"], split["trainY"]
valX, valY = split["valX"], split["valY"]
testX, testY = split["testX"], split["testY"]

metrics = Metrics()


# visulaize the data points
def visualize_data():
    # plotting the training points
    train_df = pd.DataFrame({"x": trainX["x"], "y": trainY})
    fig = px.scatter(train_df, x="x", y="y", title="Training Data")
    fig.update_layout(height=600, width=800)
    fig.show()

    # plotting the validation points
    val_df = pd.DataFrame({"x": valX["x"], "y": valY})
    fig = px.scatter(val_df, x="x", y="y", title="Validation Data")
    fig.update_layout(height=600, width=800)
    fig.show()

    # plotting the test points
    test_df = pd.DataFrame({"x": testX["x"], "y": testY})
    fig = px.scatter(test_df, x="x", y="y", title="Test Data")
    fig.update_layout(height=600, width=800)
    fig.show()

    # combined plot
    fig = px.scatter(title="Combined Data")
    fig.add_scatter(x=train_df["x"], y=train_df["y"], mode="markers", name="Train Data")
    fig.add_scatter(
        x=val_df["x"], y=val_df["y"], mode="markers", name="Validation Data"
    )
    fig.add_scatter(x=test_df["x"], y=test_df["y"], mode="markers", name="Test Data")
    fig.update_layout(height=600, width=800)
    fig.update_xaxes(title_text="X")
    fig.update_yaxes(title_text="Y")
    fig.show()


# simple model run
def run_regularization_model(degree):
    # initialize the model
    print("Model run for degree:", degree)
    model = LinearRegression(degree=degree, reg_lambda=0)
    # fit the model
    model.fit(trainX, trainY)

    print("Training Set")
    y_pred = model.predict(trainX)
    print("\tMean Square Error:", metrics.mean_squared_error(trainY, y_pred))
    print("\tStandard Deviation:", metrics.standard_deviation(trainY, y_pred))
    print("\tVariance:", metrics.variance(trainY, y_pred))

    print("Validation Set")
    y_pred = model.predict(valX)
    print("\tMean Square Error:", metrics.mean_squared_error(valY, y_pred))
    print("\tStandard Deviation:", metrics.standard_deviation(valY, y_pred))
    print("\tVariance:", metrics.variance(valY, y_pred))

    print("Test Set")
    y_pred = model.predict(testX)
    print("\tMean Square Error:", metrics.mean_squared_error(testY, y_pred))
    print("\tStandard Deviation:", metrics.standard_deviation(testY, y_pred))
    print("\tVariance:", metrics.variance(testY, y_pred))


def get_best_learning_rate(degree):
    # initialize the model
    model = LinearRegression(degree=degree, reg_lambda=0)
    learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    results = []
    for lr in learning_rates:
        model.fit(trainX, trainY, learning_rate=lr, epochs=1000)
        y_pred = model.predict(valX)
        results.append(
            {
                "Learning Rate": lr,
                "Mean Square Error": metrics.mean_squared_error(valY, y_pred),
                "Standard Deviation": metrics.standard_deviation(valY, y_pred),
                "Variance": metrics.variance(valY, y_pred),
                "Weights": model.get_weights().tolist(),
            }
        )

    results_df = pd.DataFrame(results)
    print("Results for degree:", degree)
    print(results_df)
    fig = px.line(results_df, x="Learning Rate", y="Mean Square Error")
    fig.update_xaxes(type="log")
    fig.update_layout(title="Learning Rate vs Mean Square Error", height=600, width=800)
    fig.show()

    # print top 5 results
    print("Top 5 results")
    print(results_df.sort_values(by="Mean Square Error").head(5))
    
    with open("learning_rate_results.json", "w") as f:
        json.dump(results, f)


def plot_best_fit_line(degree, learning_rate):
    model = LinearRegression(degree=degree, reg_lambda=0)
    model.fit(trainX, trainY, learning_rate=learning_rate, epochs=1000)
    c, m = model.get_weights()

    # train plot
    train_df = pd.DataFrame({"x": trainX["x"], "y": trainY})
    fig = px.scatter(train_df, x="x", y="y", title="Training Data")
    x = np.linspace(trainX["x"].min(), trainX["x"].max(), 100)
    y = m * x + c
    fig.add_scatter(x=x, y=y, mode="lines", name="Best Fit Line")
    fig.update_layout(height=600, width=800)
    fig.show()
    # val plot
    val_df = pd.DataFrame({"x": valX["x"], "y": valY})
    fig = px.scatter(val_df, x="x", y="y", title="Validation Data")
    fig.add_scatter(x=x, y=y, mode="lines", name="Best Fit Line")
    fig.update_layout(height=600, width=800)
    fig.show()
    # test plot
    test_df = pd.DataFrame({"x": testX["x"], "y": testY})
    fig = px.scatter(test_df, x="x", y="y", title="Test Data")
    fig.add_scatter(x=x, y=y, mode="lines", name="Best Fit Line")
    fig.update_layout(height=600, width=800)
    fig.show()


def get_best_degree(learning_rate):
    # initialize the model
    for calc_type in ["gradient_descent", "closed_form"]:
        deg_vals = list(range(25))
        results = []
        for deg in deg_vals:
            model = LinearRegression(degree=deg, reg_lambda=0)
            model.fit(
                trainX,
                trainY,
                learning_rate=learning_rate,
                epochs=10000,
                type=calc_type,
            )
            y_pred = model.predict(valX)
            results.append(
                {
                    "Degree": deg,
                    "Mean Square Error": metrics.mean_squared_error(valY, y_pred),
                    "Standard Deviation": metrics.standard_deviation(valY, y_pred),
                    "Variance": metrics.variance(valY, y_pred),
                    "Weights": model.get_weights().tolist(),
                }
            )

        results_df = pd.DataFrame(results)
        print(results_df)
        fig = px.line(results_df, x="Degree", y="Mean Square Error")
        fig.update_layout(title="Degree vs Mean Square Error", height=600, width=800)
        fig.show()

        # print top 5 results
        print("Top 5 results")
        print(results_df.sort_values(by="Mean Square Error").head(5))
        
        with open(f"degree_results_{calc_type}.json", "w") as f:
            json.dump(results, f)


def format_polynomial(coeffs):
    """Format polynomial coefficients into a string."""
    terms = [f"{coef:.2f}x^{i}" for i, coef in enumerate(reversed(coeffs))]
    equation = " + ".join(terms).replace("x^0", "")
    return f"{equation}"


def make_gif_plots(deg, start_zero=False, start_one=False):
    lr = 0.01
    model = LinearRegression(degree=deg, reg_lambda=0)
    model.fit(trainX, trainY, learning_rate=lr, epochs=0, type="gradient_descent")
    if start_zero:
        model.weights = np.zeros_like(model.weights)
    if start_one:
        model.weights = np.ones_like(model.weights)
    polyX = model._add_polynomial_features(trainX)
    results = [
        {
            "epoch": 0,
            "weights": model.get_weights(),
            "mse": metrics.mean_squared_error(valY, model.predict(valX)),
            "std": metrics.standard_deviation(valY, model.predict(valX)),
            "var": metrics.variance(valY, model.predict(valX)),
        }
    ]
    xs = np.linspace(trainX["x"].min(), trainX["x"].max(), 100).reshape(-1, 1)
    # create dir
    if start_zero:
        d = str(deg) + "-zero"
    elif start_one:
        d = str(deg) + "-one"
    else:
        d = str(deg)
    if not os.path.exists(f"frames-deg{d}"):
        os.makedirs(f"frames-deg{d}")
    # Generate frames
    for e in range(1, 1000):
        df = pd.DataFrame(results)

        # Create figure and subplots
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # Curve fitting subplot
        ax1.plot(
            xs,
            model.predict(xs),
            label=f"Epoch {e}: " + format_polynomial(model.get_weights()),
            c="orange",
        )
        ax1.scatter(trainX["x"], trainY, s=2, label="Data Points")
        ax1.set_title("Curve Fitting")
        ax1.legend()

        # Metrics subplots
        ax2.plot(
            df["epoch"], df["mse"], label="MSE = " + str(round(df["mse"].iloc[-1], 2))
        )
        ax2.set_title("Mean Square Error")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE")
        ax2.legend()

        ax3.plot(
            df["epoch"], df["std"], label="STD = " + str(round(df["std"].iloc[-1], 2))
        )
        ax3.set_title("Standard Deviation")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Standard Deviation")
        ax3.legend()

        ax4.plot(
            df["epoch"], df["var"], label="VAR = " + str(round(df["var"].iloc[-1], 2))
        )
        ax4.set_title("Variance")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Variance")
        ax4.legend()

        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()

        if start_zero:
            d = str(deg) + "-zero"
        elif start_one:
            d = str(deg) + "-one"
        else:
            d = str(deg)
        plt.savefig(f"frames-deg{d}/frame_{e}.png")

        # Update model and results
        model._gradient_descent(polyX, trainY, lr, 1)
        results.append(
            {
                "epoch": e,
                "weights": model.get_weights(),
                "mse": metrics.mean_squared_error(valY, model.predict(valX)),
                "std": metrics.standard_deviation(valY, model.predict(valX)),
                "var": metrics.variance(valY, model.predict(valX)),
            }
        )
        plt.close(fig)
        
        # break loop if mse or variance or standard deviation starts increasing
        if results[-1]["mse"] > results[-2]["mse"]:
            break
        if results[-1]["std"] > results[-2]["std"]:
            break
        if results[-1]["var"] > results[-2]["var"]:
            break
        


def create_gif(deg, start_zero=False, start_one=False):
    img_array = []
    if start_zero:
        deg = str(deg) + "-zero"
    elif start_one:
        deg = str(deg) + "-one"
    for i in range(len(os.listdir(f"frames-deg{deg}"))):
        img = Image.open(f"frames-deg{deg}/frame_{str(i + 1)}.png")
        img_array.append(img)
    img_array[0].save(f"animation-deg{deg}.gif",
               save_all=True,
               append_images=img_array[1:],
               duration=10,
               loop=0,
               optimize=True)
    print("Gif created")


def animation_gif():
    for k in [1,2,5,17,21]:
        make_gif_plots(k)
        create_gif(k)

def animation_initialization_gif():
    for k in [17]:
        make_gif_plots(k, start_zero=True)
        create_gif(k, start_zero=True)
        make_gif_plots(k, start_one=True)
        create_gif(k, start_one=True)
        

# visualize_data()
# run_regularization_model(degree=1)
# get_best_learning_rate(degree=1)
# plot_best_fit_line(degree=1, learning_rate=0.01)
# run_regularization_model(degree=2)
# get_best_degree(learning_rate=0.01)
# animation_gif()
# animation_initialization_gif()
