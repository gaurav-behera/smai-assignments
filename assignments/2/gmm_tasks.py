import numpy as np
import pandas as pd
import os
from utils import setup_base_dir
import plotly.express as px
import plotly.graph_objects as go
import json
import time

base_dir = setup_base_dir(levels=2)

from models.gmm.gmm import GMM, GMM_sklearn


def analyse_2d_clustering_data():
    data = pd.read_csv(
        os.path.join(base_dir, "data", "external", "2d-clustering-data.csv")
    )
    data["id"] = data.index
    X = data[["x", "y"]].to_numpy()
    # view original data with color
    map_color = {0: "rgb(0,0,355)", 1: "rgb(0,255,0)", 2: "rgb(255,0,0)"}
    data["color"] = data["color"].map(map_color)
    fig = go.Figure(
        data=go.Scatter(
            x=data["x"], y=data["y"], mode="markers", marker=dict(color=data["color"])
        )
    )
    fig.update_layout(
        title="2D Clustering Data - Original",
        xaxis_title="x",
        yaxis_title="y",
        width=800,
        height=600,
    )

    fig.show()
    # run gmm model
    gmm = GMM(k=3)
    gmm.fit(X)
    probs = (gmm.getMembership(X) * 255).astype(np.int32)
    data["cluster"] = pd.Series(
        [f"rgb({probs[i][0]},{probs[i][1]},{probs[i][2]})" for i in range(len(probs))]
    )
    # view clustered data
    fig = go.Figure(
        data=go.Scatter(
            x=data["x"], y=data["y"], mode="markers", marker=dict(color=data["cluster"])
        )
    )
    fig.update_layout(
        title="2D Clustering Data - Soft Clustered",
        xaxis_title="x",
        yaxis_title="y",
        width=800,
        height=600,
    )

    fig.show()


def aic_bic_vs_k(model="own", reduced=False):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0
    )
    if reduced:
        data = pd.read_csv(
            os.path.join(base_dir, "data", "processed", "word-embeddings-reduced.csv"),
            index_col=0,
        )
    data = data.drop(columns=['words'])
    data = data.to_numpy()

    results = []
    for _k in range(1, 15):
        match model:
            case "own":
                gmm = GMM(k=_k)
            case "sklearn":
                gmm = GMM_sklearn(k=_k)
        gmm.fit(data)
        dims = data.shape[1]
        k = _k * dims + _k * (dims * dims + dims) // 2 + _k - 1  # number of parameters
        l = gmm.getLogLikelihood(data)  # log-likelihood
        n = data.shape[0]  # number of samples
        aic = 2 * k - 2 * l
        bic = k * np.log(n) - 2 * l
        results.append(
            {
                "k": _k,
                "aic": aic,
                "bic": bic,
                "log-likelihood": l,
                "parameters": k,
                "n": n,
            }
        )

    # plot the results
    df = pd.DataFrame(results)
    # plot AIC and BIC
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["k"], y=df["aic"], mode="lines+markers", name="AIC"))
    fig.add_trace(go.Scatter(x=df["k"], y=df["bic"], mode="lines+markers", name="BIC"))
    fig.update_layout(
        title=f"AIC and BIC vs K for word-embeddings{"-reduced" if reduced else ""}",
        xaxis_title="K",
        yaxis_title="Value",
        width=800,
        height=600,
    )
    fig.show()

    with open(f"aic_bic_vs_k{"_reduced" if reduced else ""}.json", "w") as f:
        json.dump(results, f)


def sample_clustering(k, model, reduced=False):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0
    )
    if reduced:
        data = pd.read_csv(
            os.path.join(base_dir, "data", "processed", "word-embeddings-reduced.csv"),
            index_col=0,
        )
    data = data.drop(columns=['words'])
    data = data.to_numpy()

    print("GMM clustering with k =", k, f"on word-embeddings{"-reduced" if reduced else ""} data")
    gmm = model
    gmm.fit(data)
    print("\tFinal likelihood:", gmm.getLikelihood(data).round(2))
    print("\tFinal log-likelihood:", gmm.getLogLikelihood(data).round(2))
    print("\tAIC:", gmm.aic(data))
    print("\tBIC:", gmm.bic(data))


def task_4_2():
    sample_clustering(k=3, model=GMM(k=3))
    sample_clustering(k=3, model=GMM_sklearn(k=3))
    aic_bic_vs_k(model="sklearn")
    k_gmm1 = 1
    sample_clustering(k=k_gmm1, model=GMM_sklearn(k=k_gmm1))


def task_6_3():
    k_2 = 3
    sample_clustering(k=k_2, model=GMM(k=k_2))

def task_6_4():
    # aic_bic_vs_k(reduced=True)
    k_gmm3 = 3
    sample_clustering(k=k_gmm3, model=GMM(k=k_gmm3), reduced=True)
