import numpy as np
import pandas as pd
import os
from utils import setup_base_dir
import plotly.express as px
import plotly.graph_objects as go
import json
import time

base_dir = setup_base_dir(levels=2)

from models.pca.pca import PCA


def reduce_dimensions(n_components, show_words=False):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0
    )
    words = data["words"]
    data = data.drop(columns=["words"])
    data = data.to_numpy()

    # perform PCA
    pca = PCA(n_components)
    pca.fit(data)
    transformed_data = pca.transform(data)
    # verify PCA
    check = pca.checkPCA(data)
    print(f"PCA implementation check for {n_components} componenets:", check)
    # store transformed data in df
    df = pd.DataFrame(
        transformed_data, columns=[f"PC{i+1}" for i in range(n_components)]
    )
    # plot the transformed data with the words
    fig = go.Figure()
    if show_words and n_components == 2:
        fig.add_trace(
            go.Scatter(
                x=df["PC1"],
                y=df["PC2"],
                mode="text",
                text=words,
                textfont=dict(size=8, color="black")
            )
        )
        fig.update_layout(
            title="PCA on word-embeddings - 2 components", width=800, height=600
        )
        fig.show()
    if show_words and n_components == 3:
        fig.add_trace(
            go.Scatter3d(
                x=df["PC1"],
                y=df["PC2"],
                z=df["PC3"],
                mode="markers",
                text=words,
                marker=dict(size=5),
            )
        )
        fig.update_layout(
            title="PCA on word-embeddings - 3 components", width=1000, height=800
        )
        fig.show()

    if not show_words and n_components == 2:
        fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            title="PCA on word-embeddings - 2 components",
            width=800,
            height=600,
        )
        fig.show()
    if not show_words and n_components == 3:
        fig = px.scatter_3d(
            df,
            x="PC1",
            y="PC2",
            z="PC3",
            title="PCA on word-embeddings - 3 components",
            width=1000,
            height=800,
        )
        fig.show()


def scree_plot():
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0
    )
    data = data.drop(columns=["words"])
    data = data.to_numpy()

    data = data - np.mean(data, axis=0)
    data = np.matrix(data)
    u, s, vt = np.linalg.svd(data)
    s = s**2
    normalised_s = s / np.sum(s)
    df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(s))], "eigenvalues": s, "normalised_eigenvalues": normalised_s})
    fig = px.line(
        df,
        x="PC",
        y="eigenvalues",
        title="Scree Plot for word-embeddings",
        width=800,
        height=600,
    )
    fig.show()
    df.to_csv("scree-plot-word-embeddings.csv")


def save_reduced_dataset(n_components):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0
    )
    words = data["words"]
    data = data.drop(columns=["words"])
    data = data.to_numpy()

    pca = PCA(n_components)
    pca.fit(data)
    transformed_data = pca.transform(data)
    # store transformed data in df
    df = pd.DataFrame(
        transformed_data, columns=[f"PC{i+1}" for i in range(n_components)]
    )
    df["words"] = words
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(
        os.path.join(base_dir, "data", "processed", f"word-embeddings-reduced.csv")
    )


def task_5_2():
    reduce_dimensions(2)
    reduce_dimensions(3)
    k_2 = 3


def task_5_3():
    reduce_dimensions(2, show_words=True)


def task_6_2_a():
    scree_plot()
    save_reduced_dataset(5)
