import numpy as np
import pandas as pd
import os
from utils import setup_base_dir
import plotly.express as px
import json
import time

base_dir = setup_base_dir(levels=2)

from models.k_means.k_means import KMeans

def analyse_2d_clustering_data():
    data = pd.read_csv(os.path.join(base_dir, "data", "external", "2d-clustering-data.csv"))
    X = data[["x", "y"]].to_numpy()
    # view original data with color
    fig = px.scatter(data, x="x", y="y", title="2D Clustering Data - Original", width=800, height=600, color="color")
    fig.show()
    # run k_means
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    data["cluster"] = kmeans.predict(X)
    # view clustered data
    fig = px.scatter(data, x="x", y="y", title="2D Clustering Data - Clustered", width=800, height=600, color="cluster")
    fig.show()
    

analyse_2d_clustering_data()