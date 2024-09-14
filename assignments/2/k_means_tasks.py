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
    
def wcss_vs_k(reduced=False):
    data = pd.read_csv(os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0)
    if reduced:
        data = pd.read_csv(os.path.join(base_dir, "data", "processed", "word-embeddings-reduced.csv"), index_col=0)
    data = data.drop(columns=['words'])
    data = data.to_numpy()
    
    results = []
    for k in range(1, 20):
        kmeans = KMeans(k=k)
        kmeans.fit(data)
        results.append({"k": k, "wcss": kmeans.wcss})
    
    # plot the results
    df = pd.DataFrame(results)
    fig = px.line(df, x="k", y="wcss", title=f"WCSS vs K for word-embeddings{' reduced' if reduced else ''}", width=800, height=600)
    fig.show()
    
    with open(f"wcss_vs_k{'_reduced' if reduced else ''}.json", "w") as f:
        json.dump(results, f)

def sample_clustering(k, reduced=False):
    data = pd.read_csv(os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0)
    if reduced:
        data = pd.read_csv(os.path.join(base_dir, "data", "processed", "word-embeddings-reduced.csv"), index_col=0)
    words = data['words']
    data = data.drop(columns=['words'])
    data = data.to_numpy()
    
    print("KMeans clustering with k =", k, f"on word-embeddings{'-reduced' if reduced else ''} data")
    kmeans = KMeans(k=k)
    kmeans.fit(data)
    cluster_preds = kmeans.predict(data)
    print("\tFinal WCSS:", kmeans.wcss.round(2))
    print("\tPoints in each cluster:")
    for i in range(k):
        print("\t\tCluster", i, ":", np.sum(cluster_preds == i))
    
    return words, cluster_preds

# def sematic_cluster_analysis(words, cluster_preds):
#     k = np.unique(cluster_preds)

def task_3_2():
    wcss_vs_k()
    k_kmeans1 = 5
    sample_clustering(k=k_kmeans1)

def task_6_1():
    k_2 = 3
    sample_clustering(k=k_2)

def task_6_2_b():
    wcss_vs_k(reduced=True)
    k_kmeans3 = 6
    sample_clustering(k=k_kmeans3, reduced=True)

def task_7_1():
    k_means1 = 5
    k_2 = 3
    k_means3 = 6
    args1 = sample_clustering(k=k_means1)
    sematic_cluster_analysis(*args1)
    pass
# analyse_2d_clustering_data() 