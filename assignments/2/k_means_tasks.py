import numpy as np
import pandas as pd
import os
from utils import setup_base_dir
import plotly.express as px
import json
import time
from sklearn.metrics import silhouette_score

base_dir = setup_base_dir(levels=2)

from models.k_means.k_means import KMeans


def analyse_2d_clustering_data():
    data = pd.read_csv(
        os.path.join(base_dir, "data", "external", "2d-clustering-data.csv")
    )
    X = data[["x", "y"]].to_numpy()
    # view original data with color
    fig = px.scatter(
        data,
        x="x",
        y="y",
        title="2D Clustering Data - Original",
        width=800,
        height=600,
        color="color",
    )
    fig.show()
    # run k_means
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    data["cluster"] = kmeans.predict(X)
    # view clustered data
    fig = px.scatter(
        data,
        x="x",
        y="y",
        title="2D Clustering Data - Clustered",
        width=800,
        height=600,
        color="cluster",
    )
    fig.show()


def wcss_vs_k(reduced=False):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0
    )
    if reduced:
        data = pd.read_csv(
            os.path.join(base_dir, "data", "processed", "word-embeddings-reduced.csv"),
            index_col=0,
        )
    data = data.drop(columns=["words"])
    data = data.to_numpy()

    results = []
    for k in range(1, 20):
        kmeans = KMeans(k=k)
        kmeans.fit(data)
        results.append({"k": k, "wcss": kmeans.wcss})

    # plot the results
    df = pd.DataFrame(results)
    fig = px.line(
        df,
        x="k",
        y="wcss",
        title=f"WCSS vs K for word-embeddings{' reduced' if reduced else ''}",
        width=800,
        height=600,
    )
    fig.show()

    with open(f"wcss_vs_k{'_reduced' if reduced else ''}.json", "w") as f:
        json.dump(results, f)


def sample_clustering(k, reduced=False):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0
    )
    if reduced:
        data = pd.read_csv(
            os.path.join(base_dir, "data", "processed", "word-embeddings-reduced.csv"),
            index_col=0,
        )
    words = data["words"]
    data = data.drop(columns=["words"])
    data = data.to_numpy()

    print(
        "KMeans clustering with k =",
        k,
        f"on word-embeddings{'-reduced' if reduced else ''} data",
    )
    kmeans = KMeans(k=k)
    kmeans.fit(data)
    cluster_preds = kmeans.predict(data)
    print("\tFinal WCSS:", kmeans.wcss.round(2))
    print("\tPoints in each cluster:")
    for i in range(k):
        print("\t\tCluster", i, ":", np.sum(cluster_preds == i))

    return words, cluster_preds


def cluster_analysis(k, reduced=False):
    data = pd.read_csv(
        os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0
    )
    if reduced:
        data = pd.read_csv(
            os.path.join(base_dir, "data", "processed", "word-embeddings-reduced.csv"),
            index_col=0,
        )
    words = data["words"]
    data = data.drop(columns=["words"])
    data = data.to_numpy()
    
    kmeans = KMeans(k=k)
    kmeans.fit(data)
    cluster_preds = kmeans.predict(data)
    
    # print words in each cluster
    clusters = {i: [] for i in range(k)}
    for i in range(len(words)):
        clusters[cluster_preds[i]].append(words[i])
    for _k in range(k):
        print(f"Words in cluster {_k}:")
        print("\t", ", ".join(clusters[_k]))
        
    # intra-cluster distance
    intra_cluster_distances = []
    for i in range(k):
        cluster_data = data[cluster_preds == i]
        cluster_centroid = np.mean(cluster_data, axis=0)
        intra_cluster_distances.append(np.sum((cluster_data - cluster_centroid) ** 2))
    avg_intra_cluster_distance = np.mean(np.array(intra_cluster_distances))
    print("Average Intra-cluster distances:", avg_intra_cluster_distance)
    
    # inter-cluster distance
    inter_cluster_distances = []
    for i in range(k):
        for j in range(i + 1, k):
            inter_cluster_distances.append(np.sum((kmeans.centroids[i] - kmeans.centroids[j]) ** 2))
    avg_inter_cluster_distance = np.mean(np.array(inter_cluster_distances))
    print("Average Inter-cluster distances:", avg_inter_cluster_distance)
    
    # silhouette score
    avg_silhouette_score = silhouette_score(data, cluster_preds) if k>1 else np.nan
    print("Average Silhouette score:", avg_silhouette_score)
    
    # save results
    results = {
        "clusters": clusters,
        "avg_intra_cluster_distances": avg_intra_cluster_distance,
        "avg_inter_cluster_distances": avg_inter_cluster_distance,
        "avg_silhouette_score": avg_silhouette_score
    }
    with open (f"k_means_cluster_analysis_k{k}{'_reduced' if reduced else ''}.json", "w") as f:
        json.dump(results, f)

def task_3_2():
    wcss_vs_k()
    k_kmeans1 = 4
    sample_clustering(k=k_kmeans1)


def task_6_1():
    k_2 = 3
    sample_clustering(k=k_2)


def task_6_2_b():
    wcss_vs_k(reduced=True)
    k_kmeans3 = 6
    sample_clustering(k=k_kmeans3, reduced=True)


def task_7_1():
    k_means1 = 4
    k_2 = 3
    k_means3 = 6
    cluster_analysis(k_means1)
    cluster_analysis(k_2)
    cluster_analysis(k_means3, reduced=True)


# analyse_2d_clustering_data()
