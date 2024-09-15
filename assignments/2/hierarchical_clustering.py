import numpy as np
import pandas as pd
import os
from utils import setup_base_dir
import plotly.express as px
import json
import time
import scipy.cluster.hierarchy as hc
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score


base_dir = setup_base_dir(levels=2)

data = pd.read_csv(
    os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0
)
words = list(data["words"])
data = data.drop(columns=["words"])
data = data.to_numpy()


def sample_hierarchical_clustering():
    # calculate the linkage matrix
    linkage_matrix = hc.linkage(data, method="ward", metric="euclidean")
    print("Linkage Matrix:")
    print(linkage_matrix)

    # plot the dendrogram
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title("Sample Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Words")
    ax.set_ylabel("Distance")
    hc.dendrogram(linkage_matrix, labels=words, ax=ax)
    plt.show()


def compare_hierarchical_clustering():
    methods = ["single", "complete", "average"]
    metrics = ["euclidean", "cityblock", "cosine"]

    for method in methods:
        for metric in metrics:
            linkage_matrix = hc.linkage(data, method=method, metric=metric)
            fig, ax = plt.subplots(figsize=(15, 10))
            dendrogram = hc.dendrogram(linkage_matrix, labels=words, ax=ax)
            ax.set_title(f"Method: {method}, Metric: {metric}")
            ax.set_xlabel("Words")
            ax.set_ylabel("Distance")
            # save plt
            plt.savefig(
                os.path.join(
                    f"dendogram-method-{method}-metric-{metric}.png",
                )
            )


def create_clusters(method, metric, n_clusters):
    linkage_matrix = hc.linkage(data, method=method, metric=metric)
    dendogram = hc.dendrogram(linkage_matrix, labels=words)
    # get words based on the dendogram order
    # ordered_words = [words[i] for i in dendogram["leaves"]]
    cluster_preds = hc.fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1

    clusters = {i: [] for i in range(n_clusters)}
    for i in range(len(words)):
        clusters[cluster_preds[i]].append(words[i])
    for _k in range(n_clusters):
        print(f"Words in cluster {_k}:")
        print("\t", ", ".join(clusters[_k]))

    cluster_preds = np.array(cluster_preds)
    # intra-cluster distance
    intra_cluster_distances = []
    for i in range(n_clusters):
        cluster_data = data[cluster_preds == i]
        cluster_centroid = np.mean(cluster_data, axis=0)
        intra_cluster_distances.append(np.sum((cluster_data - cluster_centroid) ** 2))
    avg_intra_cluster_distance = np.mean(np.array(intra_cluster_distances))
    print("Average Intra-cluster distances:", avg_intra_cluster_distance)

    # inter-cluster distance
    inter_cluster_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            inter_cluster_distances.append(
                np.sum(
                    (
                        np.mean(data[cluster_preds == i], axis=0)
                        - np.mean(data[cluster_preds == j], axis=0)
                    )
                    ** 2
                )
            )
    avg_inter_cluster_distance = np.mean(np.array(inter_cluster_distances))
    print("Average Inter-cluster distances:", avg_inter_cluster_distance)

    # silhouette score
    avg_silhouette_score = (
        silhouette_score(data, cluster_preds) if n_clusters > 1 else np.nan
    )
    print("Average Silhouette score:", avg_silhouette_score)

    # save results
    results = {
        "clusters": clusters,
        "avg_intra_cluster_distances": avg_intra_cluster_distance,
        "avg_inter_cluster_distances": avg_inter_cluster_distance,
        "avg_silhouette_score": avg_silhouette_score,
    }
    with open(f"hierarchical_cluster_analysis_k{n_clusters}.json", "w") as f:
        json.dump(results, f)


def task_8():
    sample_hierarchical_clustering()
    compare_hierarchical_clustering()
    k_best1 = 4  # kmeans
    k_best2 = 3  # gmm
    create_clusters("complete", "euclidean", k_best1)
    create_clusters("complete", "euclidean", k_best2)
