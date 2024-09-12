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

def reduce_dimensions(n_components):
    data = pd.read_csv(os.path.join(base_dir, "data", "processed", "word-embeddings.csv"), index_col=0)
    data = data.to_numpy()
    
    # perform PCA
    pca = PCA(n_components)
    pca.fit(data)
    transformed_data = pca.transform(data)
    # verify PCA
    check = pca.checkPCA(data)
    print(f"PCA implementation check for {n_components} componenets:", check)
    # store transformed data in df
    df = pd.DataFrame(transformed_data, columns=[f"PC{i+1}" for i in range(n_components)])
    # plot the transformed data
    if n_components == 2:
        fig = px.scatter(df, x="PC1", y="PC2", title="PCA on word-embeddings - 2 components", width=800, height=600)
        fig.show()
    if n_components == 3:
        fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", title="PCA on word-embeddings - 3 components", width=1000, height=800)
        fig.show()
    
reduce_dimensions(n_components=2)
reduce_dimensions(n_components=3)
