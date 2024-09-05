import numpy as np
import pandas as pd
import os
from utils import setup_base_dir

base_dir = setup_base_dir(levels=2)
from data.operations.preprocess import process_data

df = pd.read_feather(os.path.join(base_dir, "data", "external", "word-embeddings.feather"))

df = process_data(data=df, col_expansion=["vit"], drop_columns=["words", "vit"])
df.to_csv(os.path.join(base_dir, "data","processed","word-embeddings.csv"))

print("Data saved to data/processed/word-embeddings.csv")
