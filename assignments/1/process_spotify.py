import numpy as np
import pandas as pd
import os
from utils import setup_base_dir

base_dir = setup_base_dir(levels=2)
from data.operations.preprocess import process_data

# load the data
data = pd.read_csv(
    os.path.join(base_dir, "data", "external", "spotify.csv"), index_col=0
)

data_vis = process_data(
    data=data.copy(),
    null_cols={
        "artists": "unknown",
        "album_name": "unknown",
        "track_name": "unknown",
    },
    boolean_encode=["explicit"],
    linear_norm=[
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
    ],
    drop_columns=["track_id", "artists", "album_name", "track_name"]
)
data_vis.to_csv(os.path.join(base_dir, "data", "interim", "spotify_vis.csv"))
print("Data for visulaization saved to data/interim/spotify_vis.csv")

# data for the processed folder
data_processed = process_data(
    data=data.copy(),
    boolean_encode=["explicit"],
    linear_norm=[
        "popularity",
        "energy",
        "mode",
        "acousticness",
        "instrumentalness",
        "valence",
    ],
    z_index_norm=['danceability', 'loudness'],
    drop_columns=["track_id", "artists", "album_name", "track_name", 'liveness', 'speechiness', 'loudness', 'tempo', 'time_signature', 'key', 'duration_ms']
)
data_processed.to_csv(os.path.join(base_dir, "data", "processed", "spotify.csv"))
print("Processed Data saved to data/processed/spotify.csv")

# best processed data for max accuracy
data_best = process_data(
    data=data.copy(),
    null_cols={
        "artists": "unknown",
    },
    hash_encode={"artists": 10000},
    linear_norm=[
        "energy",
        "acousticness",
        "instrumentalness",
        "valence",
    ],
    z_index_norm=['loudness'],
    drop_columns=["track_id", "album_name", "track_name", 'liveness', 'speechiness', 'loudness', 'tempo', 'time_signature', 'key', 'duration_ms', 'explicit', 'popularity', 'danceability', 'mode']
)
data_best.to_csv(os.path.join(base_dir, "data", "processed", "spotify_best.csv"))
print("Best Processed Data saved to data/processed/spotify_best.csv")
