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

vis_args = {
    "null_cols": {
        "artists": "unknown",
        "album_name": "unknown",
        "track_name": "unknown",
    },
    "boolean_encode": ["explicit"],
    "linear_norm": [
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
    "drop_columns": ["track_id", "artists", "album_name", "track_name"],
}

data_vis = process_data(data=data.copy(), **vis_args)
data_vis.to_csv(os.path.join(base_dir, "data", "interim", "spotify_vis.csv"))
print("Data for visulaization saved to data/interim/spotify_vis.csv")

# data for the processed folder
process_args = {
    "boolean_encode": ["explicit"],
    "z_index_norm": [
        "danceability",
        "loudness",
        "popularity",
        "energy",
        "mode",
        "acousticness",
        "instrumentalness",
        "valence",
        "liveness",
        "speechiness",
        "loudness",
    ],
    "drop_columns": [
        "track_id",
        "artists",
        "album_name",
        "track_name",
        "tempo",
        "time_signature",
        "key",
        "duration_ms",
    ],
}
data_processed = process_data(data=data.copy(), **process_args)
data_processed.to_csv(os.path.join(base_dir, "data", "processed", "spotify.csv"))
print("Processed Data saved to data/processed/spotify.csv")

# best processed data for max accuracy
best_process_args = {
    "null_cols": {
        "artists": "unknown",
    },
    "hash_encode": {"artists": 10000},
    "z_index_norm": [
        "loudness",
        "energy",
        "acousticness",
        "instrumentalness",
        "valence",
    ],
    "drop_columns": [
        "track_id",
        "album_name",
        "track_name",
        "liveness",
        "speechiness",
        "loudness",
        "tempo",
        "time_signature",
        "key",
        "duration_ms",
        "explicit",
        "popularity",
        "danceability",
        "mode",
    ],
}
data_best = process_data(data=data.copy(), **best_process_args)
data_best.to_csv(os.path.join(base_dir, "data", "processed", "spotify_best.csv"))
print("Best Processed Data saved to data/processed/spotify_best.csv")

# spotify 2 normal
train_data = process_data(
    data=pd.read_csv(
        os.path.join(base_dir, "data", "external", "spotify-2", "train.csv"),
        index_col=0,
    ),
    **process_args
)
val_data = process_data(
    data=pd.read_csv(
        os.path.join(base_dir, "data", "external", "spotify-2", "validate.csv"),
        index_col=0,
    ),
    **process_args
)
test_data = process_data(
    data=pd.read_csv(
        os.path.join(base_dir, "data", "external", "spotify-2", "test.csv"), index_col=0
    ),
    **process_args
)

train_data.to_csv(os.path.join(base_dir, "data", "processed", "spotify-2", "train.csv"))
val_data.to_csv(
    os.path.join(base_dir, "data", "processed", "spotify-2", "validate.csv")
)
test_data.to_csv(os.path.join(base_dir, "data", "processed", "spotify-2", "test.csv"))

print("Spotify 2 Processed Data saved to data/processed/spotify-2")

# spotify 2 best
train_data = process_data(
    data=pd.read_csv(
        os.path.join(base_dir, "data", "external", "spotify-2", "train.csv"),
        index_col=0,
    ),
    **best_process_args
)
val_data = process_data(
    data=pd.read_csv(
        os.path.join(base_dir, "data", "external", "spotify-2", "validate.csv"),
        index_col=0,
    ),
    **best_process_args
)
test_data = process_data(
    data=pd.read_csv(
        os.path.join(base_dir, "data", "external", "spotify-2", "test.csv"), index_col=0
    ),
    **best_process_args
)

train_data.to_csv(
    os.path.join(base_dir, "data", "processed", "spotify-2", "train_best.csv")
)
val_data.to_csv(
    os.path.join(base_dir, "data", "processed", "spotify-2", "validate_best.csv")
)
test_data.to_csv(
    os.path.join(base_dir, "data", "processed", "spotify-2", "test_best.csv")
)

print("Spotify 2 Best Processed Data saved to data/processed/spotify-2")