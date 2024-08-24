# Directory Structure

This folder should contain all the datasets in the following directory structure:

```
.
├── README.md
├── external
│   ├── linreg.csv
│   ├── regularisation.csv
│   ├── spotify-2
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── validate.csv
│   └── spotify.csv
├── interim
│   ├── README.md
│   └── spotify_vis.csv
├── operations
│   ├── preprocess.py
│   └── split_data.py
└── processed
    ├── linreg.csv
    ├── regularisation.csv
    ├── spotify-2
    │   ├── test.csv
    │   ├── test_best.csv
    │   ├── train.csv
    │   ├── train_best.csv
    │   ├── validate.csv
    │   └── validate_best.csv
    ├── spotify.csv
    └── spotify_best.csv
```

# Operations
There are few data operations related functions implemented in the `operations/` folder
1. `preprocess.py`: Contains functions for pre processing the dataset like removing colums, normalization, encoding, etc.
2. `split_data.py`: Contains function to split the entire dataset into training, validataion and testing sets.

# Datasets
1. Spotify Dataset
    - Raw dataset stored in `external/spotify.csv`
    - Datast with all the normialized numerical cloumns for visualization stored in `interim/spotify_vis.csv`
    - Datast with all the important columns for final processing is stored in `processed/spotify.csv`
    - Dataset with all the important columns along with columns with high mutual information score is stored in `processed/spotify_best.csv`
2. Spotify-2 Dataset
    - Raw dataset stored in `external/spotify-2/`
    - The normal processed dataset with the train, val, test split is stored in `processed/spotify-2/`
    - The best processed dataset with the train, val, test split is stored in `processed/spotify-2/`
3. Linear Regression Dataset
   - Raw dataset stored in `external/linreg.csv`
   - The same dataset is copied to `processed/linreg.csv`
4. Regularization Dataset
   - Raw dataset stored in `external/regularisation.csv`
   - The same dataset is copied to `processed/regularisation.csv`


