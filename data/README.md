# Directory Structure

This folder should contain all the datasets in the following directory structure:

```
.
├── external
│   ├── 2d-clustering-data.csv
│   ├── advertisement.csv
│   ├── diabetes.csv
│   ├── double_mnist
│   ├── fashion_mnist
│   ├── HousingData.csv
│   ├── linreg.csv
│   ├── recordings
│   ├── regularisation.csv
│   ├── self_recordings
│   ├── spotify-2
│   ├── spotify.csv
│   ├── WineQT.csv
│   └── word-embeddings.feather
├── interim
│   ├── README.md
│   └── spotify_vis.csv
├── operations
│   ├── preprocess.py
│   └── split_data.py
├── processed
│   ├── advertisement.csv
│   ├── binary_dataset.csv
│   ├── diabetes.csv
│   ├── fashion_mnist
│   ├── HousingData.csv
│   ├── linreg.csv
│   ├── regularisation.csv
│   ├── spotify-2
│   ├── spotify_best.csv
│   ├── spotify.csv
│   ├── spotify-reduced.csv
│   ├── WineQT.csv
│   ├── word-embeddings.csv
│   └── word-embeddings-reduced.csv
└── README.md
```

# Operations
There are few data operations related functions implemented in the `operations/` folder
1. `preprocess.py`: Contains functions for pre processing the dataset like removing colums, normalization, encoding, etc.
2. `split_data.py`: Contains function to split the entire dataset into training, validataion and testing sets.



