This folder contains all the models you have implemented. For each model, there is a class for it in the `<model>.py` file. 

# Directory Structure

This folder should contain all the models in the following directory structure:
```
.
├── README.md
├── knn
│   └── knn.py
└── linear_regression
    └── linreg.py
```

# Models
1. `knn.py` contains the implementation of three versions of KNN with different optimization techniques
   - `KNNInitial`: Initial KNN implementation that uses for loops for computation of the result for each data point in the test set
   - `KNNBest`: Best KNN implementation that uses numpy default vectorization for faster computation
   - `KNN`: Most optimized KNN implementation that uses parallel processing for faster computation
2. `linear_regression.py` contains the implementation of linear regression with regularization
   - `LinearRegression`: Implementation of regression and regularization for any degree polynomial. 