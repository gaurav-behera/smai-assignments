# Statistical Methods in Artificial Intelligence (Monsoon '24)

This repository contains the assignments for the SMAI course. Each assignment is organized into its own directory and contains various tasks related to machine learning models, data processing, and performance evaluation.

## Directory Structure

```
.
├── assignments
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   └── 5
├── data
│   ├── external
│   ├── interim
│   ├── operations
│   └── processed
├── models
│   ├── AutoEncoders
│   ├── cnn
│   ├── gmm
│   ├── HMM
│   ├── KDE
│   ├── k_means
│   ├── knn
│   ├── linear_regression
│   ├── MLP
│   ├── pca
│   └── RNN
└── performance_measures
```

## Assignments

### Assignment 1

This assignment focuses on K-Nearest Neighbors (KNN) and Linear Regression with Regularization.

- **KNN Models**: Implementations of KNN with different optimization techniques.
- **Linear Regression**: Analysis of linear regression models with and without regularization.
- **Data Visualization**: Visualizing the training, validation, and test data.

### Assignment 2

This assignment covers clustering techniques and dimensionality reduction.

- **K-Means Clustering**: Implementation and testing of K-Means clustering.
- **Gaussian Mixture Models (GMM)**: Implementation and testing of GMM.
- **Principal Component Analysis (PCA)**: Dimensionality reduction using PCA.

### Assignment 3

This assignment focuses on Multilayer Perceptron (MLP) models for classification and regression.

- **Single Label Classification**: MLP for single label classification tasks.
- **Multi-label Classification**: MLP for multi-label classification tasks.
- **Regression**: MLP for regression tasks.

### Assignment 4

This assignment involves Convolutional Neural Networks (CNN) for classification and regression.

- **CNN Implementation**: Implementation of CNN for image classification and regression tasks.
- **Hyperparameter Tuning**: Tuning hyperparameters for optimal performance.

### Assignment 5

This assignment covers advanced models like Kernel Density Estimation (KDE) and Hidden Markov Models (HMM).

- **KDE**: Implementation and testing of KDE.
- **HMM**: Implementation and testing of HMM for sequence data.

## Data

The [data](data)  directory contains various datasets used in the assignments. The datasets are organized into the following subdirectories:

- **external**: Raw datasets.
- **interim**: Intermediate datasets used for visualization and analysis.
- **operations**: Scripts for data preprocessing and splitting.
- **processed**: Processed datasets ready for model training and evaluation.

## Models

The [models](models) directory contains implementations of various machine learning models used in the assignments. The models are organized into the following subdirectories:

- **AutoEncoders**: Autoencoder models for dimensionality reduction.
- **cnn**: Convolutional Neural Network models.
- **gmm**: Gaussian Mixture Models.
- **HMM**: Hidden Markov Models.
- **KDE**: Kernel Density Estimation models.
- **k_means**: K-Means clustering models.
- **knn**: K-Nearest Neighbors models.
- **linear_regression**: Linear Regression models.
- **MLP**: Multilayer Perceptron models.
- **pca**: Principal Component Analysis models.
- **RNN**: Recurrent Neural Network models.

## Performance Measures

The [performance_measures](performance_measures) directory contains implementations of various evaluation metrics used to assess model performance. The metrics include accuracy, precision, recall, F1-score, mean squared error, standard deviation, and variance.
