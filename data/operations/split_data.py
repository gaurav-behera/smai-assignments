import numpy as np
import time


def split_data(data, target_column, ratio=[0.8, 0.1, 0.1]):
    """
    Split the data into training, validation and testing sets

    Parameters
    ----------
    data : pandas.DataFrame
        The input data
    target_column : str
        The target column
    ratio : list
        A list of ratios to split the data into training, validation and testing sets

    Returns
    -------
    dict : A dictionary with the following keys:
        - 'trainX': A DataFrame containing the training features.
        - 'trainY': A Series containing the training target values.
        - 'valX': A DataFrame containing the validation features.
        - 'valY': A Series containing the validation target values.
        - 'testX': A DataFrame containing the testing features.
        - 'testY': A Series containing the testing target values.
    """
    # shuffle data
    data = data.sample(frac=1)
    size = data.shape[0]

    # split the data into features and target
    X = data.drop(columns=[target_column], inplace=False)
    y = data[target_column]

    # get split indices
    train_count = int(ratio[0] * size)
    test_count = int(ratio[2] * size)
    validation_count = size - train_count - test_count

    # np.random.seed(int(time.time()))
    all_idx = np.random.permutation(size)
    train_idx = all_idx[:train_count]
    validation_idx = all_idx[train_count : train_count + validation_count]
    test_idx = all_idx[train_count + validation_count :]

    # split the data
    split_data = {
        "trainX": X.iloc[train_idx],
        "trainY": y.iloc[train_idx],
        "valX": X.iloc[validation_idx],
        "valY": y.iloc[validation_idx],
        "testX": X.iloc[test_idx],
        "testY": y.iloc[test_idx],
    }
    
    # return the split data
    return split_data
