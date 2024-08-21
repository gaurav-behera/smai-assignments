import numpy as np
import pandas as pd


def _fill_null(data, column, value):
    """
    Fill null values in a column with a specified value
    """
    data[column]=data[column].fillna(value)
    return data


def _linear_normalize(data, column):
    """
    Normalize a column using linear normalization
    """
    data[column] = (data[column] - data[column].min()) / (
        data[column].max() - data[column].min()
    )
    return data


def _z_index_normalize(data, column):
    """
    Normalize a column using z-index normalization
    """
    data[column] = (data[column] - data[column].mean()) / data[column].std()
    return data


def _encode_boolean(data, column):
    """
    Encode a boolean column as 0 or 1
    """
    data[column] = data[column].astype(int)
    return data


def _label_encode(data, column):
    """
    Label encode a column
    """
    data[column] = data[column].astype("category").cat.codes
    return data


def _one_hot_encode(data, column):
    """
    One hot encode a column as integers 0 or 1
    """
    data = pd.get_dummies(data, columns=[column], prefix=[column], dtype='int')
    # data = pd.get_dummies(data, columns=[column], prefix=[column])
    
    return data


# call all the above functions based on parameters passed
def process_data(
    data,
    null_cols={},
    label_encode=[],
    one_hot_encode=[],
    boolean_encode=[],
    linear_norm=[],
    z_index_norm=[],
):
    """
    Process the data based on the parameters passed
    Parameters:
    data : pandas.DataFrame
        The input data
    null_cols : dict
        A dictionary with column names as keys and values to fill nulls with as values
    label_encode : list
        A list of columns to label encode
    one_hot_encode : list
        A list of columns to one hot encode
    boolean_encode : list
        A list of columns to encode as boolean
    linear_norm : list
        A list of columns to linear normalize
    z_index_norm : list
        A list of columns to z-index normalize
    """
    if null_cols:
        for column, value in null_cols.items():
            data = _fill_null(data, column, value)
    for column in label_encode:
        data = _label_encode(data, column)
    for column in one_hot_encode:
        data = _one_hot_encode(data, column)
    for column in boolean_encode:
        data = _encode_boolean(data, column)
    for column in linear_norm:
        data = _linear_normalize(data, column)
    for column in z_index_norm:
        data = _z_index_normalize(data, column)
    return data
