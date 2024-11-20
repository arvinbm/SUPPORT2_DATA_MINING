from ucimlrepo import fetch_ucirepo, list_available_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from pandas.api.types import is_numeric_dtype


def import_dataset():
    """
    Fetches and loads the SUPPORT2 dataset from the UCI Machine Learning Repository.

    This function uses the `fetch_ucirepo` function to download the SUPPORT2 dataset (ID: 880) 
    and extracts its features and targets into pandas DataFrames.

    Returns:
        X (pd.DataFrame): DataFrame containing the feature variables of the dataset.
        y (pd.DataFrame): DataFrame containing the target variables of the dataset.
    """
        
    # fetch dataset 
    support2 = fetch_ucirepo(id=880) 
  
    # data
    X = support2.data.features 
    y = support2.data.targets 

    return X, y

def print_num_missing_values(X):
    missing_values = X.isnull().sum()
    print("NUM OF MISSING VALUES FOR EACH COLUMN:")
    print(missing_values)
    print()

def drop_features_with_too_many_missing_values(X):
    """
    Removes columns from the DataFrame `X` that have more than 30% missing values.

    This function calculates the number of missing values for each column in the input DataFrame `X`.
    It then iterates through all columns and checks if the proportion of missing values exceeds 30%.
    If a column has more than 30% missing values, it drops that column in place, directly modifying
    the original DataFrame.

    Args:
        X (pd.DataFrame): The input DataFrame to be processed.

    Returns:
        List[str]: A list of column names that were dropped.
    """

    missing_values = X.isnull().sum()
    num_columns = X.shape[0]
    dropped_columns = []

    # Iterate through columns and drop those with more than 35% missing values
    for col in missing_values.index:
        if missing_values[col] / num_columns > 0.30:
            dropped_columns.append(col)
            X.drop(columns=[col], inplace=True)

    return dropped_columns


def impute_missing_values(X, y):
    """
    Imputes missing values in `X` using class labels from `death` and `hospdead` columns in `y`.

    - Numerical columns: Fill missing values with the mean within each group defined by (death, hospdead).
    - Categorical columns: Fill missing values with the mode within each group defined by (death, hospdead).

    Args:
        X (pd.DataFrame): Data with potential missing values.
        y (pd.DataFrame): Class labels containing 'death' and 'hospdead' columns.

    Returns:
        None: Modifies `X` in place.
    """

    for col in X.columns:
        # Checking if the column has at least one missing value
        if X[col].isna().any():
            if is_numeric_dtype(X[col]):
                # The case of numerical feature
                X[col] = X.groupby([y['death'], y['hospdead']])[col].transform(lambda group: group.fillna(group.mean()))
            else:
                # The case of categorical feature
                X[col] = X.groupby([y['death'], y['hospdead']])[col].transform(
                    lambda group: group.fillna(group.mode().iloc[0] if not group.mode().empty else None)
)
                
def print_num_unique_values(X):
    num_unique_counts = X.nunique()
    print("NUM OF UNIQUE VALUES IN EACH COLUMNS:")
    print(num_unique_counts)
    print()

def print_data_types(X):
    print("DATATYPES OF EACH COLUMN:")
    print(X.dtypes)
    print()

def perform_one_hot_encoding(X):
    """
    Performs one-hot encoding on all non-numeric columns in the DataFrame `X`.

    - For each non-numeric column:
        - If the column has more than one unique value, it is one-hot encoded using `pd.get_dummies()`.
        - The resulting one-hot encoded columns are added to `X`.
        - The original non-numeric column is removed from `X`.

    Args:
        X (pd.DataFrame): Input DataFrame containing numeric and non-numeric columns.

    Returns:
        pd.DataFrame: Modified DataFrame with non-numeric columns replaced by their one-hot encoded counterparts.
    """
    for col in X.columns:
        if not is_numeric_dtype(X[col]):
            num_unique_values = X[col].nunique()
            if num_unique_values > 1:
                # Perform one-hot encoding
                one_hot = pd.get_dummies(X[col], prefix=col).astype(int)
                X = pd.concat([X, one_hot], axis=1)
                X.drop(columns=[col], inplace=True)
    return X

def perform_z_score_normalization(X):
    return (X - X.mean()) / X.std()

def data_preprocessing(X, y):
    print_num_missing_values(X)
    dropped_columns = drop_features_with_too_many_missing_values(X)

    print("THE COLUMNS THAT WERE DROPPED DUE TO HAVING TOO MANY MISSING VALUES:")
    print(dropped_columns)
    print()

    impute_missing_values(X, y)
    print_num_unique_values(X)

    X = perform_one_hot_encoding(X)
    X = perform_z_score_normalization(X)
    print_data_types(X)
    print(X.head())



def main():
    # Importing the data set
    X, y = import_dataset()

    # Perform Imputation for missing values
    # X = X.iloc[:, 21:]
    data_preprocessing(X, y)
    



if __name__ == "__main__":
    main()