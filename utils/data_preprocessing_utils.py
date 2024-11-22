from ucimlrepo import fetch_ucirepo, list_available_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from pandas.api.types import is_numeric_dtype
import os
from datetime import datetime

from .logger_utils import setup_logger , save_feature_log , save_dataframe




"""
Author: Arvin Bayat Manesh
Created: 2024-11-20
Last Modified: 2024-11-20
Description: This script provides a comprehensive preprocessing pipeline for the SUPPORT2 dataset from the UCI Machine Learning Repository. 
It includes functionalities for importing the dataset, analyzing feature characteristics (e.g., missing values, unique values, and data types), 
handling missing data through column removal or imputation, encoding categorical variables via one-hot encoding, normalizing numerical features, 
and preparing the dataset for analysis or machine learning tasks. Additionally, it offers logging capabilities to monitor and save preprocessing 
steps and results for traceability and debugging.
"""


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



def get_num_unique_values(X):
    """
    Getter: Calculates the number of unique values in each column of the DataFrame.

    Args:
        X (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: Series where each value represents the count of unique values in a column.
    """
    num_unique_counts = X.nunique()
    return num_unique_counts

def get_data_types(X):
    """
    Getter: Retrieves the data types of each column in the DataFrame.

    Args:
        X (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: Series where each value represents the data type of a column.
    """
    return X.dtypes

def get_num_missing_values(X):
    """
    Getter: Calculates the number of missing values in each column of the DataFrame.

    Args:
        X (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: Series where each value represents the count of missing values in a column.
    """
    missing_values = X.isnull().sum()
    return missing_values




def split_features_by_type(X):
    """
    Splits the features in the DataFrame into numeric and categorical subsets.

    Args:
        X (pd.DataFrame): Input DataFrame with mixed feature types.

    Returns:
        tuple: A tuple containing:
            - numeric_data (pd.DataFrame): DataFrame with numeric features.
            - categorical_data (pd.DataFrame): DataFrame with categorical features.
    """
    numeric_data = X.select_dtypes(include=['number'])
    categorical_data = X.select_dtypes(exclude=['number'])
    return numeric_data, categorical_data


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

    # Iterate through columns and drop those with more than 30% missing values
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


def handle_missing_values(X, y, log_dir=None, logger=None):

    missing_counts = X.isnull().sum()
    missing_df = pd.DataFrame({"Column": missing_counts.index, "Missing Values": missing_counts.values})

    if log_dir and logger:
        missing_values_path = os.path.join(log_dir, "missing_values.csv")
        save_dataframe(missing_df, missing_values_path, logger, "Missing values report")

    dropped_columns = drop_features_with_too_many_missing_values(X)
    if logger:
        if dropped_columns:
            logger.info(f"Dropped columns: {', '.join(dropped_columns)}")
        else:
            logger.info("No columns were dropped due to missing values.")

    impute_missing_values(X, y)
    if logger:
        logger.info("Missing values imputed successfully.")


def process_features(X, log_dir=None, logger=None):

    numeric_cols, categorical_cols = split_features_by_type(X)

    if log_dir and logger:
        numeric_columns_path = os.path.join(log_dir, "numeric_columns.csv")
        categorical_columns_path = os.path.join(log_dir, "categorical_columns.csv")
        save_feature_log(numeric_cols, numeric_columns_path, "Numeric columns", logger)
        save_feature_log(categorical_cols, categorical_columns_path, "Categorical columns", logger)

    # Exclude the features 'dementia' and 'diabetes' from normalization.
    # These features are categorical in nature, but in the original dataset
    # they had numeric values 0 and 1 to describe 'yes' or 'no', thus there 
    # is no need to normalize these two features.
    special_numeric_cols = numeric_cols[['dementia', 'diabetes']] if {'dementia', 'diabetes'}.issubset(numeric_cols.columns) else pd.DataFrame()
    numeric_cols_excluding_special_features = numeric_cols.drop(special_numeric_cols, axis=1)

    encoded_categorical_cols = perform_one_hot_encoding(categorical_cols)
    normalized_numeric_cols = perform_z_score_normalization(numeric_cols_excluding_special_features)

    # Add the special columns 'diabetes' and 'dementia' to the numeric columns
    normalized_numeric_cols = pd.concat([normalized_numeric_cols, special_numeric_cols], axis=1)

    # Combine the numerical and encoded categorical features
    processed_features = pd.concat([encoded_categorical_cols, normalized_numeric_cols], axis=1)
    
    if logger:
        logger.info("Features processed successfully.")
    return processed_features


def data_preprocessing(X, y, log_path=None, enable_logging=True):

    logger = None
    if enable_logging and log_path:
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, "execution_log.txt")
        logger = setup_logger(log_file)

    if logger:
        execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Execution started at: {execution_time}")

    handle_missing_values(X, y, log_path if enable_logging else None, logger)
    X_processed = process_features(X, log_path if enable_logging else None, logger)

    if logger:
        logger.info("Data preprocessing completed successfully.")
        print("Data preprocessing completed. Logs saved in the specified log folder.")
    return X_processed



