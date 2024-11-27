import pandas as pd
from sklearn.feature_selection import mutual_info_classif

import yaml

"""
Author: Arvin Bayat Manesh
Created: 2024-11-22
Last Modified: 2024-11-22
Description: This script provides a utility function to drop specified columns from a pandas DataFrame.
It is designed to validate input, handle cases where the specified columns do not exist in the DataFrame,
and optionally log the actions performed for traceability and debugging. The function can be used as part
of a preprocessing pipeline for datasets in machine learning or data analysis workflows.
"""

def drop_columns(df, columns_to_drop, logger=None):
    """
    Drops specified columns from a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (list): List of column names to drop.
        logger (optional): A logger instance for logging dropped columns (default is None).

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns removed.
    """

    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The provided input format must be a pandas DataFrame")
    if not isinstance(columns_to_drop, list):
        raise ValueError("The format of the columns to drop must be a list.")
    
    valid_columns_to_drop = []
    
    # Collect the columns that needs to be dropped
    for col in df.columns:
        if col in columns_to_drop:
            valid_columns_to_drop.append(col)
        
    # Dropped the collected columns
    df_dropped = df.drop(columns=valid_columns_to_drop)

    # If there was a column in columns_to_drop that was not in the DataFrame notify in the logger
    for col in columns_to_drop:
        if col not in df.columns:
            if logger:
                logger.warning(f"Attempted to drop non-existing column: {col}")
            else:
                print(f"Attempted to drop non-existing columns: {col}")
    
    if logger and valid_columns_to_drop:
        logger.info(f"Dropped columns: {valid_columns_to_drop}")

    return df_dropped




# Author: Amr Sharafeldin
# Created: 2024-11-24

def map_labels(row):
    """
    Maps a row of a multi-binary labeled DataFrame to a single class label based on specific conditions.

    Parameters:
        row (pd.Series): A row from a pandas DataFrame containing multiple binary label columns,
                         specifically 'death' and 'hospdead'.

    Returns:
        int: The mapped class label:
             - 0 if 'death' is 0 (Class 0: No death)
             - 1 if 'death' is not 0 but 'hospdead' is 0 (Class 1: Non-hospital-related death)
             - 2 if both 'death' and 'hospdead' are not 0 (Class 2: Hospital-related death)

    Notes:
        This function is used for reducing multi-binary labels to a single unified class label
        for classification tasks or other downstream applications.
    """
    if row['death'] == 0:
        return 0  # Class 0: No death
    elif row['hospdead'] == 0:
        return 1  # Class 1: Non-hospital-related death
    else:
        return 2  # Class 2: Hospital-related death

def load_config(config_path):
    """
    Loads a configuration file in YAML format and returns its contents as a Python dictionary.

    Parameters:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The parsed configuration data.

    Notes:
        This function is useful for dynamically configuring a script or model by reading key-value
        pairs from a YAML configuration file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)  # Safely loads the YAML file, avoiding execution of arbitrary code
    return config




def compute_and_mutual_information(X, y, random_state=42, plot=False):
    """
    Computes mutual information scores for features and optionally plots a histogram.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix. If a DataFrame is provided, feature names are retained.
        y (array-like): Target variable.
        random_state (int): Random state for reproducibility. Default is 42.
        plot (bool): Whether to plot a histogram of the mutual information scores. Default is False.

    Returns:
        pd.DataFrame: A sorted DataFrame with features and their mutual information scores.
    """
    from sklearn.feature_selection import mutual_info_classif
    import pandas as pd
    import matplotlib.pyplot as plt

    # Compute mutual information
    mi_scores = mutual_info_classif(X, y, random_state=random_state)

    # Use feature names if X is a DataFrame; otherwise, generate default names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Create a DataFrame for better interpretability
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mutual Information": mi_scores
    }).sort_values(by="Mutual Information", ascending=False)


    return feature_importance



def plot_mutual_information(feature_importance):
    """
    Plots a histogram of mutual information scores.

    Parameters:
        feature_importance (pd.DataFrame): DataFrame with feature names and mutual information scores.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance["Feature"], feature_importance["Mutual Information"], width=0.5)
    plt.xlabel("Features")
    plt.ylabel("Mutual Information Score")
    plt.title("Mutual Information Scores for Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()