import pandas as pd

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
