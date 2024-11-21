import logging
import pandas as pd
import os


"""
Author: Amr Sharafeldin
Created: 2024-11-20
Last Modified: 2024-11-20
Description: This script provides utility functions for setting up logging and saving feature summaries or dataframes during the preprocessing pipeline. 
It includes a logger setup for recording messages, feature logging for unique value analysis, and methods for saving DataFrames to CSV files with appropriate 
logging for traceability and error handling.
"""




def setup_logger(log_file):
    """
    Sets up a logger for logging messages to a specified file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("data_preprocessing")  
    logger.setLevel(logging.INFO)  


    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_file)  
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  
        file_handler.setFormatter(formatter)  
        logger.addHandler(file_handler)  

    return logger  


def save_feature_log(features, file_path, description, logger):
    """
    Logs unique value counts for features and saves to a CSV file.

    Args:
        features (pd.DataFrame): DataFrame of features to log.
        file_path (str): Path to save the CSV file.
        description (str): Description for logging.
        logger (logging.Logger): Logger instance for logging messages.
    """

    unique_values = features.nunique()

    feature_df = pd.DataFrame({
        "Columns": features.columns,
        "Unique Values": unique_values.values
    })

    save_dataframe(feature_df, file_path, logger, f"{description} logged")


def save_dataframe(dataframe, file_path, logger, description):
    """
    Saves a DataFrame to a CSV file and logs the operation.

    Args:
        dataframe (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the CSV file.
        logger (logging.Logger): Logger instance for logging messages.
        description (str): Description for logging.

    Raises:
        Exception: If saving the file fails.
    """
    try:
        dataframe.to_csv(file_path, index=False)
        logger.info(f"{description} saved to {file_path}")  
    except Exception as e:
        logger.error(f"Failed to save {description}: {e}")  
        raise
