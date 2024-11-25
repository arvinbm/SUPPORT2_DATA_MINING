import os
import pandas as pd
from utils.data_preprocessing_utils import import_dataset, data_preprocessing

def get_processed_data(log_file):
    """
    Runs the data preprocessing pipeline and returns the processed data.
    
    Parameters:
        log_file (str): Path to the logger file to log the preprocessing steps.

    Returns:
        pd.DataFrame: The processed data.
    """
    import os

    # Ensure the directory for the log file exists
    log_folder = os.path.dirname(log_file)
    os.makedirs(log_folder, exist_ok=True)

    # Import the dataset
    X, y = import_dataset()

    # Perform data preprocessing, passing the log file for logging
    processed_data = data_preprocessing(X, y, log_file)

    return processed_data, y
