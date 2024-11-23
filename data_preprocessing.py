import os
import pandas as pd
from utils.data_preprocessing_utils import import_dataset, data_preprocessing

def get_processed_data():
    """
    Runs the data preprocessing pipeline and returns the processed data.
    
    Returns:
        pd.DataFrame: The processed data.
    """

    # Define the log folder path
    log_folder = "logs/data_preprocessing_log"

    # Ensure the 'logs' directory exists
    os.makedirs(log_folder, exist_ok=True)

    # Import the dataset
    X, y = import_dataset()

    # Perform data preprocessing
    processed_data = data_preprocessing(X, y, log_folder)

    return processed_data, y
