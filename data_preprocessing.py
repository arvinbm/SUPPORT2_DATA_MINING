import os
import pandas as pd
from utils.data_preprocessing_utils import import_dataset, data_preprocessing , process_labels
from sklearn.preprocessing import LabelEncoder
def get_processed_data(log_file=None):

    import os

    # Ensure the directory for the log file exists, if provided
    if log_file:
        log_folder = os.path.dirname(log_file)
        os.makedirs(log_folder, exist_ok=True)

    # Import the dataset
    X, y = import_dataset()

    # Perform data preprocessing, passing the log file for logging if provided
    processed_data = data_preprocessing(X, y, log_file if log_file else None)
    labels = process_labels(y)

    return processed_data, labels
