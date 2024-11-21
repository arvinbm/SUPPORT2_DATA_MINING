import os
import pandas as pd
from utils.data_preprocessing_utils import import_dataset, data_preprocessing

# Define the log folder path
log_folder = "logs/data_preprocessing_log"

# Ensure the 'logs' directory exists
os.makedirs(log_folder, exist_ok=True)

# Import the dataset
X, y = import_dataset()

# Perform data preprocessing
processed_data = data_preprocessing(X, y, log_folder)

print(processed_data.head())
