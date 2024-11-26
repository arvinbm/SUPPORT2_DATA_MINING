import os
import yaml
import logging
from utils.outlier_detection_utils import *
from data_preprocessing import get_processed_data
from utils.logger_utils import setup_logger

"""
Author: Arvin Bayat Manesh
Created: 2024-11-25 
Last Modified: 2024-11-25  

Description:  
This script performs outlier detection on a preprocessed dataset using the Isolation Forest method.  
It includes functionalities for data preprocessing, applying the Isolation Forest algorithm, and visualizing the results in 2D.  
The script also saves the output and visualizations to directories specified in a YAML configuration file.

The functionalities include:
- Data preprocessing.
- Isolation Forest outlier detection with configurable parameters.
- Dimensionality reduction using PCA for visualization.
- Plotting and saving the results of outlier detection.
"""

# Load config file
config_path = "out_config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Setup logger
log_file = config["log_file"]

# Load and preprocess data
processed_data, y = get_processed_data(log_file)

logger = setup_logger(log_file)

folders = [
    config["isolation_forest_folder"]
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Isolation Forest Outlier Detection
logger.info("Starting the Isolation Forest Outlier Detection Method.")
isolation_forest_predictions = perform_isolation_forest(processed_data, config["isolation_forest_params"]["n_estimators"])

# Reducing the dimensionality of the dataset for visualizing
logger.info("Performing PCA on the dataset to prepare for visualization.")
reduced_2d_dataset = perform_pca(processed_data)

# Plotting the result of isolation forest outlier detection method
logger.info("Plotting the results of the isolation forest outlier detection methods")
plot_isolation_forest_results(isolation_forest_predictions, reduced_2d_dataset, config["isolation_forest_folder"])

