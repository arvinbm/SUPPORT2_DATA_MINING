import os
import yaml
import logging
from utils.outlier_detection_utils import *
from data_preprocessing import get_processed_data
from utils.logger_utils import setup_logger
from utils.func_utils import compute_and_mutual_information , plot_mutual_information
"""
 and saving the results of outlier detection.
"""

# Load config file
config_path = "featSel_config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Setup logger
log_file = config["log_file"]

# Load and preprocess data
processed_data, labels = get_processed_data(log_file)
y = labels['class_encoded']

logger = setup_logger(log_file)

print("Starting Feature selection script...")
logger.info("Starting Feature selection script...")

output_folder = config['histogram_folder'] 
os.makedirs(output_folder, exist_ok=True)

feature_importance = compute_and_mutual_information(processed_data, y)
plot_mutual_information(feature_importance, output_folder, output_filename="mutual_information_scores.png")


print("Feature selection script executed successfully. Logs saved in the specified folder.")
logger.info("Feature selection script executed successfully. Logs saved in the specified folder.")
