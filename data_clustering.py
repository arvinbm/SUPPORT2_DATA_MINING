import os
import yaml
from utils.data_clustering_utils import *
from data_preprocessing import get_processed_data
from utils.logger_utils import *
import logging

"""
Author: Arvin Bayat Manesh  
Created: 2024-11-24 
Last Modified: 2024-11-24  

Description:  
This script performs clustering on a preprocessed dataset using both the K-Means and DBSCAN algorithms.  
It visualizes the results and saves them to specified directories as configured in the YAML file.

The functionalities include:
- Data preprocessing.
- K-Means clustering with the Elbow Method for determining the optimal number of clusters.
- DBSCAN clustering with configurable parameters.
- PCA for visualization of clusters in 2D.
- Saving plots of clustering results and metrics.

"""

# Load config file
config_path = "clu_config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Setup logger
log_file = config["log_file"]

# Load and preprocess data
processed_data, y = get_processed_data(log_file)


logger = setup_logger(log_file)


folders = [
    config["wss_vs_clusters_folder"],
    config["kmeans_plot_folder"],
    config["dbscan_plot_folder"]
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)



# K-Means Clustering
logging.info("Starting K-Means clustering.")
wss_df = perform_elbow_method(processed_data)
logging.info("Elbow method completed. Saving WSS vs. Clusters plot.")
plot_wss_vs_clusters(wss_df, config["wss_vs_clusters_folder"])

num_clusters = 4  # Chosen based on the elbow method plot (can be updated dynamically)
logging.info(f"Performing K-Means clustering with {num_clusters} clusters.")
KMeans = perform_kmeans(processed_data, num_clusters)

logging.info("Reducing dimensions using PCA for K-Means visualization.")
reduced_data_kmeans, centroids_2d = perform_PCA_KMeans(processed_data, KMeans)

logging.info("Visualizing K-Means clustering.")
plot_KMeans_clustering(
    reduced_data_kmeans, 
    KMeans.labels_, 
    centroids_2d, 
    config["kmeans_plot_folder"]
)

# DBSCAN Clustering
logging.info("Starting DBSCAN clustering.")
dbscan_params = config.get("dbscan_params", {"eps": 0.5, "min_samples": 5})
logging.info(f"DBSCAN parameters: {dbscan_params}")
DBSCAN_clustering = perform_dbscan(processed_data, **dbscan_params)

logging.info("Reducing dimensions using PCA for DBSCAN visualization.")
reduced_data_dbscan, core_samples_mask, cluster_labels = perform_PCA_DBSCAN(
    processed_data, 
    DBSCAN_clustering
)

logging.info("Visualizing DBSCAN clustering.")
plot_DBSCAN_clustering(
    reduced_data_dbscan, 
    core_samples_mask, 
    cluster_labels, 
    config["dbscan_plot_folder"]
)

logging.info("Clustering script completed successfully.")
