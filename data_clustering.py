import os
import yaml
from utils.data_clustering_utils import *
from data_preprocessing import get_processed_data
from utils.logger_utils import *
import logging

"""
Author: Arvin Bayat Manesh & Amr Sharafeldin
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
logger.info("Starting K-Means clustering.")
wss_df = perform_elbow_method(processed_data)
logger.info("Elbow method completed. Saving WSS vs. Clusters plot.")
plot_wss_vs_clusters(wss_df, config["wss_vs_clusters_folder"])

num_clusters = 2  # Chosen based on the elbow method plot (can be updated dynamically)
logger.info(f"Performing K-Means clustering with {num_clusters} clusters.")
KMeans = perform_kmeans(processed_data, num_clusters)

logger.info("Reducing dimensions using PCA for K-Means visualization.")
reduced_data_kmeans, centroids_2d = perform_PCA_KMeans(processed_data, KMeans)

logger.info("Visualizing K-Means clustering.")
plot_KMeans_clustering(
    reduced_data_kmeans, 
    KMeans.labels_, 
    centroids_2d, 
    config["kmeans_plot_folder"]
)

# Perform grid search for hyper tuning DBSCAN parameters
# best_score, best_params = perform_grid_search(processed_data, logger) # The best parameters are used in clu_config.yaml for DBSCAN
# print(best_score)
# print(best_params)

# DBSCAN Clustering
logger.info("Starting DBSCAN clustering.")
dbscan_params = config.get("dbscan_params", {"eps": 0.5, "min_samples": 5})
logger.info(f"DBSCAN parameters: {dbscan_params}")
DBSCAN_clustering = perform_dbscan(processed_data, **dbscan_params)

logger.info("Reducing dimensions using PCA for DBSCAN visualization.")
reduced_data_dbscan, core_samples_mask, cluster_labels = perform_PCA_DBSCAN(
    processed_data, 
    DBSCAN_clustering
)

logger.info("Visualizing DBSCAN clustering.")
plot_DBSCAN_clustering(
    reduced_data_dbscan, 
    core_samples_mask, 
    cluster_labels, 
    config["dbscan_plot_folder"]
)

# Evaluate the clustering performance using the Silhouette scores
logger.info("Extracting the silhouette scores for clustering models.")
models_to_be_evaluated = [KMeans, DBSCAN_clustering]
results = evaluate_clustering_using_silhouette_scores(models_to_be_evaluated, processed_data, logger)

logger.info("Clustering script completed successfully.")
print("Data clustering executed successfully. Logs saved in the specified log folder.")
