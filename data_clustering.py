import os
from utils.data_clustering_utils import *
from data_preprocessing import get_processed_data

"""
Author: Arvin Bayat Manesh  
Created: 2024-11-24 
Last Modified: 2024-11-24  

Description:  
This script performs clustering on a preprocessed dataset using the K-Means algorithm and visualizes the results.  
It includes the following functionalities:

- Obtaining the preprocessed dataset using the `get_processed_data` function, which includes both features (`X`) and target labels (`y`).
- Using the Elbow Method to compute the Within-Cluster Sum of Squares (WSS) for a range of cluster numbers to determine the optimal number of clusters for the dataset.
- Saving the Elbow Method plot (`WSS vs. Clusters`) in a specified directory for visualization.
- Performing K-Means clustering on the dataset with a user-specified number of clusters (e.g., `num_clusters = 4`).
- Reducing the dataset and the K-Means cluster centroids to two dimensions using Principal Component Analysis (PCA) for 2D visualization.
- Visualizing the clustering results, including data points and centroids, in a 2D scatter plot saved in a specified directory.

The script uses Scikit-Learn for clustering and dimensionality reduction, Pandas for data manipulation, and Matplotlib for visualization.  
It ensures the generated plots are saved to specified directories for analysis and review.

"""

# Obtain processed data before applying clustering algorithms
processed_data, y = get_processed_data()

# Perform the elbow method to obtain the ideal number of clusters for the data set
wss_df = perform_elbow_method(processed_data)
wss_vs_clusters_folder = "wss_vs_clusters"
os.makedirs("wss_vs_clusters", exist_ok=True)
plot_wss_vs_clusters(wss_df, wss_vs_clusters_folder)

# Perform the K-means clustering algorithm
num_clusters = 4 # Chosen after examining the wss_vs_clusters plot
KMeans = perform_kmeans(processed_data, num_clusters)

# Perform PCA on the dataset for clustering visualization
reduced_data, centroids_2d = perform_PCA_KMeans(processed_data, KMeans)

# Visualize the clustering
KMeans_plot_folder = "kMeans_clustering_plot"
os.makedirs("KMeans_clustering_plot", exist_ok=True)
plot_KMeans_clustering(reduced_data, KMeans.labels_, centroids_2d, KMeans_plot_folder)


