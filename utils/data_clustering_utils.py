import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
"""
Author: Arvin Bayat Manesh & Amr Sharafeldin
Created: 2024-11-24  
Last Modified: 2024-11-24  

Description:  
This script provides functionality for performing clustering on a dataset using the K-Means algorithm and visualizing the results.  
It includes the following functionalities:  

- Performing K-Means clustering on the dataset with a specified number of clusters.
- Reducing the dataset and K-Means centroids to two dimensions using Principal Component Analysis (PCA) for 2D visualization.
- Visualizing clustering results in 2D with labeled clusters and centroids.
- Computing the Within-Cluster Sum of Squares (WSS) for multiple cluster counts to determine the optimal number of clusters using the Elbow Method.
- Plotting the Elbow Curve to visualize WSS versus the number of clusters, helping to identify the "elbow point."

The script leverages Scikit-Learn for K-Means clustering and PCA, Pandas for data manipulation, and Matplotlib for visualization.  
It ensures that results can be saved to specified directories for further analysis.
"""

def perform_kmeans(X, num_clusters):
    """
    Performs K-Means clustering on the given dataset.

    This function initializes a K-Means clustering model with the specified 
    number of clusters and a fixed random state for reproducibility. It then 
    fits the model to the input dataset and returns the trained K-Means model.

    Parameters:
        X : The dataset on which to perform K-Means clustering. Each row 
        represents a data point, and each column represents a feature.
        num_clusters: The number of clusters to form in the K-Means algorithm.

    Returns:
        sklearn.cluster.KMeans: The trained K-Means model containing cluster 
                                labels, centroids, and other attributes.
    """
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=13)
    kmeans.fit(X)
    return kmeans

def perform_PCA_KMeans(X, Kmeans):
    """
    Performs dimensionality reduction using PCA on the dataset and reduces 
    the dimensions of K-Means centroids for consistent visualization.

    This function applies Principal Component Analysis (PCA) to reduce the 
    dimensions of the input dataset `X` to two dimensions for 2D visualization. 
    It also transforms the K-Means centroids into the same PCA-reduced space, 
    ensuring the centroids can be plotted alongside the data points.

    Parameters:
        X: The original dataset on which PCA is performed. Each row represents 
            a data point, and each column represents a feature.
        Kmeans: The trained K-Means model containing the cluster centroids.

    Returns:
        tuple:
            - reduced_data: The dataset reduced to two dimensions by PCA. Shape: (n_samples, 2).
            - centroids_2d: The centroids transformed into the same PCA-reduced 2D space. Shape: (n_clusters, 2).
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X)

    # Reduce the centroids dimensions to be plotted with data
    centroids_df = pd.DataFrame(Kmeans.cluster_centers_, columns=X.columns)
    centroids_2d = pca.transform(centroids_df)

    return reduced_data, centroids_2d

def plot_KMeans_clustering(reduced_data, Kmeans_labels, Kmeans_centroids_2d, KMeans_plot_folder):
    # Scatter plot of the 2D data
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=Kmeans_labels, cmap='viridis', s=50, alpha=0.7, label="Data Points"
    )

    # Plot the centroids
    plt.scatter(
            Kmeans_centroids_2d[:, 0], Kmeans_centroids_2d[:, 1],
            c='red', marker='X', s=200, label="Centroids"
    )

    plt.title("K-Means Clustering Visualization", fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster Label")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(KMeans_plot_folder, "KMeans_Clustering.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def perform_elbow_method(X):
    """
    https://www.kaggle.com/code/funxexcel/p2-sklearn-k-means-elbow-and-silhouette-method

    Computes the Within-Cluster Sum of Squares (WSS) for a range of cluster counts 
    to assist in determining the optimal number of clusters using the Elbow Method.

    This function iteratively performs K-Means clustering on the dataset for a specified 
    range of cluster counts (1 to 11 in this case). For each value of k (number of clusters), 
    it calculates the WSS (Within-Cluster Sum of Squares) using the `inertia_` attribute 
    of the trained K-Means model. The results are returned as a DataFrame containing the 
    cluster count and the corresponding WSS value.

    Parameters:
        X: The dataset on which to perform K-Means clustering. Each row represents a data 
           point, and each column represents a feature.

    Returns:
        pandas.DataFrame: A DataFrame with two columns:
                          - 'Clusters': The number of clusters (k) tested.
                          - 'WSS': The corresponding WSS values for each k.

    Usage:
        Use the resulting DataFrame to plot an Elbow Curve, where the x-axis represents 
        the number of clusters and the y-axis represents the WSS. The "elbow" point on 
        the curve indicates the optimal number of clusters.
    """
    clusters = range (1,12)
    wss = []
    for k in clusters:
        kmeans = perform_kmeans(X, k)
        wss_iter = kmeans.inertia_
        wss.append(wss_iter)

    wss_df = pd.DataFrame({'Clusters' : clusters, 'WSS' : wss})
    return wss_df

def plot_wss_vs_clusters(wss_df, wss_vs_clusters_folder):
    """
    Plots the Within-Cluster Sum of Squares (WSS) against the number of clusters
    and saves the plot to the specified folder.

    Parameters:
        wss_df: A DataFrame containing two columns:
                - 'Clusters': Number of clusters.
                - 'WSS': Corresponding WSS values.
        wss_vs_clusters_folder: The folder path where the plot will be saved.

    Returns:
        None
    """
    plot_path = os.path.join(wss_vs_clusters_folder, "WSS_vs_Clusters.png")

    # Plot WSS vs. number of clusters
    plt.figure(figsize=(8, 6))
    plt.plot(wss_df['Clusters'], wss_df['WSS'], marker='o', linestyle='--')
    plt.title('Elbow Method: WSS vs. Number of Clusters', fontsize=14)
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (WSS)', fontsize=12)
    plt.grid(True)

    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")



def perform_dbscan(data, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering.
    
    Parameters:
    - data: array-like, shape (n_samples, n_features), the input data.
    - eps: float, the maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: int, the number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - clustering: DBSCAN object with clustering results.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    clustering = db.fit(data)
    return clustering

def perform_PCA_DBSCAN(data, clustering):
    """
    Reduce data dimensions using PCA for clustering visualization.

    Parameters:
    - data: array-like, shape (n_samples, n_features), the input data.
    - clustering: DBSCAN object, the result of the DBSCAN clustering.

    Returns:
    - reduced_data: array-like, the PCA-reduced data.
    - core_samples_mask: boolean array, mask of core samples.
    - cluster_labels: array-like, cluster labels for each point.
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    cluster_labels = clustering.labels_
    
    return reduced_data, core_samples_mask, cluster_labels

def plot_DBSCAN_clustering(reduced_data, core_samples_mask, cluster_labels, output_folder):

    import numpy as np
    import matplotlib.pyplot as plt

    unique_labels = set(cluster_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(10, 7))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (cluster_labels == k)

        # Core samples
        xy = reduced_data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', 
                 markersize=10, alpha=0.8, label=f"Cluster {k}" if k != -1 else "Noise")

        # Non-core samples
        xy = reduced_data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', 
                 markersize=6, alpha=0.5)

    plt.title('DBSCAN Clustering Visualization (PCA-reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.legend(loc='best', title="Clusters")
    plt.savefig(f"{output_folder}/dbscan_clustering_improved.png")


def evaluate_clustering_using_silhouette_scores(models_to_be_evaluated, data, logger):
    results = {}

    for model in models_to_be_evaluated:
        model_name = type(model).__name__
        
        labels = model.labels_ if hasattr(model, 'labels_') else model.predict(data)
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(data, labels)
            logger.info("====================================")
            logger.info(f"Silhouette Score for {model_name}: {silhouette_avg:.4f}")
        else:
            silhouette_avg = None
            logger.info("====================================")
            logger.warning(f"Silhouette Score for {model_name} is not applicable (only one cluster found).")

        # Store the results
        results[model_name] = {
            "silhouette_score": silhouette_avg,
            "model": model
        }
        
    logger.info("====================================")
    return results