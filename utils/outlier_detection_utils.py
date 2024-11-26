import os
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
Author: Arvin Bayat Manesh 
Created: 2024-11-25
Last Modified: 2024-11-25  

Description:  
This module provides utility functions for performing Isolation Forest-based outlier detection and visualizing the results.  
It includes implementations for applying the Isolation Forest algorithm, reducing data dimensionality using PCA,  
and plotting the results to help interpret outliers in the dataset.

The functionalities include:
- Applying Isolation Forest for outlier detection.
- Dimensionality reduction using PCA to prepare data for visualization.
- Plotting and saving the results of outlier detection for easy interpretation.
"""

def perform_isolation_forest(X, num_estimators):
    isolation_forest = IsolationForest(n_estimators=num_estimators, random_state=13, contamination='auto')
    isolation_forest.fit(X)
    predictions = isolation_forest.predict(X)
    return predictions

def perform_pca(X):
    pca = PCA(n_components=3)
    reduced_2d_dataset = pca.fit_transform(X)
    return reduced_2d_dataset

def plot_isolation_forest_results(predictions, reduced_3d_dataset, isolation_forest_folder):
    # plt.figure(figsize=(8, 6))
    # plt.scatter(reduced_2d_dataset[predictions == 1, 0], reduced_2d_dataset[predictions == 1, 1], c='blue', label='Inliers', alpha=0.5)
    # plt.scatter(reduced_2d_dataset[predictions == -1, 0], reduced_2d_dataset[predictions == -1, 1], c='red', label='Outliers', alpha=0.5)

    # plt.title("Isolation Forest: Outlier Detection Visualization (PCA-Reduced Data)")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.legend()
    # plt.grid(True)

    # # Save the graph
    # plot_path = os.path.join(isolation_forest_folder, "isolation_forest.png")
    # plt.savefig(plot_path)
    # plt.close()
    # print(f"Plot saved to {plot_path}")
        # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Separate inliers and outliers for plotting
    inliers = reduced_3d_dataset[predictions == 1]
    outliers = reduced_3d_dataset[predictions == -1]

    # Plot inliers
    ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], 
               c='blue', label='Inliers', alpha=0.5, s=20)

    # Plot outliers
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], 
               c='red', label='Outliers', alpha=0.5, s=20)

    # Set plot labels and title
    ax.set_title("Isolation Forest: Outlier Detection Visualization (3D Reduced Data)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend()

    # Save the plot
    plot_path = os.path.join(isolation_forest_folder, "isolation_forest_3d.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    print(f"3D Plot saved to {plot_path}")